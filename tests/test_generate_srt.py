"""Property tests for end-to-end SRT generation.

These build realistic segments (aligned word timings) and assert *invariants*
over the produced SRT — well-formedness, readability, synchronization, reading
speed, speaker labelling, and robustness — rather than a captured byte string.
They import only ``pysbd`` + the pure subtitle module, so they run without the
GPU/torch/whisperx stack.
"""

import re

import pytest

from whisperx_subtitles.config import (
    MAX_CPS,
    MAX_LINE_LENGTH,
    MAX_LINES,
    MIN_DURATION,
)
from whisperx_subtitles.subtitles import generate_srt

# A strict arrow line: HH:MM:SS,mmm --> HH:MM:SS,mmm with seconds in 00-59.
TIMESTAMP_RE = r"\d{2}:[0-5]\d:[0-5]\d,\d{3}"
ARROW_LINE_RE = re.compile(rf"^(?P<start>{TIMESTAMP_RE}) --> (?P<end>{TIMESTAMP_RE})$")

# Floating-point slack: timestamps are rendered to millisecond precision, so the
# parsed-back values can differ from the engine's internal floats by up to ~1ms.
EPSILON = 0.01


# --------------------------------------------------------------------------- #
# Input builders
# --------------------------------------------------------------------------- #
def _mk_words(text, t0=0.0, dur=0.4, gap=0.05, speaker=None):
    """Build aligned Word dicts with monotonically increasing timings."""
    words = []
    t = t0
    for w in text.split():
        word = {"word": w, "start": round(t, 3), "end": round(t + dur, 3), "score": 0.9}
        if speaker is not None:
            word["speaker"] = speaker
        words.append(word)
        t = t + dur + gap
    return words, t


JFK_TEXT = "And so my fellow Americans ask not what your country can do for you"
# Same line, but punctuated: the mid-sentence commas are the natural break points
# a good wrapper should prefer over splitting mid-clause.
JFK_PUNCTUATED = (
    "And so, my fellow Americans, ask not what your country can do for you."
)
LONG_TEXT = (
    "The quick brown fox jumps over the lazy dog while the sleepy cat watches "
    "from the windowsill and the birds sing in the bright morning sunshine."
)


def _segment(text, t0=0.0, speaker=None):
    words, _ = _mk_words(text, t0, speaker=speaker)
    return {"text": text, "words": words}


def _two_segments(speaker=None):
    """Two back-to-back segments with continuous, increasing timings."""
    w1, t = _mk_words(JFK_TEXT, 0.0, speaker=speaker)
    w2, _ = _mk_words(LONG_TEXT, t + 0.3, speaker=speaker)
    return [
        {"text": JFK_TEXT, "words": w1},
        {"text": LONG_TEXT, "words": w2},
    ]


# --------------------------------------------------------------------------- #
# SRT parsing helpers
# --------------------------------------------------------------------------- #
def _ts_to_seconds(ts):
    """Parse 'HH:MM:SS,mmm' to float seconds."""
    hms, millis = ts.split(",")
    hours, minutes, secs = hms.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + int(secs) + int(millis) / 1000.0


class Block:
    __slots__ = ("index", "start", "end", "text_lines")

    def __init__(self, index, start, end, text_lines):
        self.index = index
        self.start = start
        self.end = end
        self.text_lines = text_lines

    @property
    def text(self):
        return "\n".join(self.text_lines)

    @property
    def duration(self):
        return self.end - self.start


def _parse_srt(srt):
    """Parse a full SRT string into a list of Block objects.

    Asserts each block is structurally well-formed (numeric index, valid arrow
    line, at least one non-empty text line) as it goes.
    """
    blocks = []
    raw_blocks = [b for b in srt.strip().split("\n\n") if b.strip()]
    for raw in raw_blocks:
        lines = raw.splitlines()
        assert len(lines) >= 3, f"block too short: {raw!r}"
        assert lines[0].isdigit(), f"non-numeric index line: {lines[0]!r}"
        arrow = ARROW_LINE_RE.match(lines[1])
        assert arrow, f"bad arrow line: {lines[1]!r}"
        text_lines = lines[2:]
        assert any(line.strip() for line in text_lines), f"empty cue text: {raw!r}"
        blocks.append(
            Block(
                index=int(lines[0]),
                start=_ts_to_seconds(arrow.group("start")),
                end=_ts_to_seconds(arrow.group("end")),
                text_lines=text_lines,
            )
        )
    return blocks


def _strip_speaker(text):
    """Drop a leading ``[SPEAKER_xx] `` prefix from the first line, if present."""
    return re.sub(r"^\[[^\]]+\]\s*", "", text)


# --------------------------------------------------------------------------- #
# Well-formedness
# --------------------------------------------------------------------------- #
def test_blocks_sequentially_numbered_from_one():
    blocks = _parse_srt(generate_srt(_two_segments(), "en"))
    assert len(blocks) >= 2
    assert [b.index for b in blocks] == list(range(1, len(blocks) + 1))


def test_arrow_lines_and_seconds_never_60():
    # The regex itself constrains seconds to 00-59; _parse_srt enforces it.
    srt = generate_srt(_two_segments(), "en")
    for ts in re.findall(r"\d{2}:\d{2}:\d{2},\d{3}", srt):
        secs = int(ts.split(",")[0].split(":")[2])
        assert 0 <= secs <= 59, f"invalid seconds field in {ts!r}"
    _parse_srt(srt)  # also asserts strict arrow-line regex


def test_block_ends_with_blank_line():
    assert generate_srt([_segment(JFK_TEXT)], "en").endswith("\n\n")


# --------------------------------------------------------------------------- #
# Readability
# --------------------------------------------------------------------------- #
def test_line_length_within_limit():
    for blocks in (
        _parse_srt(generate_srt(_two_segments(), "en")),
        _parse_srt(generate_srt([_segment(LONG_TEXT)], "en")),
    ):
        for b in blocks:
            for line in b.text_lines:
                assert len(line) <= MAX_LINE_LENGTH, (
                    f"line over {MAX_LINE_LENGTH} chars: {line!r} ({len(line)})"
                )


def test_max_lines_per_cue():
    for blocks in (
        _parse_srt(generate_srt(_two_segments(), "en")),
        _parse_srt(generate_srt([_segment(LONG_TEXT)], "en")),
    ):
        for b in blocks:
            assert len(b.text_lines) <= MAX_LINES, (
                f"cue has {len(b.text_lines)} lines (max {MAX_LINES}): {b.text!r}"
            )


# --------------------------------------------------------------------------- #
# Linguistic line breaking (#1)
# --------------------------------------------------------------------------- #
def test_line_break_lands_at_punctuation_not_mid_clause():
    # The JFK line wraps to two lines; with a mid-sentence comma available, the
    # wrapper should break right after it (a natural clause boundary) rather than
    # in the middle of a clause. Slower per-word timings keep it a single cue
    # whose text must wrap, exercising the line-break scorer end-to-end.
    words, _ = _mk_words(JFK_PUNCTUATED, 0.0, dur=0.6)
    blocks = _parse_srt(generate_srt([{"text": JFK_PUNCTUATED, "words": words}], "en"))
    # Find the (single) cue carrying this sentence and assert it is two lines
    # whose first line ends at the comma.
    two_line = [b for b in blocks if len(b.text_lines) == 2]
    assert two_line, f"expected a wrapped 2-line cue, got {[b.text for b in blocks]}"
    b = two_line[0]
    assert b.text_lines[0].rstrip().endswith(","), (
        f"first line should break at the comma, got {b.text_lines[0]!r}"
    )
    assert b.text_lines[0].rstrip() == "And so, my fellow Americans,", (
        f"unexpected break point: {b.text_lines[0]!r}"
    )


# --------------------------------------------------------------------------- #
# Synchronization
# --------------------------------------------------------------------------- #
def test_timings_ordered_and_non_overlapping():
    blocks = _parse_srt(generate_srt(_two_segments(), "en"))
    for b in blocks:
        assert b.end >= b.start - EPSILON, f"end before start: {b.text!r}"
    for prev, nxt in zip(blocks, blocks[1:], strict=False):
        assert nxt.start >= prev.start - EPSILON, "cue starts went backwards"
        assert prev.end <= nxt.start + EPSILON, f"cue overlap: {prev.end} > {nxt.start}"


# --------------------------------------------------------------------------- #
# Reading speed
# --------------------------------------------------------------------------- #
def _assert_cps(blocks, max_cps, slack=0.5):
    for b in blocks:
        chars = len(b.text.replace("\n", ""))
        duration = b.duration
        # Cues pinned at the MIN_DURATION floor can legitimately exceed max_cps:
        # the engine guarantees readability OR the floor, not both.
        if duration <= MIN_DURATION + EPSILON:
            continue
        cps = chars / duration if duration > 0 else float("inf")
        assert cps <= max_cps + slack, (
            f"cps {cps:.2f} > {max_cps} for {b.text!r} (dur={duration:.3f})"
        )


def test_reading_speed_within_limit():
    _assert_cps(_parse_srt(generate_srt(_two_segments(), "en")), MAX_CPS)


# --------------------------------------------------------------------------- #
# Speaker labels
# --------------------------------------------------------------------------- #
def test_speaker_prefix_present_when_diarized():
    segments = _two_segments(speaker="SPEAKER_00")
    blocks = _parse_srt(generate_srt(segments, "en"))
    for b in blocks:
        assert b.text_lines[0].startswith("[SPEAKER_00] "), (
            f"missing speaker prefix: {b.text!r}"
        )
    # Once the prefix is stripped, the readability invariants still hold per line.
    for b in blocks:
        first = _strip_speaker(b.text_lines[0])
        assert len(first) <= MAX_LINE_LENGTH


def test_speaker_prefix_counts_toward_reading_time():
    # Reading-time accuracy (#4): the on-screen ``[SPEAKER_xx] `` prefix is real
    # text the viewer must read, so normalize_cues counts it toward the CPS floor.
    # The SAME short, fast-spoken line therefore stays on screen LONGER when it
    # carries a speaker label than when it doesn't (~13 extra chars of budget).
    #
    # Timings are tight (dur small) and min_duration is lowered so the cue is
    # reading-time-bound, not pinned at the MIN_DURATION floor -- otherwise both
    # would hit the same floor and the prefix's effect would be invisible.
    text = "The quick brown fox jumps over the lazy dog now"

    plain_words, _ = _mk_words(text, 0.0, dur=0.2, gap=0.02)
    diar_words, _ = _mk_words(text, 0.0, dur=0.2, gap=0.02, speaker="SPEAKER_00")

    plain = _parse_srt(
        generate_srt([{"text": text, "words": plain_words}], "en", min_duration=0.5)
    )
    diar = _parse_srt(
        generate_srt([{"text": text, "words": diar_words}], "en", min_duration=0.5)
    )

    # The short line stays a single cue in both runs, so we can compare directly.
    assert len(plain) == 1 and len(diar) == 1, (
        f"expected one cue each; got plain={len(plain)} diar={len(diar)}"
    )
    p, d = plain[0], diar[0]

    # Sanity: both are reading-time-bound, not stuck at the lowered min_duration.
    assert p.duration > 0.5 + EPSILON and d.duration > 0.5 + EPSILON, (
        f"cues hit min_duration floor (plain={p.duration}, diar={d.duration}); "
        "the prefix effect would be unobservable"
    )
    assert d.text_lines[0].startswith("[SPEAKER_00] ")

    # The prefix adds reading-time budget, so the diarized cue is on screen longer.
    assert d.duration > p.duration + EPSILON, (
        f"speaker prefix did not extend reading time: diar={d.duration:.3f} "
        f"!> plain={p.duration:.3f}"
    )


def test_no_speaker_prefix_when_undiarized():
    blocks = _parse_srt(generate_srt(_two_segments(), "en"))
    for b in blocks:
        assert not re.match(r"^\[[^\]]+\]\s", b.text_lines[0]), (
            f"unexpected speaker prefix: {b.text!r}"
        )


# --------------------------------------------------------------------------- #
# Robustness
# --------------------------------------------------------------------------- #
def test_english_produces_valid_srt():
    blocks = _parse_srt(generate_srt([_segment(JFK_TEXT)], "en"))
    assert blocks
    lowered = "\n".join(b.text for b in blocks).lower()
    for word in JFK_TEXT.split():
        assert word.lower() in lowered, f"missing spoken word {word!r}"


def test_unsupported_language_falls_back_to_valid_srt():
    # pysbd raises for an unknown language; generate_srt catches it and falls
    # back to a regex sentence split. Output must still be valid SRT.
    blocks = _parse_srt(generate_srt(_two_segments(), "zz-not-a-real-language"))
    assert len(blocks) >= 2
    assert [b.index for b in blocks] == list(range(1, len(blocks) + 1))


def test_empty_segments_returns_empty_string():
    assert generate_srt([], "en") == ""


# --------------------------------------------------------------------------- #
# Global invariants re-confirmed over the new (punctuated / diarized) inputs
# --------------------------------------------------------------------------- #
def _assert_global_invariants(blocks):
    """Lines fit, cues stay within max_lines, and timings are ordered, valid,
    and non-overlapping. (_parse_srt already enforces the strict arrow-line
    timestamp regex, incl. seconds in 00-59, as it parses.)

    Line-length is checked against the wrapped *text body*: the wrapper sizes the
    body, and the optional ``[SPEAKER_xx] `` prefix is rendered on top of the
    first line (so it may push that one line past MAX_LINE_LENGTH). This matches
    the existing test_speaker_prefix_present_when_diarized convention.
    """
    assert blocks
    for b in blocks:
        for i, line in enumerate(b.text_lines):
            body = _strip_speaker(line) if i == 0 else line
            assert len(body) <= MAX_LINE_LENGTH, (
                f"line over {MAX_LINE_LENGTH}: {body!r} ({len(body)})"
            )
        assert len(b.text_lines) <= MAX_LINES, f"too many lines: {b.text!r}"
        assert b.end >= b.start - EPSILON, f"end before start: {b.text!r}"
    assert [b.index for b in blocks] == list(range(1, len(blocks) + 1))
    for prev, nxt in zip(blocks, blocks[1:], strict=False):
        assert nxt.start >= prev.start - EPSILON, "cue starts went backwards"
        assert prev.end <= nxt.start + EPSILON, f"cue overlap: {prev.end}>{nxt.start}"


def test_global_invariants_hold_for_punctuated_and_diarized_inputs():
    punct_words, _ = _mk_words(JFK_PUNCTUATED, 0.0, dur=0.6)
    diar = _two_segments(speaker="SPEAKER_00")
    for srt in (
        generate_srt([{"text": JFK_PUNCTUATED, "words": punct_words}], "en"),
        generate_srt(diar, "en"),
    ):
        _assert_global_invariants(_parse_srt(srt))


# --------------------------------------------------------------------------- #
# Custom params
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("max_line_length", [20, 30])
def test_custom_max_line_length_respected(max_line_length):
    srt = generate_srt([_segment(LONG_TEXT)], "en", max_line_length=max_line_length)
    blocks = _parse_srt(srt)
    for b in blocks:
        for line in b.text_lines:
            assert len(line) <= max_line_length, (
                f"line over {max_line_length}: {line!r} ({len(line)})"
            )


def test_custom_low_max_cps_still_lengthens_cues():
    # A lower max_cps should never make cues *shorter*: it raises each cue's
    # reading-comfort target, so total on-screen time can only grow (or stay
    # equal where other caps bind). This holds even though the strict ceiling
    # does not (see the xfail test below).
    default = _parse_srt(generate_srt(_two_segments(), "en"))
    slow = _parse_srt(generate_srt(_two_segments(), "en", max_cps=10.0))
    assert len(slow) == len(default)
    for d, s in zip(default, slow, strict=True):
        assert s.duration >= d.duration - EPSILON, (
            f"lower max_cps shortened a cue: {s.text!r} {s.duration} < {d.duration}"
        )


@pytest.mark.xfail(
    reason=(
        "KNOWN LIMITATION in normalize_cues (reported, not fixed): the CPS "
        "ceiling is a best-effort target, not a hard guarantee. When max_cps is "
        "set low, the MAX_LEAD_OUT cap (a cue may not linger >1.5s past its last "
        "spoken word) and the no-overlap-with-next-cue rule both take priority "
        "over extending the cue to hit the CPS target. With max_cps=10.0 on the "
        "LONG segment, e.g. 'while the sleepy cat / watches from the windowsill' "
        "(47 chars) is squeezed to ~3.52s -> ~13.4 CPS. The default-max_cps path "
        "(test_reading_speed_within_limit) does hold."
    ),
    strict=True,
)
def test_custom_max_cps_respected():
    max_cps = 10.0
    blocks = _parse_srt(generate_srt(_two_segments(), "en", max_cps=max_cps))
    _assert_cps(blocks, max_cps)


# --------------------------------------------------------------------------- #
# Injectable sentence segmenter (segment_fn) — the hook the GPU side uses to
# plug in SaT/wtpsplit for Thai/CJK without importing torch into this module.
# (Cue-boundary precision is unit-tested against split_at_sentence_end in
# test_cues.py; here we assert end-to-end validity and that the hook is used.)
# --------------------------------------------------------------------------- #
def test_segment_fn_path_produces_valid_srt_for_unsupported_language():
    # With segment_fn supplied, pysbd is never constructed, so an unsupported
    # language (Thai has no pysbd support) must still yield valid SRT, no crash.
    thai = "หนึ่ง สอง สาม สี่ ห้า หก เจ็ด แปด เก้า สิบ"
    words, _ = _mk_words(thai)
    called = []

    def segment_fn(t):
        called.append(t)
        return [p for p in t.split("  ") if p.strip()] or [t]

    blocks = _parse_srt(
        generate_srt([{"text": thai, "words": words}], "th", segment_fn=segment_fn)
    )
    assert called, "segment_fn was not invoked"
    assert blocks
    assert [b.index for b in blocks] == list(range(1, len(blocks) + 1))


def test_no_segment_fn_keeps_pysbd_behaviour():
    # Sanity: omitting segment_fn reproduces the standard pysbd path.
    without = generate_srt([_segment(JFK_TEXT)], "en")
    explicit_none = generate_srt([_segment(JFK_TEXT)], "en", segment_fn=None)
    assert without == explicit_none
