"""Characterization tests for end-to-end SRT generation.

These assert the *current* behavior of ``generate_srt`` (captured by running it
first), not an idealized spec. They import only ``pysbd`` + the pure subtitle
module, so they run without the GPU/torch/whisperx stack.
"""

import re

from whisperx_subtitles.subtitles import generate_srt

# An SRT block looks like:
#   index\nHH:MM:SS,mmm --> HH:MM:SS,mmm\ntext\n\n
TIMESTAMP_RE = r"\d{2}:\d{2}:\d{2},\d{3}"
ARROW_LINE_RE = re.compile(rf"^{TIMESTAMP_RE} --> {TIMESTAMP_RE}$")


def _mk_words(text, t0=0.0, dur=0.4, gap=0.05):
    """Build aligned Word dicts with monotonically increasing timings."""
    words = []
    t = t0
    for w in text.split():
        words.append(
            {"word": w, "start": round(t, 3), "end": round(t + dur, 3), "score": 0.9}
        )
        t = t + dur + gap
    return words, t


JFK_TEXT = "And so my fellow Americans ask not what your country can do for you"


def _jfk_segment():
    words, _ = _mk_words(JFK_TEXT, 0.0)
    return {"text": JFK_TEXT, "words": words}


def _parse_blocks(srt: str):
    """Split a full SRT string into its non-empty blocks (lists of lines)."""
    return [block.splitlines() for block in srt.strip().split("\n\n") if block.strip()]


def test_generate_srt_non_empty_and_has_arrow():
    out = generate_srt([_jfk_segment()], "en")
    assert out  # non-empty
    assert "-->" in out


def test_generate_srt_blocks_numbered_from_one():
    # Two segments so we get more than one block to check sequential numbering.
    w1, t = _mk_words("Hello there my friend.", 0.0)
    w2, _ = _mk_words("How are you doing today?", t)
    segments = [
        {"text": "Hello there my friend.", "words": w1},
        {"text": "How are you doing today?", "words": w2},
    ]
    out = generate_srt(segments, "en")
    blocks = _parse_blocks(out)
    assert len(blocks) == 2
    indices = [int(block[0]) for block in blocks]
    assert indices == [1, 2]


def test_generate_srt_timestamp_format():
    out = generate_srt([_jfk_segment()], "en")
    for block in _parse_blocks(out):
        # The second line of each block is the timestamp/arrow line.
        assert ARROW_LINE_RE.match(block[1]), f"bad arrow line: {block[1]!r}"


def test_generate_srt_preserves_spoken_words():
    out = generate_srt([_jfk_segment()], "en")
    lowered = out.lower()
    for word in JFK_TEXT.split():
        assert word.lower() in lowered, f"missing word {word!r}"


def test_generate_srt_block_ends_with_blank_line():
    # Characterization: each block (including the last) is terminated by "\n\n".
    out = generate_srt([_jfk_segment()], "en")
    assert out.endswith("\n\n")


def test_generate_srt_english_exact_characterization():
    # Captured by running the current code; the cue is split into two display
    # lines by split_subtitle and spans the full word range.
    out = generate_srt([_jfk_segment()], "en")
    assert out == (
        "1\n"
        "00:00:00,000 --> 00:00:06,250\n"
        "And so my fellow Americans ask not what\n"
        "your country can do for you\n\n"
    )


def test_generate_srt_unsupported_language_falls_back():
    # pysbd.Segmenter raises for an unknown language; generate_srt catches it,
    # leaves segmenter=None, and falls back to a regex sentence split. Output is
    # still valid SRT and, for this single-sentence input, identical to "en".
    out = generate_srt([_jfk_segment()], "zz-not-a-real-language")
    assert out  # non-empty despite the failed segmenter init
    assert "-->" in out
    blocks = _parse_blocks(out)
    assert int(blocks[0][0]) == 1
    assert ARROW_LINE_RE.match(blocks[0][1])
    # The regex fallback yields the same result as the en segmenter here.
    assert out == generate_srt([_jfk_segment()], "en")


def test_generate_srt_empty_segments_returns_empty_string():
    assert generate_srt([], "en") == ""
