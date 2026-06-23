"""Property / correctness tests for the subtitle *cue pipeline*.

The engine in ``whisperx_subtitles.subtitles`` is split into two stages:
text-shaping functions (``split_at_sentence_end``, ``merge_short_cues``,
``split_long_cues_with_word_timings``) that anchor each cue to its raw
word-level start/end times and decide *text*, and a single final pass,
``normalize_cues``, that enforces all timing invariants (ordered,
non-overlapping, reading-comfort and min/max-duration bounds that never linger
far past the actual speech).

These tests assert the *intended* behavior — invariants and correctness
properties — rather than pinning observed floating-point output.

A ``Cue`` is a dict ``{"text", "start", "end", "word_data", "speaker"}``.
A ``Word`` is ``{"word", "start", "end", "score", optional "speaker"}``.

Run with (heavy GPU deps are not required):
    uvx --with pysbd --with ffmpeg-python pytest tests/test_cues.py -q
"""

from __future__ import annotations

from whisperx_subtitles.config import (
    MAX_CPS,
    MAX_DURATION,
    MAX_LEAD_OUT,
    MIN_DURATION,
    MIN_GAP,
    PAUSE_THRESHOLD,
)
from whisperx_subtitles.subtitles import (
    merge_short_cues,
    normalize_cues,
    split_at_pauses,
    split_at_sentence_end,
    split_long_cue_without_word_timings,
    split_long_cues_with_word_timings,
)


# --------------------------------------------------------------------------- #
# Small constructors for the dict-shaped types.
# --------------------------------------------------------------------------- #
def W(word: str, start, end, score: float = 0.9, speaker: str | None = None) -> dict:
    w = {"word": word, "start": start, "end": end, "score": score}
    if speaker is not None:
        w["speaker"] = speaker
    return w


def C(text, start, end, word_data=None, speaker=None) -> dict:
    return {
        "text": text,
        "start": start,
        "end": end,
        "word_data": word_data,
        "speaker": speaker,
    }


def _texts(cues):
    return [c["text"] for c in cues]


def _reassemble(cues):
    """Join every cue's text back into a flat word list (newlines -> spaces)."""
    out: list[str] = []
    for c in cues:
        out.extend(c["text"].replace("\n", " ").split())
    return out


# =========================================================================== #
# split_at_sentence_end (segmenter=None -> regex split on .!?)
# =========================================================================== #
class TestSplitAtSentenceEnd:
    def test_two_sentences_anchor_to_word_times(self):
        text = "Hello world. Goodbye now."
        wd = [
            W("Hello", 0.0, 0.5),
            W("world.", 0.5, 1.0),
            W("Goodbye", 2.0, 2.5),
            W("now.", 2.5, 3.0),
        ]
        cues = split_at_sentence_end(None, text, wd)

        assert _texts(cues) == ["Hello world.", "Goodbye now."]
        # start/end come from the first/last word of each clause.
        assert cues[0]["start"] == 0.0
        assert cues[0]["end"] == 1.0
        assert cues[1]["start"] == 2.0
        assert cues[1]["end"] == 3.0
        # word_data is the positional slice for that clause.
        assert cues[0]["word_data"] == wd[0:2]
        assert cues[1]["word_data"] == wd[2:4]

    def test_leading_none_time_word_does_not_poison_clause(self):
        # A leading word with start=None/end=None must NOT poison the clause:
        # start comes from the first word with a non-None time (0.5), and the
        # cue is anchored to real word timings (word_data preserved).
        wd = [W("um", None, None), W("hello.", 0.5, 1.0)]
        cues = split_at_sentence_end(None, "um hello.", wd)

        assert len(cues) == 1
        assert cues[0]["start"] == 0.5
        assert cues[0]["end"] == 1.0
        assert cues[0]["word_data"] == wd

    def test_trailing_none_time_word_uses_last_real_end(self):
        # end comes from the last word with a non-None time.
        wd = [W("hello", 0.5, 1.0), W("there", None, None)]
        cues = split_at_sentence_end(None, "hello there", wd)

        assert len(cues) == 1
        assert cues[0]["start"] == 0.5
        assert cues[0]["end"] == 1.0

    def test_clause_with_no_usable_times_falls_back_to_prev_end_equal(self):
        # Final clause has no usable times -> anchors to (prev_end, prev_end),
        # i.e. equal start/end (NOT prev_end + 1). normalize_cues later gives it
        # a readable duration.
        text = "Hi there. Bye."
        wd = [
            W("Hi", 0.0, 0.5),
            W("there.", 0.5, 1.0),
            W("Bye.", None, None),
        ]
        cues = split_at_sentence_end(None, text, wd)

        assert _texts(cues) == ["Hi there.", "Bye."]
        assert cues[0]["end"] == 1.0
        assert cues[1]["start"] == 1.0
        assert cues[1]["end"] == 1.0  # equal, not +1
        assert cues[1]["word_data"] is None

    def test_fallback_zero_when_no_prior_cue(self):
        wd = [W("Hi", None, None), W("there.", None, None)]
        cues = split_at_sentence_end(None, "Hi there.", wd)

        assert len(cues) == 1
        assert cues[0]["start"] == 0.0
        assert cues[0]["end"] == 0.0
        assert cues[0]["word_data"] is None

    def test_speaker_is_majority_word_speaker(self):
        wd = [
            W("Hello", 0.0, 0.5, speaker="A"),
            W("there", 0.5, 1.0, speaker="A"),
            W("friend.", 1.0, 1.5, speaker="B"),
        ]
        cues = split_at_sentence_end(None, "Hello there friend.", wd)

        assert len(cues) == 1
        assert cues[0]["speaker"] == "A"  # majority of 3 words

    def test_speaker_none_when_no_word_speakers(self):
        wd = [W("Hello", 0.0, 0.5), W("world.", 0.5, 1.0)]
        cues = split_at_sentence_end(None, "Hello world.", wd)
        assert cues[0]["speaker"] is None

    def test_empty_and_whitespace_text(self):
        assert split_at_sentence_end(None, "", []) == []
        assert split_at_sentence_end(None, "   ", []) == []

    def test_single_sentence_no_terminal_punctuation(self):
        wd = [W("just", 0.0, 0.4), W("three", 0.4, 0.8), W("words", 0.8, 1.2)]
        cues = split_at_sentence_end(None, "just three words", wd)

        assert len(cues) == 1
        assert cues[0]["text"] == "just three words"
        assert cues[0]["start"] == 0.0
        assert cues[0]["end"] == 1.2

    def test_all_cues_have_required_keys_and_ordered_times(self):
        text = "One two three. Four five six. Seven eight."
        wd = [W(w, float(i), i + 0.5) for i, w in enumerate(text.split())]
        cues = split_at_sentence_end(None, text, wd)

        assert len(cues) == 3
        for c in cues:
            assert set(c) == {"text", "start", "end", "word_data", "speaker"}
            assert c["start"] <= c["end"]


# =========================================================================== #
# merge_short_cues
# =========================================================================== #
class TestMergeShortCues:
    def test_two_short_same_speaker_cues_merge(self):
        # Both short (< min_duration & their reading time), same speaker, small
        # gap, combined fits within line/CPS budgets -> they merge into one cue
        # spanning both. Combined "Hello world how are you" is 23 chars; the
        # 1.6s combined span keeps CPS (23/1.6 ~= 14.4) under MAX_CPS=17.
        cues = [
            C("Hello world", 0.0, 0.8, speaker="A"),
            C("how are you", 0.9, 1.6, speaker="A"),
        ]
        out = merge_short_cues(cues)

        assert len(out) == 1
        assert out[0]["text"] == "Hello world how are you"
        assert out[0]["start"] == 0.0
        assert out[0]["end"] == 1.6

    def test_different_speaker_does_not_merge(self):
        cues = [
            C("Hello world", 0.0, 0.3, speaker="A"),
            C("how are you", 0.4, 0.7, speaker="B"),
        ]
        out = merge_short_cues(cues)

        assert len(out) == 2
        assert _texts(out) == ["Hello world", "how are you"]

    def test_no_merge_when_max_lines_exceeded(self):
        # Narrow width + max_lines=1: the combined text spans >1 line, so the
        # merge is rejected even though both cues are short and same speaker.
        cues = [
            C("aaaa bbbb", 0.0, 0.3, speaker="A"),
            C("cccc dddd eeee ffff", 0.4, 0.7, speaker="A"),
        ]
        out = merge_short_cues(cues, max_line_length=8, max_lines=1)

        assert len(out) == 2
        assert _texts(out) == ["aaaa bbbb", "cccc dddd eeee ffff"]

    def test_no_merge_when_gap_too_large(self):
        cues = [
            C("Hello world", 0.0, 0.3, speaker="A"),
            C("how are you", 10.0, 10.3, speaker="A"),
        ]
        out = merge_short_cues(cues)
        assert len(out) == 2

    def test_no_merge_when_prev_already_long_enough(self):
        # First cue is already long enough to read (not too_short) -> no merge.
        cues = [
            C("Hello world", 0.0, 5.0, speaker="A"),
            C("how are you", 5.1, 5.4, speaker="A"),
        ]
        out = merge_short_cues(cues)
        assert len(out) == 2

    def test_merged_word_data_concatenated(self):
        # Combined "Hi there how are you" is 20 chars; the 1.5s span keeps CPS
        # under MAX_CPS so the merge is permitted.
        wd1 = [W("Hi", 0.0, 0.5), W("there", 0.5, 0.7)]
        wd2 = [W("how", 0.8, 1.0), W("are", 1.0, 1.2), W("you", 1.2, 1.5)]
        cues = [
            C("Hi there", 0.0, 0.7, word_data=wd1, speaker="A"),
            C("how are you", 0.8, 1.5, word_data=wd2, speaker="A"),
        ]
        out = merge_short_cues(cues)
        assert len(out) == 1
        assert out[0]["word_data"] == wd1 + wd2

    def test_empty_list(self):
        assert merge_short_cues([]) == []


# =========================================================================== #
# split_long_cue_without_word_timings
# =========================================================================== #
class TestSplitLongCueWithoutWordTimings:
    def test_splits_into_max_lines_chunks(self):
        cue = C("alpha beta gamma delta epsilon zeta eta theta", 0.0, 10.0)
        out = split_long_cue_without_word_timings(cue, max_line_length=10, max_lines=2)

        for c in out:
            assert len(c["text"].split("\n")) <= 2
            assert c["word_data"] is None
        # Text fully preserved across chunks.
        assert _reassemble(out) == cue["text"].split()
        # Durations are proportional, contiguous, and span the whole cue.
        assert out[0]["start"] == 0.0
        assert out[-1]["end"] == 10.0
        for a, b in zip(out, out[1:], strict=False):
            assert a["end"] == b["start"]
            assert a["end"] >= a["start"]

    def test_inherits_speaker(self):
        cue = C(
            "alpha beta gamma delta epsilon zeta eta theta",
            0.0,
            10.0,
            speaker="SPEAKER_01",
        )
        out = split_long_cue_without_word_timings(cue, max_line_length=10, max_lines=2)
        assert len(out) > 1
        for c in out:
            assert c["speaker"] == "SPEAKER_01"

    def test_short_text_single_chunk(self):
        out = split_long_cue_without_word_timings(
            C("hi there", 0.0, 4.0), max_line_length=42, max_lines=2
        )
        assert len(out) == 1
        assert out[0]["text"] == "hi there"
        assert out[0]["start"] == 0.0
        assert out[0]["end"] == 4.0
        assert out[0]["word_data"] is None

    def test_zero_duration_chunks_collapse_to_start(self):
        out = split_long_cue_without_word_timings(
            C("alpha beta gamma delta", 5.0, 5.0), max_line_length=10, max_lines=1
        )
        assert len(out) >= 1
        for c in out:
            assert c["start"] == 5.0
            assert c["end"] == 5.0
        assert _reassemble(out) == ["alpha", "beta", "gamma", "delta"]


# =========================================================================== #
# split_long_cues_with_word_timings
# =========================================================================== #
class TestSplitLongCuesWithWordTimings:
    def test_short_cue_passes_through_unchanged(self):
        wd = [W("hi", 0.0, 0.2), W("yo", 0.3, 0.5)]
        cue = C("hi yo", 0.0, 0.5, wd, speaker="A")
        out = split_long_cues_with_word_timings([cue], max_line_length=42, max_lines=2)
        assert out == [cue]

    def test_each_output_cue_within_max_lines(self):
        words = "alpha beta gamma delta epsilon".split()
        wd = [W(w, i * 2.0, i * 2.0 + 1.8) for i, w in enumerate(words)]
        cue = C(" ".join(words), wd[0]["start"], wd[-1]["end"], wd)
        out = split_long_cues_with_word_timings([cue], max_line_length=12, max_lines=1)

        assert len(out) > 1
        for c in out:
            assert len(c["text"].split("\n")) <= 1
        # Text preserved across chunks, in order.
        assert _reassemble(out) == words
        # Each chunk anchors to real word data.
        for c in out:
            assert c["word_data"]
            assert all("word" in w for w in c["word_data"])
            assert c["start"] <= c["end"]
        starts = [c["start"] for c in out]
        assert starts == sorted(starts)

    def test_fallback_on_mismatched_word_data_length(self):
        cue = C("alpha beta gamma delta", 0.0, 8.0, word_data=[W("alpha", 0.0, 1.0)])
        out = split_long_cues_with_word_timings([cue], max_line_length=10, max_lines=2)
        # Mismatched length -> falls back to the no-timings splitter.
        assert all(c["word_data"] is None for c in out)
        assert _reassemble(out) == cue["text"].split()
        assert out == split_long_cue_without_word_timings(cue, 10, 2)

    def test_fallback_on_missing_word_data(self):
        cue = C("alpha beta gamma delta", 0.0, 8.0, word_data=None)
        out = split_long_cues_with_word_timings([cue], max_line_length=10, max_lines=2)
        assert all(c["word_data"] is None for c in out)
        assert _reassemble(out) == cue["text"].split()

    def test_empty_list(self):
        assert split_long_cues_with_word_timings([]) == []

    def test_realistic_multi_chunk_preserves_text_and_order(self):
        words = "the quick brown fox jumps over the lazy dog again now".split()
        wd = [W(w, i * 0.5, i * 0.5 + 0.45) for i, w in enumerate(words)]
        cue = C(" ".join(words), 0.0, wd[-1]["end"], wd)
        out = split_long_cues_with_word_timings([cue], max_line_length=15, max_lines=2)

        assert len(out) >= 1
        assert _reassemble(out) == words
        for c in out:
            assert len(c["text"].split("\n")) <= 2
            assert c["start"] <= c["end"]
        starts = [c["start"] for c in out]
        assert starts == sorted(starts)


# =========================================================================== #
# normalize_cues  -- the key invariant-enforcing pass
# =========================================================================== #
def _assert_normalized_invariants(
    out,
    *,
    min_gap=MIN_GAP,
    max_duration=MAX_DURATION,
    max_lead_out=MAX_LEAD_OUT,
):
    for i, c in enumerate(out):
        # end >= start always
        assert c["end"] >= c["start"] - 1e-9, f"cue {i}: end < start"
        # max duration never exceeded
        assert c["end"] - c["start"] <= max_duration + 1e-9, f"cue {i}: too long"
        if i > 0:
            # ordered + min-gap, no overlap
            assert c["start"] >= out[i - 1]["end"] + min_gap - 1e-9, (
                f"cue {i}: overlaps / gap too small"
            )
        # never linger far past the last spoken word (when word data present)
        wd = c.get("word_data")
        if wd:
            last_end = next(
                (w["end"] for w in reversed(wd) if w.get("end") is not None), None
            )
            if last_end is not None:
                assert c["end"] <= max(last_end, c["start"]) + max_lead_out + 1e-9, (
                    f"cue {i}: lingers past last word"
                )


class TestNormalizeCues:
    def test_empty_list(self):
        assert normalize_cues([]) == []

    def test_messy_cues_satisfy_all_invariants(self):
        # Deliberately messy: overlapping, out-of-order ends, zero/negative
        # durations, and a cue lingering long past its last word.
        cues = [
            C("First clause here", 0.0, 0.05, [W("First", 0.0, 0.05)]),  # too short
            C("Overlaps the first", 0.02, 5.0),  # overlaps prev
            C("Backwards end", 6.0, 5.5),  # negative duration
            C("Zero duration", 6.0, 6.0),  # zero duration
            C(
                "Lingers way past the speech end",
                7.0,
                30.0,  # ends way past last word
                [W("Lingers", 7.0, 7.2), W("past", 7.2, 7.5)],
            ),
            C("Way too long an interval", 40.0, 100.0),  # exceeds max_duration
        ]
        out = normalize_cues(cues)

        assert len(out) == len(cues)
        _assert_normalized_invariants(out)
        # Text and speaker preserved.
        assert _texts(out) == _texts(cues)

    def test_short_cue_is_stretched_toward_min_duration(self):
        # A lone short cue with plenty of room after it gets stretched to at
        # least min_duration (reading-comfort).
        cues = [C("Hello world", 0.0, 0.1)]
        out = normalize_cues(cues)
        assert out[0]["end"] - out[0]["start"] >= MIN_DURATION - 1e-9

    def test_reading_time_drives_duration_for_long_text(self):
        # Long text needs more than min_duration to read at MAX_CPS.
        text = "x" * 100  # 100 chars -> ~5.88s at 17 cps
        cues = [C(text, 0.0, 0.1)]
        out = normalize_cues(cues)
        expected = len(text) / MAX_CPS
        assert out[0]["end"] - out[0]["start"] >= expected - 1e-9
        assert out[0]["end"] - out[0]["start"] <= MAX_DURATION + 1e-9

    def test_max_lead_out_caps_end_to_last_word(self):
        wd = [W("Hi", 0.0, 0.5), W("there", 0.5, 1.0)]
        cues = [C("Hi there", 0.0, 20.0, wd)]
        out = normalize_cues(cues)
        # End cannot exceed last word end + lead out.
        assert out[0]["end"] <= 1.0 + MAX_LEAD_OUT + 1e-9

    def test_overlap_resolution_pushes_starts_forward(self):
        cues = [
            C("aaa", 0.0, 2.0),
            C("bbb", 1.0, 3.0),  # starts before prev ends
            C("ccc", 2.0, 4.0),
        ]
        out = normalize_cues(cues)
        _assert_normalized_invariants(out)
        starts = [c["start"] for c in out]
        assert starts == sorted(starts)

    def test_min_gap_enforced_between_consecutive(self):
        cues = [
            C("a", 0.0, 1.0, [W("a", 0.0, 1.0)]),
            C("b", 1.0, 2.0, [W("b", 1.0, 2.0)]),
        ]
        out = normalize_cues(cues)
        assert out[1]["start"] - out[0]["end"] >= MIN_GAP - 1e-9

    def test_text_and_speaker_preserved(self):
        cues = [
            C("hello", 0.0, 1.0, speaker="A"),
            C("world", 2.0, 3.0, speaker="B"),
        ]
        out = normalize_cues(cues)
        assert _texts(out) == ["hello", "world"]
        assert [c["speaker"] for c in out] == ["A", "B"]
