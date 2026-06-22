"""Characterization tests for the subtitle *cue pipeline*.

These tests pin down the *current* behavior of the cue functions in
``whisperx_subtitles.subtitles``. They are intentionally descriptive, not
prescriptive: every exact-value assertion below was observed by running the
real functions. Where the logic is intricate, we assert robust structural
properties (cue counts, ordering, ``start <= end``, text preservation) rather
than brittle floating-point values.

A ``Cue`` is a plain dict: ``{"text", "start", "end", "word_data}``.
A ``Word`` is ``{"word", "start", "end", "score"}``.

Run with (heavy GPU deps are not required):
    uvx --with pysbd --with ffmpeg-python pytest tests/test_cues.py -q
"""

from __future__ import annotations

from whisperx_subtitles.subtitles import (
    merge_short_cues,
    split_at_sentence_end,
    split_long_cue_without_word_timings,
    split_long_cues_with_word_timings,
)


# --------------------------------------------------------------------------- #
# Small constructors for the dict-shaped types.
# --------------------------------------------------------------------------- #
def W(word: str, start, end, score: float = 0.9) -> dict:
    return {"word": word, "start": start, "end": end, "score": score}


def C(text: str, start: float, end: float, word_data=None) -> dict:
    return {"text": text, "start": start, "end": end, "word_data": word_data}


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
    def test_two_sentences_carry_word_timings(self):
        text = "Hello world. Goodbye now."
        wd = [
            W("Hello", 0, 0.5),
            W("world.", 0.5, 1.0),
            W("Goodbye", 2.0, 2.5),
            W("now.", 2.5, 3.0),
        ]
        cues = split_at_sentence_end(None, text, wd)

        assert _texts(cues) == ["Hello world.", "Goodbye now."]
        # Timings are carried verbatim from the first/last word of each clause.
        assert cues[0]["start"] == 0
        assert cues[0]["end"] == 1.0
        assert cues[1]["start"] == 2.0
        assert cues[1]["end"] == 3.0
        # word_data is the positional slice for that clause.
        assert cues[0]["word_data"] == wd[0:2]
        assert cues[1]["word_data"] == wd[2:4]

    def test_fallback_uses_prev_cue_end_when_times_missing(self):
        # The final word has no "start"/"end" keys -> start/end can't be found,
        # so the clause falls back to (prev_end, prev_end + 1) with word_data=None.
        text = "Hi there. Bye."
        wd = [W("Hi", 0, 0.5), W("there.", 0.5, 1.0), {"word": "Bye.", "score": 0.9}]
        cues = split_at_sentence_end(None, text, wd)

        assert _texts(cues) == ["Hi there.", "Bye."]
        assert cues[0]["end"] == 1.0
        # Fallback: previous cue end (1.0) and +1.
        assert cues[1]["start"] == 1.0
        assert cues[1]["end"] == 2.0
        assert cues[1]["word_data"] is None

    def test_fallback_uses_zero_one_when_no_prior_cue(self):
        # First (and only) clause has no timing keys -> no prior cue -> (0, 1).
        wd = [{"word": "Hi", "score": 0.9}, {"word": "there.", "score": 0.9}]
        cues = split_at_sentence_end(None, "Hi there.", wd)

        assert len(cues) == 1
        assert cues[0]["start"] == 0
        assert cues[0]["end"] == 1
        assert cues[0]["word_data"] is None

    def test_present_but_none_times_still_trigger_fallback(self):
        # Quirk: the lookup uses `if "start" in word`, so a present key with a
        # None value is *selected* as the start_time, which is then None -> the
        # start_time/end_time-not-None branch fails and we hit the fallback.
        wd = [W("Hi", None, None), W("there.", None, None)]
        cues = split_at_sentence_end(None, "Hi there.", wd)

        assert len(cues) == 1
        assert cues[0]["start"] == 0
        assert cues[0]["end"] == 1
        assert cues[0]["word_data"] is None

    def test_empty_and_whitespace_text(self):
        assert split_at_sentence_end(None, "", []) == []
        assert split_at_sentence_end(None, "   ", []) == []

    def test_single_sentence_no_terminal_punctuation(self):
        text = "just three words"
        wd = [W("just", 0, 0.4), W("three", 0.4, 0.8), W("words", 0.8, 1.2)]
        cues = split_at_sentence_end(None, text, wd)

        assert len(cues) == 1
        assert cues[0]["text"] == "just three words"
        assert cues[0]["start"] == 0
        assert cues[0]["end"] == 1.2

    def test_all_cues_have_required_keys_and_ordered_times(self):
        text = "One two three. Four five six. Seven eight."
        wd = [W(w, i, i + 0.5) for i, w in enumerate(text.split())]
        cues = split_at_sentence_end(None, text, wd)

        assert len(cues) == 3
        for c in cues:
            assert set(c) == {"text", "start", "end", "word_data"}
            assert c["start"] <= c["end"]


# =========================================================================== #
# merge_short_cues
# =========================================================================== #
class TestMergeShortCues:
    def test_two_short_cues_merge_and_stretch(self):
        cues = [C("Hello world", 0, 0.5), C("how are you", 0.5, 1.0)]
        out = merge_short_cues(
            cues, min_duration=3, max_line_length=42, max_lines=2, desired_wps=4
        )

        # Both cues are short and fit within max_lines -> they merge into one.
        assert len(out) == 1
        assert out[0]["text"] == "Hello world how are you"
        assert out[0]["start"] == 0
        # Final-cue stretch: max(word_count/wps = 5/4 = 1.25, min_duration=3) = 3.
        assert out[0]["end"] == 3

    def test_no_merge_when_max_lines_exceeded(self):
        # Narrow width + max_lines=1 means the combined text spans >1 line, so
        # the merge is rejected and the cues stay separate.
        cues = [C("aaaa bbbb", 0, 0.5), C("cccc dddd eeee ffff", 0.6, 1.0)]
        out = merge_short_cues(
            cues, min_duration=3, max_line_length=8, max_lines=1, desired_wps=4
        )

        assert len(out) == 2
        assert _texts(out) == ["aaaa bbbb", "cccc dddd eeee ffff"]
        # First cue's stretch is capped at next_cue.start - 0.1 = 0.5,
        # so min(0 + 3, 0.5) leaves it unchanged at 0.5.
        assert out[0]["start"] == 0
        assert out[0]["end"] == 0.5
        # Last cue is stretched: 0.6 + max(4/4=1, 3) = 3.6.
        assert out[1]["start"] == 0.6
        assert out[1]["end"] == 3.6

    def test_single_cue_last_duration_adjustment(self):
        out = merge_short_cues([C("one two three four", 0, 0.5)], desired_wps=4)
        assert len(out) == 1
        # Last-cue stretch: max(4/4=1, min_duration=3) = 3.
        assert out[0]["end"] == 3

    def test_single_already_long_cue_unchanged(self):
        # A long-enough cue is left as-is by the final adjustment.
        out = merge_short_cues([C("a b", 0, 10)], min_duration=3, desired_wps=4)
        assert len(out) == 1
        assert out[0]["end"] == 10

    def test_empty_list(self):
        assert merge_short_cues([]) == []

    def test_output_times_ordered(self):
        cues = [
            C("alpha beta", 0, 0.4),
            C("gamma delta epsilon", 5, 5.4),
            C("zeta", 10, 10.2),
        ]
        out = merge_short_cues(cues, desired_wps=4)
        for c in out:
            assert c["start"] <= c["end"]


# =========================================================================== #
# split_long_cue_without_word_timings
# =========================================================================== #
class TestSplitLongCueWithoutWordTimings:
    def test_splits_into_max_lines_chunks(self):
        cue = C("alpha beta gamma delta epsilon zeta eta theta", 0, 10)
        out = split_long_cue_without_word_timings(cue, max_line_length=10, max_lines=2)

        # Each chunk holds at most max_lines lines.
        for c in out:
            assert len(c["text"].split("\n")) <= 2
            assert c["word_data"] is None
        # Text is fully preserved across chunks.
        assert _reassemble(out) == cue["text"].split()
        # Durations are distributed and contiguous, spanning the whole cue.
        assert out[0]["start"] == 0
        assert out[-1]["end"] == 10.0
        for a, b in zip(out, out[1:], strict=False):
            assert a["end"] == b["start"]

    def test_short_text_single_chunk(self):
        out = split_long_cue_without_word_timings(
            C("hi there", 0, 4), max_line_length=42, max_lines=2
        )
        assert len(out) == 1
        assert out[0]["text"] == "hi there"
        assert out[0]["start"] == 0
        assert out[0]["end"] == 4.0
        assert out[0]["word_data"] is None

    def test_zero_duration_chunks_collapse_to_start(self):
        # end <= start -> total_duration is 0 -> every chunk gets start==end.
        out = split_long_cue_without_word_timings(
            C("alpha beta gamma delta", 5, 5), max_line_length=10, max_lines=1
        )
        assert len(out) >= 1
        for c in out:
            assert c["start"] == 5
            assert c["end"] == 5
        assert _reassemble(out) == ["alpha", "beta", "gamma", "delta"]


# =========================================================================== #
# split_long_cues_with_word_timings
# =========================================================================== #
class TestSplitLongCuesWithWordTimings:
    def test_fallback_on_mismatched_word_data_length(self):
        cue = C("alpha beta gamma delta", 0, 8, word_data=[W("alpha", 0, 1)])
        out = split_long_cues_with_word_timings([cue], max_line_length=10, max_lines=2)
        # Mismatched length -> falls back to split_long_cue_without_word_timings,
        # which produces word_data=None cues.
        assert all(c["word_data"] is None for c in out)
        assert _reassemble(out) == cue["text"].split()
        # Matches the no-timings splitter exactly.
        assert out == split_long_cue_without_word_timings(cue, 10, 2)

    def test_fallback_on_missing_word_data(self):
        cue = C("alpha beta gamma delta", 0, 8, word_data=None)
        out = split_long_cues_with_word_timings([cue], max_line_length=10, max_lines=2)
        assert all(c["word_data"] is None for c in out)
        assert _reassemble(out) == cue["text"].split()

    def test_chunks_by_line_budget_with_word_timings(self):
        words = "alpha beta gamma delta epsilon".split()
        wd = [W(w, i * 2.0, i * 2.0 + 1.8) for i, w in enumerate(words)]
        cue = C(" ".join(words), wd[0]["start"], wd[-1]["end"], wd)
        out = split_long_cues_with_word_timings(
            [cue],
            max_line_length=12,
            max_lines=1,
            min_duration=5 / 6,
            desired_wps=4,
            max_gap_duration=1.5,
        )

        # Text is preserved and re-split into multiple cues.
        assert len(out) == 3
        assert _reassemble(out) == words
        # Each chunk's word_data is a non-empty slice of real Word dicts.
        for c in out:
            assert c["word_data"]
            assert all("word" in w for w in c["word_data"])
            assert c["start"] <= c["end"]
        # Cues are ordered by start time and non-overlapping here.
        starts = [c["start"] for c in out]
        assert starts == sorted(starts)

    def test_short_chunks_merge_within_gap(self):
        # Tightly-packed fast speech: chunks fall below min_duration and merge
        # with neighbours since the inter-word gaps are < max_gap_duration.
        words = "a b c d e f".split()
        wd = [W(w, i * 0.15, i * 0.15 + 0.1) for i, w in enumerate(words)]
        cue = C(" ".join(words), wd[0]["start"], wd[-1]["end"], wd)
        out = split_long_cues_with_word_timings(
            [cue],
            max_line_length=3,
            max_lines=1,
            min_duration=5 / 6,
            desired_wps=4,
            max_gap_duration=1.5,
        )

        # Six single-word chunks collapse into two merged cues.
        assert _texts(out) == ["a b c d", "e f"]
        assert _reassemble(out) == words
        for c in out:
            assert c["start"] <= c["end"]

    def test_single_short_chunk_gets_min_duration(self):
        # One chunk, too short to split, no neighbour to merge with -> its end
        # is pushed out to satisfy min_duration.
        wd = [W("hi", 0, 0.2), W("yo", 0.3, 0.5)]
        cue = C("hi yo", 0, 0.5, wd)
        out = split_long_cues_with_word_timings(
            [cue], max_line_length=42, max_lines=2, min_duration=5 / 6, desired_wps=4
        )
        assert len(out) == 1
        assert out[0]["text"] == "hi yo"
        assert out[0]["start"] == 0
        assert out[0]["end"] - out[0]["start"] >= 5 / 6 - 1e-9

    def test_empty_list(self):
        assert split_long_cues_with_word_timings([]) == []

    def test_realistic_multi_chunk_preserves_text_and_order(self):
        words = "the quick brown fox jumps over the lazy dog again now".split()
        wd = [W(w, i * 0.5, i * 0.5 + 0.45) for i, w in enumerate(words)]
        cue = C(" ".join(words), 0, wd[-1]["end"], wd)
        out = split_long_cues_with_word_timings([cue], max_line_length=15, max_lines=2)

        assert len(out) >= 1
        # Every original word is present, in order, exactly once.
        assert _reassemble(out) == words
        # Times are individually valid and globally non-decreasing in start.
        starts = [c["start"] for c in out]
        assert starts == sorted(starts)
        for c in out:
            assert c["start"] <= c["end"]
