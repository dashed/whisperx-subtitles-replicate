"""Tests for the translation re-timing engine (pure, GPU-free).

The translation flow is: reconstruct SOURCE sentences with time spans
(``sentences_with_spans``) -> translate each on the GPU side -> re-segment each
translation and distribute its source span proportionally
(``fit_translation_to_span`` / ``generate_translated_srt``). These tests cover
the pure pieces; the actual MT model lives in predict.py.

Run: uvx --with pysbd --with ffmpeg-python --with numpy pytest tests/test_translation.py -q
"""

from __future__ import annotations

import re

from whisperx_subtitles.config import MAX_LINE_LENGTH, MAX_LINES
from whisperx_subtitles.subtitles import (
    fit_translation_to_span,
    generate_translated_srt,
    sentences_with_spans,
)

TS = r"\d{2}:[0-5]\d:[0-5]\d,\d{3}"
ARROW = re.compile(rf"^({TS}) --> ({TS})$")


def W(word, start, end):
    return {"word": word, "start": start, "end": end, "score": 0.9}


def _ts(ts):
    hms, ms = ts.split(",")
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _parse(srt):
    blocks = []
    for raw in [b for b in srt.strip().split("\n\n") if b.strip()]:
        lines = raw.splitlines()
        assert lines[0].isdigit(), f"bad index: {lines[0]!r}"
        m = ARROW.match(lines[1])
        assert m, f"bad arrow: {lines[1]!r}"
        text_lines = lines[2:]
        assert any(t.strip() for t in text_lines)
        blocks.append((int(lines[0]), _ts(m.group(1)), _ts(m.group(2)), text_lines))
    return blocks


# --------------------------------------------------------------------------- #
# sentences_with_spans
# --------------------------------------------------------------------------- #
class TestSentencesWithSpans:
    def test_sentence_level_spans(self):
        # Two sentences in one segment; segment_fn splits on the period.
        wd = [
            W(w, float(i), float(i) + 0.8)
            for i, w in enumerate("one two three four".split())
        ]
        segs = [{"text": "one two. three four.", "words": wd}]
        out = sentences_with_spans(
            segs, segment_fn=lambda t: t.replace(". ", ".\n").split("\n")
        )
        assert [c["text"] for c in out] == ["one two.", "three four."]
        assert (out[0]["start"], out[0]["end"]) == (0.0, 1.8)  # words 0,1
        assert (out[1]["start"], out[1]["end"]) == (2.0, 3.8)  # words 2,3

    def test_does_not_clause_split(self):
        # A comma'd sentence stays ONE unit (unlike split_at_sentence_end, which
        # clause-splits) so translation gets full-sentence context.
        wd = [W(w, float(i), float(i) + 0.5) for i, w in enumerate("a b c d e".split())]
        segs = [{"text": "a b, c d e", "words": wd}]
        out = sentences_with_spans(segs, segment_fn=lambda t: [t])
        assert len(out) == 1
        assert out[0]["text"] == "a b, c d e"

    def test_no_timing_anchors_to_prev_end(self):
        wd = [W("hi", 0.0, 1.0), W("there", 1.0, 2.0)]
        segs = [{"text": "hi there", "words": wd}]
        out = sentences_with_spans(segs, segment_fn=lambda t: [t])
        assert out[0]["start"] == 0.0 and out[0]["end"] == 2.0


# --------------------------------------------------------------------------- #
# fit_translation_to_span
# --------------------------------------------------------------------------- #
class TestFitTranslationToSpan:
    def test_single_chunk_uses_whole_span(self):
        cues = fit_translation_to_span("Short line.", 1.0, 4.0)
        assert len(cues) == 1
        assert cues[0]["start"] == 1.0
        assert cues[0]["end"] == 4.0

    def test_proportional_and_anchored(self):
        # Force multiple chunks with a tiny line budget.
        cues = fit_translation_to_span(
            "aaaa bbbb cccc dddd", 0.0, 8.0, max_line_length=9, max_lines=1
        )
        assert len(cues) >= 2
        assert cues[0]["start"] == 0.0  # anchored to source start
        assert cues[-1]["end"] == 8.0  # anchored to source end
        for a, b in zip(cues, cues[1:], strict=False):
            assert a["end"] == b["start"]  # contiguous
            assert a["end"] >= a["start"]  # ordered

    def test_longer_chunk_gets_more_time(self):
        # First chunk much longer than the second -> larger share of the span.
        cues = fit_translation_to_span(
            "aaaaaaaa bbbbbbbb. cc", 0.0, 10.0, max_line_length=18, max_lines=1
        )
        assert len(cues) == 2
        assert (cues[0]["end"] - cues[0]["start"]) > (cues[1]["end"] - cues[1]["start"])

    def test_empty_text(self):
        assert fit_translation_to_span("", 0.0, 5.0) == []
        assert fit_translation_to_span("   ", 0.0, 5.0) == []


# --------------------------------------------------------------------------- #
# generate_translated_srt (end to end)
# --------------------------------------------------------------------------- #
class TestGenerateTranslatedSrt:
    def _sents(self):
        return [
            {
                "text": "This is the first translated sentence.",
                "start": 0.0,
                "end": 4.0,
                "word_data": None,
                "speaker": None,
            },
            {
                "text": "And here is a noticeably longer second sentence to translate.",
                "start": 4.5,
                "end": 11.0,
                "word_data": None,
                "speaker": None,
            },
        ]

    def test_valid_ordered_and_within_limits(self):
        blocks = _parse(generate_translated_srt(self._sents()))
        assert blocks
        assert [b[0] for b in blocks] == list(range(1, len(blocks) + 1))
        prev_end = -1.0
        for _, start, end, lines in blocks:
            assert start >= prev_end  # non-overlapping, ordered
            assert end >= start
            prev_end = end
            assert len(lines) <= MAX_LINES
            for line in lines:
                assert len(line) <= MAX_LINE_LENGTH

    def test_all_translated_words_present(self):
        blocks = _parse(generate_translated_srt(self._sents()))
        joined = " ".join(" ".join(b[3]) for b in blocks).lower()
        for word in ["first", "translated", "longer", "second"]:
            assert word in joined

    def test_empty_input(self):
        assert generate_translated_srt([]) == ""
