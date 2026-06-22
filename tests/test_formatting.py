"""Characterization tests for the subtitle *formatting primitives*.

These tests document what the code in ``whisperx_subtitles.subtitles`` ACTUALLY
does today (warts and all), not what it ideally should do. Several assertions
deliberately pin surprising/buggy behavior so that a future refactor will make
the intent of any change explicit. Such cases are flagged with ``# NOTE:``.

Covered functions:
    - format_timestamp
    - split_subtitle
    - extract_words
    - split_sentence_heuristically

Run with:
    uvx --with pysbd --with ffmpeg-python pytest tests/test_formatting.py -q
"""

from __future__ import annotations

import pytest

from whisperx_subtitles.subtitles import (
    extract_words,
    format_timestamp,
    split_sentence_heuristically,
    split_subtitle,
)

# ---------------------------------------------------------------------------
# format_timestamp
# ---------------------------------------------------------------------------


def test_format_timestamp_none_returns_zero():
    assert format_timestamp(None) == "00:00:00,000"


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "00:00:00,000"),
        (0.0, "00:00:00,000"),
        (1.5, "00:00:01,500"),
        (59.999, "00:00:59,999"),
        (3600, "01:00:00,000"),
        (3661.5, "01:01:01,500"),  # hours rollover
        (7322.123, "02:02:02,123"),
        (0.001, "00:00:00,001"),
    ],
)
def test_format_timestamp_basic(seconds, expected):
    assert format_timestamp(seconds) == expected


def test_format_timestamp_sub_millisecond_rounds_to_nearest():
    # Sub-millisecond values round to the nearest millisecond (0.0005s -> 0ms).
    assert format_timestamp(0.0005) == "00:00:00,000"


def test_format_timestamp_rolls_over_at_minute_boundary():
    # 59.9999s rounds up and rolls into the next minute instead of producing an
    # invalid "00:00:60,000" (a seconds field must never read 60).
    assert format_timestamp(59.9999) == "00:01:00,000"


def test_format_timestamp_negative_clamped_to_zero():
    # Negative input is guarded and clamped to zero rather than producing a
    # malformed, non-SRT timestamp.
    assert format_timestamp(-1.0) == "00:00:00,000"


def test_format_timestamp_uses_comma_decimal_separator():
    # SRT uses a comma (not a period) before the milliseconds.
    assert "," in format_timestamp(1.5)
    assert "." not in format_timestamp(1.5)


# ---------------------------------------------------------------------------
# split_subtitle
# ---------------------------------------------------------------------------


def test_split_subtitle_empty_string():
    assert split_subtitle("") == ""


def test_split_subtitle_whitespace_only():
    # str.split() drops all whitespace -> no words -> empty output.
    assert split_subtitle("   ") == ""


def test_split_subtitle_single_short_word():
    assert split_subtitle("hello") == "hello"


def test_split_subtitle_collapses_multiple_spaces():
    # Internal runs of whitespace are collapsed to single spaces because
    # the text is rebuilt via " ".join(words).
    assert split_subtitle("a  b   c") == "a b c"


def test_split_subtitle_greedy_wrap_default_max():
    # Exactly fills lines greedily up to max_chars (default 42).
    text = "word " * 10  # ten words of length 4
    assert split_subtitle(text, max_chars=10) == (
        "word word\nword word\nword word\nword word\nword word"
    )


@pytest.mark.parametrize(
    ("text", "max_chars", "expected"),
    [
        # "aa bb" == 5 chars fits exactly at max_chars=5, "cc" wraps.
        ("aa bb cc", 5, "aa bb\ncc"),
        # At max_chars=4, "aa bb" (5) overflows so every word is its own line.
        ("aa bb cc", 4, "aa\nbb\ncc"),
        ("a b c d e f", 3, "a b\nc d\ne f"),
    ],
)
def test_split_subtitle_boundary_lengths(text, max_chars, expected):
    assert split_subtitle(text, max_chars=max_chars) == expected


def test_split_subtitle_single_word_longer_than_max_no_leading_blank():
    # An oversize first word gets its own line with no spurious leading blank.
    result = split_subtitle("supercalifragilistic", max_chars=5)
    assert result == "supercalifragilistic"
    assert result.split("\n") == ["supercalifragilistic"]


def test_split_subtitle_oversize_word_in_middle_does_not_blank_line():
    # When the oversize word is not first, it simply gets its own line; no
    # spurious blank line because current_line was non-empty at the break.
    result = split_subtitle("hi superlongword bye", max_chars=5)
    assert result == "hi\nsuperlongword\nbye"


# ---------------------------------------------------------------------------
# extract_words
# ---------------------------------------------------------------------------


def test_extract_words_returns_set_lowercased():
    assert extract_words("Hello, world! Hello.") == {"hello", "world"}


def test_extract_words_empty():
    assert extract_words("") == set()


def test_extract_words_keeps_apostrophes():
    # The \w+ class plus an explicit apostrophe keeps contractions intact.
    assert extract_words("it's a test's") == {"it's", "a", "test's"}


def test_extract_words_underscores_and_digits_are_word_chars():
    # NOTE: \w matches underscores and digits, so "a_b" and "123" are tokens,
    # while the hyphen in "c-d" is a separator.
    assert extract_words("a_b c-d 123") == {"a_b", "c", "d", "123"}


def test_extract_words_unicode_letters():
    assert extract_words("CAFÉ café") == {"café"}


def test_extract_words_deduplicates():
    assert extract_words("the the THE The") == {"the"}


# ---------------------------------------------------------------------------
# split_sentence_heuristically
# ---------------------------------------------------------------------------


def test_ssh_short_sentence_returned_stripped_as_single_element():
    assert split_sentence_heuristically("short sentence", 42, 2) == ["short sentence"]


def test_ssh_strips_surrounding_whitespace_when_it_fits():
    assert split_sentence_heuristically("   short   ", 42, 2) == ["short"]


def test_ssh_fits_within_max_lines_is_returned_unsplit():
    # Wraps to exactly two lines at max_chars=20, so it is within max_lines=2
    # and returned as a single-element list (unsplit).
    sentence = "one two three four five six seven eight"
    assert split_sentence_heuristically(sentence, 20, 2) == [sentence]


def test_ssh_oversize_single_word_returned_unsplit():
    # A lone word can never be split on spaces/punctuation; even though it
    # technically wraps oddly, it stays a single element.
    word = "supercalifragilisticexpialidocious"
    assert split_sentence_heuristically(word, 5, 2) == [word]


def test_ssh_splits_on_conjunctions_when_overflowing():
    sentence = (
        "I went to the store and I bought some milk because we needed it "
        "for breakfast tomorrow morning before work"
    )
    # The split happens BEFORE conjunctions (and/because/before); here each
    # resulting clause already fits within max_lines so none is split further.
    assert split_sentence_heuristically(sentence, 42, 2) == [
        "I went to the store",
        "and I bought some milk",
        "because we needed it for breakfast tomorrow morning",
        "before work",
    ]


def test_ssh_splits_after_commas_and_semicolons():
    sentence = (
        "First part here is quite long, and the second part there is also "
        "fairly long indeed, third part everywhere"
    )
    assert split_sentence_heuristically(sentence, 42, 2) == [
        "First part here is quite long,",
        "and the second part there is also fairly long indeed,",
        "third part everywhere",
    ]


def test_ssh_comma_stays_attached_to_preceding_part():
    sentence = (
        "aaaa bbbb cccc dddd, eeee ffff gggg hhhh iiii jjjj kkkk llll mmmm nnnn oooo"
    )
    result = split_sentence_heuristically(sentence, 30, 2)
    # The comma is preserved on the first part (split is after it).
    assert result[0] == "aaaa bbbb cccc dddd,"


def test_ssh_overlong_part_split_recursively_to_fit():
    # No commas/semicolons/conjunctions -> a single over-long "part" that is
    # split recursively at word midpoints until every piece fits max_lines.
    sentence = (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa "
        "lambda mu nu xi omicron"
    )
    result = split_sentence_heuristically(sentence, 20, 2)
    assert result == [
        "alpha beta gamma",
        "delta epsilon zeta eta",
        "theta iota kappa lambda",
        "mu nu xi omicron",
    ]
    # Every returned part fits within max_lines=2, and words are preserved.
    assert all(len(split_subtitle(p, max_chars=20).split("\n")) <= 2 for p in result)
    assert " ".join(result).split() == sentence.split()


def test_ssh_splits_recursively_until_parts_fit():
    # The "further split" step now recurses, so no returned part exceeds
    # max_lines (previously the halving was only one level deep and a part
    # could still overflow).
    sentence = (
        "this is a very long clause with absolutely no punctuation or "
        "conjunctions at all here that keeps going and going forever"
    )
    result = split_sentence_heuristically(sentence, 20, 2)
    assert result == [
        "this is a very long",
        "clause with",
        "absolutely no punctuation",
        "or conjunctions at all",
        "here that keeps going",
        "and going forever",
    ]
    # No part overflows max_lines=2 at max_chars=20.
    overflowing = [
        p for p in result if len(split_subtitle(p, max_chars=20).split("\n")) > 2
    ]
    assert overflowing == []


def test_ssh_max_lines_one_forces_a_split():
    # With max_lines=1, a sentence that wraps to 2 lines must be split.
    sentence = "alpha beta gamma delta epsilon zeta eta theta"
    result = split_sentence_heuristically(sentence, 20, 1)
    assert len(result) >= 2
    assert " ".join(result).split() == sentence.split()
