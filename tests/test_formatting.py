"""Property / correctness tests for the subtitle *formatting primitives*.

Unlike the old characterization suite, these tests assert what the rewritten
engine in ``whisperx_subtitles.subtitles`` is *meant* to do: produce valid SRT
timestamps, balanced max-2-line wraps, and recursive sentence splits that keep
every piece within the line budget while preserving all words.

Covered functions:
    - format_timestamp
    - split_subtitle
    - split_sentence_heuristically

Run with:
    uvx --with pysbd --with ffmpeg-python pytest tests/test_formatting.py -q
"""

from __future__ import annotations

from collections import Counter

import pytest

from whisperx_subtitles.config import MAX_LINE_LENGTH, MAX_LINES
from whisperx_subtitles.subtitles import (
    format_timestamp,
    split_sentence_heuristically,
    split_subtitle,
)

# ---------------------------------------------------------------------------
# format_timestamp
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [None, -0.001, -1.0, -3600.0])
def test_format_timestamp_none_or_negative_is_zero(bad):
    # Guard clause: missing or negative input never yields a malformed timestamp.
    assert format_timestamp(bad) == "00:00:00,000"


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "00:00:00,000"),
        (0.0, "00:00:00,000"),
        (0.001, "00:00:00,001"),
        (1.5, "00:00:01,500"),
        (59.999, "00:00:59,999"),
        (3600, "01:00:00,000"),
        (3661.5, "01:01:01,500"),
        (7322.123, "02:02:02,123"),
    ],
)
def test_format_timestamp_hms_format(seconds, expected):
    assert format_timestamp(seconds) == expected


def test_format_timestamp_uses_comma_decimal_separator():
    # SRT uses a comma (not a period) before the milliseconds.
    ts = format_timestamp(1.5)
    assert "," in ts and "." not in ts


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (59.9999, "00:01:00,000"),  # rolls into next minute, not 00:00:60,000
        (0.0005, "00:00:00,000"),  # sub-ms rounds to nearest ms (down here)
        (0.0006, "00:00:00,001"),  # sub-ms rounds to nearest ms (up here)
        (3599.9999, "01:00:00,000"),  # rolls minutes -> hours
    ],
)
def test_format_timestamp_integer_ms_rounding_rolls_over(seconds, expected):
    assert format_timestamp(seconds) == expected


@pytest.mark.parametrize(
    "seconds",
    [0.0, 1.0, 59.9999, 60.0, 119.9999, 3599.9999, 3600.0, 7199.9999],
)
def test_format_timestamp_seconds_field_never_sixty(seconds):
    # The seconds field of a valid SRT timestamp must be in 00..59.
    ts = format_timestamp(seconds)
    secs = int(ts[6:8])
    assert 0 <= secs <= 59


@pytest.mark.parametrize(
    "seconds",
    [0.0, 1.5, 59.999, 3661.5, 7322.123],
)
def test_format_timestamp_well_formed_shape(seconds):
    ts = format_timestamp(seconds)
    assert len(ts) == len("00:00:00,000")
    hh, mm, rest = ts.split(":")
    ss, ms = rest.split(",")
    assert len(hh) == 2 and len(mm) == 2 and len(ss) == 2 and len(ms) == 3
    assert 0 <= int(mm) <= 59
    assert 0 <= int(ss) <= 59


# ---------------------------------------------------------------------------
# split_subtitle  (balanced, max-2-line wrapping)
# ---------------------------------------------------------------------------


def _lines(result: str) -> list[str]:
    return result.split("\n") if result else []


@pytest.mark.parametrize("empty", ["", "   ", "\t\n  "])
def test_split_subtitle_empty_or_whitespace_is_empty(empty):
    assert split_subtitle(empty) == ""


def test_split_subtitle_single_short_text_stays_one_line():
    assert split_subtitle("hello world") == "hello world"
    assert _lines(split_subtitle("hello world")) == ["hello world"]


def test_split_subtitle_collapses_internal_whitespace():
    assert split_subtitle("a  b   c") == "a b c"


@pytest.mark.parametrize(
    "text",
    [
        "the quick brown fox jumps over the lazy dog",
        "one two three four five six seven eight nine ten",
        "a b c d e f g h i j k l m n o p q r s t u v",
        "short",
        "supercalifragilistic",  # oversize single word
        "hi superlongwordthatexceeds bye",  # oversize word in the middle
    ],
)
@pytest.mark.parametrize("max_chars", [5, 10, 20, 42])
def test_split_subtitle_every_line_within_budget_when_possible(text, max_chars):
    # Every produced line must be <= max_chars, EXCEPT a single word that is
    # itself longer than max_chars (which has nowhere else to go).
    for line in _lines(split_subtitle(text, max_chars=max_chars)):
        if " " in line:
            assert len(line) <= max_chars
        else:
            # a lone word may legitimately exceed max_chars
            assert line  # never an empty line


@pytest.mark.parametrize(
    "text",
    [
        "the quick brown fox jumps over the lazy dog",
        "supercalifragilistic and tiny",
        "supercalifragilistic",
        "a b c d e f g",
    ],
)
@pytest.mark.parametrize("max_chars", [3, 5, 10, 42])
def test_split_subtitle_no_empty_or_leading_blank_line(text, max_chars):
    # Even when the first word is oversize, there must be no spurious blank line.
    result = split_subtitle(text, max_chars=max_chars)
    assert not result.startswith("\n")
    assert "" not in _lines(result)


def test_split_subtitle_words_preserved_in_order():
    text = "the quick brown fox jumps over the lazy dog again now"
    result = split_subtitle(text, max_chars=20)
    assert result.replace("\n", " ").split() == text.split()


@pytest.mark.parametrize(
    "text",
    [
        "aaaa bbbb cccc dddd eeee ffff",
        "alpha beta gamma delta epsilon",
        "the quick brown fox jumps lazy",
    ],
)
def test_split_subtitle_two_lines_are_balanced(text):
    # When the text wraps to exactly two lines, the balancer should make the two
    # line lengths reasonably close. Pick a max_chars that forces two lines but
    # leaves slack for balancing.
    max_chars = (len(text) // 2) + 6
    lines = _lines(split_subtitle(text, max_chars=max_chars))
    assert len(lines) == 2
    # Both within budget and reasonably balanced.
    assert all(len(line) <= max_chars for line in lines)
    assert abs(len(lines[0]) - len(lines[1])) <= max(len(w) for w in text.split()) + 1


def test_split_subtitle_avoids_breaking_after_function_word():
    # A greedy wrap would strand a short function word at the end of line 1.
    # The balancer should instead break before it when an alternative exists.
    text = "I really wanted to go to the beach today"
    max_chars = 22
    lines = _lines(split_subtitle(text, max_chars=max_chars))
    assert len(lines) == 2
    last_word_line1 = lines[0].split()[-1].lower().strip(",.;:!?")
    assert last_word_line1 not in {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "as",
        "by",
        "is",
        "are",
        "was",
        "were",
    }


def test_split_subtitle_uses_config_default_max():
    # A line comfortably under MAX_LINE_LENGTH stays a single line by default.
    text = "a comfortably short caption line"
    assert len(text) <= MAX_LINE_LENGTH
    assert split_subtitle(text) == text


# ---------------------------------------------------------------------------
# split_subtitle  (smart break-point scoring in _balance_two)
#
# These pin the NEW behavior: when the text wraps to two lines, the balancer
# scores candidate break points and prefers a natural linguistic boundary
# (right after punctuation, not after a short function word, not orphaning a
# single word) over a purely length-balanced split.
# ---------------------------------------------------------------------------


def test_split_subtitle_breaks_after_punctuation_jfk():
    # The canonical case: a mid-sentence comma is the most natural break point.
    # Both halves fit in 42 chars, so the punctuation bonus wins over balance.
    text = "And so, my fellow Americans, ask not what your country can do for you."
    lines = _lines(split_subtitle(text, max_chars=42))
    assert lines == [
        "And so, my fellow Americans,",
        "ask not what your country can do for you.",
    ]
    assert all(len(line) <= 42 for line in lines)


def test_split_subtitle_breaks_after_mid_sentence_comma():
    # The comma split keeps both lines within budget, so it is chosen even
    # though a more length-balanced break exists elsewhere.
    text = "We came for the food, but we stayed for the company"
    lines = _lines(split_subtitle(text, max_chars=30))
    assert len(lines) == 2
    assert lines[0] == "We came for the food,"
    assert lines[1] == "but we stayed for the company"
    assert all(len(line) <= 30 for line in lines)


def test_split_subtitle_breaks_after_semicolon():
    # A semicolon is treated like other clause-ending punctuation for breaking.
    text = "He finished the race; she cheered from the stands"
    lines = _lines(split_subtitle(text, max_chars=28))
    assert len(lines) == 2
    assert lines[0] == "He finished the race;"
    assert lines[1] == "she cheered from the stands"
    assert all(len(line) <= 28 for line in lines)


def test_split_subtitle_punctuation_break_beats_balanced_split():
    # The break after the comma is more lopsided in length than a centre split
    # would be, yet the punctuation bonus makes it the chosen break.
    text = "Yes, the meeting will start later than usual"
    lines = _lines(split_subtitle(text, max_chars=40))
    assert len(lines) == 2
    assert lines[0] == "Yes,"
    assert lines[1] == "the meeting will start later than usual"


def test_split_subtitle_avoids_orphan_word_line():
    # Without orphan/widow penalties a greedy or naive balance might leave a
    # single short word on its own line; the scorer prefers a fuller split.
    text = "Once upon a time there lived a king"
    lines = _lines(split_subtitle(text, max_chars=25))
    assert len(lines) == 2
    # Neither line is a single orphaned word.
    assert len(lines[0].split()) > 1
    assert len(lines[1].split()) > 1
    assert all(len(line) <= 25 for line in lines)


def test_split_subtitle_falls_back_to_balanced_split_without_signal():
    # No punctuation and no function-word/orphan signal to differentiate the
    # candidate breaks, so the balancer falls back to minimizing length
    # imbalance: roughly even halves.
    text = "alpha beta gamma delta epsilon zeta"
    lines = _lines(split_subtitle(text, max_chars=20))
    assert len(lines) == 2
    assert all(len(line) <= 20 for line in lines)
    # Even split: three words per line.
    assert len(lines[0].split()) == 3
    assert len(lines[1].split()) == 3


# ---------------------------------------------------------------------------
# split_sentence_heuristically  (recursive, punctuation-aware)
# ---------------------------------------------------------------------------


def _fits(part: str, max_chars: int, max_lines: int) -> bool:
    return len(_lines(split_subtitle(part, max_chars=max_chars))) <= max_lines


@pytest.mark.parametrize(
    "sentence",
    [
        "short sentence",
        "one two three four five six seven eight",  # wraps to exactly 2 lines
    ],
)
def test_ssh_fitting_sentence_returned_as_single_element(sentence):
    result = split_sentence_heuristically(sentence, 42, 2)
    assert result == [sentence.strip()]


def test_ssh_strips_surrounding_whitespace_when_it_fits():
    assert split_sentence_heuristically("   short   ", 42, 2) == ["short"]


def test_ssh_oversize_single_word_returned_unsplit():
    # A lone word cannot be split on spaces/punctuation; it stays one element.
    word = "supercalifragilisticexpialidocious"
    assert split_sentence_heuristically(word, 5, 2) == [word]


@pytest.mark.parametrize(
    "sentence",
    [
        "alpha beta gamma delta epsilon zeta eta theta iota kappa "
        "lambda mu nu xi omicron pi rho sigma tau upsilon",
        "this is a very long clause with absolutely no punctuation or "
        "conjunctions at all here that keeps going and going forever",
        "I went to the store and I bought some milk because we needed it "
        "for breakfast tomorrow morning before going to work",
        "First part here is quite long, and the second part there is also "
        "fairly long indeed, third part everywhere all at once",
    ],
)
@pytest.mark.parametrize(("max_chars", "max_lines"), [(20, 2), (42, 2), (15, 1)])
def test_ssh_every_returned_part_fits_max_lines(sentence, max_chars, max_lines):
    # The core invariant: recursion continues until no part exceeds max_lines
    # (a part consisting of a single oversize word is the only allowed exception).
    parts = split_sentence_heuristically(sentence, max_chars, max_lines)
    for part in parts:
        if not _fits(part, max_chars, max_lines):
            assert len(part.split()) == 1  # only a lone oversize word may overflow


@pytest.mark.parametrize(
    "sentence",
    [
        "alpha beta gamma delta epsilon zeta eta theta iota kappa "
        "lambda mu nu xi omicron",
        "this is a very long clause with absolutely no punctuation or "
        "conjunctions at all here that keeps going and going forever",
        "I went to the store and I bought some milk because we needed it "
        "for breakfast tomorrow morning before going to work",
    ],
)
@pytest.mark.parametrize(("max_chars", "max_lines"), [(20, 2), (42, 2)])
def test_ssh_preserves_word_multiset(sentence, max_chars, max_lines):
    # Splitting only happens at whitespace/clause boundaries, so the multiset of
    # whitespace-delimited tokens is preserved exactly.
    parts = split_sentence_heuristically(sentence, max_chars, max_lines)
    joined = " ".join(parts).split()
    assert Counter(joined) == Counter(sentence.split())


def test_ssh_splits_at_or_after_a_midpoint_comma():
    # An over-long run with a comma near the middle should split at/after the
    # comma, with the comma staying attached to the end of the preceding part.
    sentence = (
        "aaaa bbbb cccc dddd eeee ffff, gggg hhhh iiii jjjj "
        "kkkk llll mmmm nnnn oooo pppp"
    )
    parts = split_sentence_heuristically(sentence, 20, 2)
    # Some part ends with the comma (the split was taken at/after it).
    assert any(part.rstrip().endswith(",") for part in parts)
    # And the comma-bearing token "ffff," is the last token of its part.
    comma_part = next(p for p in parts if "ffff," in p)
    assert comma_part.split()[-1] == "ffff,"


def test_ssh_splits_on_clause_conjunctions_when_overflowing():
    sentence = (
        "I went to the store and I bought some milk because we needed it "
        "for breakfast tomorrow morning before work"
    )
    parts = split_sentence_heuristically(sentence, 42, 2)
    # Clauses break before coordinating/subordinating conjunctions, so several
    # parts begin with one.
    assert any(p.startswith("and ") for p in parts)
    assert any(p.startswith("because ") for p in parts)
    # And every part fits.
    assert all(_fits(p, 42, 2) for p in parts)


def test_ssh_max_lines_one_forces_a_split():
    sentence = "alpha beta gamma delta epsilon zeta eta theta"
    parts = split_sentence_heuristically(sentence, 20, 1)
    assert len(parts) >= 2
    assert Counter(" ".join(parts).split()) == Counter(sentence.split())
    assert all(_fits(p, 20, 1) for p in parts)


def test_ssh_uses_config_defaults_make_sense():
    # Sanity check the imported config defaults are usable by the splitter.
    sentence = "word " * 60
    parts = split_sentence_heuristically(sentence.strip(), MAX_LINE_LENGTH, MAX_LINES)
    assert all(_fits(p, MAX_LINE_LENGTH, MAX_LINES) for p in parts)
