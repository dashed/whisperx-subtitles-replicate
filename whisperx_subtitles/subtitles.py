"""Pure subtitle logic: sentence/clause splitting, cue merge/split, timing
normalization, and SRT rendering.

Imports only the standard library + pysbd, so it is unit-testable without the
GPU/torch/whisperx stack.

Design: the text-shaping functions (``split_at_sentence_end``,
``merge_short_cues``, ``split_long_cues_with_word_timings``) anchor each cue to
its raw word-level start/end times and decide *text*; a single final pass,
``normalize_cues``, enforces all timing invariants — ordered, non-overlapping
cues with reading-comfort (CPS) and min/max-duration bounds that never linger
far past the actual speech. ``generate_srt`` wires them together.
"""

from __future__ import annotations

import logging
import re

import pysbd

from .config import (
    LINE_BREAK_FUNCTION_WORD_PENALTY,
    LINE_BREAK_ORPHAN_PENALTY,
    LINE_BREAK_PUNCTUATION_BONUS,
    MAX_CPS,
    MAX_DURATION,
    MAX_LEAD_OUT,
    MAX_LINE_LENGTH,
    MAX_LINES,
    MERGE_MAX_GAP,
    MIN_DURATION,
    MIN_GAP,
    PAUSE_THRESHOLD,
)
from .types import Cue, Segment, Word

logger = logging.getLogger(__name__)

# Clause-break heuristic (NOTE: English-only). Split after , or ; and before a
# coordinating/subordinating conjunction.
_CONJUNCTIONS = (
    "and", "but", "or", "so", "because", "if", "when", "while", "although",
    "since", "after", "before", "unless", "until", "where", "whereas",
    "whether", "as", "though",
)  # fmt: skip
_CLAUSE_SPLIT = re.compile(
    r"(?<=[,;])\s+|(?<=\s)(?=\b(?:" + "|".join(_CONJUNCTIONS) + r")\b)"
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Short function words we avoid stranding at the end of a wrapped line.
_NO_BREAK_AFTER = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "with", "as", "by", "is", "are", "was", "were",
}  # fmt: skip


# --------------------------------------------------------------------------- #
# Timestamp formatting
# --------------------------------------------------------------------------- #
def format_timestamp(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "00:00:00,000"
    # Work in integer milliseconds so rounding rolls over correctly
    # (e.g. 59.9999s -> 00:01:00,000, not the invalid 00:00:60,000).
    total_ms = round(seconds * 1000)
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, millis = divmod(total_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# --------------------------------------------------------------------------- #
# Line wrapping (balanced, max-2-line aware)
# --------------------------------------------------------------------------- #
def _greedy_lines(words: list[str], max_chars: int) -> list[list[str]]:
    """Greedy word-wrap into lines each <= max_chars (oversize words get a line)."""
    lines: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and current_len + extra > max_chars:
            lines.append(current)
            current, current_len = [word], len(word)
        else:
            current.append(word)
            current_len += extra
    if current:
        lines.append(current)
    return lines


def _balance_two(words: list[str], max_chars: int) -> list[list[str]] | None:
    """Best 2-line split (both lines <= max_chars). Scores candidate break points
    (lower = better): start from line-length imbalance, reward breaking right
    after punctuation, penalize breaking after a short function word or leaving a
    one-word line, so the break lands at a natural linguistic boundary."""
    best: tuple[float, list[list[str]]] | None = None
    for i in range(1, len(words)):
        left, right = " ".join(words[:i]), " ".join(words[i:])
        if len(left) > max_chars or len(right) > max_chars:
            continue
        score: float = abs(len(left) - len(right))
        prev_word = words[i - 1]
        if prev_word.rstrip().endswith((",", ";", ":", ".", "!", "?", "—", "–")):
            score -= LINE_BREAK_PUNCTUATION_BONUS
        if prev_word.lower().strip(",.;:!?\"'") in _NO_BREAK_AFTER:
            score += LINE_BREAK_FUNCTION_WORD_PENALTY
        if i == 1 or i == len(words) - 1:  # one-word line on either side
            score += LINE_BREAK_ORPHAN_PENALTY
        if best is None or score < best[0]:
            best = (score, [words[:i], words[i:]])
    return best[1] if best else None


def split_subtitle(text: str, max_chars: int = MAX_LINE_LENGTH) -> str:
    words = text.split()
    if not words:
        return ""
    lines = _greedy_lines(words, max_chars)
    if len(lines) == 2:
        balanced = _balance_two(words, max_chars)
        if balanced is not None:
            lines = balanced
    return "\n".join(" ".join(line) for line in lines)


def _line_count(text: str, max_chars: int) -> int:
    words = text.split()
    return len(_greedy_lines(words, max_chars)) if words else 0


# --------------------------------------------------------------------------- #
# Sentence / clause splitting
# --------------------------------------------------------------------------- #
def _best_split_index(words: list[str], mid: int) -> int:
    """Index nearest the midpoint that splits *after* a punctuation mark."""
    for offset in range(len(words)):
        for idx in (mid + offset, mid - offset):
            if 1 <= idx < len(words) and words[idx - 1].rstrip().endswith(
                (",", ";", ":", ".", "!", "?")
            ):
                return idx
    return mid


def _split_to_fit(part: str, max_line_length: int, max_lines: int) -> list[str]:
    """Recursively split a part until each piece fits in max_lines lines,
    preferring punctuation boundaries near the midpoint over a blind halving."""
    part = part.strip()
    if not part:
        return []
    if _line_count(part, max_line_length) <= max_lines:
        return [part]
    words = part.split()
    if len(words) <= 1:
        return [part]
    split_idx = _best_split_index(words, len(words) // 2)
    left = " ".join(words[:split_idx])
    right = " ".join(words[split_idx:])
    return _split_to_fit(left, max_line_length, max_lines) + _split_to_fit(
        right, max_line_length, max_lines
    )


def split_sentence_heuristically(
    sentence: str, max_line_length: int, max_lines: int
) -> list[str]:
    if _line_count(sentence, max_line_length) <= max_lines:
        return [sentence.strip()]
    parts = [p.strip() for p in _CLAUSE_SPLIT.split(sentence) if p.strip()]
    final_parts: list[str] = []
    for part in parts:
        final_parts.extend(_split_to_fit(part, max_line_length, max_lines))
    return final_parts


# --------------------------------------------------------------------------- #
# Cue construction
# --------------------------------------------------------------------------- #
def _cue_speaker(words: list[Word] | None) -> str | None:
    """Majority speaker label among words that carry one (set by diarization)."""
    if not words:
        return None
    speakers = [w["speaker"] for w in words if w.get("speaker")]
    if not speakers:
        return None
    return max(set(speakers), key=speakers.count)


def _first_start(words: list[Word]) -> float | None:
    return next((w["start"] for w in words if w.get("start") is not None), None)


def _last_end(words: list[Word]) -> float | None:
    return next((w["end"] for w in reversed(words) if w.get("end") is not None), None)


def split_at_sentence_end(
    segmenter: pysbd.Segmenter | None,
    text: str,
    word_data: list[Word],
    max_line_length: int = MAX_LINE_LENGTH,
    max_lines: int = MAX_LINES,
) -> list[Cue]:
    if segmenter is not None:
        sentences = segmenter.segment(text)
    else:
        sentences = _SENTENCE_SPLIT.split(text)

    result: list[Cue] = []
    word_index = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        for clause in split_sentence_heuristically(
            sentence, max_line_length, max_lines
        ):
            clause = clause.strip()
            if not clause:
                continue
            count = len(clause.split())
            clause_words = word_data[word_index : word_index + count]
            word_index += count

            start = _first_start(clause_words) if clause_words else None
            end = _last_end(clause_words) if clause_words else None
            if start is not None and end is not None:
                result.append(
                    {
                        "text": clause,
                        "start": start,
                        "end": end,
                        "word_data": clause_words,
                        "speaker": _cue_speaker(clause_words),
                    }
                )
            else:
                # No usable timings: anchor to the previous cue's end; the final
                # normalize pass gives it a readable duration.
                prev_end = result[-1]["end"] if result else 0.0
                result.append(
                    {
                        "text": clause,
                        "start": prev_end,
                        "end": prev_end,
                        "word_data": None,
                        "speaker": None,
                    }
                )
    return result


# --------------------------------------------------------------------------- #
# Cue merging / splitting
# --------------------------------------------------------------------------- #
def _reading_duration(text: str, max_cps: float = MAX_CPS) -> float:
    return len(text) / max_cps if max_cps > 0 else 0.0


def _displayed_length(cue: Cue) -> int:
    """Characters actually shown on screen, including the [SPEAKER_xx] prefix."""
    length = len(cue["text"])
    speaker = cue.get("speaker")
    if speaker:
        length += len(f"[{speaker}] ")
    return length


def merge_short_cues(
    cues: list[Cue],
    max_line_length: int = MAX_LINE_LENGTH,
    max_lines: int = MAX_LINES,
    max_cps: float = MAX_CPS,
    min_duration: float = MIN_DURATION,
    max_gap: float = MERGE_MAX_GAP,
) -> list[Cue]:
    """Merge an adjacent cue into the previous one when the previous cue is too
    short to read, the merge stays within max_lines and the reading-speed (CPS)
    ceiling, the time gap is small, and the speaker doesn't change."""
    merged: list[Cue] = []
    for cue in cues:
        if not merged:
            merged.append(cue)
            continue
        prev = merged[-1]
        combined_text = prev["text"] + " " + cue["text"]
        gap = cue["start"] - prev["end"]
        prev_duration = prev["end"] - prev["start"]
        combined_duration = cue["end"] - prev["start"]
        too_short = prev_duration < max(
            min_duration, _reading_duration(prev["text"], max_cps)
        )
        fits = _line_count(combined_text, max_line_length) <= max_lines
        speaker = prev.get("speaker")
        combined_len = len(combined_text) + (len(f"[{speaker}] ") if speaker else 0)
        cps_ok = combined_duration <= 0 or combined_len <= max_cps * combined_duration
        same_speaker = prev.get("speaker") == cue.get("speaker")
        if too_short and fits and cps_ok and 0 <= gap <= max_gap and same_speaker:
            prev["text"] = combined_text
            prev["end"] = cue["end"]
            prev_wd, cue_wd = prev.get("word_data"), cue.get("word_data")
            prev["word_data"] = (
                prev_wd + cue_wd if prev_wd is not None and cue_wd is not None else None
            )
        else:
            merged.append(cue)
    return merged


def _make_chunk_cue(words: list[str], word_data: list[Word], parent: Cue) -> Cue:
    start = _first_start(word_data)
    end = _last_end(word_data)
    return {
        "text": " ".join(words),
        "start": start if start is not None else parent["start"],
        "end": end if end is not None else parent["end"],
        "word_data": word_data,
        "speaker": _cue_speaker(word_data),
    }


def split_long_cue_without_word_timings(
    cue: Cue, max_line_length: int = MAX_LINE_LENGTH, max_lines: int = MAX_LINES
) -> list[Cue]:
    """Split a cue with no usable word timings into max_lines-line chunks,
    distributing the cue's duration proportionally by chunk length."""
    lines = split_subtitle(cue["text"], max_chars=max_line_length).split("\n")
    chunks: list[str] = []
    current: list[str] = []
    for line in lines:
        current.append(line)
        if len(current) == max_lines:
            chunks.append("\n".join(current))
            current = []
    if current:
        chunks.append("\n".join(current))

    total_len = sum(len(c.replace("\n", " ")) for c in chunks)
    start = cue["start"]
    total_duration = max(cue["end"] - cue["start"], 0.0)
    new_cues: list[Cue] = []
    for chunk in chunks:
        proportion = len(chunk.replace("\n", " ")) / total_len if total_len > 0 else 0
        chunk_end = start + total_duration * proportion
        new_cues.append(
            {
                "text": chunk,
                "start": start,
                "end": chunk_end,
                "word_data": None,
                "speaker": cue.get("speaker"),
            }
        )
        start = chunk_end
    return new_cues


def split_at_pauses(
    cues: list[Cue], pause_threshold: float = PAUSE_THRESHOLD
) -> list[Cue]:
    """Split each cue at internal silences >= pause_threshold that also fall on a
    clause boundary (the preceding word ends with punctuation), so cue boundaries
    land on natural pauses without fragmenting a phrase mid-clause."""
    out: list[Cue] = []
    for cue in cues:
        out.extend(_split_cue_at_pauses(cue, pause_threshold))
    return out


def _split_cue_at_pauses(cue: Cue, pause_threshold: float) -> list[Cue]:
    word_data = cue.get("word_data")
    words = cue["text"].split()
    if not word_data or len(words) != len(word_data) or len(word_data) < 2:
        return [cue]
    boundaries = [0]
    for i in range(1, len(word_data)):
        prev_end = word_data[i - 1].get("end")
        cur_start = word_data[i].get("start")
        ends_clause = (
            words[i - 1].rstrip().endswith((",", ";", ":", ".", "!", "?", "—", "–"))
        )
        if (
            prev_end is not None
            and cur_start is not None
            and cur_start - prev_end >= pause_threshold
            and ends_clause
        ):
            boundaries.append(i)
    boundaries.append(len(word_data))
    if len(boundaries) <= 2:  # no qualifying pause found
        return [cue]
    return [
        _make_chunk_cue(words[a:b], word_data[a:b], cue)
        for a, b in zip(boundaries, boundaries[1:], strict=False)
    ]


def split_long_cues_with_word_timings(
    cues: list[Cue],
    max_line_length: int = MAX_LINE_LENGTH,
    max_lines: int = MAX_LINES,
) -> list[Cue]:
    """Split cues that exceed max_lines into chunks, anchoring each chunk to its
    own word-level timings. Cues without aligned word data fall back to the
    proportional splitter."""
    new_cues: list[Cue] = []
    for cue in cues:
        words = cue["text"].split()
        word_data = cue.get("word_data")
        if not word_data or len(words) != len(word_data):
            if _line_count(cue["text"], max_line_length) <= max_lines:
                new_cues.append(cue)
            else:
                new_cues.extend(
                    split_long_cue_without_word_timings(cue, max_line_length, max_lines)
                )
            continue

        if _line_count(cue["text"], max_line_length) <= max_lines:
            new_cues.append(cue)
            continue

        current_words: list[str] = []
        current_wd: list[Word] = []
        for word, timing in zip(words, word_data, strict=True):
            trial = current_words + [word]
            if (
                current_words
                and _line_count(" ".join(trial), max_line_length) > max_lines
            ):
                new_cues.append(_make_chunk_cue(current_words, current_wd, cue))
                current_words, current_wd = [word], [timing]
            else:
                current_words.append(word)
                current_wd.append(timing)
        if current_words:
            new_cues.append(_make_chunk_cue(current_words, current_wd, cue))
    return new_cues


# --------------------------------------------------------------------------- #
# Timing normalization (the synchronization safety net)
# --------------------------------------------------------------------------- #
def normalize_cues(
    cues: list[Cue],
    min_duration: float = MIN_DURATION,
    max_duration: float = MAX_DURATION,
    max_cps: float = MAX_CPS,
    min_gap: float = MIN_GAP,
    max_lead_out: float = MAX_LEAD_OUT,
) -> list[Cue]:
    """Final pass guaranteeing cues are ordered and non-overlapping, with
    reading-comfort durations that never linger far past the actual speech.

    Invariants on the result: start[i] >= end[i-1] + min_gap; end >= start;
    end <= next_start - min_gap; duration bounded by [min_duration, max_duration]
    where ordering allows; end <= last_spoken_word_end + max_lead_out."""
    out: list[Cue] = []
    n = len(cues)
    for i, cue in enumerate(cues):
        start = float(cue["start"])
        end = float(cue["end"])

        if out:  # no overlap with previous
            start = max(start, out[-1]["end"] + min_gap)
        end = max(end, start)

        # reading-comfort floor: the cue must stay long enough to read (counting
        # the displayed length, including any speaker prefix)
        reading_secs = _displayed_length(cue) / max_cps if max_cps > 0 else 0.0
        reading_end = start + max(min_duration, reading_secs)
        end = max(end, reading_end)

        # soft cap: prefer not to linger far past the spoken audio, but never
        # below the reading floor (readability wins over a tight lead-out)
        word_data = cue.get("word_data")
        word_end = _last_end(word_data) if word_data else None
        if word_end is not None:
            end = min(end, max(word_end + max_lead_out, reading_end))

        # hard caps: absolute max duration, then never overlap the next cue
        end = min(end, start + max_duration)
        if i + 1 < n:
            end = min(end, float(cues[i + 1]["start"]) - min_gap)

        end = max(end, start)  # overlap-avoidance may have squeezed below start
        out.append(
            {
                "text": cue["text"],
                "start": start,
                "end": end,
                "word_data": cue.get("word_data"),
                "speaker": cue.get("speaker"),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Top-level SRT generation
# --------------------------------------------------------------------------- #
def generate_srt(
    segments: list[Segment],
    language: str,
    *,
    max_line_length: int = MAX_LINE_LENGTH,
    max_lines: int = MAX_LINES,
    max_cps: float = MAX_CPS,
    min_duration: float = MIN_DURATION,
    max_duration: float = MAX_DURATION,
) -> str:
    segmenter = None
    try:
        segmenter = pysbd.Segmenter(language=language, clean=False)
    except Exception as e:  # pysbd raises for unsupported languages
        logger.warning(
            "pysbd segmenter unavailable for language %r (%s); using regex fallback",
            language,
            e,
        )

    cues: list[Cue] = []
    for segment in segments:
        cues.extend(
            split_at_sentence_end(
                segmenter,
                segment["text"],
                segment.get("words", []),
                max_line_length,
                max_lines,
            )
        )

    cues = merge_short_cues(cues, max_line_length, max_lines, max_cps, min_duration)
    cues = split_at_pauses(cues)
    cues = split_long_cues_with_word_timings(cues, max_line_length, max_lines)
    cues = normalize_cues(cues, min_duration, max_duration, max_cps)

    output_srt = ""
    for index, cue in enumerate(cues, start=1):
        speaker = cue.get("speaker")
        prefix = f"[{speaker}] " if speaker else ""
        # Wrap the prefix together with the text so the [SPEAKER_xx] tag counts
        # toward the line width (otherwise the first line could overflow).
        text = split_subtitle(f"{prefix}{cue['text']}", max_chars=max_line_length)
        output_srt += f"{index}\n"
        output_srt += (
            f"{format_timestamp(cue['start'])} --> {format_timestamp(cue['end'])}\n"
        )
        output_srt += f"{text}\n\n"
    return output_srt
