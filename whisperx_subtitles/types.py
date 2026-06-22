"""Shared TypedDict definitions for transcription segments and subtitle cues."""

from typing import NotRequired, TypedDict


class Word(TypedDict):
    end: float | None
    word: str
    score: float | None
    start: float | None
    speaker: NotRequired[str]  # set by diarization (whisperx.assign_word_speakers)


class Segment(TypedDict):
    end: float
    text: str
    start: float
    words: list[Word]
    speaker: NotRequired[str]


class Cue(TypedDict):
    text: str
    start: float
    end: float
    word_data: list[Word] | None
    speaker: NotRequired[str | None]


Segments = list[Segment]
