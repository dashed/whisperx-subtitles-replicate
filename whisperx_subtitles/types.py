"""Shared TypedDict definitions for transcription segments and subtitle cues."""

from typing import TypedDict


class Word(TypedDict):
    end: float | None
    word: str
    score: float | None
    start: float | None


class Segment(TypedDict):
    end: float
    text: str
    start: float
    words: list[Word]


class Cue(TypedDict):
    text: str
    start: float
    end: float
    word_data: list[Word] | None


Segments = list[Segment]
