"""Smoke test: the pure modules import and run without the GPU/torch stack."""

from whisperx_subtitles.audio import distribute_segments_equally
from whisperx_subtitles.subtitles import format_timestamp, split_subtitle


def test_format_timestamp_zero_and_none():
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(None) == "00:00:00,000"


def test_split_subtitle_respects_max_chars():
    out = split_subtitle("a b c d e f g h", max_chars=5)
    assert all(len(line) <= 5 for line in out.split("\n"))


def test_distribute_segments_equally_basic():
    starts = distribute_segments_equally(120_000, 30_000, 3)
    assert len(starts) == 3
    assert starts[0] == 0
    assert starts[-1] == 90_000
