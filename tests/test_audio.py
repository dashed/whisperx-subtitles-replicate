"""Characterization tests for the audio timing/probing helpers.

``distribute_segments_equally`` is pure and tested directly. ``get_audio_duration``
and ``extract_audio_segment`` shell out to ffmpeg, so we monkeypatch the ffmpeg
calls and never invoke the real binary.
"""

import pytest

import whisperx_subtitles.audio as audio
from whisperx_subtitles.audio import (
    distribute_segments_equally,
    extract_audio_segment,
    get_audio_duration,
)

# ---------------------------------------------------------------------------
# distribute_segments_equally (pure)
# ---------------------------------------------------------------------------


def test_distribute_single_iteration_returns_zero_only():
    # iterations == 1 -> spacing 0, single start at 0.
    assert distribute_segments_equally(120_000, 30_000, 1) == [0]


def test_distribute_two_iterations():
    # First start is 0; last is forced to total - segments.
    assert distribute_segments_equally(120_000, 30_000, 2) == [0, 90_000]


def test_distribute_three_iterations_evenly_spaced():
    assert distribute_segments_equally(120_000, 30_000, 3) == [0, 45_000, 90_000]


def test_distribute_last_element_is_total_minus_segments():
    starts = distribute_segments_equally(100_000, 10_000, 4)
    assert starts[0] == 0
    assert starts[-1] == 100_000 - 10_000


@pytest.mark.parametrize(
    "total,segs,iters,expected",
    [
        (120_000, 30_000, 1, [0]),
        (120_000, 30_000, 2, [0, 90_000]),
        (120_000, 30_000, 3, [0, 45_000, 90_000]),
        (100_000, 10_000, 4, [0, 30_000, 60_000, 90_000]),
        # Floor-division spacing: available=70, spacing=70//2=35, then the last
        # element is overwritten with total - segments = 70.
        (100, 30, 3, [0, 35, 70]),
    ],
)
def test_distribute_parametrized(total, segs, iters, expected):
    assert distribute_segments_equally(total, segs, iters) == expected


def test_distribute_length_matches_iterations():
    for iters in range(1, 8):
        assert len(distribute_segments_equally(500_000, 20_000, iters)) == iters


# ---------------------------------------------------------------------------
# get_audio_duration (ffmpeg.probe monkeypatched)
# ---------------------------------------------------------------------------


def test_get_audio_duration_converts_seconds_to_ms(monkeypatch):
    fake_probe = {
        "streams": [
            {"codec_type": "video", "duration": "999.0"},
            {"codec_type": "audio", "duration": "12.5"},
        ]
    }

    def fake_probe_fn(file_path):
        return fake_probe

    monkeypatch.setattr(audio.ffmpeg, "probe", fake_probe_fn)

    # 12.5 s -> 12500.0 ms; first *audio* stream is selected, not the video one.
    assert get_audio_duration("whatever.wav") == 12_500.0


def test_get_audio_duration_raises_without_audio_stream(monkeypatch):
    fake_probe = {"streams": [{"codec_type": "video", "duration": "10.0"}]}

    monkeypatch.setattr(audio.ffmpeg, "probe", lambda file_path: fake_probe)

    with pytest.raises(ValueError, match="No audio stream found"):
        get_audio_duration("novideo.mp4")


# ---------------------------------------------------------------------------
# extract_audio_segment (ffmpeg.input chain monkeypatched; no real ffmpeg)
# ---------------------------------------------------------------------------


def test_extract_audio_segment_returns_temp_path_with_suffix(monkeypatch):
    calls = {}

    class FakeStream:
        def output(self, out_path, **kwargs):
            calls["output_path"] = out_path
            calls["output_kwargs"] = kwargs
            return self

        def run(self, **kwargs):
            calls["run_kwargs"] = kwargs
            return (b"", b"")

    def fake_input(path, **kwargs):
        calls["input_path"] = path
        calls["input_kwargs"] = kwargs
        return FakeStream()

    monkeypatch.setattr(audio.ffmpeg, "input", fake_input)

    result = extract_audio_segment(
        "/tmp/song.mp3", start_time_ms=2000, duration_ms=5000
    )

    # Returns a Path to a temp file that preserves the input extension.
    assert result.suffix == ".mp3"
    # ms are converted to seconds for ffmpeg's ss / t options.
    assert calls["input_kwargs"]["ss"] == 2.0
    assert calls["output_kwargs"]["t"] == 5.0
    # output target is the same temp file path that is returned.
    assert calls["output_path"] == str(result)

    # Clean up the empty temp file created by NamedTemporaryFile.
    result.unlink(missing_ok=True)
