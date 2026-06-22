"""Tests for the recursion/selection logic of ``detect_language``.

``whisperx_subtitles.transcription`` imports ``torch`` and ``whisperx`` at module
import time, neither of which is installed in this test environment, so we stub
them into ``sys.modules`` before importing the module under test.
"""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub heavy / unavailable dependencies before importing the module under test.
# ---------------------------------------------------------------------------
for name in ("torch", "whisperx", "whisperx.audio", "whisperx.diarize"):
    sys.modules.setdefault(name, MagicMock())

# whisperx.audio must expose the names that transcription imports by name.
audio_mod = sys.modules["whisperx.audio"]
audio_mod.N_SAMPLES = 480000
audio_mod.log_mel_spectrogram = MagicMock()

# whisperx.diarize must expose DiarizationPipeline (imported by name).
sys.modules["whisperx.diarize"].DiarizationPipeline = MagicMock()

from whisperx_subtitles import transcription  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------
def make_whisper_model(scripted_results):
    """Build a fake whisper_model whose ``.model.detect_language`` returns the
    next scripted ``[[("<<xx>>", prob)]]`` result on each successive call.

    ``scripted_results`` is a list of ``(token, probability)`` tuples, one per
    expected detect_language call.
    """
    model = MagicMock()
    model.feat_kwargs = {"feature_size": 80}

    call_results = [[[(token, prob)]] for token, prob in scripted_results]
    model.model.detect_language.side_effect = call_results
    # encode just needs to return something to feed detect_language.
    model.encode.return_value = MagicMock(name="encoder_output")
    return model


@pytest.fixture(autouse=True)
def patch_audio(monkeypatch):
    """Neutralise all real audio/IO work performed by detect_language."""
    # extract_audio_segment returns a fake path whose .unlink() is a no-op.
    fake_path = MagicMock(name="audio_segment_path")
    monkeypatch.setattr(
        transcription, "extract_audio_segment", MagicMock(return_value=fake_path)
    )

    # whisperx.load_audio -> a real small numpy array so audio[:N_SAMPLES] and
    # audio.shape[0] behave like the production code expects.
    fake_audio = np.zeros(1000, dtype=np.float32)
    monkeypatch.setattr(
        transcription.whisperx, "load_audio", MagicMock(return_value=fake_audio)
    )

    # log_mel_spectrogram -> any array-like; only fed into model.encode (mocked).
    monkeypatch.setattr(
        transcription, "log_mel_spectrogram", MagicMock(return_value=MagicMock())
    )

    return fake_path


# A generous list of window start offsets so recursion never indexes past it.
SEGMENT_STARTS = [0, 30000, 60000, 90000, 120000]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_returns_dict_with_expected_keys():
    model = make_whisper_model([("<<en>>", 0.99)])
    result = transcription.detect_language(
        model,
        "audio.wav",
        SEGMENT_STARTS,
        language_detection_min_prob=0.9,
        language_detection_max_tries=3,
    )
    assert set(result.keys()) == {"language", "probability", "iterations"}


def test_stops_early_when_probability_meets_threshold():
    """First window already clears min_prob -> exactly one iteration, no recursion."""
    model = make_whisper_model([("<<en>>", 0.99), ("<<fr>>", 0.5)])
    result = transcription.detect_language(
        model,
        "audio.wav",
        SEGMENT_STARTS,
        language_detection_min_prob=0.9,
        language_detection_max_tries=3,
    )
    assert result["language"] == "en"
    assert result["probability"] == pytest.approx(0.99)
    assert result["iterations"] == 1
    # Only the first window should ever have been evaluated.
    assert model.model.detect_language.call_count == 1


def test_token_is_stripped():
    """A token like ``<<en>>`` must be reduced to ``en`` via [2:-2]."""
    model = make_whisper_model([("<<de>>", 0.95)])
    result = transcription.detect_language(
        model,
        "audio.wav",
        SEGMENT_STARTS,
        language_detection_min_prob=0.9,
        language_detection_max_tries=3,
    )
    assert result["language"] == "de"


def test_recurses_to_max_tries_and_returns_most_probable():
    """Probabilities stay below min_prob, so it recurses up to max_tries and
    returns the single most probable result across all iterations."""
    # probs [0.3, 0.8, 0.5]; min_prob unreachable -> recurse all 3 tries.
    model = make_whisper_model([("<<en>>", 0.3), ("<<fr>>", 0.8), ("<<de>>", 0.5)])
    result = transcription.detect_language(
        model,
        "audio.wav",
        SEGMENT_STARTS,
        language_detection_min_prob=0.99,
        language_detection_max_tries=3,
    )
    assert model.model.detect_language.call_count == 3
    # Most probable across iterations is the 0.8 / "fr" one (iteration 2).
    assert result["language"] == "fr"
    assert result["probability"] == pytest.approx(0.8)
    assert result["iterations"] == 2


def test_max_tries_cap_is_respected():
    """Even when nothing clears the threshold, recursion stops at max_tries."""
    model = make_whisper_model([("<<en>>", 0.1), ("<<fr>>", 0.2), ("<<de>>", 0.15)])
    result = transcription.detect_language(
        model,
        "audio.wav",
        SEGMENT_STARTS,
        language_detection_min_prob=0.99,
        language_detection_max_tries=3,
    )
    assert model.model.detect_language.call_count == 3
    # Best is the 0.2 "fr" from iteration 2.
    assert result["language"] == "fr"
    assert result["probability"] == pytest.approx(0.2)


def test_first_iteration_wins_ties_against_later():
    """The recursion keeps a later result only if its probability is strictly
    greater, so an equal-probability later window does NOT override the first."""
    model = make_whisper_model([("<<en>>", 0.5), ("<<fr>>", 0.5), ("<<de>>", 0.5)])
    result = transcription.detect_language(
        model,
        "audio.wav",
        SEGMENT_STARTS,
        language_detection_min_prob=0.99,
        language_detection_max_tries=3,
    )
    assert result["language"] == "en"
    assert result["iterations"] == 1


def test_model_is_reused_not_reloaded():
    """The cost fix: detect_language must never reload the model. It should use
    the passed-in whisper_model and never call whisperx.load_model."""
    model = make_whisper_model([("<<en>>", 0.99)])
    transcription.detect_language(
        model,
        "audio.wav",
        SEGMENT_STARTS,
        language_detection_min_prob=0.9,
        language_detection_max_tries=3,
    )
    # The passed model is the one that did the work.
    assert model.model.detect_language.called
    # And no reload happened.
    assert not transcription.whisperx.load_model.called


def test_default_feature_size_when_missing():
    """When feat_kwargs has no feature_size, code falls back to n_mels=80 and
    still completes normally."""
    model = make_whisper_model([("<<en>>", 0.99)])
    model.feat_kwargs = {}  # .get("feature_size") -> None -> fallback to 80
    result = transcription.detect_language(
        model,
        "audio.wav",
        SEGMENT_STARTS,
        language_detection_min_prob=0.9,
        language_detection_max_tries=3,
    )
    assert result["language"] == "en"
    # n_mels argument should have fallen back to 80.
    _, kwargs = transcription.log_mel_spectrogram.call_args
    assert kwargs["n_mels"] == 80
