"""WhisperX-backed transcription helpers: language detection, alignment, diarization."""

import gc
import logging
import time

import torch
import whisperx
from whisperx.audio import N_SAMPLES, log_mel_spectrogram
from whisperx.diarize import DiarizationPipeline

from .audio import extract_audio_segment
from .config import device

logger = logging.getLogger(__name__)


def detect_language(
    whisper_model,
    full_audio_file_path,
    segments_starts,
    language_detection_min_prob,
    language_detection_max_tries,
    iteration=1,
):
    """Detect the spoken language on successive 30s windows, keeping the most
    probable result. Reuses the already-loaded WhisperModel (no per-call reload)."""
    start_ms = segments_starts[iteration - 1]
    audio_segment_file_path = extract_audio_segment(
        full_audio_file_path, start_ms, 30000
    )
    audio = whisperx.load_audio(audio_segment_file_path)

    model_n_mels = whisper_model.feat_kwargs.get("feature_size")
    segment = log_mel_spectrogram(
        audio[:N_SAMPLES],
        n_mels=model_n_mels if model_n_mels is not None else 80,
        padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0],
    )
    encoder_output = whisper_model.encode(segment)
    results = whisper_model.model.detect_language(encoder_output)
    language_token, language_probability = results[0][0]
    language = language_token[2:-2]

    logger.info(
        "Iteration %d - detected language: %s (%.2f)",
        iteration,
        language,
        language_probability,
    )
    audio_segment_file_path.unlink()

    detected_language = {
        "language": language,
        "probability": language_probability,
        "iterations": iteration,
    }

    if (
        language_probability >= language_detection_min_prob
        or iteration >= language_detection_max_tries
    ):
        return detected_language

    next_detected = detect_language(
        whisper_model,
        full_audio_file_path,
        segments_starts,
        language_detection_min_prob,
        language_detection_max_tries,
        iteration + 1,
    )
    if next_detected["probability"] > detected_language["probability"]:
        return next_detected
    return detected_language


def align(audio, result, debug, align_cache=None):
    """Align transcription to word-level timestamps. If align_cache (a dict) is
    given, the per-language alignment model is loaded once and reused."""
    start_time = time.time_ns() / 1e6
    language = result["language"]

    if align_cache is not None and language in align_cache:
        model_a, metadata = align_cache[language]
    else:
        model_a, metadata = whisperx.load_align_model(
            language_code=language, device=device
        )
        if align_cache is not None:
            align_cache[language] = (model_a, metadata)

    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    if debug:
        logger.info("Alignment took %.2f ms", time.time_ns() / 1e6 - start_time)

    if align_cache is None:
        del model_a
        gc.collect()
        torch.cuda.empty_cache()

    return result


def diarize(audio, result, debug, huggingface_access_token, min_speakers, max_speakers):
    start_time = time.time_ns() / 1e6

    diarize_model = DiarizationPipeline(token=huggingface_access_token, device=device)
    diarize_segments = diarize_model(
        audio, min_speakers=min_speakers, max_speakers=max_speakers
    )
    result = whisperx.assign_word_speakers(diarize_segments, result)

    if debug:
        logger.info("Diarization took %.2f ms", time.time_ns() / 1e6 - start_time)

    del diarize_model
    gc.collect()
    torch.cuda.empty_cache()

    return result
