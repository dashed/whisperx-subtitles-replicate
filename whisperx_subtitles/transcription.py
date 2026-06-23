"""WhisperX-backed transcription helpers: language detection, alignment, diarization."""

import gc
import logging
import time

import torch
import whisperx
from whisperx.audio import N_SAMPLES, log_mel_spectrogram
from whisperx.diarize import DiarizationPipeline

from .audio import extract_audio_segment
from .config import MMS_ALIGN_MODEL, device

logger = logging.getLogger(__name__)

# whisper emits ISO-639-1 (2-letter) codes; uroman/MMS want ISO-639-3 (3-letter).
# uroman tolerates an unknown code (falls back to script-based rules), so this map
# only needs the common languages that hit the MMS fallback.
_ISO1_TO_ISO3 = {
    "th": "tha", "vi": "vie", "id": "ind", "ms": "msa", "sw": "swa",
    "am": "amh", "bn": "ben", "gu": "guj", "kn": "kan", "mr": "mar",
    "ne": "nep", "pa": "pan", "si": "sin", "ta": "tam", "my": "mya",
    "km": "khm", "lo": "lao", "hy": "hye", "az": "aze", "kk": "kaz",
    "ky": "kir", "uz": "uzb", "mn": "mon", "yo": "yor", "ha": "hau",
    "ig": "ibo", "zu": "zul", "tg": "tgk", "ps": "pus", "sd": "snd",
    "so": "som", "tt": "tat", "ba": "bak", "is": "isl",
}  # fmt: skip


def _iso2_to_iso3(code):
    return _ISO1_TO_ISO3.get(code, code) if code else None


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


def align_mms(audio, result, debug, align_cache=None):
    """Multilingual forced-alignment FALLBACK for languages outside whisperx's
    built-in alignment set, so they still get word-level timestamps.

    Uses MMS-300M via ctc-forced-aligner (1000+ languages). NOTE: the MMS weights
    are CC-BY-NC 4.0 (non-commercial). Never raises: on any failure each segment
    is returned with word-less timings, i.e. it degrades to segment-level timing
    exactly as the no-alignment path did.

    `audio`  : 16kHz mono float32 numpy array (from whisperx.load_audio).
    `result` : dict with result["language"] and result["segments"].
    Returns `result` with each segment carrying a `words` list of
    {"word", "start", "end", "score"} (start/end/score may be None).
    """
    from ctc_forced_aligner import (
        generate_emissions,
        get_alignments,
        get_spans,
        load_alignment_model,
        postprocess_results,
        preprocess_text,
    )

    start_time = time.time_ns() / 1e6
    language = result["language"]
    segments = result.get("segments", [])
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32

    if align_cache is not None and "__mms__" in align_cache:
        model_a, tokenizer = align_cache["__mms__"]
    else:
        model_a, tokenizer = load_alignment_model(
            device, model_path=MMS_ALIGN_MODEL, dtype=dtype
        )
        if align_cache is not None:
            align_cache["__mms__"] = (model_a, tokenizer)

    # Flat transcript in segment order; remember how many words each segment owns
    # so the flat aligned-word list can be redistributed back to its segment.
    seg_word_counts = [len((seg.get("text") or "").split()) for seg in segments]
    full_text = " ".join(
        " ".join((seg.get("text") or "").split()) for seg in segments
    ).strip()

    def _wordless():
        for seg in segments:
            seg["words"] = [
                {"word": w, "start": None, "end": None, "score": None}
                for w in (seg.get("text") or "").split()
            ]
        return result

    if not full_text:
        for seg in segments:
            seg["words"] = []
        return result

    try:
        waveform = torch.as_tensor(audio).to(device=device, dtype=dtype)
        emissions, stride = generate_emissions(model_a, waveform, batch_size=1)
        tokens_starred, text_starred = preprocess_text(
            full_text, romanize=True, language=_iso2_to_iso3(language)
        )
        aligns, scores, blank = get_alignments(emissions, tokens_starred, tokenizer)
        spans = get_spans(tokens_starred, aligns, blank)
        word_ts = postprocess_results(text_starred, spans, stride, scores)
    except Exception as exc:  # alignment must never break the request
        logger.warning("MMS alignment failed (%s); returning word-less segments", exc)
        return _wordless()

    cursor = 0
    n = len(word_ts)
    for seg, count in zip(segments, seg_word_counts, strict=False):
        words = []
        for _ in range(count):
            if cursor < n:
                wt = word_ts[cursor]
                words.append(
                    {
                        "word": wt.get("text", ""),
                        "start": wt.get("start"),
                        "end": wt.get("end"),
                        "score": wt.get("score"),
                    }
                )
                cursor += 1
            else:
                words.append({"word": "", "start": None, "end": None, "score": None})
        seg["words"] = words

    if debug:
        logger.info("MMS alignment took %.2f ms", time.time_ns() / 1e6 - start_time)

    if align_cache is None:
        del model_a, tokenizer
        gc.collect()
        if str(device).startswith("cuda"):
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
