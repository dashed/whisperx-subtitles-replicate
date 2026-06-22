import gc
import logging
import math
import os

import torch
import whisperx
from cog import BaseModel, BaseRunner, Input, Path
from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF, DEFAULT_ALIGN_MODELS_TORCH

from whisperx_subtitles.audio import distribute_segments_equally, get_audio_duration
from whisperx_subtitles.config import (
    MAX_CPS,
    MAX_DURATION,
    MAX_LINE_LENGTH,
    MAX_LINES,
    MIN_DURATION,
    compute_type,
    device,
    whisper_arch,
)
from whisperx_subtitles.subtitles import generate_srt
from whisperx_subtitles.transcription import align, detect_language, diarize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Output(BaseModel):
    detected_language: str
    srt_output: str
    srt_file: Path


class Runner(BaseRunner):
    def setup(self):
        # Load the heavy WhisperModel once at boot; each request builds a cheap
        # pipeline wrapper around it (whisperx.load_model(..., model=...)) so the
        # ~1.5GB model is not reloaded per prediction. Alignment models are
        # cached lazily per language.
        self.whisper_model = whisperx.load_model(
            whisper_arch, device, compute_type=compute_type
        ).model
        self.align_models: dict = {}

    def run(
        self,
        audio_file: Path = Input(description="Audio file"),
        language: str | None = Input(
            description="ISO code of the language spoken in the audio, specify None to perform language detection",
            default=None,
        ),
        language_detection_min_prob: float = Input(
            description="If language is not specified, then the language will be detected recursively on different "
            "parts of the file until it reaches the given probability",
            default=0,
        ),
        language_detection_max_tries: int = Input(
            description="If language is not specified, then the language will be detected following the logic of "
            "language_detection_min_prob parameter, but will stop after the given max retries. If max "
            "retries is reached, the most probable language is kept.",
            default=5,
        ),
        initial_prompt: str | None = Input(
            description="Optional text to provide as a prompt for the first window",
            default=None,
        ),
        batch_size: int = Input(
            description="Parallelization of input audio transcription", default=64
        ),
        temperature: float = Input(
            description="Temperature to use for sampling", default=0
        ),
        vad_onset: float = Input(description="VAD onset", default=0.500),
        vad_offset: float = Input(description="VAD offset", default=0.363),
        align_output: bool = Input(
            description="Aligns whisper output to get accurate word-level timestamps",
            default=True,
        ),
        diarization: bool = Input(
            description="Assign speaker ID labels", default=False
        ),
        huggingface_access_token: str | None = Input(
            description="To enable diarization, please enter your HuggingFace token (read). You need to accept "
            "the user agreement for the models specified in the README.",
            default=None,
        ),
        min_speakers: int | None = Input(
            description="Minimum number of speakers if diarization is activated (leave blank if unknown)",
            default=None,
        ),
        max_speakers: int | None = Input(
            description="Maximum number of speakers if diarization is activated (leave blank if unknown)",
            default=None,
        ),
        max_line_length: int = Input(
            description="Maximum number of characters per subtitle line",
            default=MAX_LINE_LENGTH,
        ),
        max_lines: int = Input(
            description="Maximum number of lines per subtitle cue", default=MAX_LINES
        ),
        max_cps: float = Input(
            description="Maximum reading speed in characters per second",
            default=MAX_CPS,
        ),
        min_duration: float = Input(
            description="Minimum seconds a subtitle stays on screen",
            default=MIN_DURATION,
        ),
        max_duration: float = Input(
            description="Maximum seconds a subtitle stays on screen",
            default=MAX_DURATION,
        ),
        debug: bool = Input(
            description="Print out compute/inference times and memory usage information",
            default=False,
        ),
    ) -> Output:
        if diarization and not huggingface_access_token:
            raise ValueError(
                "huggingface_access_token is required when diarization is enabled."
            )

        with torch.inference_mode():
            asr_options = {
                "temperatures": [temperature],
                "initial_prompt": initial_prompt,
            }
            vad_options = {"vad_onset": vad_onset, "vad_offset": vad_offset}

            audio_duration = get_audio_duration(audio_file)

            if (
                language is None
                and language_detection_min_prob > 0
                and audio_duration > 30000
            ):
                segments_duration_ms = 30000
                language_detection_max_tries = min(
                    language_detection_max_tries,
                    math.floor(audio_duration / segments_duration_ms),
                )
                segments_starts = distribute_segments_equally(
                    audio_duration, segments_duration_ms, language_detection_max_tries
                )
                logger.info(
                    "Detecting language on segments starting at %s",
                    ", ".join(map(str, segments_starts)),
                )
                details = detect_language(
                    self.whisper_model,
                    audio_file,
                    segments_starts,
                    language_detection_min_prob,
                    language_detection_max_tries,
                )
                logger.info(
                    "Detected language %s (%.2f) after %d iterations",
                    details["language"],
                    details["probability"],
                    details["iterations"],
                )
                language = details["language"]

            # Reuse the cached heavy model; only the lightweight pipeline wrapper
            # (and per-request ASR/VAD options) is built here.
            model = whisperx.load_model(
                whisper_arch,
                device,
                compute_type=compute_type,
                language=language,
                asr_options=asr_options,
                vad_options=vad_options,
                model=self.whisper_model,
            )
            audio = whisperx.load_audio(audio_file)
            result = model.transcribe(audio, batch_size=batch_size)
            detected_language = result["language"]
            del model
            gc.collect()
            torch.cuda.empty_cache()

            if not result["segments"]:
                logger.warning("No speech detected; returning empty subtitles.")
            else:
                if align_output:
                    if (
                        detected_language in DEFAULT_ALIGN_MODELS_TORCH
                        or detected_language in DEFAULT_ALIGN_MODELS_HF
                    ):
                        result = align(audio, result, debug, self.align_models)
                    else:
                        logger.warning(
                            "Cannot align output: language %s is not supported for alignment.",
                            detected_language,
                        )

                if diarization:
                    result = diarize(
                        audio,
                        result,
                        debug,
                        huggingface_access_token,
                        min_speakers,
                        max_speakers,
                    )

            if debug:
                logger.info(
                    "max gpu memory allocated over runtime: %.2f GB",
                    torch.cuda.max_memory_reserved() / (1024**3),
                )

        audio_basename = os.path.basename(str(audio_file)).rsplit(".", 1)[0]
        srt_file = f"/tmp/{audio_basename}.{detected_language}.srt"
        srt_output = generate_srt(
            result["segments"],
            language=detected_language,
            max_line_length=max_line_length,
            max_lines=max_lines,
            max_cps=max_cps,
            min_duration=min_duration,
            max_duration=max_duration,
        )
        with open(srt_file, "w", encoding="utf-8") as srt:
            srt.write(srt_output)

        return Output(
            detected_language=detected_language,
            srt_output=srt_output,
            srt_file=Path(srt_file),
        )
