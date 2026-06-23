# whisperx-subtitles-replicate

> Generates synchronized, readable SRT subtitles from transcribed audio with WhisperX (faster-whisper-large-v3)

This a fork of:

- https://github.com/victor-upmeet/whisperx-replicate
- https://replicate.com/victor-upmeet/whisperx

This script processes transcribed audio data to generate properly formatted subtitle files (`.srt`). It handles the splitting and merging of subtitle cues based on duration, line length, line count, and desired reading speed, ensuring that the resulting subtitles are readable and synchronized with the audio.

Code based on: https://github.com/m-bain/whisperX/issues/883

Here’s a high-level overview of how it achieves this:

1. **Generate transcription:** Uses WhisperX (faster-whisper-large-v3) to generate transcription with word-level timestamps.
2. **Sentence Segmentation:** Splits the transcribed text into sentences with `PySBD` (Python Sentence Boundary Disambiguation) for the languages it supports; for languages it does not (e.g. Thai), it falls back to the neural [`SaT`/`wtpsplit`](https://github.com/segment-any-text/wtpsplit) segmenter, which handles non-space-delimited and unpunctuated scripts.
3. **Initial Cue Creation:** For each sentence, the script creates an initial subtitle cue, including start and end times based on word-level timings.
4. **Cue Merging:**
   - Merges short cues that don't meet a minimum duration (e.g., 3 seconds) to ensure subtitles are displayed long enough for viewers to read.
   - Merges are performed without exceeding maximum line lengths or line counts.
   - Considers time gaps between cues to avoid merging cues that are too far apart in time.
5. **Cue Splitting:**
   - Splits long cues that exceed maximum line lengths or line counts into smaller cues.
   - Uses word-level timings to maintain accurate synchronization when splitting.
   - Avoids splitting in the middle of words or sentences when possible.
6. **Adjusting Cue Durations:**
   - Adjusts cue durations based on the desired words per second (e.g., 4 WPS) for comfortable reading.
   - Ensures that cue durations are not shorter than the minimum duration and do not exceed the maximum duration.
   - Re-adjusts durations after merging to match the new speech/reading rate.
7. **Handling Time Gaps:**
   - When merging cues or chunks, the script considers the time gap between them.
   - Avoids merging cues/chunks if the time gap exceeds a maximum acceptable duration (e.g., 1.5 seconds).
8. **SRT File Generation:** Formats each cue according to the SRT specification, including numbering, timing, and text formatting.

# Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management,
[ruff](https://docs.astral.sh/ruff/) for linting/formatting, and
[ty](https://docs.astral.sh/ty/) for type checking. `pyproject.toml` is the
single source of truth for dependencies.

## Dependencies

The runtime stack (`torch`, `whisperx`, `pyannote.audio`, ...) targets CUDA 12.8
on Linux/x86_64 — the Replicate GPU image — and has no macOS wheels, so the
lockfile is resolved for that platform only. Cog still builds from
`requirements.txt`, which is generated from the lock:

```sh
uv lock                  # resolve / update uv.lock from pyproject.toml
uv export --frozen --no-dev --no-emit-project --no-hashes \
  --format requirements-txt -o requirements.txt
```

After exporting, re-add the `--extra-index-url https://download.pytorch.org/whl/cu128`
line near the top of `requirements.txt` (uv omits explicit indexes on export, and
pip needs it to find the `+cu128` torch wheels). See the header in that file.

## Lint, format and type-check

These run without installing the heavy GPU stack, so they work on any machine:

```sh
uvx ruff check .          # lint
uvx ruff format .         # format
uvx ty check              # type check
```

(On the Linux GPU image, where the full dependencies are installed, you can also
run them via `uv run ruff ...` / `uv run ty check` with full import resolution.)

## Tests

The subtitle/formatting logic lives in the `whisperx_subtitles` package and is
deliberately free of the GPU stack, so the test suite runs on any machine
without installing torch/whisperx:

```sh
uvx --with pysbd --with ffmpeg-python --with numpy pytest   # any machine
uv run pytest                                                # on the Linux image (full env)
```

## Project layout

```
predict.py                     # Cog entry point: Runner (run/setup) + Output (thin glue)
whisperx_subtitles/
  config.py                    # runtime + subtitle-formatting constants (line length, CPS, durations)
  types.py                     # Word / Segment / Cue TypedDicts
  subtitles.py                 # pure subtitle logic (split, merge, timing normalization, SRT)
  audio.py                     # ffmpeg probing + pure segment-timing math
  transcription.py             # whisperx glue: language detection, alignment, diarization
tests/                         # pytest suite for the pure modules + mocked pipeline
```

## Subtitle formatting

Readability follows EBU-TT / Netflix-style guidelines, all configurable as model
inputs (`max_line_length`, `max_lines`, `max_cps`, `min_duration`, `max_duration`;
defaults in `config.py`): max characters per line, max 2 lines, a
characters-per-second reading-speed ceiling, and min/max on-screen duration. The
`normalize_cues` pass guarantees cues are ordered and non-overlapping and don't
linger far past the spoken audio. When `diarization` is enabled, each cue is
prefixed with its `[SPEAKER_xx]` label.

## Multilingual

Transcription (Whisper large-v3) covers ~99 languages. Word-level timestamps —
which the subtitle engine needs to time cues — are produced by forced alignment:

- **WhisperX's built-in alignment** (wav2vec2) covers ~41 languages natively.
- For any **other** language, the model falls back to **MMS forced alignment**
  (via [`ctc-forced-aligner`](https://github.com/MahmoudAshraf97/ctc-forced-aligner),
  1000+ languages) so those languages still get word-level cue timing instead of
  segment-level only. If alignment fails for any reason it degrades gracefully to
  segment-level timing — it never breaks a request.

> ⚠️ **License:** the MMS alignment weights
> ([`MahmoudAshraf/mms-300m-1130-forced-aligner`](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner),
> derived from Meta MMS) are **CC-BY-NC 4.0 (non-commercial)**. As of 2026 there is
> no permissively-licensed broad multilingual aligner. If you need commercial use,
> swap in a permissively-licensed per-language CTC model for the languages you care
> about. The SaT segmenter weights (`sat-3l-sm`) and `wtpsplit` are MIT-licensed.

Known limitation: for scripts with no word spacing (Thai, Lao, Khmer, Burmese),
ASR output is often a single unsegmented run, so MMS produces one timestamp span
rather than true per-word timing. Per-word timing there needs a word segmenter
(e.g. `pythainlp`), which is out of scope for now.

## Translation

Set the `translate_to` input (an ISO code like `en`, `es`, `ja`) to output
subtitles in a different language than the audio. Naive per-cue translation is
deliberately **not** used — it loses sentence context and breaks on word-order
differences (e.g. Korean SOV → English SVO) and length changes. Instead the model
runs a **cascade**:

1. Transcribe + align in the **source** language → word-level timestamps.
2. Reconstruct full **sentences** with their `[start, end]` spans.
3. Translate each whole sentence (context preserved) with
   [MADLAD-400](https://huggingface.co/google/madlad400-3b-mt) (Apache-2.0, 400+
   languages, any→any).
4. Re-segment each translation into cues and distribute the source sentence's
   time span across them **proportionally by character count**, so the
   translation stays synced to the audio.

The translation model is loaded lazily (only when `translate_to` is set), so
normal transcription requests don't pay its cost. For higher translation quality
you can swap `MT_MODEL` in `config.py` for a larger MADLAD or an LLM translator.

Limitation: translation re-timing is proportional, not word-aligned (the
translated words aren't in the audio); if a translated sentence is much longer
than its source span, reading speed may exceed the target CPS.

## Download models

```sh
./build.sh
```

The multilingual model weights (SaT segmenter and the MMS alignment model) are
not fetched by `build.sh` — they are pre-cached into the image at build time by
the `build.run` steps in `cog.yaml`, so cold starts don't download them.

## Publish to cog

```sh
cog login
cog push r8.im/dashed/whisperx-subtitles-replicate
```

# Usage

Extract audio with:

```
ffmpeg -i input_video.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 output_audio.wav
ffmpeg -i input_video.mp4 -vn -acodec aac -b:a 192k output_audio.m4a
```

# Model Information

WhisperX provides fast automatic speech recognition (70x realtime with large-v3) with word-level timestamps and speaker diarization.

Whisper is an ASR model developed by OpenAI, trained on a large dataset of diverse audio. Whilst it does produces highly accurate transcriptions, the corresponding timestamps are at the utterance-level, not per word, and can be inaccurate by several seconds. OpenAI’s whisper does not natively support batching, but WhisperX does.

Model used is for transcription is large-v3 from faster-whisper.

For more information about WhisperX, including implementation details, see the [WhisperX github repo](https://github.com/m-bain/whisperX).

## Diarization

When `diarization` is enabled, WhisperX uses pyannote's
[`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
pipeline. You must accept that model's user agreement on Hugging Face and pass a
read token via `huggingface_access_token`.

# Citation

```
@misc{bain2023whisperx,
      title={WhisperX: Time-Accurate Speech Transcription of Long-Form Audio},
      author={Max Bain and Jaesung Huh and Tengda Han and Andrew Zisserman},
      year={2023},
      eprint={2303.00747},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
