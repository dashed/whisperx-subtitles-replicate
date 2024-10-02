# whisperx-subtitles-replicate

> Generates synchronized, readable SRT subtitles from transcribed audio with WhisperX (faster-whisper-large-v3)

This a fork of:

- https://github.com/victor-upmeet/whisperx-replicate
- https://replicate.com/victor-upmeet/whisperx

This processes transcribed audio data to generate properly formatted subtitle files (`.srt`). It handles the splitting and merging of subtitle cues based on duration, line length, and line count constraints, ensuring that the resulting subtitles are readable and synchronized with the audio.

Code based on: https://github.com/m-bain/whisperX/issues/883

Here’s a high-level overview of how it achieves this:

1. **Generate transcription:** Uses WhisperX (faster-whisper-large-v3) to generate transcription with word-level timestamps.
2. **Sentence Segmentation:** It uses the `PySBD` (Python Sentence Boundary Disambiguation) library to split the transcribed text into sentences, respecting language-specific punctuation and sentence boundaries.
3. **Initial Cue Creation:** For each sentence, the script creates an initial subtitle cue, including start and end times based on word-level timings if available.
4. **Cue Merging:**
   - Merges short cues that don't meet a minimum duration (e.g., 3 seconds) to ensure subtitles are displayed long enough for viewers to read.
   - It carefully merges cues without exceeding maximum line lengths or line counts.
5. **Cue Splitting:**
   - Long cues that exceed maximum line lengths or line counts are split into smaller cues.
   - When splitting, the script uses word-level timings to maintain accurate synchronization with the audio.
   - It handles cases where word timings are missing by distributing timing proportionally.
6. **Adjusting for Minimum Duration:**
   - After splitting, the script checks if any cues are shorter than the minimum duration.
   - It attempts to merge these short cues with adjacent ones while ensuring formatting constraints are met.
7. **SRT File Generation:** The script formats each cue according to the SRT specification, including numbering, timing, and text formatting.

# Development

Python environment:

```sh
python3.11 -m venv venv
source venv/bin/activate
```

Download models:

```sh
./build.sh
```

Publish to cog:

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
