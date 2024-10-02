# whisperx-subtitles-replicate

> Generates synchronized, readable SRT subtitles from transcribed audio with WhisperX (faster-whisper-large-v3)

This a fork of:

- https://github.com/victor-upmeet/whisperx-replicate
- https://replicate.com/victor-upmeet/whisperx

This script processes transcribed audio data to generate properly formatted subtitle files (`.srt`). It handles the splitting and merging of subtitle cues based on duration, line length, line count, and desired reading speed, ensuring that the resulting subtitles are readable and synchronized with the audio.

Code based on: https://github.com/m-bain/whisperX/issues/883

Here’s a high-level overview of how it achieves this:

1. **Generate transcription:** Uses WhisperX (faster-whisper-large-v3) to generate transcription with word-level timestamps.
2. **Sentence Segmentation:** Utilizes the `PySBD` (Python Sentence Boundary Disambiguation) library to split the transcribed text into sentences, respecting language-specific punctuation and sentence boundaries.
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
