# whisperX on Replicate

This a fork of:

- https://github.com/victor-upmeet/whisperx-replicate
- https://replicate.com/victor-upmeet/whisperx

This generates readalbe subtitles (`.srt`) based on: https://github.com/m-bain/whisperX/issues/883

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
