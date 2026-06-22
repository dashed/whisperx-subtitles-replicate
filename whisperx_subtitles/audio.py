"""Audio probing/segmentation helpers (ffmpeg) and pure segment-timing math."""

import tempfile
from pathlib import Path

import ffmpeg


def get_audio_duration(file_path):
    probe = ffmpeg.probe(file_path)
    stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "audio"), None
    )
    if stream is None:
        raise ValueError(f"No audio stream found in {file_path}")
    return float(stream["duration"]) * 1000


def extract_audio_segment(input_file_path, start_time_ms, duration_ms):
    input_file_path = (
        Path(input_file_path)
        if not isinstance(input_file_path, Path)
        else input_file_path
    )
    file_extension = input_file_path.suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file_path = Path(temp_file.name)

        print(f"Extracting from {input_file_path.name} to {temp_file.name}")

        try:
            (
                ffmpeg.input(input_file_path, ss=start_time_ms / 1000)
                .output(temp_file.name, t=duration_ms / 1000)
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
        except ffmpeg.Error as e:
            print("ffmpeg error occurred: ", e.stderr.decode("utf-8"))
            raise e

    return temp_file_path


def distribute_segments_equally(total_duration, segments_duration, iterations):
    available_duration = total_duration - segments_duration

    if iterations > 1:
        spacing = available_duration // (iterations - 1)
    else:
        spacing = 0

    start_times = [i * spacing for i in range(iterations)]

    if iterations > 1:
        start_times[-1] = total_duration - segments_duration

    return start_times
