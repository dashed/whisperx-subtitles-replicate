"""Audio probing/segmentation helpers (ffmpeg) and pure segment-timing math."""

import logging
import tempfile
from pathlib import Path

import ffmpeg

logger = logging.getLogger(__name__)


def get_audio_duration(file_path):
    probe = ffmpeg.probe(file_path)
    stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "audio"),
        None,
    )
    if stream is None:
        raise ValueError(f"No audio stream found in {file_path}")
    # Some containers omit a per-stream duration; fall back to the format-level one.
    duration = stream.get("duration") or probe.get("format", {}).get("duration")
    if duration is None:
        raise ValueError(f"Could not determine audio duration for {file_path}")
    return float(duration) * 1000


def extract_audio_segment(input_file_path, start_time_ms, duration_ms):
    input_file_path = (
        Path(input_file_path)
        if not isinstance(input_file_path, Path)
        else input_file_path
    )
    file_extension = input_file_path.suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file_path = Path(temp_file.name)

        logger.debug("Extracting from %s to %s", input_file_path.name, temp_file.name)

        try:
            (
                ffmpeg.input(input_file_path, ss=start_time_ms / 1000)
                .output(temp_file.name, t=duration_ms / 1000)
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
        except ffmpeg.Error as e:
            logger.error("ffmpeg error: %s", e.stderr.decode("utf-8"))
            raise

    return temp_file_path


def distribute_segments_equally(total_duration, segments_duration, iterations):
    available_duration = max(total_duration - segments_duration, 0)

    if iterations > 1:
        spacing = available_duration // (iterations - 1)
    else:
        spacing = 0

    start_times = [i * spacing for i in range(iterations)]

    if iterations > 1:
        start_times[-1] = available_duration

    return start_times
