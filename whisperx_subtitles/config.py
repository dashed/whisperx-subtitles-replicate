"""Runtime configuration constants."""

from pathlib import Path

compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)
device = "cuda"

# Absolute path to the bundled faster-whisper model (resolved relative to this
# file so it works regardless of the current working directory).
whisper_arch = str(
    Path(__file__).resolve().parent.parent / "models" / "faster-whisper-large-v3"
)

# --- Subtitle formatting defaults (aligned with EBU-TT / Netflix guidelines) ---
MAX_LINE_LENGTH = 42  # max characters per line
MAX_LINES = 2  # max lines per cue (hard cap)
MAX_CPS = 17.0  # max reading speed in characters per second
MIN_DURATION = 1.0  # min seconds a cue stays on screen
MAX_DURATION = 7.0  # max seconds a cue stays on screen
MIN_GAP = 0.083  # min gap between consecutive cues (~2 frames @ 24fps)
MAX_LEAD_OUT = 1.5  # max seconds a cue may extend past its last spoken word

# Deprecated: retained until the subtitle engine fully migrates to MAX_CPS.
DESIRED_WPS = 4  # Words per second for comfortable reading
