"""Runtime configuration constants."""

from pathlib import Path

compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)
device = "cuda"

# Absolute path to the bundled faster-whisper model (resolved relative to this
# file so it works regardless of the current working directory).
whisper_arch = str(
    Path(__file__).resolve().parent.parent / "models" / "faster-whisper-large-v3"
)

# Multilingual forced-alignment fallback (ctc-forced-aligner / MMS-300M), used
# only for languages outside whisperx's built-in alignment set so they still get
# word-level timestamps. NOTE: the MMS weights are CC-BY-NC 4.0 (NON-COMMERCIAL).
MMS_ALIGN_MODEL = "MahmoudAshraf/mms-300m-1130-forced-aligner"

# Neural multilingual sentence segmenter (SaT / wtpsplit, MIT-licensed). Used in
# place of pysbd for languages pysbd does not support (e.g. Thai), which keeps
# sentence boundaries sane for non-space-delimited and unsupported scripts.
SAT_MODEL = "sat-3l-sm"

# Text machine-translation model for the `translate_to` feature (MADLAD-400,
# Apache-2.0, 400+ languages, any->any via a "<2{target}>" token). Loaded lazily
# (only when a translation is requested). Swap for a larger MADLAD (7b/10b) or an
# LLM translator for higher quality.
MT_MODEL = "google/madlad400-3b-mt"

# --- Subtitle formatting defaults (aligned with EBU-TT / Netflix guidelines) ---
MAX_LINE_LENGTH = 42  # max characters per line
MAX_LINES = 2  # max lines per cue (hard cap)
MAX_CPS = 17.0  # max reading speed in characters per second
MIN_DURATION = 1.0  # min seconds a cue stays on screen
MAX_DURATION = 7.0  # max seconds a cue stays on screen
MIN_GAP = 0.083  # min gap between consecutive cues (~2 frames @ 24fps)
MAX_LEAD_OUT = 1.5  # max seconds a cue may extend past its last spoken word

# Treat an inter-word silence >= PAUSE_THRESHOLD as a natural pause: a cue is
# split there, and two cues are never merged across it. Kept fairly high so only
# genuine pauses split a phrase (a lower value over-fragments dramatic speech).
PAUSE_THRESHOLD = 1.0
MERGE_MAX_GAP = PAUSE_THRESHOLD  # max gap across which two short cues may merge

# Line-break scoring (lower score = preferred break point); see _balance_two.
LINE_BREAK_PUNCTUATION_BONUS = 25  # reward breaking right after , ; : . ! ? —
LINE_BREAK_FUNCTION_WORD_PENALTY = 30  # avoid breaking after a/the/of/to/...
LINE_BREAK_ORPHAN_PENALTY = 20  # avoid a one-word line (widow/orphan)
