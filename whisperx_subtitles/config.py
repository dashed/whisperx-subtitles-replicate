"""Runtime configuration constants."""

compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)
device = "cuda"
whisper_arch = "./models/faster-whisper-large-v3"

DESIRED_WPS = 4  # Words per second for comfortable reading
