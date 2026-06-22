#!/bin/bash

set -e

download() {
  local file_url="$1"
  local destination_path="$2"

  if [ ! -e "$destination_path" ]; then
    wget -O "$destination_path" "$file_url"
  else
    echo "$destination_path already exists. No need to download."
  fi
}

# Download faster-whisper-large-v3 model
faster_whisper_model_dir=models/faster-whisper-large-v3
mkdir -p $faster_whisper_model_dir

download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/config.json" "$faster_whisper_model_dir/config.json"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/model.bin" "$faster_whisper_model_dir/model.bin"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/preprocessor_config.json" "$faster_whisper_model_dir/preprocessor_config.json"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/tokenizer.json" "$faster_whisper_model_dir/tokenizer.json"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/vocabulary.json" "$faster_whisper_model_dir/vocabulary.json"

# The VAD segmentation model no longer needs to be pre-downloaded: current
# whisperx ships it inside the package (whisperx/assets/pytorch_model.bin) and
# loads it from there, so it is already present once dependencies are installed.
