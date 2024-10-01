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

# Download fullstop-punctuation-multilang-large model
punctuation_model_dir=models/fullstop-punctuation-multilang-large
mkdir -p $punctuation_model_dir

https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large
https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large/commit/88db183baebdc61e721ff1d2d77c1a07ca7ccd11
download "https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large/resolve/main/config.json" "$punctuation_model_dir/config.json"
download "https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large/resolve/main/pytorch_model.bin" "$punctuation_model_dir/pytorch_model.bin"
download "https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large/resolve/main/sentencepiece.bpe.model" "$punctuation_model_dir/sentencepiece.bpe.model"
download "https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large/resolve/main/special_tokens_map.json" "$punctuation_model_dir/special_tokens_map.json"
download "https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large/resolve/main/tokenizer_config.json" "$punctuation_model_dir/tokenizer_config.json"
download "https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large/resolve/main/training_args.bin" "$punctuation_model_dir/training_args.bin"

# Download whisperx-vad-segmentation model
pip install -U git+https://github.com/m-bain/whisperx.git

vad_model_dir=models/vad
mkdir -p $vad_model_dir

download $(python3 ./get_vad_model_url.py) "$vad_model_dir/whisperx-vad-segmentation.bin"

# cog run python
