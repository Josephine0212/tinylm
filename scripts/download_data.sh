#!/usr/bin/env bash
# Download TinyStories (train ~2.1GB, valid ~22MB) into data/.
set -euo pipefail
mkdir -p data && cd data
BASE=https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main
for f in TinyStoriesV2-GPT4-train.txt TinyStoriesV2-GPT4-valid.txt; do
  if [ ! -f "$f" ]; then
    echo "Downloading $f"
    curl -L --fail -o "$f" "$BASE/$f"
  else
    echo "$f already present"
  fi
done
