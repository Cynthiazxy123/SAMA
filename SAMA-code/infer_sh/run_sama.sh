#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0

# Edit these paths before running.
MODEL_ROOT=/path/to/Wan2.1-T2V-14B
STATE_DICT=/path/to/sama_checkpoint.safetensors
SRC_VIDEO=./inference_example/1526909-hd_1920_1080_24fps.mp4
PROMPT="Replace the spotted baby seal on the sand with a red crab."
OUTPUT_DIR=./outputs/seed_1

python infer_sh/inference.py \
  --model-root "$MODEL_ROOT" \
  --state-dict "$STATE_DICT" \
  --src-video "$SRC_VIDEO" \
  --prompt "$PROMPT" \
  --output-dir "$OUTPUT_DIR" \
  --max-frames 49 \
  --seed 1 \
  --prompt-prefix \
  --tiled \
  --overwrite
