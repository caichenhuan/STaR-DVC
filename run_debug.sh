#!/usr/bin/env bash
set -euo pipefail

SOCCERNET_PATH="${SOCCERNET_PATH:-/path/to/soccernet}"
POSITION_PATH="${POSITION_PATH:-/path/to/positions}"
GPT_PATH="${GPT_PATH:-gpt2}"
MODEL_NAME="${MODEL_NAME:-gpt2-train}"
FEATURES="${FEATURES:-224p_5fps.npy}"
GPU="${GPU:-0}"
GPT_TYPE="${GPT_TYPE:-gpt2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
FRAMERATE="${FRAMERATE:-1}"
EVAL_FREQ="${EVAL_FREQ:-10}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
POOL="${POOL:-PerceiverResamplerGlobalPosition}"
NMS_THRESHOLD="${NMS_THRESHOLD:-0.70}"
WINDOW_SIZE_SPOTTING="${WINDOW_SIZE_SPOTTING:-30}"
WINDOW_SIZE_CAPTION="${WINDOW_SIZE_CAPTION:-30}"
NMS_WINDOW="${NMS_WINDOW:-30}"
STAGE="${STAGE:-classifying}"

python main.py \
  --SoccerNet_path "$SOCCERNET_PATH" \
  --Position_path "$POSITION_PATH" \
  --model_name "$MODEL_NAME" \
  --gpt_path "$GPT_PATH" \
  --features "$FEATURES" \
  --GPU "$GPU" \
  --model_type gpt \
  --gpt_type "$GPT_TYPE" \
  --batch_size "$BATCH_SIZE" \
  --framerate "$FRAMERATE" \
  --evaluation_frequency "$EVAL_FREQ" \
  --max_epochs "$MAX_EPOCHS" \
  --pool "$POOL" \
  --NMS_threshold "$NMS_THRESHOLD" \
  --window_size_spotting "$WINDOW_SIZE_SPOTTING" \
  --window_size_caption "$WINDOW_SIZE_CAPTION" \
  --NMS_window "$NMS_WINDOW" \
  --debug \
  --stage "$STAGE"
