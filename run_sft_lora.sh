#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
LLM_PYTHON="${LLM_PYTHON:-/opt/conda/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/notebook/data/group/LLMs_data/Qwen3-4B-Instruct-2507}"
DATA_PATH="${DATA_PATH:-sft_data/official_gpt4o/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-sft_outputs/qwen3_4b_lora}"
"$LLM_PYTHON" train_lora.py --model "$MODEL_PATH" --data "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" --max-length "${MAX_LENGTH:-8192}" \
  --epochs "${EPOCHS:-1}" --batch-size "${BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${GRAD_ACCUM:-16}" \
  --merge-output "${MERGE_OUTPUT:-sft_outputs/qwen3_4b_merged}" "$@"
