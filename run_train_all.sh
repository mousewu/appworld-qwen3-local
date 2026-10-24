#!/usr/bin/env bash
set -euo pipefail

# Defaults match the user's server; environment variables can override each path.
LLM_PYTHON="${LLM_PYTHON:-/opt/conda/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/notebook/data/group/LLMs_data/Qwen3-4B-Instruct-2507}"
APPWORLD_PYTHON="${APPWORLD_PYTHON:-/opt/conda/envs/appworld/bin/python}"
APPWORLD_ROOT="${APPWORLD_ROOT:-/home/notebook/data/group/decision_model/code/appworld}"

[[ -x "$LLM_PYTHON" ]] || { echo "Missing LLM Python: $LLM_PYTHON" >&2; exit 1; }
[[ -x "$APPWORLD_PYTHON" ]] || { echo "Missing AppWorld Python: $APPWORLD_PYTHON" >&2; exit 1; }
[[ -d "$MODEL_PATH" ]] || { echo "Missing model: $MODEL_PATH" >&2; exit 1; }
[[ -d "$APPWORLD_ROOT/data" ]] || { echo "Missing AppWorld data: $APPWORLD_ROOT/data" >&2; exit 1; }

args=(
  --backend transformers
  --model "$MODEL_PATH"
  --device cuda
  --appworld-python "$APPWORLD_PYTHON"
  --appworld-root "$APPWORLD_ROOT"
  --split train
  --all-tasks
  --concurrency "${CONCURRENCY:-4}"
  --inference-batch-size "${INFERENCE_BATCH_SIZE:-4}"
)
exec "$LLM_PYTHON" "$(dirname "$0")/run_agent.py" "${args[@]}" "$@"
