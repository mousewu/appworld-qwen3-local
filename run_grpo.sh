#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
SLIME_PYTHON="${SLIME_PYTHON:-python}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
for source_dir in "${SLIME_DIR:-}" "${MEGATRON_DIR:-}"; do
  if [[ -n "$source_dir" ]]; then
    source_dir="$(cd "$source_dir" && pwd)"
    export PYTHONPATH="$source_dir:$PYTHONPATH"
  fi
done
: "${MODEL_PATH:?Set MODEL_PATH to your SFT HF checkpoint}"
args=(--model "$MODEL_PATH"
  --appworld-python "${APPWORLD_PYTHON:-/opt/conda/envs/appworld/bin/python}"
  --appworld-root "${APPWORLD_ROOT:-/home/notebook/data/group/decision_model/code/appworld}"
  --output-dir "${OUTPUT_DIR:-$PWD/grpo_runs/run_$(date +%Y%m%d_%H%M%S)}"
  --num-gpus "${NUM_GPUS:-2}" --tp-size "${TP_SIZE:-2}" --rollout-tp "${ROLLOUT_TP:-2}"
  --tasks-per-batch "${TASKS_PER_BATCH:-8}" --updates "${UPDATES:-100}"
  --context-length "${CONTEXT_LENGTH:-16384}" --max-steps "${MAX_STEPS:-50}"
  --swanlab-mode "${SWANLAB_MODE:-offline}")
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then args+=(--init-checkpoint "$INIT_CHECKPOINT"); fi
if [[ -n "${REF_CHECKPOINT:-}" ]]; then args+=(--ref-checkpoint "$REF_CHECKPOINT"); fi
if [[ "${SAVE_HF:-1}" == 0 ]]; then args+=(--no-save-hf); fi
exec "$SLIME_PYTHON" run_grpo.py "${args[@]}" "$@"
