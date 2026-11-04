#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
case "${1:-}" in
  24|32) export MAX_ENVS="$1"; shift ;;
  *) echo "Usage: bash run_grpo_async_compare.sh {24|32} [launcher options]" >&2; exit 2 ;;
esac
export NUM_GPUS=4 ACTOR_GPUS=2 TP_SIZE=2 ROLLOUT_TP=1 TASKS_PER_BATCH=8
export UPDATES="${UPDATES:-12}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-12}"
export ASYNC_IMPLEMENTATION=pipeline
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PWD/grpo_runs/compare_envs${MAX_ENVS}_$(date +%Y%m%d_%H%M%S)}"
exec bash run_grpo_async.sh --env-execute-budget-seconds "${ENV_EXECUTE_BUDGET_SECONDS:-300}" "$@"
