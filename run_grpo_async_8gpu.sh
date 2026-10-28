#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# Two actor GPUs (TP2), six independent rollout engines (TP1).
export NUM_GPUS=8
export ACTOR_GPUS=2
export TP_SIZE=2
export ROLLOUT_TP=1
# Preserve batch size to isolate hardware effects. Tune concurrency after profiling.
export TASKS_PER_BATCH="${TASKS_PER_BATCH:-8}"
export MAX_ENVS="${MAX_ENVS:-24}"
exec bash run_grpo_async.sh "$@"
