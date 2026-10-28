#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# Four allocated GPUs: 2 for training (TP2), 2 independent TP1 rollout engines.
export NUM_GPUS="${NUM_GPUS:-4}"
export TP_SIZE="${TP_SIZE:-2}"
export ROLLOUT_TP="${ROLLOUT_TP:-1}"
export TASKS_PER_BATCH="${TASKS_PER_BATCH:-8}"
# 400 GB RAM is provided by the job platform, not allocated by this script.
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-16384}"
exec bash run_grpo.sh --async-training --actor-gpus "${ACTOR_GPUS:-2}" \
  --async-implementation "${ASYNC_IMPLEMENTATION:-auto}" \
  --max-envs "${MAX_ENVS:-24}" --server-concurrency "${SERVER_CONCURRENCY:-16}" \
  --sglang-memory "${SGLANG_MEMORY:-0.70}" --save-interval "${SAVE_INTERVAL:-20}" \
  --gpu-util-target 70 --gpu-monitor-interval 5 --gpu-monitor-warmup 120 "$@"
