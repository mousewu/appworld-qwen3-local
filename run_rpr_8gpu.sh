#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RPR_PYTHON="${SLIME_PYTHON:-python}"
# Migration profile: two TP2 actor/Critic training GPUs and six TP1 rollout GPUs.
exec "$RPR_PYTHON" "$PROJECT_DIR/run_rpr.py" \
  --num-gpus 8 \
  --actor-gpus 2 \
  --tp-size 2 \
  --rollout-tp 1 \
  --batch-size "${RPR_BATCH_SIZE:-32}" \
  --max-envs "${MAX_ENVS:-48}" \
  --server-concurrency "${SERVER_CONCURRENCY:-16}" \
  --async-group-capacity "${RPR_GROUP_CAPACITY:-64}" \
  --async-inflight-groups "${RPR_INFLIGHT_GROUPS:-48}" \
  --async-max-policy-lag "${RPR_MAX_POLICY_LAG:-1}" \
  --async-group-retries "${RPR_GROUP_RETRIES:-3}" \
  --async-max-consecutive-timeouts "${RPR_MAX_CONSECUTIVE_TIMEOUTS:-3}" \
  --repeated-execution-error-limit "${REPEATED_EXECUTION_ERROR_LIMIT:-2}" \
  --swanlab-mode "${SWANLAB_MODE:-offline}" \
  "$@"
