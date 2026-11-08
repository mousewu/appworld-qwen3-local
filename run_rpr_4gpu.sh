#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RPR_PYTHON="${SLIME_PYTHON:-python}"
exec "$RPR_PYTHON" "$PROJECT_DIR/run_rpr.py" \
  --num-gpus 4 \
  --actor-gpus 2 \
  --tp-size 2 \
  --rollout-tp 1 \
  --batch-size "${RPR_BATCH_SIZE:-16}" \
  --max-envs "${MAX_ENVS:-24}" \
  --server-concurrency "${SERVER_CONCURRENCY:-16}" \
  --swanlab-mode "${SWANLAB_MODE:-offline}" \
  "$@"
