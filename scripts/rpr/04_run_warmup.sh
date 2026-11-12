#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

print_stage "[04] Train the cold Critic with the SFT Actor frozen"
run_rpr_stage warmup "$WARMUP_LOAD_ROOT" "$WARMUP_RUN_DIR" \
  "$WARMUP_UPDATES" 0 "$WARMUP_SAVE_INTERVAL" "${EXPERIMENT_NAME}-warmup"

latest_checkpoint_iter "$WARMUP_RUN_DIR/checkpoints_critic"
