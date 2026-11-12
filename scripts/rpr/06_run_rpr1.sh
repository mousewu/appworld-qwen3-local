#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

print_stage "[06] Run asynchronous RPR-1 with FDR and inherited Critic"
run_rpr_stage rpr1 "$RPR1_LOAD_ROOT" "$RPR1_RUN_DIR" \
  "$RPR1_UPDATES" "$RPR1_CRITIC_ONLY_STEPS" "$RPR_SAVE_INTERVAL" \
  "${EXPERIMENT_NAME}-rpr1"

latest_checkpoint_iter "$RPR1_RUN_DIR/checkpoints"
latest_checkpoint_iter "$RPR1_RUN_DIR/checkpoints_critic"
