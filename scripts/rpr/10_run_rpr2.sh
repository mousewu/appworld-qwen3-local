#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

print_stage "[10] Run asynchronous RPR-2 with SPR and inherited Critic"
run_rpr_stage rpr2 "$RPR2_LOAD_ROOT" "$RPR2_RUN_DIR" \
  "$RPR2_UPDATES" "$RPR2_CRITIC_ONLY_STEPS" "$RPR_SAVE_INTERVAL" \
  "${EXPERIMENT_NAME}-rpr2"

latest_checkpoint_iter "$RPR2_RUN_DIR/checkpoints"
latest_checkpoint_iter "$RPR2_RUN_DIR/checkpoints_critic"
