#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

print_stage "[09] Pair the RSR-2 Actor with the unchanged RPR-1 Critic"
prepare_parent_dirs
require_absent "$RPR2_LOAD_ROOT"
require_absent "${RPR2_LOAD_ROOT}_critic"
ACTOR_ITER="$(latest_checkpoint_iter "$RSR2_SFT_RUN_DIR/checkpoints")"
CRITIC_ITER="$(latest_checkpoint_iter "$RPR1_RUN_DIR/checkpoints_critic")"

"$SLIME_PYTHON" "$PROJECT_DIR/prepare_rpr_load_root.py" \
  --actor-iter "$ACTOR_ITER" \
  --critic-iter "$CRITIC_ITER" \
  --output "$RPR2_LOAD_ROOT"
