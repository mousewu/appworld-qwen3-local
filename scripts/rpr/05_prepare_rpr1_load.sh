#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

print_stage "[05] Pair the original SFT Actor with the Warm Critic"
prepare_parent_dirs
require_absent "$RPR1_LOAD_ROOT"
require_absent "${RPR1_LOAD_ROOT}_critic"
SFT_ITER="$(resolve_sft_actor_iter)"
CRITIC_ITER="$(latest_checkpoint_iter "$WARMUP_RUN_DIR/checkpoints_critic")"

"$SLIME_PYTHON" "$PROJECT_DIR/prepare_rpr_load_root.py" \
  --actor-iter "$SFT_ITER" \
  --critic-iter "$CRITIC_ITER" \
  --output "$RPR1_LOAD_ROOT"
