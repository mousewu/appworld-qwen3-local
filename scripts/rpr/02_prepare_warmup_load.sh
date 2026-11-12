#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

print_stage "[02] Prepare SFT Actor-only load root for Critic warm-up"
prepare_parent_dirs
require_absent "$WARMUP_LOAD_ROOT"
require_absent "${WARMUP_LOAD_ROOT}_critic"
SFT_ITER="$(resolve_sft_actor_iter)"

"$SLIME_PYTHON" "$PROJECT_DIR/prepare_rpr_load_root.py" \
  --actor-iter "$SFT_ITER" \
  --output "$WARMUP_LOAD_ROOT"
