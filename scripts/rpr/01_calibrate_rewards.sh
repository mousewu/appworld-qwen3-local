#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

print_stage "[01] Calibrate immutable AppWorld FDR/SPR reward scales"
prepare_parent_dirs
require_absent "$REWARD_SCALES"

"$SLIME_PYTHON" "$PROJECT_DIR/calibrate_rpr_rewards.py" \
  --appworld-python "$APPWORLD_PYTHON" \
  --appworld-root "$APPWORLD_ROOT" \
  --split train \
  --output "$REWARD_SCALES" \
  --env-timeout "$ENV_TIMEOUT"

echo "Reward scales: $REWARD_SCALES"
