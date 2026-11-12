#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

print_stage "[07] Replay-verify RPR-1 successes and build RSR-2 SFT data"
require_dir "$RPR1_RUN_DIR/episodes"
require_absent "$RSR2_DATA_DIR"

"$SLIME_PYTHON" "$PROJECT_DIR/build_rsr2_sft.py" \
  --episodes-dir "$RPR1_RUN_DIR/episodes" \
  --appworld-python "$APPWORLD_PYTHON" \
  --appworld-root "$APPWORLD_ROOT" \
  --output-dir "$RSR2_DATA_DIR" \
  --max-trajectories-per-task "$RSR2_MAX_TRAJECTORIES_PER_TASK" \
  --env-timeout "$ENV_TIMEOUT"

"$SLIME_PYTHON" "$PROJECT_DIR/prepare_slime_sft.py" \
  --model "$MODEL" \
  --data "$RSR2_DATA_DIR/train.jsonl" \
  --output "$RSR2_SLIME_DATA" \
  --max-length "$RSR2_MAX_LENGTH"

echo "RSR-2 samples: $(wc -l < "$RSR2_SLIME_DATA" | tr -d ' ')"
