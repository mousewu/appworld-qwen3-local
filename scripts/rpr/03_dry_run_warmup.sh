#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

print_stage "[03] Materialize and validate the Warm Critic command"
run_rpr_stage warmup "$WARMUP_LOAD_ROOT" "$WARMUP_DRY_DIR" \
  "$WARMUP_UPDATES" 0 "$WARMUP_SAVE_INTERVAL" "${EXPERIMENT_NAME}-warmup-dry" \
  --dry-run

echo "Generated command: $WARMUP_DRY_DIR/command.json"
