#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

stages=(
  00_preflight.sh
  01_calibrate_rewards.sh
  02_prepare_warmup_load.sh
)
if [[ "$RUN_WARMUP_DRY_RUN" == "1" ]]; then
  stages+=(03_dry_run_warmup.sh)
fi
stages+=(
  04_run_warmup.sh
  05_prepare_rpr1_load.sh
  06_run_rpr1.sh
  07_build_rsr2_data.sh
  08_run_rsr2_sft.sh
  09_prepare_rpr2_load.sh
  10_run_rpr2.sh
  11_status.sh
)

for stage in "${stages[@]}"; do
  echo "Running $stage"
  bash "$SCRIPT_DIR/$stage"
done
