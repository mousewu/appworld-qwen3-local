#!/usr/bin/env bash
set -euo pipefail

RPR_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$RPR_SCRIPT_DIR/../.." && pwd)"

# shellcheck source=config.sh
source "$RPR_SCRIPT_DIR/config.sh"
if [[ -f "$RPR_SCRIPT_DIR/config.local.sh" ]]; then
  # shellcheck source=config.local.example.sh
  source "$RPR_SCRIPT_DIR/config.local.sh"
fi

case "$NUM_GPUS" in
  4)
    RPR_LAUNCHER="${RPR_LAUNCHER:-$PROJECT_DIR/run_rpr_4gpu.sh}"
    RPR_BATCH_SIZE="${RPR_BATCH_SIZE:-16}"
    MAX_ENVS="${MAX_ENVS:-24}"
    ;;
  8)
    RPR_LAUNCHER="${RPR_LAUNCHER:-$PROJECT_DIR/run_rpr_8gpu.sh}"
    RPR_BATCH_SIZE="${RPR_BATCH_SIZE:-32}"
    MAX_ENVS="${MAX_ENVS:-48}"
    ;;
  *)
    echo "NUM_GPUS must be 4 or 8, got: $NUM_GPUS" >&2
    exit 2
    ;;
esac

RPR_GROUP_CAPACITY="${RPR_GROUP_CAPACITY:-$((2 * RPR_BATCH_SIZE))}"
RPR_INFLIGHT_GROUPS="${RPR_INFLIGHT_GROUPS:-$MAX_ENVS}"
RPR_WORK_ROOT="${RPR_WORK_ROOT:-$PROJECT_DIR/rpr_experiments/$EXPERIMENT_NAME}"
if [[ "$RPR_WORK_ROOT" != /* ]]; then
  RPR_WORK_ROOT="$PROJECT_DIR/$RPR_WORK_ROOT"
fi

RPR_DATA_DIR="$RPR_WORK_ROOT/data"
RPR_LOAD_DIR="$RPR_WORK_ROOT/load_roots"
RPR_RUN_DIR="$RPR_WORK_ROOT/runs"
REWARD_SCALES="$RPR_DATA_DIR/train_reward_scales.json"
WARMUP_LOAD_ROOT="$RPR_LOAD_DIR/warmup"
WARMUP_DRY_DIR="$RPR_RUN_DIR/warmup_dry"
WARMUP_RUN_DIR="$RPR_RUN_DIR/warmup"
RPR1_LOAD_ROOT="$RPR_LOAD_DIR/rpr1"
RPR1_RUN_DIR="$RPR_RUN_DIR/rpr1"
RSR2_DATA_DIR="$RPR_DATA_DIR/rsr2_verified"
RSR2_SLIME_DATA="$RSR2_DATA_DIR/slime_train.jsonl"
RSR2_SFT_RUN_DIR="$RPR_RUN_DIR/rsr2_sft"
RPR2_LOAD_ROOT="$RPR_LOAD_DIR/rpr2"
RPR2_RUN_DIR="$RPR_RUN_DIR/rpr2"

export TERMINAL_RSI_DIR SLIME_PYTHON MODEL APPWORLD_PYTHON APPWORLD_ROOT
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$PROJECT_DIR:$TERMINAL_RSI_DIR${PYTHONPATH:+:$PYTHONPATH}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "Missing file: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "Missing directory: $1"
}

require_executable() {
  [[ -x "$1" ]] || die "Missing executable: $1"
}

require_absent() {
  [[ ! -e "$1" ]] || die "Refusing to overwrite existing path: $1"
}

latest_checkpoint_iter() {
  local root="$1"
  require_dir "$root"
  require_file "$root/latest_checkpointed_iteration.txt"
  "$SLIME_PYTHON" - "$root" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).expanduser().resolve()
tracker = (root / "latest_checkpointed_iteration.txt").read_text().strip()
if tracker == "release":
    path = root / "release"
    if not path.exists():
        raise SystemExit(f"Missing release checkpoint: {path}")
    print(path.resolve())
    raise SystemExit
if not tracker.isdigit():
    raise SystemExit(f"Unsupported checkpoint tracker value {tracker!r} in {root}")
iteration = int(tracker)
matches = []
for path in root.glob("iter_*"):
    suffix = path.name.removeprefix("iter_")
    if path.is_dir() and suffix.isdigit() and int(suffix) == iteration:
        matches.append(path)
if len(matches) != 1:
    raise SystemExit(f"Expected one iter directory for {iteration} in {root}, got {matches}")
print(matches[0].resolve())
PY
}

resolve_sft_actor_iter() {
  if [[ -n "$SFT_ACTOR_ITER" ]]; then
    require_dir "$SFT_ACTOR_ITER"
    (cd "$SFT_ACTOR_ITER" && pwd)
    return
  fi
  [[ -n "$SFT_CKPT_ROOT" ]] || die \
    "Set SFT_CKPT_ROOT or SFT_ACTOR_ITER in scripts/rpr/config.local.sh"
  latest_checkpoint_iter "$SFT_CKPT_ROOT"
}

prepare_parent_dirs() {
  mkdir -p "$RPR_DATA_DIR" "$RPR_LOAD_DIR" "$RPR_RUN_DIR"
}

print_stage() {
  echo
  echo "============================================================"
  echo "$1"
  echo "Experiment: $EXPERIMENT_NAME"
  echo "Work root : $RPR_WORK_ROOT"
  echo "============================================================"
}

run_rpr_stage() {
  local phase="$1"
  local load_root="$2"
  local output_dir="$3"
  local updates="$4"
  local critic_only_steps="$5"
  local save_interval="$6"
  local swanlab_name="$7"
  shift 7

  require_dir "$load_root"
  require_file "$load_root/latest_checkpointed_iteration.txt"
  require_file "$REWARD_SCALES"
  require_absent "$output_dir"

  RPR_BATCH_SIZE="$RPR_BATCH_SIZE" \
  MAX_ENVS="$MAX_ENVS" \
  SERVER_CONCURRENCY="$SERVER_CONCURRENCY" \
  RPR_GROUP_CAPACITY="$RPR_GROUP_CAPACITY" \
  RPR_INFLIGHT_GROUPS="$RPR_INFLIGHT_GROUPS" \
  RPR_MAX_POLICY_LAG="$RPR_MAX_POLICY_LAG" \
  RPR_GROUP_RETRIES="$RPR_GROUP_RETRIES" \
  RPR_MAX_CONSECUTIVE_TIMEOUTS="$RPR_MAX_CONSECUTIVE_TIMEOUTS" \
  REPEATED_EXECUTION_ERROR_LIMIT="$REPEATED_EXECUTION_ERROR_LIMIT" \
  SWANLAB_MODE="$SWANLAB_MODE" \
  bash "$RPR_LAUNCHER" \
    --phase "$phase" \
    --model "$MODEL" \
    --load-root "$load_root" \
    --terminal-rsi-dir "$TERMINAL_RSI_DIR" \
    --appworld-python "$APPWORLD_PYTHON" \
    --appworld-root "$APPWORLD_ROOT" \
    --reward-scales "$REWARD_SCALES" \
    --output-dir "$output_dir" \
    --updates "$updates" \
    --critic-only-steps "$critic_only_steps" \
    --context-length "$CONTEXT_LENGTH" \
    --train-max-tokens "$TRAIN_MAX_TOKENS" \
    --max-steps "$MAX_STEPS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --observation-tokens "$OBSERVATION_TOKENS" \
    --env-timeout "$ENV_TIMEOUT" \
    --env-timeout-retries "$ENV_TIMEOUT_RETRIES" \
    --env-execute-budget-seconds "$ENV_EXECUTE_BUDGET_SECONDS" \
    --temperature "$TEMPERATURE" \
    --critic-lr "$CRITIC_LR" \
    --gamma-turn "$GAMMA_TURN" \
    --lambda-turn "$LAMBDA_TURN" \
    --save-interval "$save_interval" \
    --sglang-memory "$SGLANG_MEMORY" \
    --gpu-util-target "$GPU_UTIL_TARGET" \
    --swanlab-mode "$SWANLAB_MODE" \
    --swanlab-project "$SWANLAB_PROJECT" \
    --swanlab-name "$swanlab_name" \
    "$@"
}
