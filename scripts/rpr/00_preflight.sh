#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

print_stage "[00] RPR environment and checkpoint preflight"
require_executable "$SLIME_PYTHON"
require_executable "$APPWORLD_PYTHON"
require_dir "$MODEL"
require_file "$MODEL/config.json"
require_dir "$APPWORLD_ROOT/data"
require_file "$TERMINAL_RSI_DIR/train.py"
require_file "$TERMINAL_RSI_DIR/train_async.py"
require_file "$TERMINAL_RSI_DIR/slime_plugins/harbor/tb_gae.py"
require_file "$PROJECT_DIR/run_rpr.py"
require_file "$RPR_LAUNCHER"

SFT_ITER="$(resolve_sft_actor_iter)"
echo "SFT actor iteration: $SFT_ITER"

"$SLIME_PYTHON" - "$NUM_GPUS" "$MODEL/config.json" <<'PY'
import json
import sys
import torch
import ray
import megatron.core
import transformer_engine
import sglang
import slime

required = int(sys.argv[1])
config = json.load(open(sys.argv[2]))
visible = torch.cuda.device_count()
if visible < required:
    raise SystemExit(f"Need {required} visible GPUs, found {visible}")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("Visible GPUs do not support BF16")
if config.get("model_type") != "qwen3" or config.get("hidden_size") != 2560:
    raise SystemExit("MODEL is not the supported Qwen3-4B architecture")
print(f"Slime runtime OK: GPUs={visible}, BF16=True, model={config.get('model_type')}")
PY

"$SLIME_PYTHON" "$PROJECT_DIR/run_agent.py" --check-env \
  --appworld-python "$APPWORLD_PYTHON" \
  --appworld-root "$APPWORLD_ROOT"

echo "Preflight passed."
