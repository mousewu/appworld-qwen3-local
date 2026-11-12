#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

print_stage "[11] Show RPR stage artifacts and exit diagnoses"
"$SLIME_PYTHON" - "$RPR_WORK_ROOT" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
print(f"root: {root}")
for name in ("warmup", "rpr1", "rsr2_sft", "rpr2"):
    run = root / "runs" / name
    status = run / "run_status.json"
    if status.is_file():
        row = json.loads(status.read_text())
        print(f"{name:10} state={row.get('state')} diagnosis={row.get('diagnosis')} returncode={row.get('returncode')}")
    elif run.exists():
        print(f"{name:10} directory exists; no RPR run_status.json")
    else:
        print(f"{name:10} not started")
    for checkpoint_name in ("checkpoints", "checkpoints_critic"):
        tracker = run / checkpoint_name / "latest_checkpointed_iteration.txt"
        if tracker.is_file():
            print(f"  {checkpoint_name}: {tracker.read_text().strip()}")
summary = root / "data" / "rsr2_verified" / "summary.json"
if summary.is_file():
    row = json.loads(summary.read_text())
    print("rsr2 data:", json.dumps(row.get("counts", {}), ensure_ascii=False))
PY
