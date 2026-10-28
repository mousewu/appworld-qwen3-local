"""Recover a diagnosis after abrupt launcher death; optionally upload existing evidence."""
import argparse
import json
from pathlib import Path
from resource_monitor import diagnose
from grpo_tracking import SwanLabBridge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--returncode', type=int, help='Actual exit code from job platform, if known')
    parser.add_argument('--swanlab-mode', choices=['disabled', 'cloud', 'offline'], default='disabled')
    parser.add_argument('--swanlab-name', help='New diagnostic experiment name (does not resume original run)')
    args = parser.parse_args()
    root = args.output_dir.resolve()
    if not root.is_dir():
        parser.error('Output directory does not exist')
    status = root/'run_status.json'
    saved = json.loads(status.read_text()) if status.is_file() else {}
    report = diagnose(root, args.returncode if args.returncode is not None else saved.get('returncode'),
                      saved.get('received_signal'))
    print(json.dumps({key:value for key,value in report.items() if key not in ('resource', 'log_tail')}, indent=2))
    if args.swanlab_mode != 'disabled':
        config = json.loads((root/'project_config.json').read_text())
        bridge = SwanLabBridge(root, args.swanlab_mode, config.get('swanlab_project', 'appworld-grpo'),
            args.swanlab_name or root.name+'_diagnosis', config)
        bridge.finish()


if __name__ == '__main__':
    main()
