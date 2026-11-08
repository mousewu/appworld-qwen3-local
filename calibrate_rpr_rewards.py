"""Build frozen AppWorld family divisors for Terminal-RSI FDR/SPR rewards."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import uuid

from appworld_rpr import task_family
from run_agent import WorldProcess


def nearest_rank_percentile(values, percentile):
    if not values:
        raise ValueError('Cannot calculate a percentile of an empty sequence')
    if not 0 < percentile <= 100:
        raise ValueError('Percentile must be in (0, 100]')
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile/100*len(ordered))-1)]


def summarize(rows, percentile=90):
    families = defaultdict(list)
    for row in rows:
        total = row['total_checks']
        if type(total) is not int or total <= 0:
            raise ValueError(f"Invalid total_checks for {row['task_id']}: {total!r}")
        families[row['family_id']].append(total)
    all_totals = [row['total_checks'] for row in rows]
    return {
        'schema': 'appworld_rpr_reward_scales_v1',
        'divisor_definition': f'nearest-rank p{percentile:g} of total_checks',
        'family_rule': "task_id.rsplit('_', 1)[0]",
        'percentile': percentile,
        'task_count': len(rows),
        'family_count': len(families),
        'global_divisor': float(nearest_rank_percentile(all_totals, percentile)),
        'family_divisors': {family: float(nearest_rank_percentile(values, percentile))
                            for family, values in sorted(families.items())},
        'families': {family: {'tasks': len(values), 'min_total_checks': min(values),
                              'max_total_checks': max(values)}
                     for family, values in sorted(families.items())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--appworld-python', default=sys.executable)
    parser.add_argument('--appworld-root', required=True, type=Path)
    parser.add_argument('--split', default='train', choices=['train', 'dev', 'test_normal', 'test_challenge'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--percentile', type=float, default=90)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--env-timeout', type=float, default=120)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; reward calibration must be immutable')
    if not 0 < args.percentile <= 100 or args.env_timeout <= 0:
        parser.error('Percentile must be in (0,100] and timeout must be positive')
    root = args.appworld_root.expanduser().resolve()
    worker = WorldProcess(args.appworld_python, root, args.env_timeout)
    rows = []
    experiment = 'rpr_reward_calibration/' + uuid.uuid4().hex
    try:
        task_ids = worker.call('list', split=args.split)
        for index, task_id in enumerate(task_ids):
            worker.call('reset', task_id=task_id, experiment_name=experiment,
                        max_interactions=1, seed=args.seed)
            evaluation = worker.call('evaluate')
            rows.append(dict(task_id=task_id, family_id=task_family(task_id),
                             total_checks=evaluation['total_checks']))
            print(f'[{index+1}/{len(task_ids)}] {task_id}: {evaluation["total_checks"]} checks', flush=True)
    finally:
        worker.close()
    report = summarize(rows, args.percentile)
    report.update(split=args.split, appworld_root=str(root), seed=args.seed,
                  created_at=datetime.now(timezone.utc).isoformat(), tasks=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in
        ('task_count', 'family_count', 'global_divisor', 'divisor_definition')}, indent=2))


if __name__ == '__main__':
    main()
