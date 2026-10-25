"""Replay official Train execution logs and export successful next-code targets."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import uuid

from run_agent import WorldProcess, extract_code, execution_feedback, prompt_messages

DEFAULT_EXPERIMENT = 'react_gpt4o_train'


def convert_task(worker, task_id, source, experiment, seed):
    # Parse with the installed SDK, not a regex over potentially nested fences.
    entries = worker.call('read_trajectory', path=str(source))
    if not entries:
        raise ValueError('Empty trajectory')
    codes = []
    for entry in entries:
        code = entry['input'].strip()
        if extract_code(code) != code:
            raise ValueError('Execution input unexpectedly contains formatting')
        codes.append(code)
    task = worker.call('reset', task_id=task_id, experiment_name=experiment,
                       max_interactions=len(codes) + 1, seed=seed)
    messages = prompt_messages(task['supervisor'], task['instruction'])
    rows, done = [], False
    for step, code in enumerate(codes, 1):
        if done:
            raise ValueError('Execution found after declared completion')
        # Use fresh observations: teacher observations may differ by SDK version.
        output = worker.call('execute', code=code)
        rows.append(dict(task_id=task_id, scenario_id=task_id.rsplit('_', 1)[0],
                         step=step, messages=messages.copy(), completion=code))
        messages = messages + [dict(role='assistant', content=code),
            dict(role='user', content=execution_feedback(output['observation'])[0])]
        done = output['declared_done']
    evaluation = worker.call('evaluate')
    if evaluation['success'] is not True:
        return [], evaluation, task['runtime']
    return rows, evaluation, task['runtime']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--appworld-python', default=sys.executable)
    p.add_argument('--appworld-root', required=True, type=Path)
    p.add_argument('--experiment', default=DEFAULT_EXPERIMENT,
                   help='Official Train experiment name under experiments/outputs')
    p.add_argument('--output-dir', type=Path, default=Path('sft_data/official_gpt4o'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--env-timeout', type=float, default=120)
    a = p.parse_args()
    root = a.appworld_root.expanduser().resolve()
    source = (root / 'experiments/outputs' / a.experiment).resolve()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if any(a.output_dir.iterdir()):
        p.error('Output directory must be empty; choose a new directory')
    if not source.is_dir():
        p.error(f'Missing downloaded experiment: {source}')
    run_name = 'sft_replay/' + uuid.uuid4().hex
    worker = WorldProcess(a.appworld_python, root, a.env_timeout)
    counts = Counter()
    try:
        train_ids = worker.call('list', split='train')
        with (a.output_dir / 'train.jsonl').open('w') as data, \
             (a.output_dir / 'audit.jsonl').open('w') as audit:
            for task_id in train_ids:
                path = source / 'tasks' / task_id / 'logs/environment_io.md'
                report = dict(task_id=task_id, source=str(path))
                rows = []
                try:
                    if not path.is_file():
                        report['status'] = 'missing'
                    else:
                        report['source_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                        rows, evaluation, runtime = convert_task(worker, task_id, path,
                                                                 run_name, a.seed)
                        report.update(status='success' if rows else 'failed_evaluation',
                                      evaluation=evaluation, runtime=runtime)
                        for row in rows:
                            row['source_sha256'] = report['source_sha256']
                            data.write(json.dumps(row, ensure_ascii=False) + '\n')
                except Exception as exc:
                    report.update(status='error', error=f'{type(exc).__name__}: {exc}')
                counts[report['status']] += 1
                counts['samples'] += len(rows)
                audit.write(json.dumps(report, ensure_ascii=False) + '\n')
                audit.flush()
                data.flush()
                print(task_id, report['status'], flush=True)
    finally:
        worker.close()
    summary = dict(counts=counts, source=str(source), replay_experiment=run_name,
        prompt_sha256=hashlib.sha256(Path(__file__).with_name('prompt_template.txt').read_bytes()).hexdigest(),
        split='train', seed=a.seed, success_definition='fresh world.evaluate().success is True')
    (a.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not counts['success']:
        raise SystemExit('No successful trajectories exported; inspect audit.jsonl')


if __name__ == '__main__':
    main()
