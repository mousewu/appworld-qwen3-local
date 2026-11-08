"""Replay successful RPR episodes and export verified RSR-2 SFT examples."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import uuid

from run_agent import WorldProcess, execution_feedback, extract_code, prompt_messages


def select_episodes(paths, max_per_task=1):
    candidates = defaultdict(list)
    for path in paths:
        row = json.loads(path.read_text())
        if row.get('evaluation', {}).get('success') is True and row.get('turns'):
            candidates[row['task_id']].append((len(row['turns']), path, row))
    selected = []
    for task_id in sorted(candidates):
        selected.extend(item[1:] for item in sorted(candidates[task_id], key=lambda item:(item[0], str(item[1])))
                        [:max_per_task])
    return selected


def replay_episode(worker, path, episode, experiment, seed):
    codes = [extract_code(turn['raw_completion']) for turn in episode['turns']]
    task = worker.call('reset', task_id=episode['task_id'], experiment_name=experiment,
                       max_interactions=len(codes)+1, seed=seed)
    messages = prompt_messages(task['supervisor'], task['instruction'])
    rows, done = [], False
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    for step, code in enumerate(codes, 1):
        if done:
            raise ValueError('Episode contains actions after AppWorld declared completion')
        result = worker.call('execute', code=code)
        rows.append(dict(task_id=episode['task_id'], scenario_id=episode['task_id'].rsplit('_', 1)[0],
                         step=step, messages=messages.copy(), completion=code,
                         source_episode=episode.get('summary', {}).get('episode_id'),
                         source_sha256=source_sha))
        messages = messages + [dict(role='assistant', content=code),
            dict(role='user', content=execution_feedback(result['observation'])[0])]
        done = result['declared_done']
    evaluation = worker.call('evaluate')
    return rows if evaluation['success'] is True else [], evaluation, task['runtime']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes-dir', required=True, type=Path)
    parser.add_argument('--appworld-python', default=sys.executable)
    parser.add_argument('--appworld-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--max-trajectories-per-task', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--env-timeout', type=float, default=120)
    args = parser.parse_args()
    if args.max_trajectories_per_task <= 0 or args.env_timeout <= 0:
        parser.error('Trajectory count and timeout must be positive')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Output directory must be empty')
    selected = select_episodes(args.episodes_dir.glob('*.json'), args.max_trajectories_per_task)
    if not selected:
        raise SystemExit('No successful episode JSON files found')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    worker = WorldProcess(args.appworld_python, args.appworld_root, args.env_timeout)
    counts = Counter()
    experiment = 'rsr2_replay/' + uuid.uuid4().hex
    try:
        with (args.output_dir/'train.jsonl').open('w') as output, \
             (args.output_dir/'audit.jsonl').open('w') as audit:
            for path, episode in selected:
                report = dict(task_id=episode['task_id'], source=str(path))
                rows = []
                try:
                    rows, evaluation, runtime = replay_episode(
                        worker, path, episode, experiment+'/'+uuid.uuid4().hex, args.seed)
                    report.update(status='success' if rows else 'failed_replay',
                                  evaluation=evaluation, runtime=runtime)
                    for row in rows:
                        output.write(json.dumps(row, ensure_ascii=False)+'\n')
                except Exception as exc:
                    report.update(status='error', error=f'{type(exc).__name__}: {exc}')
                counts[report['status']] += 1
                counts['samples'] += len(rows)
                audit.write(json.dumps(report, ensure_ascii=False)+'\n')
                print(episode['task_id'], report['status'], flush=True)
    finally:
        worker.close()
    summary = dict(counts=counts, selected_trajectories=len(selected),
        unique_tasks=len({episode['task_id'] for _, episode in selected}), seed=args.seed,
        success_definition='fresh AppWorld replay followed by world.evaluate().success is True')
    (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not counts['success']:
        raise SystemExit('No trajectory survived fresh replay verification')


if __name__ == '__main__':
    main()
