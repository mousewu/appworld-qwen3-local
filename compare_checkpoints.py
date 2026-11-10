"""Evaluate five HF checkpoints sequentially on one frozen AppWorld task set."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from run_agent import WorldProcess


CHECKPOINTS = ('sft', 'previous-final', 'rl-100', 'rl-200', 'rl-300')


def write_summary(output, rows):
    for filename in ('comparison.json', 'comparison.csv'):
        path = output / filename
        temp = path.with_suffix(path.suffix + '.tmp')
        if path.suffix == '.json':
            temp.write_text(json.dumps(rows, indent=2) + '\n')
        else:
            with temp.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        temp.replace(path)


def summarize(label, model, result_dir, tasks):
    rows = json.loads((result_dir / 'summary.json').read_text())['results']
    if [row['task_id'] for row in rows] != tasks:
        raise RuntimeError(f'{label}: results do not cover the frozen task list in order')
    successes = sum(row['evaluated'] and row['success'] for row in rows)
    repeated = sum(row['termination'] == 'repeated_execution_error' for row in rows)
    return dict(checkpoint=label, model=str(model), results=str(result_dir),
                tasks=len(tasks), evaluated=sum(row['evaluated'] for row in rows),
                errors=sum(row['termination'] == 'error' for row in rows),
                successes=successes, success_rate=successes / len(tasks),
                average_steps=sum(row['steps'] for row in rows) / len(tasks),
                repeated_error_terminations=repeated,
                repeated_error_termination_rate=repeated / len(tasks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for label in CHECKPOINTS:
        parser.add_argument(f'--{label}', type=Path, required=True, help='Standalone Hugging Face model directory')
    parser.add_argument('--appworld-root', type=Path, required=True)
    parser.add_argument('--appworld-python', default=sys.executable)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split', choices=['dev', 'test_normal', 'test_challenge'], default='dev')
    parser.add_argument('--limit', type=int, default=0, help='0 uses the full split; positive values are for smoke tests')
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--context-length', type=int, default=16384)
    parser.add_argument('--observation-tokens', type=int, default=2048)
    parser.add_argument('--repeated-execution-error-limit', type=int, default=2)
    parser.add_argument('--env-timeout', type=float, default=120)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--inference-batch-size', type=int, default=4)
    parser.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'], default='cuda')
    args = parser.parse_args()

    models = {label: getattr(args, label.replace('-', '_')).expanduser().resolve() for label in CHECKPOINTS}
    for label, model in models.items():
        if not (model / 'config.json').is_file() or not any(
                path.is_file() for pattern in ('model*.safetensors', 'pytorch_model*.bin')
                for path in model.glob(pattern)):
            parser.error(f'--{label} must contain a standalone HF config and weights: {model}')
        max_positions = json.loads((model / 'config.json').read_text()).get('max_position_embeddings')
        if isinstance(max_positions, int) and max_positions < args.context_length:
            parser.error(f'--{label} supports only {max_positions} positions; lower --context-length for all models')
    if len(set(models.values())) != len(models):
        parser.error('Checkpoint paths must resolve to distinct directories')
    if args.limit < 0:
        parser.error('--limit must be nonnegative')

    output = args.output_dir.expanduser().resolve()
    if output.exists():
        parser.error(f'Choose a fresh output directory: {output}')
    args.appworld_root = args.appworld_root.expanduser().resolve()
    probe = WorldProcess(args.appworld_python, args.appworld_root, args.env_timeout)
    try:
        tasks = list(dict.fromkeys(probe.call('list', split=args.split)))
    finally:
        probe.close()
    tasks = tasks[:args.limit] if args.limit else tasks
    if not tasks:
        parser.error('Selected split contains no tasks')

    output.mkdir(parents=True)
    shared = {key: getattr(args, key) for key in (
        'appworld_python', 'appworld_root', 'split', 'max_steps', 'max_new_tokens',
        'context_length', 'observation_tokens', 'repeated_execution_error_limit',
        'env_timeout', 'seed', 'concurrency', 'inference_batch_size', 'device')}
    shared.update(temperature=0.0, top_p=1.0)
    (output / 'manifest.json').write_text(json.dumps(
        dict(tasks=tasks, settings=shared, checkpoints=models), indent=2, default=str) + '\n')
    common_args = [part for key, value in shared.items()
                   for part in ('--' + key.replace('_', '-'), str(value))]
    task_args = [part for task in tasks for part in ('--task-id', task)]
    agent = Path(__file__).resolve().with_name('run_agent.py')
    rows = []
    for label, model in models.items():
        print(f'Evaluating {label} on {len(tasks)} {args.split} tasks', flush=True)
        target = output / label
        process = subprocess.run([sys.executable, str(agent), '--model', str(model),
                                  '--output-dir', str(target), '--all-tasks', *common_args, *task_args])
        summaries = list(target.glob('*/summary.json'))
        if len(summaries) != 1:
            raise RuntimeError(f'{label}: expected one result directory, exit code {process.returncode}')
        row = summarize(label, model, summaries[0].parent, tasks)
        if process.returncode not in (0, 1) or (process.returncode and not row['errors']):
            raise RuntimeError(f'{label}: evaluator exited with code {process.returncode}')
        rows.append(row)
        write_summary(output, rows)
        print(json.dumps(row), flush=True)
    print(f'Comparison: {output / "comparison.csv"}', flush=True)
    return int(any(row['errors'] for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())
