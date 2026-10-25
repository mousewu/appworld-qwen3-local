"""Build whole-trajectory 8K/16K/32K datasets from replay-verified Train JSONL."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import tempfile

from prepare_slime_sft import write_csv
from sft_diagnostics import complete_task_actions
from slime_sft import prepare_row, tokenizer_signature, SLIME_REVISION

LIMITS = {'complete_8k': 8192, 'complete_16k': 16384, 'complete_32k': 32768}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                yield line_number, json.loads(line)


def fingerprint(messages):
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_lengths(stats_path, source_hash, signature):
    stats = json.loads(stats_path.read_text())
    if stats['source_sha256'] != source_hash or stats['tokenizer_signature'] != signature:
        raise ValueError('Diagnostic source/tokenizer signature mismatch; regenerate diagnostics')
    suffix = '.stats.json'
    if not str(stats_path).endswith(suffix):
        raise ValueError('--diagnostics-stats must end in .stats.json')
    csv_path = Path(str(stats_path)[:-len(suffix)] + '.lengths.csv')
    records = {}
    with csv_path.open() as stream:
        for row in csv.DictReader(stream):
            line = int(row['source_line'])
            if line in records:
                raise ValueError('Duplicate source line in cached lengths')
            for key in ['step', 'total_tokens', 'prompt_tokens', 'completion_tokens']:
                row[key] = int(row[key])
            if (min(row['prompt_tokens'], row['completion_tokens']) <= 0
                    or row['total_tokens'] != row['prompt_tokens']+row['completion_tokens']):
                raise ValueError('Invalid cached token lengths')
            records[line] = row
    return records


def add_task_row(tasks, row, line, lengths):
    task = tasks.setdefault(row['task_id'], dict(task_id=row['task_id'], samples=0,
        max_total_tokens=0, complete_task_samples=0, complete_task_calls=0,
        action_categories=Counter(), valid_history=True, anomalies=[]))
    step = row['step']
    if type(step) is not int or step != task['samples']+1:
        task['valid_history'] = False
    messages = row['messages']
    if task['samples'] and (len(messages) != task['_expected_length']+1
            or fingerprint(messages[:-1]) != task['_expected_prefix']
            or messages[-1].get('role') != 'user'):
        task['valid_history'] = False
    actions = complete_task_actions(row['completion'])
    task['samples'] += 1
    task['max_total_tokens'] = max(task['max_total_tokens'], lengths['total_tokens'])
    task['complete_task_samples'] += int(bool(actions))
    task['complete_task_calls'] += sum(actions.values())
    task['action_categories'].update(actions)
    task['_final_has_complete_task'] = bool(actions)
    expected = messages + [dict(role='assistant', content=row['completion'])]
    task['_expected_length'] = len(expected)
    task['_expected_prefix'] = fingerprint(expected)
    return task


def rejection_reasons(task, limit, threshold):
    reasons = []
    if task['max_total_tokens'] >= threshold:
        reasons.append('million_token_anomaly')
    if not task['valid_history']:
        reasons.append('noncontiguous_steps_or_inconsistent_history')
    if not task['_final_has_complete_task']:
        reasons.append('missing_final_complete_task')
    if task['max_total_tokens'] > limit:
        reasons.append('trajectory_exceeds_length')
    return reasons


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--data', required=True, type=Path,
                   help='Original replay-verified train.jsonl, never an already filtered dataset')
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--diagnostics-stats', type=Path,
                   help='Optional existing .stats.json and sibling .lengths.csv; avoids re-encoding excluded histories')
    p.add_argument('--anomaly-threshold', type=int, default=1000000)
    a = p.parse_args()
    if a.anomaly_threshold <= 0:
        p.error('--anomaly-threshold must be positive')
    if a.output_dir.exists():
        p.error('Output directory must not exist; choose a new path')
    source_hash = file_hash(a.data)
    signature = tokenizer_signature(a.model)
    cached = load_lengths(a.diagnostics_stats, source_hash, signature) if a.diagnostics_stats else None
    tokenizer = None
    if cached is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    tasks, seen = {}, set()
    for line_number, row in read_rows(a.data):
        if cached is not None:
            lengths = cached.get(line_number)
            if lengths is None or (lengths['task_id'], lengths['step']) != (row['task_id'], row['step']):
                raise ValueError(f'Cached lengths do not match source line {line_number}')
        else:
            lengths = {}
            prepare_row(tokenizer, row, 1, signature, diagnostics=lengths)
        seen.add(line_number)
        task = add_task_row(tasks, row, line_number, lengths)
        if lengths['total_tokens'] >= a.anomaly_threshold:
            task['anomalies'].append(dict(source_line=line_number, step=row['step'],
                                          total_tokens=lengths['total_tokens']))
    if not tasks:
        p.error('Input contains no samples')
    if cached is not None and seen != set(cached):
        raise ValueError('Cached lengths contain extra rows not in source')
    selected = {name: {tid for tid, task in tasks.items()
                      if not rejection_reasons(task, limit, a.anomaly_threshold)}
                for name, limit in LIMITS.items()}
    a.output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Stage all outputs, then publish together so a failed conversion cannot leave partial datasets.
    with tempfile.TemporaryDirectory(prefix='complete-sft-', dir=a.output_dir.parent) as directory:
        stage = Path(directory)
        summaries, audit = {}, []
        for name, limit in LIMITS.items():
            target = stage/name
            target.mkdir()
            kept = [tasks[tid] for tid in sorted(selected[name])]
            categories = Counter()
            for task in kept:
                categories.update(task['action_categories'])
            summaries[name] = dict(max_length=limit, tasks=len(kept),
                samples=sum(t['samples'] for t in kept),
                complete_task_samples=sum(t['complete_task_samples'] for t in kept),
                complete_task_calls=sum(t['complete_task_calls'] for t in kept),
                complete_task_call_categories=categories, supervised_tokens=0,
                task_ids=sorted(selected[name]))
            for tid, task in sorted(tasks.items()):
                audit.append(dict(version=name, task_id=tid, samples=task['samples'],
                    max_total_tokens=task['max_total_tokens'], complete_task_calls=task['complete_task_calls'],
                    retained=tid in selected[name],
                    reasons=';'.join(rejection_reasons(task, limit, a.anomaly_threshold))))
        union = set().union(*selected.values())
        if union and tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
        from contextlib import ExitStack
        with ExitStack() as stack:
            raw = {name: stack.enter_context((stage/name/'train.jsonl').open('w')) for name in LIMITS}
            tokens = {name: stack.enter_context((stage/name/'slime_train.jsonl').open('w')) for name in LIMITS}
            for line_number, row in read_rows(a.data):
                tid = row['task_id']
                if tid not in union:
                    continue
                sample = prepare_row(tokenizer, row, max(LIMITS.values()), signature)
                if sample is None:
                    raise ValueError(f'Task {tid}: actual tokenization exceeds cached length')
                actual = len(sample['metadata']['tokens'])
                if cached is not None and (
                        actual != cached[line_number]['total_tokens'] or
                        sample['metadata']['response_length'] != cached[line_number]['completion_tokens']):
                    raise ValueError(f'Task {tid}: cached lengths differ from actual encoding')
                for name, limit in LIMITS.items():
                    if tid in selected[name]:
                        if actual > limit:
                            raise ValueError(f'Task {tid}: cached length inconsistent with tokenizer')
                        raw[name].write(json.dumps(row, ensure_ascii=False)+'\n')
                        tokens[name].write(json.dumps(sample, ensure_ascii=False)+'\n')
                        summaries[name]['supervised_tokens'] += sample['metadata']['response_length']
        if file_hash(a.data) != source_hash:
            raise ValueError('Source changed during processing')
        summary = dict(source=str(a.data.resolve()), source_sha256=source_hash,
            tokenizer_signature=signature, slime_revision=SLIME_REVISION,
            anomaly_threshold=a.anomaly_threshold, input_tasks=len(tasks), input_samples=len(seen),
            excluded_anomaly_tasks=[dict(task_id=tid, max_total_tokens=t['max_total_tokens'],
                anomalies=t['anomalies']) for tid,t in sorted(tasks.items()) if t['anomalies']],
            complete_task_definition='AST occurrences in target code; not runtime execution counts; aliases not resolved',
            versions=summaries)
        for name in LIMITS:
            (stage/name/'stats.json').write_text(json.dumps(summaries[name], indent=2))
        write_csv(stage/'task_audit.csv', audit)
        write_csv(stage/'summary.csv', [{k:v for k,v in dict(version=name, **s).items()
            if k not in ['task_ids', 'complete_task_call_categories']} for name,s in summaries.items()])
        (stage/'summary.json').write_text(json.dumps(summary, indent=2))
        stage.rename(a.output_dir)
    print((a.output_dir/'summary.csv').read_text())
    print('Excluded anomaly tasks:', [t['task_id'] for t in summary['excluded_anomaly_tasks']])
    print('Reports:', a.output_dir)


if __name__ == '__main__':
    main()
