"""Tokenize replay-verified AppWorld next-code examples for Slime SFT."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from slime_sft import prepare_row, tokenizer_signature, SLIME_REVISION
from sft_diagnostics import Diagnostics, describe_anomaly


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-length', type=int, default=8192)
    p.add_argument('--anomaly-threshold', type=int, default=1000000,
                   help='Diagnose examples with total token length >= this value')
    p.add_argument('--diagnostic-preview-chars', type=int, default=0,
                   help='Optional bounded content previews in LOCAL anomaly report; default no raw text')
    a = p.parse_args()
    if min(a.max_length, a.anomaly_threshold) <= 0 or a.diagnostic_preview_chars < 0:
        p.error('Lengths/threshold must be positive; preview length must be nonnegative')
    artifacts = [a.output] + [a.output.with_suffix(s) for s in [
        '.stats.json', '.lengths.csv', '.tasks.csv', '.anomalies.jsonl']]
    temporary = a.output.with_suffix(a.output.suffix + '.tmp')
    if any(path.exists() for path in artifacts + [temporary]):
        p.error('Output or diagnostic report already exists; choose a new output path')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    signature = tokenizer_signature(a.model)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(read=0, samples=0, supervised_tokens=0, skipped_overlength=0,
                     anomaly_samples=0)
    diagnostics = Diagnostics()
    try:
        with a.data.open() as source, temporary.open('x') as out, \
             a.output.with_suffix('.anomalies.jsonl').open('x') as anomalies:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                lengths = {}
                sample = prepare_row(tokenizer, row, a.max_length, signature, diagnostics=lengths)
                record = diagnostics.add(row, line_number, lengths, sample is not None)
                counts['read'] += 1
                if lengths['total_tokens'] >= a.anomaly_threshold:
                    counts['anomaly_samples'] += 1
                    anomaly = describe_anomaly(tokenizer, row, record, a.diagnostic_preview_chars)
                    anomalies.write(json.dumps(anomaly, ensure_ascii=False) + '\n')
                    print(f"Anomaly: task={row['task_id']} step={row['step']} source_line={line_number} "
                          f"tokens={lengths['total_tokens']} "
                          f"largest_message_index={anomaly['largest_messages'][0]['message_index'] if anomaly['largest_messages'] else 'none'}",
                          flush=True)
                if sample is None:
                    counts['skipped_overlength'] += 1
                    continue
                out.write(json.dumps(sample, ensure_ascii=False) + '\n')
                counts['samples'] += 1
                counts['supervised_tokens'] += sample['metadata']['response_length']
        summary = diagnostics.summary()
        stats = dict(counts=counts, unique_tasks=summary['tasks']['with_retained_samples'],
            max_length=a.max_length, anomaly_threshold=a.anomaly_threshold,
            tokenizer_signature=signature, slime_revision=SLIME_REVISION,
            source=str(a.data.resolve()), source_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(),
            **summary)
        write_csv(a.output.with_suffix('.lengths.csv'), diagnostics.records)
        write_csv(a.output.with_suffix('.tasks.csv'), diagnostics.task_rows())
        a.output.with_suffix('.stats.json').write_text(json.dumps(stats, indent=2))
        if not counts['samples']:
            raise SystemExit('No samples fit max-length; diagnostic reports were saved')
        temporary.rename(a.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    # Keep the console compact; detailed distributions and top-20 are on disk.
    print(json.dumps({k: stats[k] for k in ['counts', 'unique_tasks', 'max_length',
        'tokenizer_signature', 'slime_revision', 'source_sha256', 'tasks', 'complete_task']}, indent=2))
    print('Diagnostic reports:', ', '.join(str(path) for path in artifacts[1:]))


if __name__ == '__main__':
    main()
