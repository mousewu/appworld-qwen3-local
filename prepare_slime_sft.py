"""Tokenize replay-verified AppWorld next-code examples for Slime SFT."""
import argparse
from collections import Counter
import json
from pathlib import Path

from slime_sft import prepare_row, tokenizer_signature, SLIME_REVISION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-length', type=int, default=8192)
    a = p.parse_args()
    if a.max_length <= 0:
        p.error('--max-length must be positive')
    if a.output.exists():
        p.error('Output already exists; choose a new path')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    signature = tokenizer_signature(a.model)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.output.with_suffix(a.output.suffix + '.tmp')
    counts = Counter()
    tasks = set()
    with a.data.open() as source, temporary.open('x') as out:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            sample = prepare_row(tokenizer, row, a.max_length, signature)
            counts['read'] += 1
            if sample is None:
                counts['skipped_overlength'] += 1
                continue
            out.write(json.dumps(sample, ensure_ascii=False) + '\n')
            counts['samples'] += 1
            counts['supervised_tokens'] += sample['metadata']['response_length']
            tasks.add(row['task_id'])
    if not counts['samples']:
        temporary.unlink()
        raise SystemExit('No samples fit max-length')
    temporary.rename(a.output)
    stats = dict(counts=counts, unique_tasks=len(tasks), max_length=a.max_length,
                 tokenizer_signature=signature, slime_revision=SLIME_REVISION,
                 source_sha256=__import__('hashlib').sha256(a.data.read_bytes()).hexdigest())
    a.output.with_suffix('.stats.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
