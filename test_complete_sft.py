import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import build_complete_sft as builder
from slime_sft import prepare_row, tokenizer_signature
from test_sft import Tokenizer


def trajectory(task_id, size=1):
    messages = [dict(role='user', content='task')]
    rows = [dict(task_id=task_id, step=1, messages=messages, completion='x=1')]
    messages = messages + [dict(role='assistant', content='x=1'), dict(role='user', content='o'*size)]
    rows.append(dict(task_id=task_id, step=2, messages=messages,
                     completion='apis.supervisor.complete_task()'))
    return rows


class CompleteTests(unittest.TestCase):
    def test_full_cli_nested_sets_exclusion_and_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'config.json').write_text('{}')
            signature = tokenizer_signature(root)
            rows = trajectory('a_1') + trajectory('b_1', 9000) + trajectory('c_1', 18000)
            rows += trajectory('million_1')
            rows += trajectory('gap_1')
            rows[-1]['step'] = 3
            rows += trajectory('missing_end_1')
            rows[-1]['completion'] = 'print(x)'
            rows += trajectory('bad_history_1')
            rows[-1]['messages'][0] = dict(role='user', content='changed')
            data = root/'source.jsonl'
            data.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            lengths = []
            for i, row in enumerate(rows, 1):
                values = {}
                prepare_row(Tokenizer(), row, 1, signature, diagnostics=values)
                if row['task_id'] == 'million_1' and row['step'] == 2:
                    values = dict(total_tokens=1000000, prompt_tokens=999995, completion_tokens=5)
                lengths.append(dict(source_line=i, task_id=row['task_id'], step=row['step'], **values))
            with (root/'cached.lengths.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(lengths[0]))
                writer.writeheader()
                writer.writerows(lengths)
            stats = root/'cached.stats.json'
            stats.write_text(json.dumps(dict(source_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
                                             tokenizer_signature=signature)))
            fake = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k:Tokenizer()))
            out = root/'complete'
            with patch.dict(sys.modules, transformers=fake), patch.object(sys, 'argv', [
                'build', '--model', str(root), '--data', str(data), '--output-dir', str(out),
                '--diagnostics-stats', str(stats)]):
                builder.main()
            summary = json.loads((out/'summary.json').read_text())
            self.assertEqual([s['tasks'] for s in summary['versions'].values()], [1, 2, 3])
            self.assertEqual([s['samples'] for s in summary['versions'].values()], [2, 4, 6])
            self.assertEqual([s['complete_task_calls'] for s in summary['versions'].values()], [1, 2, 3])
            self.assertEqual(summary['excluded_anomaly_tasks'][0]['task_id'], 'million_1')
            for name, expected in zip(builder.LIMITS, [{'a_1'}, {'a_1','b_1'}, {'a_1','b_1','c_1'}]):
                actual = [json.loads(line) for line in (out/name/'train.jsonl').read_text().splitlines()]
                self.assertEqual({r['task_id'] for r in actual}, expected)
                self.assertEqual(len(actual), 2*len(expected))
                tokens = [json.loads(line) for line in (out/name/'slime_train.jsonl').read_text().splitlines()]
                self.assertEqual(len(tokens), len(actual))

    def test_threshold_boundary_and_length_boundary(self):
        task = dict(max_total_tokens=8192, valid_history=True, _final_has_complete_task=True)
        self.assertEqual(builder.rejection_reasons(task, 8192, 1000000), [])
        task['max_total_tokens'] = 1000000
        self.assertIn('million_token_anomaly', builder.rejection_reasons(task, 2000000, 1000000))

    def test_cache_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            stats = Path(directory)/'test.stats.json'
            stats.write_text(json.dumps(dict(source_sha256='old',tokenizer_signature='sig')))
            with self.assertRaises(ValueError):
                builder.load_lengths(stats, 'new', 'sig')


if __name__ == '__main__':
    unittest.main()
