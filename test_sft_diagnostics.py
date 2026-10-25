import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sft_diagnostics import Diagnostics, distribution, complete_task_actions, describe_anomaly
from test_sft import Tokenizer


class DiagnosticsTests(unittest.TestCase):
    def test_histogram_boundaries_and_percentiles(self):
        stats = distribution([4096, 8192, 8193, 1000000, 1000001])
        self.assertEqual(sum(stats['histogram'].values()), 5)
        self.assertEqual(stats['histogram']['4097-8192'], 1)
        self.assertEqual(stats['histogram']['1000001+'], 1)
        self.assertEqual(stats['percentiles']['p50'], 8193)
        self.assertIsNone(distribution([])['mean'])

    def test_ast_actions_ignore_comments_and_strings(self):
        code = '''# apis.supervisor.complete_task()
s = "apis.supervisor.complete_task()"
apis.supervisor.complete_task(answer=1)
apis.supervisor.complete_task(status="fail")
apis.supervisor.complete_task(**kwargs)
'''
        self.assertEqual(complete_task_actions(code), dict(default_status=1, explicit_fail=1,
                                                         other_or_dynamic_status=1))

    def test_task_and_action_denominators(self):
        diag = Diagnostics()
        lengths = dict(total_tokens=20, prompt_tokens=15, completion_tokens=5)
        diag.add(dict(task_id='a_1', step=1, completion='x=1'), 1, lengths, True)
        diag.add(dict(task_id='a_1', step=2, completion='apis.supervisor.complete_task()'),
                 2, lengths, False)
        diag.add(dict(task_id='b_1', step=1, completion='x=2'), 3, lengths, False)
        self.assertEqual(diag.task_rows()[0]['retention_rate'], .5)
        self.assertIsNone(diag.task_rows()[1]['complete_task_retention_rate'])
        summary = diag.summary()
        self.assertEqual(summary['complete_task']['sample_retention_rate'], 0)
        self.assertEqual(summary['tasks']['fully_dropped'], 1)
        self.assertEqual(summary['tasks']['partially_retained'], 1)

    def test_anomaly_localization_without_raw_text(self):
        row = dict(messages=[dict(role='assistant', content='print(data)'),
                            dict(role='user', content='x'*100), dict(role='user', content='')])
        report = describe_anomaly(Tokenizer(), row, dict(source_line=4), 0)
        self.assertEqual(report['largest_messages'][0]['message_index'], 1)
        self.assertEqual(report['largest_messages'][0]['preceding_assistant_index'], 0)
        self.assertNotIn('preview', report['largest_messages'][0])

    def test_cli_reports_even_when_all_dropped(self):
        import prepare_slime_sft as script
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'config.json').write_text('{}')
            data = root/'input.jsonl'
            row = dict(task_id='a_1', step=1, messages=[dict(role='user', content='x'*100)],
                       completion='apis.supervisor.complete_task()')
            data.write_text('\n'+json.dumps(row)+'\n')
            out = root/'train.jsonl'
            fake = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k:Tokenizer()))
            with patch.dict(sys.modules, transformers=fake), patch.object(sys, 'argv', [
                'prepare', '--model', str(root), '--data', str(data), '--output', str(out),
                '--max-length', '10', '--anomaly-threshold', '100']):
                with self.assertRaises(SystemExit):
                    script.main()
            stats = json.loads(out.with_suffix('.stats.json').read_text())
            self.assertEqual(stats['counts']['anomaly_samples'], 1)
            anomaly = json.loads(out.with_suffix('.anomalies.jsonl').read_text())
            self.assertEqual(anomaly['source_line'], 2)
            self.assertFalse(out.exists())
            self.assertFalse(out.with_suffix('.jsonl.tmp').exists())
            self.assertTrue(out.with_suffix('.tasks.csv').is_file())


if __name__ == '__main__':
    unittest.main()
