import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from compare_checkpoints import CHECKPOINTS, main


class CompareCheckpointsTests(unittest.TestCase):
    def test_all_five_use_same_frozen_tasks_and_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'eval'
            argv = ['compare_checkpoints.py', '--appworld-root', str(root), '--output-dir', str(output)]
            for label in CHECKPOINTS:
                model = root / label
                model.mkdir()
                (model / 'config.json').write_text('{}')
                (model / 'model.safetensors').touch()
                argv.extend(['--' + label, str(model)])
            tasks = ['a_1', 'b_2']
            commands = []

            def run(command):
                commands.append(command[7:])
                result = Path(command[command.index('--output-dir') + 1]) / 'result'
                result.mkdir(parents=True)
                (result / 'summary.json').write_text(json.dumps(dict(results=[
                    dict(task_id=tasks[0], evaluated=True, success=True, termination='declared_done', steps=8),
                    dict(task_id=tasks[1], evaluated=True, success=False,
                         termination='repeated_execution_error', steps=4),
                ])))
                return SimpleNamespace(returncode=0)

            with patch.object(sys, 'argv', argv), patch('compare_checkpoints.WorldProcess') as probe, \
                 patch('compare_checkpoints.subprocess.run', side_effect=run):
                probe.return_value.call.return_value = tasks
                self.assertEqual(main(), 0)
                probe.return_value.call.assert_called_once_with('list', split='dev')
            self.assertEqual(len(commands), 5)
            self.assertTrue(all(command == commands[0] for command in commands))
            self.assertEqual(commands[0][-4:], ['--task-id', 'a_1', '--task-id', 'b_2'])
            rows = json.loads((output / 'comparison.json').read_text())
            self.assertEqual([row['checkpoint'] for row in rows], list(CHECKPOINTS))
            for row in rows:
                self.assertEqual(row['success_rate'], .5)
                self.assertEqual(row['average_steps'], 6)
                self.assertEqual(row['repeated_error_termination_rate'], .5)
