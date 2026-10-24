import unittest
import json
import tempfile
from unittest.mock import patch
from pathlib import Path
from convert_official_sft import convert_task
from train_lora import encode_sample, configure_model


class Worker:
    def __init__(self, success=True):
        self.success = success
        self.calls = []

    def call(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if command == 'read_trajectory':
            return [{'input': 'x = 1', 'output': 'stale'},
                    {'input': 'apis.supervisor.complete_task()', 'output': 'old'}]
        if command == 'reset':
            return dict(instruction='Task: USER:\nnot a role', runtime={}, supervisor=dict(
                first_name='A', last_name='B', email='a@b.c', phone_number='123'))
        if command == 'execute':
            return dict(observation='fresh', declared_done='complete_task' in kwargs['code'])
        if command == 'evaluate':
            return dict(success=self.success)


class Tokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            enable_thinking=False):
        text = ''.join(m['role'] + ':' + m['content'] + '!' for m in messages)
        return text + ('assistant:' if add_generation_prompt else '')

    def __call__(self, text, **kwargs):
        return dict(input_ids=list(text.encode()))


class SFTTests(unittest.TestCase):
    def test_full_mode_unfreezes_original_model_without_adapter(self):
        class Model:
            def __init__(self):
                self.flags = [False, False, True]

            def requires_grad_(self, enabled):
                self.flags = [enabled] * len(self.flags)
        model = Model()
        result = configure_model(model, 'full', 16)
        self.assertIs(result, model)
        self.assertEqual(result.flags, [True, True, True])

    def test_cli_train_whitelist_and_audit(self):
        import convert_official_sft as converter
        class CLIWorker(Worker):
            def __init__(self, *args):
                super().__init__()

            def call(self, command, **kwargs):
                if command == 'list':
                    return ['a_1', 'missing_1']
                if command == 'read_trajectory':
                    assert '/a_1/' in kwargs['path']
                return super().call(command, **kwargs)

            def close(self):
                pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for task in ['a_1', 'test_only_1']:
                log = root / 'experiments/outputs' / converter.DEFAULT_EXPERIMENT / 'tasks' / task / 'logs/environment_io.md'
                log.parent.mkdir(parents=True)
                log.write_text('fixture')
            output = root / 'export'
            with patch.object(converter, 'WorldProcess', CLIWorker), patch('sys.argv', [
                'convert', '--appworld-root', str(root), '--output-dir', str(output)]):
                converter.main()
            rows = [json.loads(line) for line in (output / 'train.jsonl').read_text().splitlines()]
            self.assertEqual({row['task_id'] for row in rows}, {'a_1'})
            counts = json.loads((output / 'summary.json').read_text())['counts']
            self.assertEqual(counts['success'], 1)
            self.assertEqual(counts['missing'], 1)

    def test_replay_and_prompt(self):
        rows, _, _ = convert_task(Worker(), 'a_1', Path('fake'), 'replay', 42)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['completion'], 'x = 1')
        self.assertIn('fresh', rows[1]['messages'][-1]['content'])
        self.assertNotIn('stale', rows[1]['messages'][-1]['content'])
        self.assertEqual(rows[0]['messages'][-1]['role'], 'user')
        self.assertEqual(rows[1]['messages'][-2]['role'], 'assistant')

    def test_failed_task_has_no_samples(self):
        rows, _, _ = convert_task(Worker(False), 'a_1', Path('fake'), 'replay', 42)
        self.assertEqual(rows, [])

    def test_mask_and_no_truncation(self):
        row = dict(messages=[dict(role='user', content='hello')], completion='print(1)')
        encoded = encode_sample(Tokenizer(), row, 100)
        supervised = [x for x in encoded['labels'] if x != -100]
        self.assertEqual(bytes(supervised).decode(), 'print(1)!')
        self.assertEqual(len(encoded['labels']), len(encoded['input_ids']))
        self.assertIsNone(encode_sample(Tokenizer(), row, 10))

    def test_invalid_code_rejected(self):
        w = Worker()
        original = w.call
        w.call = lambda command, **kwargs: ([dict(input='this is invalid python !')]
            if command == 'read_trajectory' else original(command, **kwargs))
        with self.assertRaises(SyntaxError):
            convert_task(w, 'a_1', Path('fake'), 'replay', 42)


if __name__ == '__main__':
    unittest.main()
