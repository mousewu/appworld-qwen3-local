import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from slime_sft import prepare_row, populate_sample, generate_rollout, tokenizer_signature
from test_sft import Tokenizer


class SlimeSFTTests(unittest.TestCase):
    def row(self):
        return dict(task_id='a_1', step=2, messages=[
            dict(role='user', content='Task'),
            dict(role='assistant', content='x = 1'),
            dict(role='user', content='Execution output')], completion='print(x)')

    def test_only_final_code_is_supervised(self):
        row = prepare_row(Tokenizer(), self.row(), 1000, 'signature')
        sample = populate_sample(SimpleNamespace(metadata=row['metadata']), 'signature')
        self.assertEqual(bytes(sample.tokens[-sample.response_length:]).decode(), 'print(x)!')
        mask = [0]*(len(sample.tokens)-sample.response_length) + sample.loss_mask
        self.assertEqual(len(mask), len(sample.tokens))
        self.assertEqual(sum(mask), len('print(x)!'))
        self.assertEqual(sample.reward, 0)

    def test_overlength_is_skipped(self):
        self.assertIsNone(prepare_row(Tokenizer(), self.row(), 10, 'signature'))

    def test_mismatched_tokenizer_rejected(self):
        row = prepare_row(Tokenizer(), self.row(), 1000, 'one')
        with self.assertRaises(ValueError):
            populate_sample(SimpleNamespace(metadata=row['metadata']), 'two')

    def test_malformed_completion_rejected(self):
        row = self.row()
        row['completion'] = 'Here is code:\n```python\nprint(1)\n```'
        with self.assertRaises(ValueError):
            prepare_row(Tokenizer(), row, 1000, 'signature')

    def test_rollout_contract_and_group_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'config.json').write_text('{}')
            signature = tokenizer_signature(directory)
            row = prepare_row(Tokenizer(), self.row(), 1000, signature)
            sample = SimpleNamespace(metadata=row['metadata'])
            groups = [[sample]]
            buffer = SimpleNamespace(get_samples=lambda n: groups)
            args = SimpleNamespace(hf_checkpoint=directory, rollout_batch_size=1)
            self.assertIs(generate_rollout(args, 0, buffer), groups)
            self.assertEqual(sample.loss_mask, [1]*sample.response_length)
            groups[0].append(sample)
            with self.assertRaises(ValueError):
                generate_rollout(args, 1, buffer)


if __name__ == '__main__':
    unittest.main()
