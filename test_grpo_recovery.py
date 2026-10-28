"""Recovery must preserve world history and the sampled policy trajectory."""
import asyncio
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from appworld_grpo import run_episode
from test_grpo import GRPOTests, Worker, ChatTokenizer


class RecoveryTests(unittest.TestCase):
    def run_case(self, mode):
        with tempfile.TemporaryDirectory() as directory:
            class Flaky(Worker):
                attempts = 0
                def call(self, command, **kwargs):
                    if command == 'execute' and 'complete_task' in kwargs['code']:
                        Flaky.attempts += 1
                        if mode == 'exhaust' or Flaky.attempts == 1:
                            raise TimeoutError('execute timed out')
                    result = super().call(command, **kwargs)
                    if mode == 'diverge' and command == 'execute' and Flaky.attempts:
                        result = dict(result, observation='different replay result')
                    return result
            generated = []
            async def post(*args):
                code = ['x=1', 'apis.supervisor.complete_task()'][len(generated)]
                generated.append(code)
                return dict(text=code, meta_info=dict(finish_reason=dict(type='stop'),
                    output_token_logprobs=[[-.5,t,None] for t in (code+'!').encode()]))
            cfg = GRPOTests().config(directory)
            with patch.dict(os.environ, APPWORLD_GRPO_CONFIG=str(cfg)):
                operation = run_episode(SimpleNamespace(sglang_router_ip='x', sglang_router_port=1),
                    GRPOTests().sample(), {}, ChatTokenizer(), post, Flaky)
                if mode == 'recover':
                    result = asyncio.run(operation)
                    self.assertEqual(result.reward, 1)
                    self.assertEqual(result.metadata['appworld']['timeout_recoveries'], 1)
                    self.assertEqual(sum(result.loss_mask), sum(len(c)+1 for c in generated))
                    self.assertIn('cleanup', result.metadata['appworld']['timings'])
                else:
                    with self.assertRaises((RuntimeError, TimeoutError)):
                        asyncio.run(operation)
            events = [json.loads(line) for line in next((Path(directory)/'episodes').glob('*.jsonl')).read_text().splitlines()]
            self.assertIn('timeout', [e['event'] for e in events])
            self.assertEqual(events[-1]['event'], 'episode_closed')
            self.assertEqual(len(generated), 2)  # recovery never resamples policy tokens
            if mode != 'recover':
                self.assertIn('episode_error', [e['event'] for e in events])
            self.assertTrue(Worker.instances[-1].closed)

    def test_recover_with_exact_replay(self): self.run_case('recover')
    def test_exhaustion_preserves_failure_journal(self): self.run_case('exhaust')
    def test_replay_divergence_fails_closed(self): self.run_case('diverge')

    def test_live_event_bridge_handles_partial_lines_and_deduplicates(self):
        from grpo_tracking import SwanLabBridge
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'episodes').mkdir()
            journal = root/'episodes/a.jsonl'
            journal.write_text('{"event":"timeout"}\n{"event":')
            logs = []
            bridge = SwanLabBridge(directory, 'disabled', 'test', 'test', {})
            bridge.swanlab = SimpleNamespace(log=lambda values, **kw: logs.append(values))
            bridge.poll_episode_events()
            bridge.poll_episode_events()
            self.assertEqual(len(logs), 1)
            with journal.open('a') as stream:
                stream.write('"episode_error"}\n')
            bridge.poll_episode_events()
            self.assertEqual(logs[-1]['environment_events/timeout'], 1)
            self.assertEqual(logs[-1]['environment_events/episode_error'], 1)

    def test_eight_gpu_actor_rollout_split(self):
        from test_async_grpo import arguments
        from run_grpo import training_args
        args = arguments()
        args.num_gpus = 8
        cfg = dict(model_type='qwen3', hidden_size=2560, rope_theta=5000000,
                   max_position_embeddings=262144)
        command = training_args(args, Path('/data'), Path('/out'), cfg)
        self.assertEqual(command[command.index('--rollout-num-gpus')+1], '6')
        self.assertEqual(command[command.index('--actor-num-gpus-per-node')+1], '2')
