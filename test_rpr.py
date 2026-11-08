import asyncio
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from appworld_grpo import run_episode
from appworld_rpr import fdr_rewards, shaped_reward, spr_rewards
from build_rsr2_sft import replay_episode, select_episodes
from calibrate_rpr_rewards import nearest_rank_percentile, summarize
from prepare_rpr_load_root import prepare
from run_rpr import architecture_args, main as rpr_main, training_args, write_role_config, write_task_config
from test_grpo import ChatTokenizer, Worker


class RPRTests(unittest.TestCase):
    def config(self, directory, scale_file):
        path = Path(directory)/'config.json'
        path.write_text(json.dumps(dict(appworld_python='fake', appworld_root=directory,
            env_timeout=1, experiment='rpr-test', max_steps=3, seed=42,
            context_length=100000, max_new_tokens=100, observation_tokens=128,
            output_dir=directory, enable_tb_gae=True, reward_scale_file=str(scale_file),
            reward_mode='fdr')))
        return path

    def sample(self, success, passed=2, total=4, task='family_1'):
        return SimpleNamespace(metadata=dict(task_id=task, appworld=dict(family_id='family',
            episode_id=task, evaluation=dict(success=success, passed_checks=passed,
                                             total_checks=total))))

    def test_fdr_spr_and_binary_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            scales = Path(directory)/'scales.json'
            scales.write_text(json.dumps(dict(global_divisor=8, family_divisors={'family':4})))
            cfg = self.config(directory, scales)
            with patch.dict(os.environ, APPWORLD_GRPO_CONFIG=str(cfg)):
                self.assertEqual(shaped_reward(self.sample(False), 'fdr')[0], .5)
                self.assertEqual(shaped_reward(self.sample(True), 'spr')[0], 1+2/6)
                fallback = self.sample(True, passed=None, total=None)
                self.assertEqual(shaped_reward(fallback, 'fdr')[0], 1)
                raw, processed = fdr_rewards(None, [self.sample(False), self.sample(True)])
                self.assertEqual(raw, processed)
                self.assertEqual(spr_rewards(None, [self.sample(False)])[0], [2/6])

    def test_rollout_emits_turn_and_terminal_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            scales = Path(directory)/'scales.json'
            scales.write_text(json.dumps(dict(global_divisor=1, family_divisors={})))
            cfg = self.config(directory, scales)
            codes = ['x=1', 'apis.supervisor.complete_task()']
            calls = []
            async def post(url, payload):
                code = codes[len(calls)]
                calls.append(code)
                ids = list((code+'!').encode())
                return dict(text=code, meta_info=dict(finish_reason=dict(type='stop'),
                    output_token_logprobs=[[-.5, token, None] for token in ids]))
            sample = SimpleNamespace(metadata=dict(task_id='family_1'), index=3, group_index=0,
                Status=SimpleNamespace(COMPLETED='completed'))
            with patch.dict(os.environ, APPWORLD_GRPO_CONFIG=str(cfg)):
                result = asyncio.run(run_episode(SimpleNamespace(sglang_router_ip='x',
                    sglang_router_port=1), sample, {}, ChatTokenizer(), post, Worker))
            self.assertEqual(sum(result.metadata['turn_end_mask']), 2)
            self.assertEqual(sum(result.metadata['is_real_done']), 1)
            self.assertEqual(len(result.metadata['turn_end_mask']), result.response_length)
            self.assertTrue(result.metadata['is_real_done'][-1])
            self.assertEqual(result.metadata['tb_boundary']['chunk_count'], 1)

    def test_scale_summary_uses_frozen_family_p90(self):
        rows = [dict(task_id='a_1', family_id='a', total_checks=value) for value in [1,2,10]]
        rows += [dict(task_id='b_1', family_id='b', total_checks=4)]
        report = summarize(rows, 90)
        self.assertEqual(nearest_rank_percentile([1,2,10], 90), 10)
        self.assertEqual(report['family_divisors'], {'a':10.0, 'b':4.0})
        self.assertEqual(report['global_divisor'], 10.0)

    def test_prepare_paired_load_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor, critic = root/'actor_iter', root/'critic_iter'
            actor.mkdir(); critic.mkdir()
            load, critic_load = prepare(actor, critic, root/'load')
            self.assertEqual((load/'release').resolve(), actor.resolve())
            self.assertEqual((critic_load/'release').resolve(), critic.resolve())
            self.assertEqual((load/'latest_checkpointed_iteration.txt').read_text(), 'release\n')

    def test_rsr2_selects_shortest_success_and_drops_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                ('long.json', 'a_1', True, 3),
                ('short.json', 'a_1', True, 1),
                ('failed.json', 'b_1', False, 1),
            ]
            paths = []
            for name, task, success, turns in rows:
                path = root/name
                path.write_text(json.dumps(dict(task_id=task,
                    evaluation=dict(success=success), turns=[{'raw_completion':'x=1'}]*turns)))
                paths.append(path)
            selected = select_episodes(paths)
            self.assertEqual([(path.name, row['task_id']) for path, row in selected],
                             [('short.json', 'a_1')])

    def test_rsr2_replays_before_exporting_targets(self):
        class ReplayWorker:
            def __init__(self):
                self.actions = []
            def call(self, command, **kwargs):
                if command == 'reset':
                    return dict(instruction='do it', supervisor=dict(first_name='A', last_name='B',
                        email='a@b', phone_number='1'), runtime={'appworld_version':'test'})
                if command == 'execute':
                    self.actions.append(kwargs['code'])
                    return dict(observation='ok', declared_done=len(self.actions) == 2)
                if command == 'evaluate':
                    return dict(success=True, passed_checks=2, total_checks=2)
                raise AssertionError(command)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'episode.json'
            episode = dict(task_id='a_1', summary=dict(episode_id='source'),
                turns=[dict(raw_completion='x=1'),
                       dict(raw_completion='apis.supervisor.complete_task()')])
            path.write_text(json.dumps(episode))
            worker = ReplayWorker()
            rows, evaluation, runtime = replay_episode(worker, path, episode, 'replay/test', 42)
            self.assertEqual(worker.actions, ['x=1', 'apis.supervisor.complete_task()'])
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[-1]['completion'], 'apis.supervisor.complete_task()')
            self.assertTrue(evaluation['success'])
            self.assertEqual(runtime['appworld_version'], 'test')

    def test_launcher_builds_ppo_tb_gae_and_separate_critic_stage(self):
        model_config = dict(model_type='qwen3', num_hidden_layers=36, hidden_size=2560,
            intermediate_size=9728, num_attention_heads=32, num_key_value_heads=8,
            head_dim=128, vocab_size=151936, rope_theta=1000000,
            max_position_embeddings=32768)
        self.assertIn('--qk-layernorm', architecture_args(model_config))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            role, task = root/'role.yaml', root/'task.yaml'
            write_role_config(role, 5e-7, 1.5e-5)
            write_task_config(task, 8192, 1, 1)
            args = SimpleNamespace(phase='rpr1', num_gpus=4, actor_gpus=2, rollout_tp=1,
                tp_size=2, batch_size=16, updates=12, critic_only_steps=4,
                model=root/'model', load_root=root/'load', save_interval=2,
                context_length=16384, temperature=.8, gamma_turn=1., lambda_turn=1.,
                actor_lr=5e-7, train_max_tokens=8192, sglang_memory=.55,
                server_concurrency=16, seed=42)
            command = training_args(args, root/'prompts.jsonl', root, role, task, model_config)
            self.assertEqual(command[command.index('--advantage-estimator')+1], 'ppo')
            self.assertIn('--enable-tb-gae', command)
            self.assertEqual(command[command.index('--num-critic-only-steps')+1], '4')
            self.assertEqual(command[command.index('--rollout-num-gpus')+1], '2')
            self.assertEqual(command[command.index('--n-samples-per-prompt')+1], '1')

    def test_rpr_dry_run_materializes_reproducible_command(self):
        class ListWorker:
            def __init__(self, *args): pass
            def call(self, command, **kwargs):
                self.assert_command = command
                return [f'task_{index}' for index in range(16)]
            def close(self): pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root/'model'; model.mkdir()
            model_config = dict(model_type='qwen3', num_hidden_layers=36, hidden_size=2560,
                intermediate_size=9728, num_attention_heads=32, num_key_value_heads=8,
                head_dim=128, vocab_size=151936, rope_theta=1000000,
                max_position_embeddings=32768)
            (model/'config.json').write_text(json.dumps(model_config))
            load = root/'load'; load.mkdir()
            (load/'latest_checkpointed_iteration.txt').write_text('release\n')
            critic = root/'load_critic'; critic.mkdir()
            (critic/'latest_checkpointed_iteration.txt').write_text('release\n')
            terminal = root/'terminal'
            for name in ['train.py', 'train_async.py', 'slime/utils/tb_gae_config.py',
                         'slime_plugins/harbor/tb_gae.py', 'slime_plugins/harbor/train_loss.py']:
                path = terminal/name; path.parent.mkdir(parents=True, exist_ok=True); path.touch()
            scales = root/'scales.json'
            scales.write_text(json.dumps(dict(schema='appworld_rpr_reward_scales_v1',
                global_divisor=4, family_divisors={})))
            output = root/'run'
            with patch('run_rpr.WorldProcess', ListWorker), patch('sys.stdout', io.StringIO()):
                rpr_main(['--phase','rpr1','--model',str(model),'--load-root',str(load),
                    '--terminal-rsi-dir',str(terminal),'--appworld-python','fake',
                    '--appworld-root',str(root/'appworld'),'--reward-scales',str(scales),
                    '--output-dir',str(output),'--dry-run'])
            command = json.loads((output/'command.json').read_text())
            self.assertEqual(command[2], str(terminal.resolve()/'train_async.py'))
            self.assertIn('appworld_rpr.fdr_rewards', command)
            self.assertEqual(json.loads((output/'project_config.json').read_text())['reward_mode'], 'fdr')


if __name__ == '__main__':
    unittest.main()
