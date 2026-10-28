import asyncio
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from run_grpo import training_args
from gpu_monitor import GPUMonitor, select_gpus
from slime_runtime import find_async_training_script
import test_grpo as fixtures
from appworld_grpo import run_episode


def arguments(implementation='pipeline'):
    return SimpleNamespace(tasks_per_batch=8,async_training=True,actor_gpus=2,num_gpus=4,
        tp_size=2,rollout_tp=1,model=Path('/model'),init_checkpoint=None,ref_checkpoint=None,
        save_interval=20,updates=100,context_length=16384,temperature=.8,kl_coef=.01,
        learning_rate=1e-6,sglang_memory=.7,seed=42,server_concurrency=16,
        async_implementation=implementation,no_save_hf=False,train_max_tokens=16384)


class AsyncTests(unittest.TestCase):
    def test_disjoint_gpu_topology_pipeline_and_fully_flags(self):
        cfg=dict(model_type='qwen3',hidden_size=2560,rope_theta=5000000,max_position_embeddings=262144)
        for mode in ['pipeline','fully']:
            args=training_args(arguments(mode),Path('/data'),Path('/out'),cfg)
            self.assertNotIn('--colocate',args)
            self.assertEqual(args[args.index('--actor-num-gpus-per-node')+1],'2')
            self.assertEqual(args[args.index('--rollout-num-gpus')+1],'2')
            self.assertEqual(args[args.index('--rollout-num-gpus-per-engine')+1],'1')
            self.assertIn('--use-tis',args)
            self.assertIn('--use-dynamic-batch-size',args)
            self.assertEqual('--update-weights-interval' in args,mode=='pipeline')
            self.assertEqual('--rollout-function-path' in args,mode=='fully')

    def test_native_async_entry_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'train.py').touch();(root/'train_async.py').touch()
            script,mode=find_async_training_script('auto',{'SLIME_DIR':directory})
            self.assertEqual(script.name,'train_async.py');self.assertEqual(mode,'pipeline')
            (root/'train_async.py').unlink()
            (root/'slime/rollout').mkdir(parents=True)
            (root/'slime/rollout/fully_async_rollout.py').touch()
            script,mode=find_async_training_script('auto',{'SLIME_DIR':directory})
            self.assertEqual(script.name,'train.py');self.assertEqual(mode,'fully')

    def test_allocated_gpu_identity(self):
        rows=[dict(index=i,uuid=f'GPU-{i}') for i in range(8)]
        self.assertEqual(len(select_gpus(rows,4,'0,1,2,3')),4)
        with self.assertRaises(ValueError): select_gpus(rows,4,'')

    def test_time_weighted_utilization_and_target(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor=GPUMonitor(directory,4,warmup=5,target=70)
            monitor.started=0
            def rows(util):
                return [dict(index=i,uuid=f'GPU-{i}',utilization=util,memory_used_mb=100,memory_total_mb=80000) for i in range(4)]
            with patch.object(monitor,'_read',side_effect=[rows(60),rows(80),rows(90)]), \
                 patch('gpu_monitor.time.monotonic',side_effect=[0,10,20]):
                for _ in range(3):monitor._sample()
            report=json.loads((Path(directory)/'gpu_summary.json').read_text())
            self.assertAlmostEqual(report['steady_all_gpu_mean'],1100/15)
            self.assertTrue(report['target_met'])
            self.assertEqual(set(report['full_run_per_gpu_mean'].values()),{70})

    def test_async_aborted_episode_not_rewarded_or_executed(self):
        helper=fixtures.GRPOTests()
        with tempfile.TemporaryDirectory() as directory:
            path=helper.config(directory)
            cfg=json.loads(path.read_text());cfg['async_training']=True;path.write_text(json.dumps(cfg))
            async def post(*args):return dict(meta_info=dict(finish_reason=dict(type='abort')))
            sample=helper.sample();sample.Status.ABORTED='aborted'
            with patch.dict(os.environ,APPWORLD_GRPO_CONFIG=str(path)):
                result=asyncio.run(run_episode(SimpleNamespace(sglang_router_ip='x',sglang_router_port=1),
                    sample,{},fixtures.ChatTokenizer(),post,fixtures.Worker))
            self.assertIsNone(result.reward)
            self.assertEqual(result.status,'aborted')
            self.assertTrue(fixtures.Worker.instances[-1].closed)
            self.assertFalse(any(cmd in ['execute','evaluate'] for cmd,_ in fixtures.Worker.instances[-1].calls))


if __name__=='__main__':unittest.main()
