import asyncio
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from appworld_grpo import run_episode, append_segment, log_rollout_data
from test_sft import Tokenizer


class ChatTokenizer(Tokenizer):
    def decode(self, ids, skip_special_tokens=False):
        return bytes(ids).decode()

    def convert_tokens_to_ids(self, token):
        return ord('!')


class Worker:
    instances = []
    success = True
    fail_reset = False

    def __init__(self, *args):
        self.calls, self.closed = [], False
        Worker.instances.append(self)

    def call(self, command, **kwargs):
        self.calls.append((command,kwargs))
        if command == 'reset':
            if self.fail_reset:
                raise RuntimeError('infrastructure unavailable')
            return dict(instruction='Find answer',supervisor=dict(first_name='A',last_name='B',
                email='a@b',phone_number='123'))
        if command == 'execute':
            return dict(observation='result',declared_done='complete_task' in kwargs['code'])
        if command == 'evaluate':
            return dict(success=self.success)

    def close(self):
        self.closed = True


class GRPOTests(unittest.TestCase):
    def config(self, directory):
        cfg = dict(appworld_python='fake',appworld_root=directory,env_timeout=1,
            experiment='test',max_steps=3,seed=42,context_length=100000,
            max_new_tokens=100,observation_tokens=128,output_dir=directory)
        path = Path(directory)/'config.json'
        path.write_text(json.dumps(cfg))
        return path

    def sample(self):
        return SimpleNamespace(metadata=dict(task_id='a_1'),index=3,group_index=0,
            Status=SimpleNamespace(COMPLETED='completed'))

    def test_multiturn_mask_exact_ids_reward_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            codes = ['x=1','apis.supervisor.complete_task()']
            payloads = []
            async def post(url,payload):
                payloads.append(payload)
                code = codes[len(payloads)-1]
                ids = list((code+'!').encode())
                return dict(text=code,meta_info=dict(finish_reason=dict(type='stop'),
                    output_token_logprobs=[[-.5,token,None] for token in ids]))
            with patch.dict(os.environ,APPWORLD_GRPO_CONFIG=str(self.config(directory))):
                sample = asyncio.run(run_episode(SimpleNamespace(sglang_router_ip='localhost',
                    sglang_router_port=123),self.sample(),{'seed':99},ChatTokenizer(),post,Worker))
            self.assertEqual(sample.reward,1)
            self.assertEqual(sum(sample.loss_mask),sum(len(code)+1 for code in codes))
            self.assertTrue(0 in sample.loss_mask)
            self.assertEqual(len(sample.loss_mask),len(sample.rollout_log_probs))
            self.assertGreater(len(payloads[1]['input_ids']),len(payloads[0]['input_ids']))
            self.assertTrue(all('seed' not in p['sampling_params'] for p in payloads))
            self.assertTrue(all('sampling_seed' in p['sampling_params'] for p in payloads))
            self.assertNotEqual(payloads[0]['sampling_params']['sampling_seed'],
                                payloads[1]['sampling_params']['sampling_seed'])
            action_logps = [p for p,m in zip(sample.rollout_log_probs,sample.loss_mask) if m]
            self.assertTrue(all(p==-.5 for p in action_logps))
            self.assertTrue(Worker.instances[-1].closed)

    def test_declared_done_is_not_reward(self):
        with tempfile.TemporaryDirectory() as directory:
            async def post(*args):
                return dict(text='apis.supervisor.complete_task()',meta_info=dict(
                    finish_reason=dict(type='stop'),output_token_logprobs=[[-.5,33,None]]))
            with patch.dict(os.environ,APPWORLD_GRPO_CONFIG=str(self.config(directory))), \
                 patch.object(Worker,'success',False):
                sample=asyncio.run(run_episode(SimpleNamespace(sglang_router_ip='x',sglang_router_port=1),
                    self.sample(),{},ChatTokenizer(),post,Worker))
            self.assertEqual(sample.reward,0)

    def test_infrastructure_error_not_zero_reward(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ,APPWORLD_GRPO_CONFIG=str(self.config(directory))), \
                 patch.object(Worker,'fail_reset',True):
                with self.assertRaises(RuntimeError):
                    asyncio.run(run_episode(SimpleNamespace(),self.sample(),{},ChatTokenizer(),None,Worker))
            self.assertTrue(Worker.instances[-1].closed)

    def test_format_error_not_submitted_and_feedback_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            payloads=[]
            async def post(url,payload):
                payloads.append(payload)
                code='Here is code:' if len(payloads)==1 else 'apis.supervisor.complete_task()'
                return dict(text=code,meta_info=dict(finish_reason=dict(type='stop'),
                    output_token_logprobs=[[-.5,t,None] for t in (code+'!').encode()]))
            with patch.dict(os.environ,APPWORLD_GRPO_CONFIG=str(self.config(directory))):
                sample=asyncio.run(run_episode(SimpleNamespace(sglang_router_ip='x',sglang_router_port=1),
                    self.sample(),{},ChatTokenizer(),post,Worker))
            calls=[kw for cmd,kw in Worker.instances[-1].calls if cmd=='execute']
            self.assertEqual(len(calls),1)
            self.assertEqual(sample.metadata['appworld']['format_errors'],1)
            self.assertIn('OUTPUT FORMAT ERROR',bytes(payloads[1]['input_ids']).decode())

    def test_length_limited_code_is_not_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            async def post(*args):
                return dict(text='print(',meta_info=dict(finish_reason=dict(type='length'),
                    output_token_logprobs=[[-.5,112,None]]))
            with patch.dict(os.environ,APPWORLD_GRPO_CONFIG=str(self.config(directory))), \
                 patch.object(Worker,'success',False):
                sample=asyncio.run(run_episode(SimpleNamespace(sglang_router_ip='x',sglang_router_port=1),
                    self.sample(),{},ChatTokenizer(),post,Worker))
            self.assertFalse(any(cmd=='execute' for cmd,kw in Worker.instances[-1].calls))
            self.assertEqual(sample.reward,0)
            self.assertEqual(sample.metadata['appworld']['termination'],'generation_limit')

    def test_group_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            samples=[]
            for i,reward in enumerate([0,1,0,0]):
                s=self.sample();s.reward=reward;s.loss_mask=[1,0,1]
                s.metadata['appworld']=dict(steps=2,format_errors=0,execution_errors=0,
                    observation_truncations=0,termination='declared_done')
                samples.append(s)
            with patch.dict(os.environ,APPWORLD_GRPO_CONFIG=str(self.config(directory))):
                self.assertFalse(log_rollout_data(0,None,samples,{},1))
            row=json.loads((Path(directory)/'project_metrics/00000000.json').read_text())
            self.assertEqual(row['metrics']['appworld/mixed_group_rate'],1)
            self.assertEqual(row['metrics']['appworld/success_rate'],.25)

    def test_reject_misaligned_log_probs(self):
        with self.assertRaises(ValueError):
            append_segment([],[],[],[1,2],True,[-.5])

class TrackingTests(unittest.TestCase):
    def test_bridge_deduplicates_project_and_native_metrics(self):
        from grpo_tracking import SwanLabBridge
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'project_metrics').mkdir()
            (root/'project_metrics/0.json').write_text(json.dumps(dict(step=0,metrics={'appworld/success_rate':.25})))
            (root/'tensorboard').mkdir()
            (root/'tensorboard/events.out.tfevents.test').touch()
            class Acc:
                def __init__(self,*args,**kwargs): pass
                def Reload(self): pass
                def Tags(self): return dict(scalars=['train/loss'])
                def Scalars(self,tag): return [SimpleNamespace(step=0,wall_time=1,value=.4)]
            logged=[]
            bridge=SwanLabBridge.__new__(SwanLabBridge)
            bridge.root=root;bridge.seen_events=set();bridge.seen_rollouts=set();bridge.accumulators={}
            bridge.event_class=Acc
            bridge.swanlab=SimpleNamespace(log=lambda data,step:logged.append((data,step)))
            bridge.poll();bridge.poll()
            self.assertEqual(len(logged),2)
            self.assertIn(({'slime/train/loss':.4},0),logged)


if __name__ == '__main__':
    unittest.main()
