"""Project-local multi-turn generation and metric hooks for Slime GRPO."""
import asyncio
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import uuid
import weakref

from observation_filter import bound_observation
from run_agent import (WorldProcess, chat_input_ids, extract_code, execution_feedback,
                       format_error_feedback, normalize_input_ids, prompt_messages)


def config():
    return json.loads(Path(os.environ['APPWORLD_GRPO_CONFIG']).read_text())


def append_segment(tokens, mask, log_probs, ids, trainable, values=None):
    if not ids:
        return
    if trainable and (values is None or len(ids) != len(values)):
        raise ValueError('Sampled token IDs and log probabilities must align exactly')
    if trainable and not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite sampled log probabilities')
    tokens.extend(ids)
    mask.extend([int(trainable)]*len(ids))
    log_probs.extend(values if trainable else [0.0]*len(ids))




async def run_episode(args, sample, sampling_params, tokenizer, post, worker_factory=WorldProcess):
    cfg = config()
    if getattr(args, 'partial_rollout', False):
        raise ValueError('AppWorld GRPO requires complete synchronous rollouts, not partial rollout')
    task_id = sample.metadata['task_id']
    episode_id = uuid.uuid4().hex
    # An aborted episode is restarted, so its old policy-version history is no longer relevant.
    if hasattr(sample, 'weight_versions'):
        sample.weight_versions = []
    worker = worker_factory(cfg['appworld_python'], cfg['appworld_root'], cfg['env_timeout'])
    tokens, mask, log_probs, turns = [], [], [], []
    reason = 'step_limit'
    try:
        task = await asyncio.to_thread(worker.call, 'reset', task_id=task_id,
            experiment_name=cfg['experiment']+'/'+episode_id,
            max_interactions=cfg['max_steps'], seed=cfg['seed'])
        messages = prompt_messages(task['supervisor'], task['instruction'])
        initial = chat_input_ids(tokenizer, messages)
        tokens = initial.copy()
        for turn in range(cfg['max_steps']):
            remaining = cfg['context_length']-len(tokens)
            if remaining < 16:
                reason = 'context_limit'
                break
            params = dict(sampling_params)
            # /generate uses SamplingParams.sampling_seed, not the OpenAI-style seed.
            params.pop('seed', None)
            params.update(max_new_tokens=min(cfg['max_new_tokens'], remaining),
                          sampling_seed=(cfg['seed']+int(sample.index or 0)*1000003+turn) % (2**31))
            output = await post(f'http://{args.sglang_router_ip}:{args.sglang_router_port}/generate',
                dict(input_ids=tokens.copy(), sampling_params=params, return_logprob=True))
            finish = output['meta_info']['finish_reason']['type']
            if finish == 'abort':
                if not cfg.get('async_training'):
                    raise RuntimeError('SGLang aborted a synchronous AppWorld rollout')
                sample.tokens = tokens
                sample.response_length = len(mask)
                sample.loss_mask = mask
                sample.rollout_log_probs = log_probs
                sample.response = tokenizer.decode(tokens[len(initial):], skip_special_tokens=False)
                sample.reward = None
                sample.status = sample.Status.ABORTED
                # Slime requeues aborted groups. Never score them as failures. A retried
                # unfinished episode starts from a fresh world rather than replaying side effects.
                return sample
            entries = output['meta_info']['output_token_logprobs']
            ids, values = [int(e[1]) for e in entries], [float(e[0]) for e in entries]
            if not ids or len(tokens)+len(ids) > cfg['context_length']:
                raise ValueError('Empty generation or response exceeds context budget')
            append_segment(tokens, mask, log_probs, ids, True, values)
            version = output['meta_info'].get('weight_version')
            if version is not None:
                if not hasattr(sample, 'weight_versions'):
                    sample.weight_versions = []
                sample.weight_versions.append(str(version))
            raw = output['text']
            if finish == 'length':
                reason = 'generation_limit'
                turns.append(dict(turn=turn+1, raw_completion=raw, feedback_kind='generation_limit'))
                break  # Never submit a potentially incomplete code block.
            if finish != 'stop':
                raise ValueError(f'Unsupported SGLang finish reason: {finish}')
            try:
                code = extract_code(raw)
            except (ValueError, SyntaxError) as exc:
                observation, done, kind = format_error_feedback(exc), False, 'format_error'
            else:
                result = await asyncio.to_thread(worker.call, 'execute', code=code)
                observation, failed = execution_feedback(result['observation'])
                done = result['declared_done']
                kind = 'execution_error' if failed else 'execution_output'
            observation, clipped = bound_observation(tokenizer, observation, cfg['observation_tokens'])
            turns.append(dict(turn=turn+1, raw_completion=raw, observation=observation,
                              feedback_kind=kind, observation_truncated=clipped))
            if done:
                reason = 'declared_done'
                break
            if turn+1 == cfg['max_steps']:
                break
            # Qwen3 chat boundary is context, never an invented sampled action.
            end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
            boundary = [] if tokens[-1] == end_id else [end_id]
            rendered = '\n'+tokenizer.apply_chat_template([dict(role='user', content=observation)],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            boundary += normalize_input_ids(tokenizer(rendered, add_special_tokens=False,
                                                      return_attention_mask=False))
            if len(tokens)+len(boundary)+16 > cfg['context_length']:
                reason = 'context_limit'
                break
            append_segment(tokens, mask, log_probs, boundary, False)
        evaluation = await asyncio.to_thread(worker.call, 'evaluate')
        if not mask or not any(mask):
            raise ValueError('No action tokens: increase context_length or check the prompt')
        sample.tokens = tokens
        sample.prompt = tokenizer.decode(initial, skip_special_tokens=False)
        sample.response_length = len(mask)
        sample.response = tokenizer.decode(tokens[len(initial):], skip_special_tokens=False)
        sample.loss_mask = mask
        sample.rollout_log_probs = log_probs
        sample.reward = float(evaluation['success'] is True)
        # Limits are valid scored outcomes; do not let generic truncation filtering discard failures.
        sample.status = sample.Status.COMPLETED
        sample.metadata = dict(sample.metadata, appworld=dict(termination=reason, steps=len(turns),
            format_errors=sum(t['feedback_kind']=='format_error' for t in turns),
            execution_errors=sum(t['feedback_kind']=='execution_error' for t in turns),
            observation_truncations=sum(t.get('observation_truncated', False) for t in turns)))
        assert len(tokens)-len(initial) == len(mask) == len(log_probs)
        path = Path(cfg['output_dir'])/'episodes'/f'{episode_id}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(task_id=task_id, group_index=sample.group_index,
            sample_index=sample.index, reward=sample.reward, evaluation=evaluation,
            summary=sample.metadata['appworld'], turns=turns), ensure_ascii=False))
        return sample
    finally:
        await asyncio.to_thread(worker.close)


_ENV_SEMAPHORES = weakref.WeakKeyDictionary()


async def generate(args, sample, sampling_params, evaluation=False):
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.http_utils import post
    cfg = config()
    loop = asyncio.get_running_loop()
    if loop not in _ENV_SEMAPHORES:
        _ENV_SEMAPHORES[loop] = asyncio.Semaphore(cfg.get('max_envs', 32))
    async with _ENV_SEMAPHORES[loop]:
        return await run_episode(args, sample, sampling_params, GenerateState(args).tokenizer, post)


async def reward(args, sample, **kwargs):
    if sample.reward not in (0.0, 1.0):
        raise ValueError('Missing fresh AppWorld evaluation reward')
    return sample.reward


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    groups = defaultdict(list)
    flattened = []
    for item in samples:
        flattened.extend(item if isinstance(item, list) else [item])
    for sample in flattened:
        groups[sample.group_index].append(float(sample.reward))
    if any(len(values) != 4 for values in groups.values()):
        raise ValueError('Expected exactly 4 complete trajectories per task group')
    metrics = dict(success_rate=sum(s.reward for s in flattened)/len(flattened),
        mixed_group_rate=sum(min(v)!=max(v) for v in groups.values())/len(groups),
        all_fail_group_rate=sum(max(v)==0 for v in groups.values())/len(groups),
        all_success_group_rate=sum(min(v)==1 for v in groups.values())/len(groups),
        average_turns=sum(s.metadata['appworld']['steps'] for s in flattened)/len(flattened),
        action_tokens=sum(sum(s.loss_mask) for s in flattened), rollout_seconds=rollout_time)
    for key in ['format_errors', 'execution_errors', 'observation_truncations']:
        metrics[key] = sum(s.metadata['appworld'][key] for s in flattened)
    for reason, count in Counter(s.metadata['appworld']['termination'] for s in flattened).items():
        metrics['termination/'+reason] = count
    path = Path(config()['output_dir'])/'project_metrics'/f'{rollout_id:08d}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(step=rollout_id, metrics={'appworld/'+k:v for k,v in metrics.items()})))
    return False  # Keep Slime's native logs/TensorBoard reporting too.
