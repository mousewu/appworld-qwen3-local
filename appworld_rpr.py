"""AppWorld adapters for Terminal-RSI actor-critic PPO and TB-GAE.

The rollout itself stays in :mod:`appworld_grpo`; this module adds the strict
RPR contracts: explicit turn boundaries, frozen family reward scales, FDR/SPR
reward post-processing, and trajectory-deduplicated metrics after TB chunking.
"""
from collections import Counter
from functools import lru_cache
import json
import math
from pathlib import Path

from appworld_grpo import config, generate as generate_appworld


def task_family(task_id):
    if not isinstance(task_id, str) or not task_id:
        raise ValueError('AppWorld task_id must be a nonempty string')
    return task_id.rsplit('_', 1)[0]


def _flatten(node):
    if isinstance(node, list):
        for item in node:
            yield from _flatten(item)
    else:
        yield node


@lru_cache(maxsize=4)
def _load_scale_config(path_string):
    path = Path(path_string)
    scales = json.loads(path.read_text())
    global_divisor = scales.get('global_divisor')
    if isinstance(global_divisor, bool) or not isinstance(global_divisor, (int, float)) \
            or not math.isfinite(global_divisor) or global_divisor <= 0:
        raise ValueError('Reward scale file needs a positive finite global_divisor')
    table = scales.get('family_divisors', {})
    if not isinstance(table, dict):
        raise ValueError('family_divisors must be an object')
    return float(global_divisor), table


def _scale_config():
    return _load_scale_config(str(Path(config()['reward_scale_file']).resolve()))


def _reward_inputs(sample, global_divisor, table):
    metadata = sample.metadata.get('appworld', {})
    evaluation = metadata.get('evaluation', {})
    success = evaluation.get('success') is True
    passed, total = evaluation.get('passed_checks'), evaluation.get('total_checks')
    family = metadata.get('family_id') or task_family(sample.metadata['task_id'])
    divisor = table.get(family, global_divisor)
    if isinstance(divisor, bool) or not isinstance(divisor, (int, float)) \
            or not math.isfinite(divisor) or divisor <= 0:
        raise ValueError(f'Invalid reward divisor for family {family!r}: {divisor!r}')
    valid_progress = (type(passed) is int and passed >= 0 and type(total) is int
                      and total > 0 and passed <= total)
    return success, passed if valid_progress else None, total if valid_progress else None, family, float(divisor)


def shaped_reward(sample, mode):
    """Return paper-style FDR or SPR with binary fallback for invalid progress."""
    global_divisor, table = _scale_config()
    success, passed, total, family, divisor = _reward_inputs(sample, global_divisor, table)
    if passed is None:
        reward = float(success)
    elif mode == 'fdr':
        reward = passed / divisor
    elif mode == 'spr':
        reward = float(success) + passed / (passed + divisor)
    else:
        raise ValueError(f'Unknown RPR reward mode: {mode}')
    if not math.isfinite(reward) or reward < 0:
        raise ValueError(f'Invalid {mode.upper()} reward: {reward!r}')
    return reward, dict(mode=mode, family_id=family, divisor=divisor,
                        passed_checks=passed, total_checks=total, success=success)


def _post_process(samples, mode):
    raw = []
    for sample in _flatten(samples):
        reward, details = shaped_reward(sample, mode)
        sample.metadata.setdefault('appworld', {})['rpr_reward'] = dict(details, value=reward)
        raw.append(reward)
    # PPO sends the absolute reward to the critic and TB-GAE.  GRPO-style group
    # centering here would destroy the value target and is intentionally absent.
    return raw, raw.copy()


def fdr_rewards(args, samples, **kwargs):
    return _post_process(samples, 'fdr')


def spr_rewards(args, samples, **kwargs):
    return _post_process(samples, 'spr')


async def generate(args, sample, sampling_params, evaluation=False):
    result = await generate_appworld(args, sample, sampling_params, evaluation=evaluation)
    if result.status == getattr(result.Status, 'ABORTED', None):
        return result
    metadata = result.metadata or {}
    response = result.response_length
    for key in ('turn_end_mask', 'is_real_done', 'tb_boundary'):
        if key not in metadata:
            raise ValueError(f'AppWorld RPR rollout is missing {key}')
    if len(metadata['turn_end_mask']) != response or len(metadata['is_real_done']) != response:
        raise ValueError('TB-GAE masks are not response aligned')
    if sum(metadata['is_real_done']) != 1:
        raise ValueError('AppWorld finite-horizon episode needs exactly one terminal action')
    return result


def _trajectory_samples(samples):
    """Deduplicate TB chunks so environment metrics count complete episodes."""
    trajectories = {}
    for sample in _flatten(samples):
        appworld = sample.metadata.get('appworld', {})
        episode = appworld.get('episode_id') or sample.metadata.get('tb_boundary', {}).get('episode')
        if not episode:
            raise ValueError('RPR sample is missing episode identity')
        trajectories.setdefault(episode, sample)
    return list(trajectories.values())


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    trajectories = _trajectory_samples(samples)
    mode = config()['reward_mode']
    shaped = [shaped_reward(sample, mode)[0] for sample in trajectories]
    checks = [sample.metadata['appworld']['evaluation'] for sample in trajectories]
    progress = [row['passed_checks']/row['total_checks'] for row in checks
                if type(row.get('passed_checks')) is int and type(row.get('total_checks')) is int
                and row['total_checks'] > 0]
    summaries = [sample.metadata['appworld'] for sample in trajectories]
    metrics = {
        'success_rate': sum(row.get('success') is True for row in checks)/len(checks),
        'shaped_reward_mean': sum(shaped)/len(shaped),
        'check_fraction_mean': sum(progress)/len(progress) if progress else 0.0,
        'average_turns': sum(row['steps'] for row in summaries)/len(summaries),
        'trajectory_count': len(trajectories),
        'training_chunk_count': len(list(_flatten(samples))),
        'action_tokens': sum(sum(sample.loss_mask) for sample in _flatten(samples)),
        'rollout_seconds': rollout_time,
        'execute_budget_trigger_rate': sum(bool(row.get('execute_budget_triggered'))
                                           for row in summaries)/len(summaries),
        'timeout_recoveries': sum(row.get('timeout_recoveries', 0) for row in summaries),
    }
    for reason, count in Counter(row['termination'] for row in summaries).items():
        metrics['termination/'+reason] = count
    path = Path(config()['output_dir'])/'project_metrics'/f'{rollout_id:08d}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(step=rollout_id,
        metrics={'appworld_rpr/'+key:value for key, value in metrics.items()})))
    return False
