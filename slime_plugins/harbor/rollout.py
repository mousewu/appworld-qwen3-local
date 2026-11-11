"""Bounded AppWorld scheduler at Terminal-RSI's TB-GAE validated path.

Terminal-RSI deliberately restricts TB-GAE to a reviewed rollout entry point.
This project preserves that path while replacing only its internal scheduling:
complete prompt groups are continuously supplied, aborted groups have bounded
retries, and isolated exhausted execute timeouts are replaced by fresh groups.
The per-sample interaction remains the configured ``appworld_rpr.generate``.
"""
import asyncio

from appworld_async_rollout import GroupQueue
from appworld_grpo import config


_queue = None


async def _collect(args, rollout_id, data_source):
    from slime.rollout.base_types import RolloutFnTrainOutput
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.misc import load_function

    global _queue
    if _queue is None:
        cfg = config()
        generate_sample = load_function(args.custom_generate_function_path)

        async def generate_group(group):
            params = GenerateState(args).sampling_params
            async with asyncio.TaskGroup() as tasks:
                results = [tasks.create_task(generate_sample(args, sample, params.copy()))
                           for sample in group]
            return [result.result() for result in results]

        _queue = GroupQueue(
            data_source,
            generate_group,
            capacity=cfg['async_group_capacity'],
            concurrency=cfg['async_inflight_groups'],
            max_lag=cfg['async_max_policy_lag'],
            max_retries=cfg['async_group_retries'],
            max_consecutive_timeouts=cfg['async_max_consecutive_timeouts'],
            group_size=args.n_samples_per_prompt,
        )
    try:
        final = rollout_id == args.num_rollout - 1
        groups, metrics = await _queue.collect(rollout_id, args.rollout_batch_size, final=final)
        if args.rollout_all_samples_process_path is not None:
            from slime.utils.misc import load_function

            process = load_function(args.rollout_all_samples_process_path)
            process(args, groups, data_source)
        if final:
            _queue = None
        return RolloutFnTrainOutput(samples=groups, metrics=metrics)
    except BaseException:
        await _queue.close()
        _queue = None
        raise


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    if evaluation:
        from slime.rollout.sglang_rollout import generate_rollout as native_generate_rollout

        return native_generate_rollout(args, rollout_id, data_source, evaluation=True)
    from slime.utils.async_utils import run

    return run(_collect(args, rollout_id, data_source))
