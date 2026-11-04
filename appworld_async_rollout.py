"""Bounded, persistent group producer for Slime's project rollout hook."""
import asyncio
from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass

from appworld_grpo import config, generate


@dataclass
class Work:
    original: list
    stale_retries: int = 0
    abort_retries: int = 0


class GroupQueue:
    def __init__(self, source, generate_group, *, capacity, concurrency, max_lag, max_retries):
        self.source = source
        self.generate_group = generate_group
        self.capacity = capacity
        self.concurrency = concurrency
        self.max_lag = max_lag
        self.max_retries = max_retries
        self.pending = deque()
        self.deferred = deque()
        self.active = set()
        self.ready = deque()
        self.changed = asyncio.Event()
        self.epoch = None
        self.closed = False
        self.collecting = False
        self.protected = None
        self.error = None
        self.counts = Counter()

    @property
    def size(self):
        return len(self.pending) + len(self.deferred) + len(self.active) + len(self.ready)

    def _retry_abort(self, work):
        if work.abort_retries >= self.max_retries:
            raise RuntimeError(f'AppWorld async group {work.original[0].group_index} '
                               'exhausted retries after aborted_resampled')
        work.abort_retries += 1
        self.counts['aborted_resampled'] += 1
        # Background aborts can be caused by a weight publication in progress.
        queue = self.pending if self.collecting else self.deferred
        queue.appendleft(work)

    def _fill_slots(self):
        while not self.closed and not self.error and self.pending and len(self.active) < self.concurrency:
            work = self.pending.popleft()
            # Epoch is a conservative lower bound on the policy version of every
            # action, including actions started during the next weight update.
            epoch = self.epoch
            task = asyncio.create_task(self.generate_group(deepcopy(work.original)))
            self.active.add(task)
            task.add_done_callback(lambda done, work=work, epoch=epoch: self._complete(done, work, epoch))

    def _complete(self, task, work, epoch):
        self.active.remove(task)
        if self.closed:
            return
        try:
            group = task.result()
            if len(group) != 4:
                raise ValueError('AppWorld async requires exactly four trajectories per group')
            if any(s.status == s.Status.ABORTED for s in group):
                self._retry_abort(work)
            else:
                if any(s.status != s.Status.COMPLETED or s.reward not in (0.0, 1.0) for s in group):
                    raise ValueError('AppWorld async received an incomplete or unscored trajectory')
                enqueue = self.ready.appendleft if work is self.protected else self.ready.append
                enqueue((work, epoch, group))
        except (Exception, asyncio.CancelledError) as exc:
            self.error = exc
        self._fill_slots()
        self.changed.set()

    async def collect(self, epoch, target, *, final=False):
        if self.closed or not 0 < target <= self.capacity:
            raise ValueError('Async group queue is closed or smaller than the rollout batch')
        if self.epoch is not None and epoch != self.epoch + 1:
            raise ValueError('Project async requires consecutive rollout IDs and one weight sync per update')
        self.epoch = epoch
        self.collecting = True
        self.pending.extendleft(reversed(self.deferred))
        self.deferred.clear()
        groups, ages = [], []
        while True:
            self.changed.clear()
            if self.error is not None:
                raise self.error
            while self.ready and len(groups) < target:
                work, started, group = self.ready[0]
                if self.protected is not None and work is not self.protected and len(groups) == target-1:
                    break  # Keep the last seat open for the protected group's fresh attempt.
                age = epoch - started
                if age > self.max_lag and work.stale_retries >= self.max_retries and self.protected is not None:
                    break  # Only one protected attempt at a time.
                self.ready.popleft()
                if age > self.max_lag:
                    if work.stale_retries >= self.max_retries:
                        self.protected = work
                        self.counts['stale_protected'] += 1
                    else:
                        work.stale_retries += 1
                        self.counts['stale_resampled'] += 1
                    if self.protected is not None and work is not self.protected:
                        self.pending.append(work)
                    else:
                        self.pending.appendleft(work)
                else:
                    groups.append(group)
                    ages.append(age)
                    if work is self.protected:
                        self.protected = None
            remaining = target - len(groups)
            # Only the consumer advances Slime's data source. During actor
            # training callbacks use already reserved prompts, so a concurrent
            # Slime checkpoint never races a data-source read.
            desired = remaining if final else self.capacity
            while self.size < desired:
                originals = self.source.get_samples(1)
                if len(originals) != 1 or len(originals[0]) != 4:
                    raise ValueError('Slime data source must provide one complete four-sample prompt group')
                self.pending.append(Work(originals[0]))
            if not remaining:
                break
            self._fill_slots()
            await self.changed.wait()

        metrics = {
            'async/policy_lag_max': max(ages),
            'async/policy_lag_mean': sum(ages) / len(ages),
            'async/ready_groups': len(self.ready),
            'async/pending_groups': len(self.pending) + len(self.deferred),
            'async/active_groups': len(self.active),
            'async/stale_protected_total': self.counts['stale_protected'],
            'async/stale_resampled_total': self.counts['stale_resampled'],
            'async/aborted_resampled_total': self.counts['aborted_resampled'],
        }
        self.collecting = False
        if final:
            await self.close()
        else:
            self._fill_slots()
        return groups, metrics

    async def close(self):
        self.closed = True
        active = list(self.active)
        for task in active:
            task.cancel()
        await asyncio.gather(*active, return_exceptions=True)
        self.ready.clear()
        self.pending.clear()
        self.deferred.clear()


_queue = None


async def _collect(args, rollout_id, data_source):
    from slime.rollout.base_types import RolloutFnTrainOutput
    from slime.rollout.sglang_rollout import GenerateState

    global _queue
    if _queue is None:
        cfg = config()

        async def generate_group(group):
            params = GenerateState(args).sampling_params
            async with asyncio.TaskGroup() as tasks:
                results = [tasks.create_task(generate(args, sample, params.copy())) for sample in group]
            return [result.result() for result in results]

        _queue = GroupQueue(data_source, generate_group, capacity=cfg['async_group_capacity'],
                            concurrency=cfg['async_inflight_groups'], max_lag=cfg['async_max_policy_lag'],
                            max_retries=cfg['async_group_retries'])
    try:
        final = rollout_id == args.num_rollout - 1
        groups, metrics = await _queue.collect(rollout_id, args.rollout_batch_size, final=final)
        if final:
            _queue = None
        return RolloutFnTrainOutput(samples=groups, metrics=metrics)
    except BaseException:
        await _queue.close()
        _queue = None
        raise


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    """Entry point for native train.py and its persistent asyncio runner."""
    from slime.utils.async_utils import run

    if evaluation:
        raise ValueError('Project async rollout hook is training-only')
    return run(_collect(args, rollout_id, data_source))
