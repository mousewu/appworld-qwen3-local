import asyncio
from collections import Counter
from types import SimpleNamespace
import unittest

from appworld_async_rollout import GroupQueue
from appworld_grpo import ExecuteRecoveryExhaustedError


STATUS = SimpleNamespace(COMPLETED='complete', ABORTED='aborted')


class Source:
    def __init__(self):
        self.next_id = 0

    def get_samples(self, count):
        assert count == 1
        gid = self.next_id
        self.next_id += 1
        return [[SimpleNamespace(group_index=gid, index=gid*4+i, status=None,
                                 reward=None, Status=STATUS, metadata={'task_id': str(gid)}) for i in range(4)]]


async def complete(group):
    for sample in group:
        sample.status = STATUS.COMPLETED
        sample.reward = 1.0
    return group


def ids(groups):
    return [group[0].group_index for group in groups]


async def wait_for_ready(queue, count):
    async with asyncio.timeout(1):
        while len(queue.ready) < count:
            await asyncio.sleep(0)


class GroupQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_surplus_retained_and_prefetch_bounded_while_actor_busy(self):
        source = Source()
        queue = GroupQueue(source, complete, capacity=4, concurrency=2, max_lag=2, max_retries=2)
        self.addAsyncCleanup(queue.close)
        first, _ = await asyncio.wait_for(queue.collect(10, 2), 1)
        await wait_for_ready(queue, 4)
        self.assertEqual(queue.size, 4)
        self.assertEqual(source.next_id, 6)
        self.assertEqual(ids(first), [0, 1])
        second, metrics = await asyncio.wait_for(queue.collect(11, 2), 1)
        self.assertEqual(ids(second), [2, 3])
        self.assertEqual(metrics['async/policy_lag_max'], 1)
        third, _ = await asyncio.wait_for(queue.collect(12, 2, final=True), 1)
        self.assertEqual(ids(third), [4, 5])
        self.assertEqual(queue.size, 0)

    async def test_ready_groups_bypass_a_slow_inflight_group(self):
        release = asyncio.Event()
        async def generate(group):
            if ids([group]) == [0]:
                await release.wait()
            return await complete(group)
        queue = GroupQueue(Source(), generate, capacity=4, concurrency=2, max_lag=2, max_retries=2)
        self.addAsyncCleanup(queue.close)
        first, _ = await asyncio.wait_for(queue.collect(0, 2), 1)
        self.assertEqual(ids(first), [1, 2])
        self.assertTrue(queue.active)
        release.set()

    async def test_stale_group_regenerated_from_pristine_original(self):
        starts = Counter()
        async def generate(group):
            self.assertTrue(all(s.reward is None and s.status is None for s in group))
            starts[group[0].group_index] += 1
            return await complete(group)
        queue = GroupQueue(Source(), generate, capacity=2, concurrency=2, max_lag=0, max_retries=2)
        self.addAsyncCleanup(queue.close)
        await asyncio.wait_for(queue.collect(0, 1), 1)
        await wait_for_ready(queue, 2)
        second, metrics = await asyncio.wait_for(queue.collect(1, 1, final=True), 1)
        self.assertEqual(starts[ids(second)[0]], 2)
        self.assertEqual(metrics['async/policy_lag_max'], 0)
        self.assertEqual(metrics['async/stale_resampled_total'], 2)

    async def test_abort_retries_whole_group_and_infrastructure_error_propagates(self):
        attempts = 0
        async def generate(group):
            nonlocal attempts
            attempts += 1
            self.assertTrue(all(s.reward is None for s in group))
            await complete(group)
            if attempts == 1:
                group[0].status = STATUS.ABORTED
            elif attempts == 3:
                raise TimeoutError('worker unavailable')
            return group
        queue = GroupQueue(Source(), generate, capacity=1, concurrency=1, max_lag=1, max_retries=1)
        self.addAsyncCleanup(queue.close)
        first, metrics = await asyncio.wait_for(queue.collect(0, 1), 1)
        self.assertEqual(ids(first), [0])
        self.assertEqual(metrics['async/aborted_resampled_total'], 1)
        with self.assertRaisesRegex(TimeoutError, 'worker unavailable'):
            await asyncio.wait_for(queue.collect(1, 1), 1)

    async def test_background_abort_waits_for_next_weight_epoch(self):
        release = asyncio.Event()
        aborted = asyncio.Event()
        attempts = Counter()

        async def generate(group):
            gid = group[0].group_index
            attempts[gid] += 1
            await complete(group)
            if gid == 1 and attempts[gid] == 1:
                await release.wait()
                group[0].status = STATUS.ABORTED
                aborted.set()
            return group

        queue = GroupQueue(Source(), generate, capacity=1, concurrency=1, max_lag=1, max_retries=1)
        self.addAsyncCleanup(queue.close)
        await asyncio.wait_for(queue.collect(0, 1), 1)
        release.set()
        await asyncio.wait_for(aborted.wait(), 1)
        await asyncio.sleep(0)
        self.assertEqual(attempts[1], 1)
        self.assertEqual(queue.size, 1)
        groups, metrics = await asyncio.wait_for(queue.collect(1, 1, final=True), 1)
        self.assertEqual(ids(groups), [1])
        self.assertEqual(attempts[1], 2)
        self.assertEqual(metrics['async/policy_lag_max'], 0)

    async def test_exhausted_stale_group_reserves_slot_without_losing_other_groups(self):
        release = asyncio.Event()
        started = asyncio.Event()
        attempts = Counter()

        async def generate(group):
            gid = ids([group])[0]
            attempts[gid] += 1
            if gid == 1 and attempts[gid] == 2:
                started.set()
                await release.wait()
            return await complete(group)

        queue = GroupQueue(Source(), generate, capacity=3, concurrency=3, max_lag=0, max_retries=0)
        self.addAsyncCleanup(queue.close)
        await asyncio.wait_for(queue.collect(0, 1), 1)
        await wait_for_ready(queue, 3)
        collecting = asyncio.create_task(queue.collect(1, 1))
        await asyncio.wait_for(started.wait(), 1)
        self.assertFalse(collecting.done())
        release.set()
        groups, metrics = await asyncio.wait_for(collecting, 1)
        self.assertEqual(ids(groups), [1])
        self.assertEqual(metrics['async/stale_protected_total'], 1)
        self.assertEqual(metrics['async/policy_lag_max'], 0)
        self.assertEqual(queue.size, 3)

    async def test_abort_exhaustion_replaces_entire_group(self):
        async def generate(group):
            await complete(group)
            if ids([group]) == [0]:
                group[0].status = STATUS.ABORTED
            return group

        queue = GroupQueue(Source(), generate, capacity=1, concurrency=1, max_lag=2, max_retries=1)
        self.addAsyncCleanup(queue.close)
        groups, metrics = await asyncio.wait_for(queue.collect(0, 1, final=True), 1)
        self.assertEqual(ids(groups), [1])
        self.assertEqual(len(groups[0]), 4)
        self.assertEqual(metrics['async/aborted_resampled_total'], 1)
        self.assertEqual(metrics['async/aborted_skipped_total'], 1)

    async def test_systemic_abort_does_not_spin_forever(self):
        async def generate(group):
            group[0].status = STATUS.ABORTED
            return group

        queue = GroupQueue(Source(), generate, capacity=1, concurrency=1, max_lag=2, max_retries=0)
        self.addAsyncCleanup(queue.close)
        with self.assertRaisesRegex(RuntimeError, 'without accepting a complete group'):
            await asyncio.wait_for(queue.collect(0, 1), 1)

    async def test_exhausted_execute_timeout_replaces_whole_group(self):
        async def generate(group):
            if ids([group]) == [0]:
                raise ExceptionGroup('TaskGroup', [ExecuteRecoveryExhaustedError('execute')])
            return await complete(group)

        queue = GroupQueue(Source(), generate, capacity=1, concurrency=1, max_lag=1, max_retries=1)
        self.addAsyncCleanup(queue.close)
        groups, metrics = await asyncio.wait_for(queue.collect(0, 1, final=True), 1)
        self.assertEqual(ids(groups), [1])
        self.assertEqual(len(groups[0]), 4)
        self.assertEqual(metrics['async/execute_timeout_skipped_total'], 1)
        self.assertEqual(metrics['async/execute_timeout_skipped_rate'], .5)
        self.assertEqual(metrics['async/aborted_skipped_total'], 0)

    async def test_consecutive_execute_timeout_groups_stop_collection(self):
        async def generate(group):
            raise ExecuteRecoveryExhaustedError('execute')

        source = Source()
        queue = GroupQueue(source, generate, capacity=1, concurrency=1, max_lag=1, max_retries=1,
                           max_consecutive_timeouts=2)
        self.addAsyncCleanup(queue.close)
        with self.assertRaisesRegex(ExecuteRecoveryExhaustedError, '2 consecutive task groups'):
            await asyncio.wait_for(queue.collect(0, 1), 1)
        self.assertEqual(source.next_id, 2)

    async def test_mixed_exception_group_propagates(self):
        async def generate(group):
            raise ExceptionGroup('mixed', [ExecuteRecoveryExhaustedError('execute'), ValueError('invalid')])

        queue = GroupQueue(Source(), generate, capacity=1, concurrency=1, max_lag=1, max_retries=1)
        self.addAsyncCleanup(queue.close)
        with self.assertRaisesRegex(ExceptionGroup, 'mixed'):
            await asyncio.wait_for(queue.collect(0, 1), 1)
        self.assertEqual(queue.counts['execute_timeout_skipped'], 0)

    async def test_single_trajectory_groups_support_rpr_timeout_replacement(self):
        class SingleSource(Source):
            def get_samples(self, count):
                groups = super().get_samples(count)
                return [[group[0]] for group in groups]

        async def generate(group):
            if ids([group]) == [0]:
                raise ExecuteRecoveryExhaustedError('execute')
            return await complete(group)

        queue = GroupQueue(SingleSource(), generate, capacity=2, concurrency=2,
                           max_lag=1, max_retries=1, group_size=1)
        self.addAsyncCleanup(queue.close)
        groups, metrics = await asyncio.wait_for(queue.collect(0, 1, final=True), 1)
        self.assertEqual(ids(groups), [1])
        self.assertEqual(len(groups[0]), 1)
        self.assertEqual(metrics['async/execute_timeout_skipped_total'], 1)

    async def test_single_trajectory_groups_bound_rpr_abort_retries(self):
        class SingleSource(Source):
            def get_samples(self, count):
                groups = super().get_samples(count)
                return [[group[0]] for group in groups]

        attempts = Counter()

        async def generate(group):
            gid = ids([group])[0]
            attempts[gid] += 1
            await complete(group)
            if gid == 0:
                group[0].status = STATUS.ABORTED
            return group

        queue = GroupQueue(SingleSource(), generate, capacity=1, concurrency=1,
                           max_lag=1, max_retries=1, group_size=1)
        self.addAsyncCleanup(queue.close)
        groups, metrics = await asyncio.wait_for(queue.collect(0, 1, final=True), 1)
        self.assertEqual(ids(groups), [1])
        self.assertEqual(attempts[0], 2)
        self.assertEqual(metrics['async/aborted_skipped_total'], 1)
