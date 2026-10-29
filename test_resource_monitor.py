import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from resource_monitor import cgroup_snapshot, cgroup_paths, diagnose, ResourceMonitor
from grpo_tracking import SwanLabBridge
from types import SimpleNamespace


class ResourceTests(unittest.TestCase):
    def test_cgroup_v2_memory_cpu_and_unlimited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, text in {'memory.current':'200', 'memory.max':'max', 'memory.peak':'300',
                    'memory.events':'oom 2\noom_kill 1\n', 'cpu.stat':'usage_usec 9000\nnr_throttled 3\n',
                    'cpu.max':'200000 100000'}.items():
                (root/name).write_text(text)
            row = cgroup_snapshot({'v2':root})
            self.assertIsNone(row['memory_limit_bytes'])
            self.assertEqual(row['memory_events']['oom_kill'], 1)
            self.assertEqual(row['cpu_quota_cores'], 2)

    def test_v1_and_mount_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'self').mkdir()
            (root/'self/cgroup').write_text('0::/job\n')
            mount = root/'mount'
            (mount/'job').mkdir(parents=True)
            info = root/'mountinfo'
            info.write_text(f'1 0 0:1 / {mount} rw - cgroup2 cgroup rw\n')
            self.assertEqual(cgroup_paths(root, info)['v2'], mount/'job')
            (mount/'memory.usage_in_bytes').write_text('45')
            (mount/'memory.failcnt').write_text('3')
            row = cgroup_snapshot({'memory':mount})
            self.assertEqual(row['memory_bytes'], 45)
            self.assertEqual(row['memory_events']['failcnt'], 3)

    def test_reason_evidence_precedence_and_no_false_oom_from_signal(self):
        cases = [({}, '', 137, 'external_signal_unconfirmed'),
                 ({'oom':2}, '', 1, 'application_error'),
                 ({'oom_kill':1}, '', 137, 'container_oom_kill_observed'),
                 ({}, 'torch.OutOfMemoryError: CUDA out of memory', 1, 'cuda_oom'),
                 ({}, 'AppWorld worker exceeded 120s during execute', 1, 'environment_timeout'),
                 ({}, '', None, 'incomplete_or_unobserved_exit'),
                 ({}, '', 0, 'completed')]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for delta, log, code, expected in cases:
                (root/'resource_summary.json').write_text(json.dumps(dict(timestamp=1, memory_events_delta=delta)))
                (root/'training.log').write_text(log)
                self.assertEqual(diagnose(root, code)['category'], expected)

    def test_counter_baseline_and_bridge_dedup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = dict(memory_events={'oom_kill':8})
            final = dict(memory_events={'oom_kill':9}, memory_bytes=100, memory_limit_bytes=200)
            with patch('resource_monitor.cgroup_snapshot', side_effect=[initial, final]):
                monitor = ResourceMonitor(root)
                row = monitor.sample()
            self.assertEqual(row['memory_events_delta']['oom_kill'], 1)
            logs = []
            bridge = SwanLabBridge(root, 'disabled', 'test', 'test', {})
            bridge.swanlab = SimpleNamespace(log=lambda values, **kw:logs.append(values))
            bridge.poll_resource_metrics()
            bridge.poll_resource_metrics()
            self.assertEqual(len(logs), 1)
            self.assertEqual(logs[0]['resources/cgroup_memory_percent'], 50)
            diagnose(root, 137)
            bridge.poll_resource_metrics()
            self.assertEqual(logs[-1]['exit/container_oom_kill_observed'], 1)

    def test_terminal_exception_overrides_earlier_timeout_and_swanlab_category(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root/'training.log'
            for final in ('RuntimeError: Recovery reset differs from original task',
                          'RuntimeError: Recovery replay diverged; refusing inconsistent training sample'):
                log.write_text('TimeoutError: AppWorld worker exceeded 120s during execute\n'
                               'During handling of the above exception, another exception occurred:\n'+final+'\n')
                report = diagnose(root, 1)
                self.assertEqual(report['category'], 'recovery_validation_failed')
                self.assertEqual(report['terminal_error'], final)
                logs = []
                bridge = SwanLabBridge(root, 'disabled', 'test', 'test', {})
                bridge.swanlab = SimpleNamespace(log=lambda values, **kw: logs.append(values))
                bridge.poll_resource_metrics()
                self.assertEqual(logs[-1]['exit/recovery_validation_failed'], 1)
                self.assertEqual(logs[-1]['exit/environment_timeout'], 0)

    def test_stale_timeout_does_not_hide_other_terminal_error_or_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'training.log').write_text('TimeoutError: AppWorld worker exceeded 120s during execute\n'
                                           'ValueError: later unrelated failure\n')
            self.assertEqual(diagnose(root, 1)['category'], 'application_error')
            self.assertEqual(diagnose(root, 0)['category'], 'completed')
            self.assertEqual(diagnose(root, -15)['category'], 'external_signal_unconfirmed')
            (root/'training.log').write_text('RuntimeError: earlier error\n'
                'TimeoutError: AppWorld timeout recovery budget exhausted; inspect journal\n')
            self.assertEqual(diagnose(root, 1)['category'], 'environment_timeout')
