"""Independent CPU/RAM/cgroup evidence and conservative exit diagnosis."""
import json
import os
from pathlib import Path
import threading
import time


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def read_number(path):
    try:
        raw = Path(path).read_text().strip()
        return int(raw) if raw != 'max' else None
    except (OSError, ValueError):
        return None


def read_pairs(path):
    try:
        return {k:int(v) for k,v in (line.split() for line in Path(path).read_text().splitlines())}
    except (OSError, ValueError):
        return {}


def cgroup_paths(proc=Path('/proc'), mounts=Path('/proc/self/mountinfo')):
    """Resolve mount roots, including namespaced cgroups; support v1 and v2."""
    try:
        memberships = [line.split(':', 2) for line in (proc/'self/cgroup').read_text().splitlines()]
        lines = mounts.read_text().splitlines()
    except OSError:
        return {}
    result = {}
    for line in lines:
        left, right = line.split(' - ', 1)
        fields, tail = left.split(), right.split()
        fs, root, mount = tail[0], fields[3], fields[4]
        for _, controllers, member in memberships:
            if fs == 'cgroup2' and not controllers:
                names = ['v2']
            elif fs == 'cgroup':
                names = [c for c in controllers.split(',') if c in tail[2].split(',')]
            else:
                continue
            relative = os.path.relpath(member, root)
            candidate = Path(mount)/relative
            # Namespace membership may refer to an ancestor outside this mount.
            if '..' in Path(relative).parts or not candidate.is_dir():
                candidate = Path(mount)
            for name in names:
                result[name] = candidate
    return result


def cgroup_snapshot(paths):
    memory = paths.get('v2', paths.get('memory'))
    cpu = paths.get('v2', paths.get('cpu'))
    result = {}
    if memory:
        v2 = 'v2' in paths
        names = {'memory_bytes':'memory.current' if v2 else 'memory.usage_in_bytes',
                 'memory_limit_bytes':'memory.max' if v2 else 'memory.limit_in_bytes',
                 'memory_peak_bytes':'memory.peak' if v2 else 'memory.max_usage_in_bytes'}
        for key, filename in names.items():
            value = read_number(memory/filename)
            # v1 represents unlimited with a very large integer.
            result[key] = None if key == 'memory_limit_bytes' and value and value >= 2**60 else value
        result['memory_events'] = read_pairs(memory/'memory.events') if v2 else {
            'failcnt':read_number(memory/'memory.failcnt') or 0,
            **read_pairs(memory/'memory.oom_control')}
        result['memory_scope'] = str(memory)
    if cpu:
        result['cpu_stat'] = read_pairs(cpu/'cpu.stat')
        if 'v2' not in paths and 'cpuacct' in paths:
            usage_ns = read_number(paths['cpuacct']/'cpuacct.usage')
            if usage_ns is not None:
                result['cpu_stat']['usage_usec'] = usage_ns//1000
        if 'v2' in paths:
            try:
                quota, period = (cpu/'cpu.max').read_text().split()
                result['cpu_quota_cores'] = None if quota == 'max' else int(quota)/int(period)
            except (OSError, ValueError):
                pass
    return result


class ResourceMonitor:
    def __init__(self, output, interval=5):
        self.root = Path(output)
        self.interval = interval
        self.stop = threading.Event()
        self.thread = None
        self.samples = 0
        self.paths = cgroup_paths()
        self.baseline = cgroup_snapshot(self.paths).get('memory_events', {})
        self.previous = None
        self.error = None

    def sample(self):
        import psutil
        now = time.time()
        cg = cgroup_snapshot(self.paths)
        events = cg.get('memory_events', {})
        deltas = {key:max(0, value-self.baseline.get(key, value)) for key,value in events.items()}
        vm = psutil.virtual_memory()
        processes = []
        for process in psutil.Process().children(recursive=True)+[psutil.Process()]:
            try:
                processes.append(dict(pid=process.pid, rss_bytes=process.memory_info().rss,
                    cpu_seconds=sum(process.cpu_times()[:2])))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        cpu = cg.get('cpu_stat', {})
        usage = cpu.get('usage_usec')
        cg_cpu_cores = None
        if self.previous and usage is not None and self.previous[1] is not None:
            cg_cpu_cores = max(0, usage-self.previous[1])/1e6/max(now-self.previous[0], 1e-6)
        self.previous = (now, usage)
        self.samples += 1
        row = dict(sample=self.samples, timestamp=now, launcher_pid=os.getpid(),
            host_cpu_percent=psutil.cpu_percent(), host_memory_available_bytes=vm.available,
            host_memory_percent=vm.percent, process_tree_rss_bytes=sum(p['rss_bytes'] for p in processes),
            process_count=len(processes), processes=processes, cgroup=cg,
            memory_events_delta=deltas, cgroup_cpu_used_cores=cg_cpu_cores, monitor_error=self.error)
        with (self.root/'resource_usage.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        atomic_json(self.root/'resource_summary.json', row)
        return row

    def _loop(self):
        while not self.stop.wait(self.interval):
            try:
                self.sample()
            except Exception as exc:
                self.error = repr(exc)
                atomic_json(self.root/'resource_monitor_error.json', dict(error=self.error, timestamp=time.time()))

    def start(self):
        # Sample before launching Ray so early OOM evidence has a baseline.
        self.sample()
        self.thread = threading.Thread(target=self._loop, daemon=True, name='resource-monitor')
        self.thread.start()

    def finish(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=10)
        self.sample()


def diagnose(output, returncode=None, received_signal=None, launcher_error=None):
    root = Path(output)
    try:
        resource = json.loads((root/'resource_summary.json').read_text())
    except (OSError, ValueError):
        resource = {}
    path = root/'training.log'
    text = ''
    if path.is_file():
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size-256*1024))
            text = stream.read().decode(errors='replace')
    lower = text.lower()
    delta = resource.get('memory_events_delta', {})
    evidence = []
    if delta.get('oom_kill', 0) > 0 or delta.get('oom_group_kill', 0) > 0:
        category = 'container_oom_kill_observed'
        evidence.append('Cgroup OOM kill counter increased since monitor start; victim identity is not proven.')
    elif 'cuda out of memory' in lower or 'torch.outofmemoryerror' in lower:
        category = 'cuda_oom'
        evidence.append('CUDA OOM found in trailing training log.')
    elif 'appworld' in lower and ('timeout recovery budget exhausted' in lower or 'appworld worker exceeded' in lower):
        category = 'environment_timeout'
        evidence.append('AppWorld timeout found in trailing training log.')
    elif 'outofmemoryerror' in lower or 'memory pressure' in lower or 'node was running low on memory' in lower:
        category = 'memory_pressure_reported'
        evidence.append('Memory pressure/Ray OOM message found; does not prove kernel OOM kill.')
    elif received_signal or returncode in (-9, -15, 137, 143):
        category = 'external_signal_unconfirmed'
        evidence.append('Termination signal observed; platform cancellation, manual kill and OOM require external evidence to distinguish.')
    elif returncode == 0 and not launcher_error:
        category = 'completed'
    elif returncode is None:
        category = 'incomplete_or_unobserved_exit'
        evidence.append('No observed child exit code; last heartbeat alone cannot prove platform termination.')
    else:
        category = 'application_error'
        evidence.append('Nonzero exit; inspect training log and episode journals.')
    report = dict(category=category, returncode=returncode, received_signal=received_signal,
        launcher_error=launcher_error, evidence=evidence, resource=resource,
        last_heartbeat_age_seconds=time.time()-resource['timestamp'] if resource else None,
        generated_at=time.time(), log_tail=text[-12000:],
        limitation='SIGKILL/whole-container termination prevents final diagnosis/upload. Use persisted heartbeat and platform exit events; no inference of platform cause without evidence.')
    atomic_json(root/'exit_diagnosis.json', report)
    return report
