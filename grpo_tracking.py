"""Project-side SwanLab bridge; Slime only writes its native TensorBoard events."""
import json
import time
import sys
from pathlib import Path


class SwanLabBridge:
    def __init__(self, output_dir, mode, project, name, config):
        self.root = Path(output_dir)
        self.seen_events, self.seen_rollouts = set(), set()
        self.accumulators = {}
        self.uploaded_files = {}
        self.last_file_sync = 0.0
        self.file_sync_interval = config.get("log_upload_interval", 60)
        self.warned_no_save = False
        self.last_gpu_sample = None
        self.swanlab = None
        if mode != 'disabled':
            import swanlab
            # Import before launching training; report a missing bridge dependency early.
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            self.event_class = EventAccumulator
            self.swanlab = swanlab
            swanlab.init(project=project, experiment_name=name, mode=mode,
                         logdir=str(self.root/'swanlab'), config=config)

    def poll(self):
        if self.swanlab is None:
            return
        self.poll_resource_metrics()
        self.poll_episode_events()
        for path in sorted((self.root/'project_metrics').glob('*.json')):
            if path in self.seen_rollouts:
                continue
            try:
                row = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue  # Writer may not have closed the current file yet.
            self.swanlab.log(row['metrics'], step=int(row['step']))
            self.seen_rollouts.add(path)
        gpu_path = self.root/'gpu_summary.json'
        if gpu_path.is_file():
            report = json.loads(gpu_path.read_text())
            if report['samples'] != getattr(self, 'last_gpu_sample', None):
                gpu_metrics = {'hardware/gpu_mean_steady':report['steady_all_gpu_mean'],
                               'hardware/monitor_seconds':report['monitored_seconds']}
                if report['full_run_per_gpu_mean']:
                    gpu_metrics['hardware/gpu_mean_full'] = sum(report['full_run_per_gpu_mean'].values())/len(report['full_run_per_gpu_mean'])
                for row in report.get('latest_gpus', []):
                    gpu_metrics[f'hardware/gpu_{row["index"]}/utilization'] = row['utilization']
                    gpu_metrics[f'hardware/gpu_{row["index"]}/memory_used_mb'] = row['memory_used_mb']
                gpu_metrics = {k:v for k,v in gpu_metrics.items() if v is not None}
                self.swanlab.log(gpu_metrics, step=report['samples'])
                self.last_gpu_sample = report['samples']
        events = []
        for path in (self.root/'tensorboard').rglob('events.out.tfevents.*'):
            if str(path) not in self.accumulators:
                self.accumulators[str(path)] = self.event_class(str(path), size_guidance={'scalars':0})
            acc = self.accumulators[str(path)]
            acc.Reload()
            for tag in acc.Tags().get('scalars', []):
                for event in acc.Scalars(tag):
                    key = (str(path), tag, event.step, event.wall_time)
                    if key not in self.seen_events:
                        events.append((event.step, event.wall_time, tag, event.value, key))
        for step, _, tag, value, key in sorted(events):
            self.swanlab.log({'slime/'+tag:value}, step=int(step))
            self.seen_events.add(key)

    def poll_resource_metrics(self):
        path = self.root/'resource_summary.json'
        if path.is_file():
            row = json.loads(path.read_text())
            if row['sample'] != getattr(self, 'last_resource_sample', None):
                cg = row.get('cgroup', {})
                metrics = {'resources/host_cpu_percent':row['host_cpu_percent'],
                    'resources/host_memory_percent':row['host_memory_percent'],
                    'resources/host_available_gb':row['host_memory_available_bytes']/2**30,
                    'resources/process_tree_rss_gb':row['process_tree_rss_bytes']/2**30,
                    'resources/process_count':row['process_count'],
                    'resources/cgroup_cpu_used_cores':row.get('cgroup_cpu_used_cores')}
                for key in ('memory_bytes', 'memory_limit_bytes', 'memory_peak_bytes'):
                    if cg.get(key) is not None:
                        metrics['resources/cgroup_'+key.replace('_bytes', '_gb')] = cg[key]/2**30
                if cg.get('memory_limit_bytes'):
                    metrics['resources/cgroup_memory_percent'] = 100*cg['memory_bytes']/cg['memory_limit_bytes']
                for key, value in row.get('memory_events_delta', {}).items():
                    metrics['resources/memory_events_delta/'+key] = value
                for key, value in cg.get('cpu_stat', {}).items():
                    if 'throttl' in key:
                        metrics['resources/cpu_stat/'+key] = value
                self.swanlab.log({key:value for key,value in metrics.items() if value is not None}, step=row['sample'])
                self.last_resource_sample = row['sample']
        path = self.root/'exit_diagnosis.json'
        if path.is_file() and not getattr(self, 'diagnosis_logged', False):
            report = json.loads(path.read_text())
            categories = ['container_oom_kill_observed', 'cuda_oom', 'environment_timeout',
                'memory_pressure_reported', 'recovery_validation_failed', 'external_signal_unconfirmed', 'completed',
                'incomplete_or_unobserved_exit', 'application_error']
            self.swanlab.log({'exit/'+category:int(report['category'] == category)
                             for category in categories}, step=0)
            self.diagnosis_logged = True

    def poll_episode_events(self):
        # Separate event axis: includes failed/in-flight episodes, not only scored batches.
        if not hasattr(self, 'journal_offsets'):
            self.journal_offsets, self.journal_counts, self.journal_step = {}, {}, 0
        changed = False
        for path in (self.root/'episodes').glob('*.jsonl'):
            with path.open() as stream:
                stream.seek(self.journal_offsets.get(str(path), 0))
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line or not line.endswith('\n'):
                        stream.seek(offset)
                        break
                    event = json.loads(line)['event']
                    if event in ('timeout', 'recovery_ready', 'recovery_failed',
                                 'episode_error', 'episode_complete'):
                        self.journal_counts[event] = self.journal_counts.get(event, 0)+1
                        changed = True
                self.journal_offsets[str(path)] = stream.tell()
        if changed:
            self.swanlab.log({'environment_events/'+key: self.journal_counts.get(key, 0)
                for key in ('timeout', 'recovery_ready', 'recovery_failed',
                            'episode_error', 'episode_complete')}, step=self.journal_step)
            self.journal_step += 1

    def sync_files(self, force=False):
        if self.swanlab is None:
            return
        now = time.monotonic()
        if not force and now-self.last_file_sync < self.file_sync_interval:
            return
        save = getattr(self.swanlab, 'save', None)
        if not callable(save):
            if not self.warned_no_save:
                print('Installed SwanLab SDK has no save() API; full files remain local.', file=sys.stderr)
                self.warned_no_save = True
            return
        # Explicit allowlist: never upload model weights, arbitrary files, or credentials.
        paths = [self.root/name for name in ['training.log', 'project_config.json',
                 'command.json', 'train_tasks.jsonl', 'run_status.json',
                 'gpu_utilization.jsonl', 'gpu_summary.json', 'resource_usage.jsonl',
                 'resource_summary.json', 'resource_monitor_error.json', 'exit_diagnosis.json']]
        paths += list((self.root/'episodes').glob('*.json'))
        paths += list((self.root/'episodes').glob('*.jsonl'))
        paths += list((self.root/'project_metrics').glob('*.json'))
        for path in paths:
            if not path.is_file():
                continue
            stat = path.stat()
            stamp = (stat.st_size, stat.st_mtime_ns)
            if self.uploaded_files.get(str(path)) == stamp:
                continue
            save(str(path), base_path=str(self.root), policy='now')
            self.uploaded_files[str(path)] = stamp
        self.last_file_sync = now

    def finish(self):
        if self.swanlab is not None:
            try:
                self.poll()
                self.sync_files(force=True)
            finally:
                self.swanlab.finish()
