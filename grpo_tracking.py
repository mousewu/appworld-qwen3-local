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
        for path in sorted((self.root/'project_metrics').glob('*.json')):
            if path in self.seen_rollouts:
                continue
            try:
                row = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue  # Writer may not have closed the current file yet.
            self.swanlab.log(row['metrics'], step=int(row['step']))
            self.seen_rollouts.add(path)
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
                 'command.json', 'train_tasks.jsonl', 'run_status.json']]
        paths += list((self.root/'episodes').glob('*.json'))
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
