"""Measure allocated GPUs and host/cgroup memory; report measured target attainment."""
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def select_gpus(rows, expected, visible=None):
    visible = visible if visible is not None else os.environ.get('GPU_MONITOR_DEVICES',os.environ.get('CUDA_VISIBLE_DEVICES',''))
    if visible and visible not in ('all','-1'):
        requested = [x.strip() for x in visible.split(',')]
        rows = [row for row in rows if str(row['index']) in requested or row['uuid'] in requested]
    if len(rows) != expected:
        raise ValueError(f'Cannot identify exactly {expected} allocated GPUs, found {len(rows)}; set GPU_MONITOR_DEVICES to physical indices or GPU UUIDs')
    return rows


class GPUMonitor:
    def __init__(self, output, count, interval=5, warmup=120, target=70):
        self.root=Path(output);self.count=count;self.interval=interval
        self.warmup=warmup;self.target=target;self.stop=threading.Event()
        self.thread=None;self.started=None;self.previous=None;self.integrals={};self.duration=0
        self.steady_integrals={};self.steady_duration=0;self.samples=0;self.error=None

    def start(self):
        self.started=time.monotonic()
        self.thread=threading.Thread(target=self._loop,daemon=True,name='gpu-util-monitor')
        self.thread.start()

    def _read(self):
        text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'],text=True,timeout=10)
        rows=[]
        for line in text.splitlines():
            index,uuid,util,used,total=[x.strip() for x in line.split(',')]
            rows.append(dict(index=int(index),uuid=uuid,utilization=float(util),memory_used_mb=float(used),memory_total_mb=float(total)))
        return select_gpus(rows,self.count)

    def _sample(self):
        rows=self._read();elapsed=time.monotonic()-self.started
        if self.previous:
            last_time,last_rows=self.previous
            dt=elapsed-last_time
            steady=max(0,elapsed-max(last_time,self.warmup))
            if {r['uuid'] for r in rows}!={r['uuid'] for r in last_rows}:
                raise ValueError('GPU device set changed during monitoring')
            self.duration+=dt;self.steady_duration+=steady
            for row in last_rows:
                key=row['uuid']
                self.integrals[key]=self.integrals.get(key,0)+dt*row['utilization']
                self.steady_integrals[key]=self.steady_integrals.get(key,0)+steady*row['utilization']
        self.previous=(elapsed,rows);self.samples+=1
        memory={}
        try:
            import psutil
            vm=psutil.virtual_memory()
            memory=dict(host_used_gb=vm.used/2**30,host_total_gb=vm.total/2**30,
                host_percent=vm.percent,
                appworld_processes=sum('world_worker.py' in ' '.join(p.info.get('cmdline') or [])
                    for p in psutil.process_iter(['cmdline'])))
        except Exception:
            pass
        for key in ['memory.current','memory.max','memory.peak']:
            path=Path('/sys/fs/cgroup')/key
            if path.is_file():
                raw=path.read_text().strip()
                if raw.isdigit(): memory['cgroup_'+key.split('.')[-1]+'_gb']=int(raw)/2**30
        record=dict(sample=self.samples,elapsed_seconds=elapsed,gpus=rows,memory=memory)
        with (self.root/'gpu_utilization.jsonl').open('a') as out:
            out.write(json.dumps(record)+'\n')
        self._report()

    def _report(self):
        full={k:v/self.duration for k,v in self.integrals.items()} if self.duration else {}
        steady={k:v/self.steady_duration for k,v in self.steady_integrals.items()} if self.steady_duration else {}
        average=sum(steady.values())/len(steady) if steady else None
        full_average=sum(full.values())/len(full) if full else None
        report=dict(samples=self.samples,interval_seconds=self.interval,warmup_seconds=self.warmup,
            monitored_seconds=self.duration,steady_seconds=self.steady_duration,
            full_run_per_gpu_mean=full,steady_per_gpu_mean=steady,
            steady_all_gpu_mean=average,full_all_gpu_mean=full_average,target_percent=self.target,
            target_met=full_average>=self.target if full_average is not None and self.error is None else None,
            steady_target_met=average>=self.target if average is not None and self.error is None else None,
            latest_gpus=self.previous[1] if self.previous else [],
            error=self.error,definition='Time-weighted sampled nvidia-smi utilization averaged equally over allocated GPUs; excludes warmup only for steady metrics')
        temporary=self.root/'gpu_summary.tmp'
        temporary.write_text(json.dumps(report,indent=2))
        temporary.replace(self.root/'gpu_summary.json')

    def _loop(self):
        try:
            while not self.stop.is_set():
                self._sample()
                self.stop.wait(self.interval)
        except Exception as exc:
            self.error=str(exc);self._report()
            print(f'GPU monitoring unavailable: {exc}',flush=True)

    def finish(self):
        if self.thread:
            self.stop.set();self.thread.join(timeout=15)
            if not self.thread.is_alive() and self.error is None:
                try: self._sample()
                except Exception as exc: self.error=str(exc)
            self._report()
