"""Launch synchronous four-trajectory AppWorld GRPO using installed Slime."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from grpo_tracking import SwanLabBridge
from process_logging import ProcessLogRelay
from run_agent import WorldProcess
from slime_runtime import find_training_script

HERE = Path(__file__).resolve().parent


def training_args(a, prompt_file, output, model_config):
    batch = a.tasks_per_batch*4
    if a.num_gpus % a.tp_size or batch % (a.num_gpus//a.tp_size):
        raise ValueError('GPU count must be divisible by TP and global batch by DP')
    rope = model_config.get('rope_theta', model_config.get('rope_parameters', {}).get('rope_theta'))
    if model_config.get('model_type') != 'qwen3' or model_config.get('hidden_size') != 2560:
        raise ValueError('This launcher supports Qwen3-4B checkpoints')
    args = ['--actor-num-nodes','1','--actor-num-gpus-per-node',str(a.num_gpus),
        '--rollout-num-gpus',str(a.num_gpus),'--colocate',
        '--rollout-num-gpus-per-engine',str(a.rollout_tp),
        '--hf-checkpoint',str(a.model),'--load',str(a.init_checkpoint or a.model),
        '--ref-load',str(a.ref_checkpoint or a.model),
        '--save',str(output/'checkpoints'),'--save-interval',str(a.save_interval),
        '--prompt-data',str(prompt_file),'--input-key','prompt','--metadata-key','metadata',
        '--rollout-shuffle','--num-rollout',str(a.updates),
        '--rollout-batch-size',str(a.tasks_per_batch),'--n-samples-per-prompt','4',
        '--global-batch-size',str(batch),'--num-steps-per-rollout','1','--rollout-max-response-len',str(a.context_length),
        '--rollout-temperature',str(a.temperature),'--rollout-top-p','1',
        '--custom-generate-function-path','appworld_grpo.generate',
        '--custom-rm-path','appworld_grpo.reward',
        '--custom-rollout-log-function-path','appworld_grpo.log_rollout_data',
        '--advantage-estimator','grpo','--pg-loss-type','ppo',
        '--eps-clip','0.2','--eps-clip-high','0.2',
        '--use-kl-loss','--kl-loss-coef',str(a.kl_coef),'--kl-loss-type','low_var_kl',
        '--entropy-coef','0','--tensor-model-parallel-size',str(a.tp_size),
        '--pipeline-model-parallel-size','1','--context-parallel-size','1',
        '--expert-model-parallel-size','1','--expert-tensor-parallel-size','1',
        '--micro-batch-size','1','--use-distributed-optimizer','--bf16',
        '--recompute-granularity','full','--recompute-method','uniform','--recompute-num-layers','1',
        '--optimizer','adam','--lr',str(a.learning_rate),'--lr-decay-style','constant',
        '--weight-decay','0.01','--adam-beta1','0.9','--adam-beta2','0.95',
        '--attention-dropout','0','--hidden-dropout','0','--attention-backend','flash',
        '--accumulate-allreduce-grads-in-fp32','--attention-softmax-in-fp32',
        '--rotary-base',str(rope),'--seq-length',str(a.context_length),
        '--max-position-embeddings',str(model_config['max_position_embeddings']),
        '--sglang-mem-fraction-static',str(a.sglang_memory),
        '--use-tensorboard','--tb-project-name','appworld_grpo','--seed',str(a.seed)]
    if a.tp_size > 1:
        args.append('--sequence-parallel')
    if not a.no_save_hf:
        args += ['--save-hf',str(output/'hf')]
    return args


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path, help='SFT HF checkpoint, also frozen KL reference')
    p.add_argument('--appworld-python', required=True)
    p.add_argument('--appworld-root', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--num-gpus', type=int, default=2)
    p.add_argument('--tp-size', type=int, default=2)
    p.add_argument('--rollout-tp', type=int, default=2)
    p.add_argument('--tasks-per-batch', type=int, default=8)
    p.add_argument('--updates', type=int, default=100)
    p.add_argument('--max-steps', type=int, default=50)
    p.add_argument('--context-length', type=int, default=16384)
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--observation-tokens', type=int, default=2048)
    p.add_argument('--env-timeout', type=float, default=120)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--learning-rate', type=float, default=1e-6)
    p.add_argument('--kl-coef', type=float, default=0.01)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-interval', type=int, default=10)
    p.add_argument('--sglang-memory', type=float, default=0.5)
    p.add_argument('--init-checkpoint', type=Path, help='Optional converted Megatron SFT checkpoint')
    p.add_argument('--ref-checkpoint', type=Path, help='Optional converted Megatron copy of the SAME SFT model')
    p.add_argument('--no-save-hf', action='store_true')
    p.add_argument('--swanlab-mode', choices=['offline','cloud','disabled'], default='offline')
    p.add_argument('--swanlab-project', default='appworld-grpo')
    p.add_argument('--swanlab-name', default=None)
    p.add_argument('--log-upload-interval', type=float, default=60,
                   help='Seconds between full-log/episode file syncs to SwanLab')
    p.add_argument('--dry-run', action='store_true', help='Validate/create configs and print command without allocating GPUs')
    a = p.parse_args()
    for key in ['model','appworld_root','output_dir','init_checkpoint','ref_checkpoint']:
        if getattr(a,key) is not None:
            setattr(a,key,getattr(a,key).expanduser().resolve())
    if min(a.num_gpus,a.tp_size,a.rollout_tp,a.tasks_per_batch,a.updates,a.max_steps,a.max_new_tokens) <= 0:
        p.error('Counts and sizes must be positive')
    if a.num_gpus % a.rollout_tp or a.observation_tokens < 128 or a.context_length < 1024 or a.temperature <= 0:
        p.error('Invalid rollout TP, observation/context budget or sampling temperature')
    if a.log_upload_interval <= 0:
        p.error('--log-upload-interval must be positive')
    if a.learning_rate <= 0 or a.kl_coef < 0 or a.env_timeout <= 0 or not 0 < a.sglang_memory < 1:
        p.error('Invalid learning rate, KL coefficient, timeout or SGLang memory fraction')
    if a.output_dir.exists():
        p.error('Output directory must not exist; use a fresh run directory')
    model_config = json.loads((a.model/'config.json').read_text())
    if a.context_length > model_config['max_position_embeddings']:
        p.error('Context budget exceeds model config')
    script = find_training_script()
    worker = WorldProcess(a.appworld_python, a.appworld_root, a.env_timeout)
    try:
        task_ids = worker.call('list', split='train')
    finally:
        worker.close()
    if len(task_ids) < a.tasks_per_batch:
        p.error('Train set smaller than tasks-per-batch')
    a.output_dir.mkdir(parents=True)
    cfg = {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    cfg['experiment'] = 'grpo/'+uuid.uuid4().hex
    cfg_path = a.output_dir/'project_config.json'
    cfg_path.write_text(json.dumps(cfg,indent=2))
    prompts = a.output_dir/'train_tasks.jsonl'
    prompts.write_text(''.join(json.dumps(dict(prompt=tid,metadata=dict(task_id=tid,source_name='appworld_train')))+'\n' for tid in task_ids))
    # Use a local architecture script only; no upstream edits or version restrictions.
    architecture = subprocess.check_output(['bash','-c',
        'source "$1"; printf "%s\\0" "${MODEL_ARGS[@]}"','bash',str(HERE/'configs/qwen3_4b_slime.sh')])
    model_args = [x.decode() for x in architecture.split(b'\0') if x]
    command = [sys.executable,str(HERE/'slime_grpo_driver.py'),str(script),*model_args,
               *training_args(a,prompts,a.output_dir,model_config)]
    (a.output_dir/'command.json').write_text(json.dumps(command,indent=2))
    print(' '.join(command),flush=True)
    if a.dry_run:
        return
    env = os.environ.copy()
    env.update(APPWORLD_GRPO_CONFIG=str(cfg_path),TENSORBOARD_DIR=str(a.output_dir/'tensorboard'),
        PYTHONPATH=str(HERE)+os.pathsep+env.get('PYTHONPATH',''), CUDA_DEVICE_MAX_CONNECTIONS='1',
        PYTHONUNBUFFERED='1', RAY_DEDUP_LOGS='0')
    bridge = SwanLabBridge(a.output_dir,a.swanlab_mode,a.swanlab_project,
                           a.swanlab_name or a.output_dir.name,cfg)
    proc = None
    relay = None
    started = time.time()
    try:
        print('Starting Slime command:', ' '.join(command), flush=True)
        proc = subprocess.Popen(command,env=env,cwd=script.parent,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        relay = ProcessLogRelay(proc.stdout, a.output_dir/'training.log').start()
        while proc.poll() is None:
            if relay.error is not None:
                raise RuntimeError(f'Training log relay failed: {relay.error}')
            try:
                bridge.poll()
                bridge.sync_files()
            except Exception as exc:
                print(f'SwanLab forwarding failed; native files retained: {exc}', file=sys.stderr)
            time.sleep(5)
        if proc.returncode:
            raise SystemExit(proc.returncode)
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if relay is not None:
            try:
                relay.finish()
            except Exception as exc:
                print(f'Training log finalization failed: {exc}', file=sys.stderr)
        (a.output_dir/'run_status.json').write_text(json.dumps(dict(
            returncode=proc.returncode if proc else None, started_at=started, finished_at=time.time()), indent=2))
        try:
            bridge.finish()
        except Exception as exc:
            print(f'SwanLab final flush failed; native files retained: {exc}', file=sys.stderr)


if __name__ == '__main__':
    main()
