"""Launch Qwen3-4B AppWorld Warm-Critic/RPR training on Terminal-RSI Slime."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

from gpu_monitor import GPUMonitor
from grpo_tracking import SwanLabBridge
from process_logging import ProcessLogRelay
from resource_monitor import ResourceMonitor, atomic_json, diagnose
from run_agent import WorldProcess


HERE = Path(__file__).resolve().parent
PHASES = {
    'warmup': dict(reward='fdr', actor_lr=5e-7, asynchronous=False),
    'rpr1': dict(reward='fdr', actor_lr=5e-7, asynchronous=True),
    'rpr2': dict(reward='spr', actor_lr=1e-7, asynchronous=True),
}


def validate_terminal_rsi(root):
    root = Path(root).expanduser().resolve()
    required = ['train.py', 'train_async.py', 'slime/utils/tb_gae_config.py',
                'slime_plugins/harbor/tb_gae.py', 'slime_plugins/harbor/train_loss.py']
    missing = [name for name in required if not (root/name).is_file()]
    if missing:
        raise ValueError(f'TERMINAL_RSI_DIR is missing RPR runtime files: {missing}')
    return root


def architecture_args(model_config):
    rope = model_config.get('rope_theta', model_config.get('rope_parameters', {}).get('rope_theta'))
    expected = {'model_type':'qwen3', 'num_hidden_layers':36, 'hidden_size':2560,
                'intermediate_size':9728, 'num_attention_heads':32,
                'num_key_value_heads':8, 'head_dim':128, 'vocab_size':151936}
    bad = {key:(model_config.get(key), value) for key, value in expected.items()
           if model_config.get(key) != value}
    if bad or rope is None:
        raise ValueError(f'Unsupported Qwen3-4B checkpoint config: mismatches={bad}, rope={rope!r}')
    return ['--swiglu', '--num-layers','36', '--hidden-size','2560',
        '--ffn-hidden-size','9728', '--num-attention-heads','32',
        '--group-query-attention', '--num-query-groups','8',
        '--use-rotary-position-embeddings', '--disable-bias-linear',
        '--normalization','RMSNorm', '--norm-epsilon','1e-6',
        '--rotary-base',str(rope), '--vocab-size','151936', '--kv-channels','128',
        '--qk-layernorm']


def write_role_config(path, actor_lr, critic_lr):
    path.write_text(f"""megatron:
  - name: appworld-qwen3-4b-actor
    role: actor
    overrides:
      lr: {actor_lr:.8g}
      min_lr: {actor_lr:.8g}
  - name: appworld-qwen3-4b-critic
    role: critic
    overrides:
      lr: {critic_lr:.8g}
      min_lr: {critic_lr:.8g}
      value_clip: 4.0
      clip_grad: 100.0
""")


def write_task_config(path, train_token_cap, gamma, lambd):
    path.write_text(f"""enable_tb_gae: true
tb_gae_gamma_turn: {gamma:.8g}
tb_gae_lambda_turn: {lambd:.8g}
harbor_sample_mode: trajectory_retokenized
harbor_include_subagent_rollouts: false
harbor_max_train_total_tokens: {train_token_cap}
""")


def training_args(a, prompt_file, output, role_config, task_config, model_config):
    phase = PHASES[a.phase]
    rollout_gpus = a.num_gpus-a.actor_gpus
    if rollout_gpus <= 0 or rollout_gpus % a.rollout_tp:
        raise ValueError('Rollout GPU count must be positive and divisible by rollout TP')
    if a.actor_gpus % a.tp_size:
        raise ValueError('Actor GPU count must be divisible by actor TP')
    if a.batch_size % (a.actor_gpus//a.tp_size):
        raise ValueError('Batch size must be divisible by actor data parallel size')
    critic_only = a.updates if a.phase == 'warmup' else a.critic_only_steps
    reward_hook = 'appworld_rpr.fdr_rewards' if phase['reward'] == 'fdr' else 'appworld_rpr.spr_rewards'
    args = [
        '--actor-num-nodes','1', '--actor-num-gpus-per-node',str(a.actor_gpus),
        '--rollout-num-gpus',str(rollout_gpus),
        '--rollout-num-gpus-per-engine',str(a.rollout_tp),
        '--hf-checkpoint',str(a.model), '--load',str(a.load_root), '--ref-load',str(a.load_root),
        '--no-load-optim','--no-load-rng', '--save',str(output/'checkpoints'),
        '--save-interval',str(a.save_interval), '--start-rollout-id','0',
        '--prompt-data',str(prompt_file), '--input-key','prompt', '--metadata-key','metadata',
        '--rollout-shuffle', '--num-rollout',str(a.updates),
        '--rollout-batch-size',str(a.batch_size), '--n-samples-per-prompt','1',
        '--global-batch-size',str(a.batch_size), '--num-steps-per-rollout','1', '--balance-data',
        '--rollout-max-response-len',str(a.context_length),
        '--rollout-max-context-len',str(a.context_length),
        '--rollout-temperature',str(a.temperature), '--rollout-top-p','1',
        '--rollout-function-path','slime_plugins.harbor.rollout.generate_rollout',
        '--custom-generate-function-path','appworld_rpr.generate',
        '--custom-rm-path','appworld_grpo.reward',
        '--custom-reward-post-process-path',reward_hook,
        '--custom-rollout-log-function-path','appworld_rpr.log_rollout_data',
        '--rollout-all-samples-process-path',
        'slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_samples',
        '--rollout-data-postprocess-path',
        'slime_plugins.harbor.sample_postprocess.cap_total_tokens_allow_response_trim_in_rollout_data',
        '--custom-config-path',str(task_config),
        '--advantage-estimator','ppo', '--enable-tb-gae',
        '--tb-gae-gamma-turn',str(a.gamma_turn), '--tb-gae-lambda-turn',str(a.lambda_turn),
        '--num-critic-only-steps',str(critic_only),
        '--eps-clip','0.3', '--eps-clip-high','5.0', '--keep-old-actor',
        '--loss-type','custom_loss', '--custom-loss-function-path',
        'slime_plugins.harbor.train_loss.policy_loss_no_entropy_when_disabled',
        '--kl-loss-coef','0', '--kl-coef','0', '--entropy-coef','0',
        '--megatron-config-path',str(role_config),
        '--optimizer','adam', '--lr',str(a.actor_lr), '--lr-decay-style','constant',
        '--weight-decay','0.1', '--adam-beta1','0.9', '--adam-beta2','0.98',
        '--use-precision-aware-optimizer', '--optimizer-cpu-offload',
        '--overlap-cpu-optimizer-d2h-h2d', '--use-distributed-optimizer', '--bf16',
        '--tensor-model-parallel-size',str(a.tp_size), '--pipeline-model-parallel-size','1',
        '--context-parallel-size','1', '--expert-model-parallel-size','1',
        '--expert-tensor-parallel-size','1', '--micro-batch-size','1',
        '--recompute-granularity','full', '--recompute-method','uniform',
        '--recompute-num-layers','1', '--qkv-format','thd',
        '--seq-length',str(a.context_length),
        '--max-position-embeddings',str(model_config['max_position_embeddings']),
        '--use-dynamic-batch-size', '--max-tokens-per-gpu',str(a.train_max_tokens),
        '--data-pad-size-multiplier','2048', '--log-probs-chunk-size','64',
        '--attention-dropout','0', '--hidden-dropout','0', '--attention-backend','flash',
        '--accumulate-allreduce-grads-in-fp32', '--attention-softmax-in-fp32',
        '--sglang-mem-fraction-static',str(a.sglang_memory),
        '--sglang-context-length',str(a.context_length),
        '--sglang-max-prefill-tokens',str(a.context_length),
        '--sglang-chunked-prefill-size',str(min(8192, a.context_length)),
        '--sglang-disable-custom-all-reduce', '--sglang-server-concurrency',str(a.server_concurrency),
        '--update-weights-interval','1', '--use-fault-tolerance',
        '--rollout-health-check-interval','30', '--rollout-health-check-timeout','120',
        '--rollout-health-check-first-wait','600',
        '--use-tensorboard', '--tb-project-name','appworld_rpr', '--seed',str(a.seed),
    ]
    if a.tp_size > 1:
        args.append('--sequence-parallel')
    return args


def launch(command, a, cfg, script):
    if a.dry_run:
        return
    env = os.environ.copy()
    env.update(APPWORLD_GRPO_CONFIG=str(a.output_dir/'project_config.json'),
        TENSORBOARD_DIR=str(a.output_dir/'tensorboard'),
        PYTHONPATH=str(HERE)+os.pathsep+str(a.terminal_rsi_dir)+os.pathsep+env.get('PYTHONPATH',''),
        CUDA_DEVICE_MAX_CONNECTIONS='1', PYTHONUNBUFFERED='1', RAY_DEDUP_LOGS='0')
    bridge = SwanLabBridge(a.output_dir, a.swanlab_mode, a.swanlab_project,
                           a.swanlab_name or a.output_dir.name, cfg)
    monitor = GPUMonitor(a.output_dir, a.num_gpus, a.gpu_monitor_interval,
                         a.gpu_monitor_warmup, a.gpu_util_target)
    resources = ResourceMonitor(a.output_dir, a.gpu_monitor_interval)
    received_signal = None
    def terminate(signum, frame):
        nonlocal received_signal
        received_signal = signum
        raise SystemExit(128+signum)
    old_handlers = {sig:signal.signal(sig, terminate) for sig in (signal.SIGTERM, signal.SIGINT)}
    proc = relay = None
    started = time.time()
    atomic_json(a.output_dir/'run_status.json',
                dict(state='running', started_at=started, launcher_pid=os.getpid()))
    try:
        resources.start()
        print('Starting Terminal-RSI command:', ' '.join(command), flush=True)
        proc = subprocess.Popen(command, env=env, cwd=script.parent,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        relay = ProcessLogRelay(proc.stdout, a.output_dir/'training.log').start()
        monitor.start()
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
        launcher_error = repr(sys.exc_info()[1]) if sys.exc_info()[1] else None
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
        if relay is not None:
            relay.finish()
        monitor.finish()
        try:
            resources.finish()
        except Exception as exc:
            print(f'Resource monitor finalization failed: {exc}', file=sys.stderr)
        diagnosis = diagnose(a.output_dir, proc.returncode if proc else None,
                             received_signal, launcher_error)
        (a.output_dir/'run_status.json').write_text(json.dumps(dict(state='finished',
            diagnosis=diagnosis['category'], received_signal=received_signal,
            returncode=proc.returncode if proc else None, started_at=started,
            finished_at=time.time()), indent=2))
        try:
            bridge.finish()
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=PHASES)
    parser.add_argument('--model', required=True, type=Path, help='Qwen3-4B HuggingFace model/tokenizer')
    parser.add_argument('--load-root', required=True, type=Path,
                        help='Megatron actor root; RPR phases require sibling <root>_critic')
    parser.add_argument('--terminal-rsi-dir', type=Path,
                        default=os.environ.get('TERMINAL_RSI_DIR'))
    parser.add_argument('--appworld-python', required=True)
    parser.add_argument('--appworld-root', required=True, type=Path)
    parser.add_argument('--reward-scales', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--num-gpus', type=int, default=4)
    parser.add_argument('--actor-gpus', type=int, default=2)
    parser.add_argument('--tp-size', type=int, default=2)
    parser.add_argument('--rollout-tp', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--updates', type=int, default=12)
    parser.add_argument('--critic-only-steps', type=int, default=4)
    parser.add_argument('--context-length', type=int, default=16384)
    parser.add_argument('--train-max-tokens', type=int, default=8192,
                        help='TB-GAE hard chunk cap; may be below rollout context')
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--observation-tokens', type=int, default=2048)
    parser.add_argument('--max-envs', type=int, default=24)
    parser.add_argument('--server-concurrency', type=int, default=16)
    parser.add_argument('--async-group-capacity', type=int,
                        help='Bounded pending/active/ready task groups; default: 2 * batch-size')
    parser.add_argument('--async-inflight-groups', type=int,
                        help='Concurrent task groups; default: max-envs')
    parser.add_argument('--async-max-policy-lag', type=int, default=1)
    parser.add_argument('--async-group-retries', type=int, default=3,
                        help='Maximum retries of an aborted task group before replacing it')
    parser.add_argument('--async-max-consecutive-timeouts', type=int, default=3,
                        help='Stop after this many consecutive exhausted execute-timeout groups')
    parser.add_argument('--env-timeout', type=float, default=120)
    parser.add_argument('--env-timeout-retries', type=int, default=1)
    parser.add_argument('--env-execute-budget-seconds', type=float, default=300)
    parser.add_argument('--repeated-execution-error-limit', type=int, default=2,
                        help='End and evaluate after the same action/error repeats (0 disables)')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--actor-lr', type=float)
    parser.add_argument('--critic-lr', type=float, default=1.5e-5)
    parser.add_argument('--gamma-turn', type=float, default=1.0)
    parser.add_argument('--lambda-turn', type=float, default=1.0)
    parser.add_argument('--save-interval', type=int, default=2)
    parser.add_argument('--sglang-memory', type=float, default=0.55)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu-util-target', type=float, default=70)
    parser.add_argument('--gpu-monitor-interval', type=float, default=5)
    parser.add_argument('--gpu-monitor-warmup', type=float, default=120)
    parser.add_argument('--swanlab-mode', choices=['offline','cloud','disabled'], default='offline')
    parser.add_argument('--swanlab-project', default='appworld-rpr')
    parser.add_argument('--swanlab-name')
    parser.add_argument('--log-upload-interval', type=float, default=60)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.terminal_rsi_dir is None:
        parser.error('Set --terminal-rsi-dir or TERMINAL_RSI_DIR')
    args.actor_lr = args.actor_lr or PHASES[args.phase]['actor_lr']
    if args.async_group_capacity is None:
        args.async_group_capacity = 2 * args.batch_size
    if args.async_inflight_groups is None:
        args.async_inflight_groups = args.max_envs
    for key in ('model','load_root','terminal_rsi_dir','appworld_root','reward_scales','output_dir'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    return args


def main(argv=None):
    a = parse_args(argv)
    positive = [a.num_gpus,a.actor_gpus,a.tp_size,a.rollout_tp,a.batch_size,a.updates,
                a.context_length,a.train_max_tokens,a.max_steps,a.max_new_tokens,a.max_envs,
                a.server_concurrency,a.async_group_capacity,a.async_inflight_groups,
                a.async_max_consecutive_timeouts,a.env_timeout,a.actor_lr,a.critic_lr,a.save_interval]
    if any(value <= 0 for value in positive) or not 0 < a.gamma_turn <= 1 or not 0 < a.lambda_turn <= 1:
        raise SystemExit('Counts/LRs must be positive and turn discounts must be in (0,1]')
    if a.critic_only_steps < 0 or a.critic_only_steps >= a.updates and a.phase != 'warmup':
        raise SystemExit('RPR critic-only steps must leave at least one joint actor/critic update')
    if a.train_max_tokens > a.context_length or not 0 < a.sglang_memory < 1:
        raise SystemExit('Train token cap must not exceed context; SGLang memory must be in (0,1)')
    if a.async_group_capacity < a.batch_size or a.async_inflight_groups > a.async_group_capacity:
        raise SystemExit('Async group capacity must cover the batch and concurrent groups')
    if (a.async_max_policy_lag < 0 or a.async_group_retries < 0
            or a.repeated_execution_error_limit < 0):
        raise SystemExit('Async lag/retry and repeated-error limits must be nonnegative')
    if a.output_dir.exists():
        raise SystemExit('Output directory must not exist')
    a.terminal_rsi_dir = validate_terminal_rsi(a.terminal_rsi_dir)
    if not (a.model/'config.json').is_file() or not a.reward_scales.is_file():
        raise SystemExit('Missing model config or reward scale file')
    if not (a.load_root/'latest_checkpointed_iteration.txt').is_file():
        raise SystemExit('load-root is not a Megatron checkpoint root')
    critic_root = Path(str(a.load_root)+'_critic')
    if a.phase == 'warmup' and critic_root.exists():
        raise SystemExit('Warm-up requires a cold Critic; use an actor root without <root>_critic')
    if a.phase != 'warmup' and not (critic_root/'latest_checkpointed_iteration.txt').is_file():
        raise SystemExit('RPR phase requires inherited Critic at <load-root>_critic')
    model_config = json.loads((a.model/'config.json').read_text())
    if a.context_length > model_config['max_position_embeddings']:
        raise SystemExit('Context length exceeds model maximum')
    scales = json.loads(a.reward_scales.read_text())
    if scales.get('schema') != 'appworld_rpr_reward_scales_v1':
        raise SystemExit('Unsupported reward scale schema; run calibrate_rpr_rewards.py')
    worker = WorldProcess(a.appworld_python, a.appworld_root, a.env_timeout)
    try:
        task_ids = worker.call('list', split='train')
    finally:
        worker.close()
    if len(task_ids) < a.batch_size:
        raise SystemExit('Train set is smaller than batch-size')
    a.output_dir.mkdir(parents=True)
    copied_scales = a.output_dir/'reward_scales.json'
    shutil.copy2(a.reward_scales, copied_scales)
    role_config = a.output_dir/'rpr_roles.yaml'
    task_config = a.output_dir/'rpr_task_config.yaml'
    write_role_config(role_config, a.actor_lr, a.critic_lr)
    write_task_config(task_config, a.train_max_tokens, a.gamma_turn, a.lambda_turn)
    prompts = a.output_dir/'train_tasks.jsonl'
    prompts.write_text(''.join(json.dumps(dict(prompt=task_id,
        metadata=dict(task_id=task_id, source_name='appworld_train')))+'\n' for task_id in task_ids))
    cfg = {key:(str(value) if isinstance(value, Path) else value) for key, value in vars(a).items()}
    cfg.update(experiment='rpr/'+uuid.uuid4().hex, async_training=PHASES[a.phase]['asynchronous'],
               enable_tb_gae=True, reward_mode=PHASES[a.phase]['reward'],
               reward_scale_file=str(copied_scales))
    cfg_path = a.output_dir/'project_config.json'
    cfg_path.write_text(json.dumps(cfg, indent=2))
    script = a.terminal_rsi_dir/('train_async.py' if PHASES[a.phase]['asynchronous'] else 'train.py')
    command = [sys.executable, str(HERE/'slime_grpo_driver.py'), str(script),
               *architecture_args(model_config),
               *training_args(a, prompts, a.output_dir, role_config, task_config, model_config)]
    (a.output_dir/'command.json').write_text(json.dumps(command, indent=2))
    print(' '.join(command), flush=True)
    launch(command, a, cfg, script)


if __name__ == '__main__':
    main()
