# Four-GPU asynchronous GRPO (400 GB job memory)

Request **4 A100 GPUs and 400 GB RAM in the job platform**. This repository cannot
change the platform resource allocation. It uses Slime's native asynchronous scheduler,
without modifying Slime or pinning its version.

## Launch

Activate the existing Slime environment and retain your SFT model/checkpoint variables:

```bash
git pull
export MODEL_PATH=/path/to/sft/hf
export APPWORLD_PYTHON=/opt/conda/envs/appworld/bin/python
export APPWORLD_ROOT=/home/notebook/data/group/decision_model/code/appworld
# Optional when not already installed/importable:
# export SLIME_DIR=/path/to/slime
# export MEGATRON_DIR=/path/to/Megatron-LM
# If direct HF loading is unsupported, set INIT_CHECKPOINT and REF_CHECKPOINT
# to converted checkpoints of this same SFT model.

OUTPUT_DIR="$PWD/grpo_runs/async_preflight" bash run_grpo_async.sh --dry-run
OUTPUT_DIR="$PWD/grpo_runs/async_4gpu" SWANLAB_MODE=cloud bash run_grpo_async.sh
```

The first command validates paths/data and prints the actual entry and arguments;
it does not test GPU initialization. Use a distinct output directory for training.
To run only a few updates, append `--updates 5`. The two environments remain separate.
The model's reward, four trajectories per task and action-only mask are unchanged.

## Native Slime selection

`ASYNC_IMPLEMENTATION=auto` (default):

- If the installed checkout contains `train_async.py`, use it directly (pipeline mode).
  This overlaps the next rollout batch with the current optimizer step, then waits for
  in-flight rollout before publishing weights every update. It is Slime's native legacy
  asynchronous training entry, with approximately one prefetched batch rather than an
  unbounded background dataset. The startup command records which path was selected.
- Otherwise use installed `train.py` with the native
  `slime.rollout.fully_async_rollout.generate_rollout_fully_async` extension. Its presence
  is checked by file capability, not a version comparison. No synchronous fallback is
  silently used. This path has a bounded native ready queue but does **not** guarantee
  that every trajectory's lag is at most one update; monitor native staleness metrics.

Set `ASYNC_IMPLEMENTATION=pipeline` to require `train_async.py`, or `fully` to explicitly
choose continuous generation if your installation supports it. Different releases can
expose different arguments; no versions are forced. This integration requires native
custom generation, complete-group GRPO, rollout logprobs and TIS interfaces.

## Starting parameters

| Resource / setting | Default |
|---|---|
| Allocated GPUs | 4, **disjoint** training/rollout resources |
| Actor training | 2 GPUs, TP=2, DP=1 |
| SGLang | 2 GPUs, two independent TP=1 4B replicas |
| Global batch | 8 tasks × 4 trajectories = 32 |
| Active AppWorld processes | at most 24 in the local rollout event loop |
| SGLang request concurrency | 16 per engine; actual activity also bounded by AppWorld cap |
| Train batching | dynamic token batching, 16384 tokens/GPU budget |
| Context / max action | 16384 total / 512 generated tokens per turn |
| Observations / turns | 2048 tokens per feedback / 50 attempts |
| Learning rate / KL | 1e-6 / 0.01 against frozen SFT reference |
| Weight publication | every update (native scheduling decides synchronization boundaries) |
| SGLang static memory fraction | 0.70 |
| Checkpoint interval | every 20 updates and final save according to installed Slime |

There is **no --colocate** in async mode: rollout no longer waits for its GPUs to be
borrowed by the trainer. No CPU optimizer-offload flag is added. BF16, distributed
optimizer and activation recomputation remain enabled. Two TP1 serving replicas avoid
TP communication for the small 4B model and admit parallel requests; dynamic training
microbatches pack short trajectories more effectively than a fixed sample per batch.
Actual gains depend on A100 memory size, CPU tools and sequence lengths.

The AppWorld cap is acquired before creating the subprocess and held until it closes.
This bounds live database worlds even when the native sampler prefetches multiple
batches. This recipe uses a single-node/local producer; with distributed producers a
process-local cap is not a cluster-wide semaphore. Partial rollout is not supported.
If the native async scheduler aborts an unfinished request, its world is closed and the
Sample is marked ABORTED with **no reward**, for native group requeueing. A retry starts
a fresh world. This avoids pretending external side effects can be resumed from text.
Other SDK/router failures still raise rather than being rewarded as task failures.

Async uses the actual sampled logprobs with `--use-tis --tis-clip-low 0 --tis-clip 2`.
PPO clipping remains 0.2. If SGLang returns weight-version metadata, versions are retained
on the Sample for native staleness reporting. Importance correction limits mismatch;
it does not make arbitrarily stale rollouts on-policy. Do not enable arbitrary deep
prefetch or long weight-sync intervals solely to improve utilization.

## Utilization measurement and the 70% target

This is a throughput-oriented **starting configuration**, not a measured guarantee of
70%. Without access to the four-GPU server we cannot truthfully certify the target.
The launcher now measures actual allocated devices every 5 seconds using nvidia-smi:

- `gpu_utilization.jsonl`: per-GPU utilization/VRAM and host/cgroup memory observations;
- `gpu_summary.json`: time-weighted per-GPU and equal-weight all-GPU averages, full-run
  and separately after 120 seconds warmup; `target_met` refers to the **full-run** average,
  `steady_target_met` to the warmup-excluded average. Errors/no data yield null flags.
- SwanLab: `hardware/gpu_mean_full`, `hardware/gpu_mean_steady`, and per-GPU instant
  utilization/used VRAM; full telemetry files are also synchronized as run artifacts.

Only allocated GPUs are counted. If nvidia-smi exposes more GPUs than the job requested
and CUDA_VISIBLE_DEVICES does not identify them, monitoring reports an error rather
than averaging the wrong set. Set `GPU_MONITOR_DEVICES=0,1,2,3` (physical indices) or
comma-separated full GPU UUIDs when needed. The utilization definition is sampled SM
busy percentage, not FLOP utilization or throughput. Initialization, checkpointing and
shutdown affect the full-run average; report both definitions, not only the better one.

The platform's full system-memory graph may differ from a container's memory quota;
cgroup current/max/peak are recorded when accessible. The GPU count alone does not
ensure a 400 GB quota; check the job resource request before launch. Monitoring failure
does not kill the scientific run, but means utilization attainment is unknown.

## Tune after a representative run

Inspect reward, mixed-group fraction, native TIS clipping / rollout-train mismatch,
memory peak and each GPU, not just the aggregate percentage. The all-GPU average is
approximately `(mean_actor_util + mean_rollout_util)/2` in the 2+2 split. One side being
saturated cannot hide the other side sitting idle.

- Actor idle while rollout is slow: if RAM has substantial headroom, try
  `MAX_ENVS=32 SERVER_CONCURRENCY=16` (still two engines); evaluate the memory peak.
- Serving idle while training is saturated: increase the dynamic packing budget only
  when VRAM permits, e.g. append `--train-max-tokens 24576`; this may reduce trainer time.
- Both low while environment CPU activity is high: inspect AppWorld execution latency,
  process startup and CPU allocation. Increasing GPU batch size alone cannot remove
  a CPU bottleneck.
- RAM approaches quota: reduce `MAX_ENVS` to 16 first; preserve the four samples per
  task and global batch while serializing more worlds. Do not disable Ray OOM protection.

Never add dummy computation, extra optimizer steps or excessive policy lag merely to
inflate the utilization score. Record task-success-per-hour alongside GPU utilization.
The async entry/topology, utilization arithmetic, abort handling and option dispatch
are CPU-tested; GPU utilization, live queue behavior and memory limits require real
server validation. Refer to upstream native scheduling documentation:
https://thudm.github.io/slime/advanced/rollout-scheduling.html

### Timeout recovery and diagnosis

The default remains four GPUs (two actor TP2, two rollout TP1). Environment
calls use a 120-second deadline and one fresh-world recovery per episode:

```bash
SWANLAB_MODE=cloud bash run_grpo_async.sh --env-timeout 120 --env-timeout-retries 1
```

Keep the existing MODEL_PATH, APPWORLD_ROOT, APPWORLD_PYTHON and Slime setup.
Use a fresh OUTPUT_DIR. No Slime or Megatron source modification/version pin is
required. `--env-timeout-retries 0` disables recovery.

On timeout, the worker is closed, a unique experiment is reset with the same
seed, and every completed executable action is replayed. Reset information and
replay results must match exactly before retrying the timed-out operation.
Policy tokens/log probabilities are not regenerated or duplicated, and the
four-trajectory group remains intact. Replay can fail for nondeterministic
operations. Exhausted timeouts, replay divergence and other infrastructure
errors still stop training explicitly; they are never silently scored zero or
dropped from a group. This is bounded recovery, not a guarantee against persistent
bad loops or a checkpoint-resume implementation.

`episodes/<episode_id>.jsonl` is flushed for every event, including code before
execution, raw generation with token log probabilities, raw environment results,
timeouts, recovery and terminal failures. These files may be large. Existing
completed `.json` episode summaries remain available. Files are synced through
the existing SwanLab file bridge (default every 60 seconds and at shutdown);
cloud mode is needed for online visibility. Hard process/host kills can prevent
final upload; already flushed local journal lines remain available.

SwanLab metrics:

- `appworld/timing/{reset,generation,execute,evaluate,recovery,cleanup,env_slot_wait}_{mean,max}_seconds`
  measure accumulated wall time per episode, averaged/maximized over a rollout
  batch. Generation includes router/HTTP wait; execute includes timed-out calls;
  recovery includes worker shutdown, reset and replay. Concurrent episode times
  overlap: do not add these averages to infer batch wall time.
- `appworld/timing/episode_mean_seconds`, `appworld/rollout_seconds` and
  `appworld/timeout_recoveries` describe scored batches. Episode time excludes
  semaphore wait and final cleanup.
- `environment_events/*` are cumulative live journal counts, including failures
  before a scored batch exists. Their x-axis is bridge event polls, not updates.
- Existing `slime/*` native TensorBoard metrics retain training/weight-sync timings
  exposed by the installed Slime. This bridge does not invent unavailable native
  timing fields or patch the framework.

For a later eight-GPU comparison (six TP1 rollout engines, two TP2 actor GPUs):

```bash
SWANLAB_MODE=cloud bash run_grpo_async_8gpu.sh --env-timeout 120 --env-timeout-retries 1
```

The batch remains eight tasks times four trajectories and MAX_ENVS remains 24,
so the initial comparison isolates topology. Increase MAX_ENVS separately only
after checking CPU pressure and timeout frequency. Eight GPUs do not guarantee
70% utilization. Request GPUs/CPU/RAM from the job platform separately.
