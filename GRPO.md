# SFT → synchronous multi-turn AppWorld GRPO

All integration code is in this repository; no Slime source edits, version checks or
Monkey patches. The launcher executes your installed Slime `train.py` (synchronous
entry), using its normal SGLang rollout, GRPO advantages, PPO clipping, Megatron
optimizer and weight synchronization. It does not use Slime's fully-async entry point.

## Environment

Use the existing working Slime training environment, plus project tracking dependencies:

```bash
python -m pip install -r requirements-grpo.txt
```

No packages are installed automatically during launch. AppWorld stays in its separate
Python environment. SwanLab credentials for cloud logging are supplied by your normal
SwanLab login/environment; never put API keys in this repo. If the platform already
installs SwanLab, do not reinstall it unnecessarily. This launcher does not alter the
framework's protobuf/grpc requirements.

Slime must expose custom-generate, custom-RM, custom-rollout-log, native GRPO/PPO-loss,
TensorBoard and co-located SGLang APIs. Versions are not constrained; missing APIs
produce normal installed-framework errors. `SLIME_DIR`/`MEGATRON_DIR` remain optional
source paths, as in SFT. `SLIME_TRAIN_SCRIPT` can select the actual synchronous train.py.
Do not point it at train_async.py. Only single-node training is configured here.

## Start from the SFT checkpoint

`MODEL_PATH` must be your **complete HF-format SFT model**, including tokenizer and
config. It initializes Actor and provides the frozen KL reference, not the base Qwen.

```bash
git pull
export MODEL_PATH=/path/to/sft_output/hf
# Optional if these modules are not installed as importable packages:
# export SLIME_DIR=/path/to/slime
# export MEGATRON_DIR=/path/to/Megatron-LM
export APPWORLD_PYTHON=/opt/conda/envs/appworld/bin/python
export APPWORLD_ROOT=/home/notebook/data/group/decision_model/code/appworld

# Check data enumeration/config/command only (still needs the CPU AppWorld SDK).
OUTPUT_DIR="$PWD/grpo_runs/check_config" bash run_grpo.sh --dry-run

# Fresh directory for real training; activates the Python from your Slime environment.
OUTPUT_DIR="$PWD/grpo_runs/first_run" SWANLAB_MODE=offline \
NUM_GPUS=2 TP_SIZE=2 ROLLOUT_TP=2 TASKS_PER_BATCH=8 UPDATES=100 \
bash run_grpo.sh
```

Use `SWANLAB_MODE=cloud` for a logged-in cloud run, or `disabled` to omit SwanLab
imports/logging. Native TensorBoard files and project rollout metrics still get written.

Initial weights can be HF-loaded only when your Slime build supports that path. For
older checkpoint loaders, set **both** `INIT_CHECKPOINT` and `REF_CHECKPOINT` to
converted Megatron checkpoints of this same SFT model. To omit unsupported direct HF
export, use `SAVE_HF=0`; use your framework's export tool before evaluation.

Default settings:

- Train split only, 8 tasks per rollout batch, exactly 4 independent trajectories
  per task, global batch 32, one optimizer batch per completed rollout batch.
- Temperature 0.8, top-p 1, 50 interaction attempts, at most 512 generated tokens per
  turn, 16K total trajectory budget, per-observation cap 2048 tokens.
- GRPO advantage estimator and PPO-style clipping 0.2; learning rate 1e-6; frozen SFT
  reference with KL loss coefficient 0.01. No Critic, GAE or chunk splitting.
- BF16 full tuning, microbatch 1, distributed optimizer and activation recomputation.
  Training/inference share GPUs using Slime's co-location/offload mechanisms.
- Full loss reduction uses Slime's default per-trajectory weighting. We do not enable
  the SFT per-token outer reduction, which could overweight long failed trajectories.

The four env instances are separate subprocesses with unique experiment paths and
the same task/world seed. Sampling seeds differ per trajectory and turn. Slime waits
for the batch before training and syncing weights; concurrent generation *within*
the batch is expected. The batch is not an asynchronous policy-update queue.

Start with high-memory CUDA GPUs and measure actual peak memory. The two-GPU command
is a topology example, not a guarantee for 24 GB cards. For smaller smoke runs use
`TASKS_PER_BATCH=1 UPDATES=2`, with batch/TP compatible with the number of GPUs.
Ray/SGLang may run multiple environment subprocesses concurrently; allow CPU/RAM for
32 independent worlds at the default batch size, plus the dataset and model. No extra
GPU is reserved for a reward model.

## Exact rollout/reward contract

The dataset is built from the installed `load_task_ids('train')`; no Dev/Test task
trajectory or ground truth is used for policy generation. Each episode resets a fresh
world and renders the same project prompt used for SFT and evaluation.

`/generate` receives token IDs. The action IDs and rollout log probabilities come
**directly** from SGLang `output_token_logprobs`; generated text is never re-tokenized
to stand in for sampled actions. Only those IDs have loss mask 1. Initial examples,
environment feedback, injected turn boundaries and subsequent assistant generation
headers have loss mask 0. Dummy feedback log probabilities are zero and masked out.
The response span starts after the initial prompt and includes observations between
actions; Slime prepends the initial-prompt zero mask during training.

This adapter uses Qwen3 chat boundaries (`<|im_end|>`) and the local tokenizer's
non-thinking user/assistant template. It preserves generated stop tokens; if the
server omitted the closing token, the delimiter is inserted as zero-loss context.
Do not substitute a different chat model/tokenizer. Malformed code gets the existing
explicit "CODE NOT EXECUTED" feedback. A length-limited generation is not submitted
as potentially incomplete Python. Context/step limits terminate the episode, then
still evaluate the resulting environment state.

Only `world.evaluate().success is True` earns 1; everything evaluated unsuccessful
earns 0. `complete_task`, partial check fractions and login success do not earn reward.
No step penalty or intermediate shaping is added. The KL regularizer affects the
optimization objective, not the reported binary environment reward. Scored step/context
limits are kept as valid failures; `metadata.appworld.termination` carries their reason.
A subprocess/SDK/router/evaluation failure raises and stops the batch instead of
silently adding a false reward-0 trajectory. First fix the infrastructure and rerun
into a fresh output directory; automatic retry/resume is not implemented here.

Huge observations are bounded before tokenization and then token-capped with an
explicit truncation notice. This changes only the feedback shown to the policy;
variables and AppWorld state are retained. The policy can issue new pagination,
filtering or summary code. Initial prompts and sampled actions are not truncated.

## SwanLab without framework changes

Slime writes its built-in TensorBoard events using `--use-tensorboard`. The project
parent process polls the scalar event files and mirrors available tags into SwanLab
under `slime/...`, including whichever training loss/KL/gradient metrics your installed
Slime emits. It does not parse stdout, replace logging functions, or import SwanLab
inside GPU actors. TensorBoard's flush interval can delay appearance of these metrics.
Only metrics actually emitted as scalar events are forwarded.

The custom rollout logging hook also writes project metrics under `appworld/...`:

- success rate, all-fail/all-success/mixed-reward group fractions;
- average interaction turns, trained action-token count;
- format/execution errors, observation truncations and termination counts.

Groups are checked to contain four trajectories. Binary uniform groups have no
outcome-reward relative advantage (KL can still contribute); monitor mixed-group
fraction before scaling training. Project rollout counters use rollout IDs; native
TensorBoard tags preserve their emitted steps. Each scalar/event is deduplicated
within the running bridge process. This is fresh-run logging, not resume support.

## Outputs and evaluation

```text
OUTPUT_DIR/
  project_config.json, command.json, train_tasks.jsonl
  episodes/*.json            # code, bounded feedback, reward, task/group/index and termination
  project_metrics/*.json     # rollout batch metrics
  tensorboard/               # native Slime scalar events
  swanlab/                   # project-side logs (offline/cloud mode)
  checkpoints/               # Megatron checkpoints
  hf/                        # latest HF export, if enabled/supported
```

Do not publish episodes or derived task data unencrypted. The directory is ignored by
git under the default `grpo_runs/` location.

Evaluate SFT and RL checkpoints with identical observation handling and budgets. The
existing evaluation runner now supports `--observation-tokens` (0 keeps old behavior).
For example:

```bash
python run_agent.py --backend transformers --model /path/to/grpo_run/hf \
  --device cuda --appworld-python "$APPWORLD_PYTHON" --appworld-root "$APPWORLD_ROOT" \
  --split dev --all-tasks --context-length 16384 --max-steps 50 \
  --max-new-tokens 512 --observation-tokens 2048 --temperature 0
```

Run the same command with the SFT model path first. Do not compare RL with clipped
feedback against the old SFT score using unlimited feedback and call the whole change
a policy gain. API-backend evaluation can also use the cap, but requires `--model`
pointing to the corresponding local tokenizer even when inference runs on a server.

## Validation boundary

CPU unit/integration tests cover exact action ID/logprob masks, multi-turn state,
reward from evaluation rather than declaration, group metrics, subprocess cleanup,
raising infrastructure failures and SwanLab event deduplication. Tests use simulated
AppWorld/SGLang/TensorBoard objects; there is no CUDA, live SGLang or Megatron training
on the development Mac. Use a two-update server smoke run before a full experiment;
check a saved episode, mask lengths, finite native losses, weight sync and HF export.

## Full console and episode-file synchronization to SwanLab

The launcher now pipes the training child's **merged stdout/stderr** back through the
parent process's SwanLab console proxy. This includes Slime/Ray worker output forwarded
to the driver, native warnings and Python exception stacks. Ray starts with
`log_to_driver=True` and `RAY_DEDUP_LOGS=0`, so repeated worker errors are not intentionally
collapsed by Ray in this run. Output is drained on a dedicated thread while metric
polling continues, and also saved without line truncation to `OUTPUT_DIR/training.log`.
The child's nonzero exit status is preserved. Graceful error/interrupt handling drains
remaining output before calling SwanLab finish.

SwanLab's Logs UI/SDK can impose per-line length and display limits. To retain the
complete original stream, the project additionally calls the public `swanlab.save()`
API for a file allowlist, periodically and again before finishing:

- `training.log`;
- `episodes/*.json` (every completed episode's stored turns, code and bounded feedback);
- `project_metrics/*.json`;
- `project_config.json`, `command.json`, `train_tasks.jsonl`, `run_status.json`.

No model checkpoint or arbitrary directory is uploaded. Files already submitted with
unchanged size/mtime are not resubmitted; the growing training.log is resubmitted when
changed. Default file-sync interval is 60 seconds. Override it with:

```bash
SWANLAB_MODE=cloud bash run_grpo.sh --log-upload-interval 30
```

In cloud mode, console output and saved files are sent to the current run; in offline
mode they remain in the SDK's local run storage for later sync. Disabled mode still
writes the full local training.log, but does not send anything to SwanLab. Check the
run's Logs view for console output and its file listing for saved artifacts (UI placement
varies). This requires a SwanLab SDK exposing public `save(glob_str, base_path, policy)`;
if absent, the project explicitly reports that files are local only. SDK submission
and the final SDK flush are not independent verification of server-side retention.

Only output written/forwarded to the child process's stdout/stderr is in training.log.
Private Ray log files never forwarded to the driver are not automatically harvested.
Episode files contain the actual **bounded** observations the policy received, not
unlimited pre-truncation API payloads. An episode that never completes due to an
infrastructure exception may have no episode JSON; its exception remains in the console
log. Hard machine termination/SIGKILL cannot guarantee a final upload; periodic sync
limits how much remains only on disk.

A new launch is required to enable subprocess capture; this does not retroactively
collect stdout from an already-running process. Existing metric forwarding continues
unchanged. Full-log/file synchronization is covered by CPU subprocess tests, including
stderr, Unicode, lines longer than the UI limit, exit status, file allowlisting and
resubmission after change. Cloud delivery and a live GPU/Ray run still need verification
in the server environment.
