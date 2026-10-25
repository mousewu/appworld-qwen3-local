# AppWorld → Slime full-parameter SFT

This is the current SFT entry point. `run_sft_full.sh` now forwards to
`run_sft_slime.sh`; the old HF shell launcher is `run_sft_full_hf.sh`. `train_full.py`/`train_lora.py` remain as legacy
HF Trainer alternatives; **they are not invoked by Slime training**. This integration
uses Slime's Megatron full-parameter training backend, custom file rollout, and native
`sft_loss`. It is not a LoRA implementation. Model checkpoint loading/saving, distributed
optimizer, backward, gradient reduction and parameter updates are owned by Slime.

## 1. Training environment (separate from AppWorld)

Pin the upstream source/API checked for this implementation:

```bash
git clone https://github.com/THUDM/slime.git
cd slime
git checkout 18868a79b0b083a093798a94ed0f8cb4f71b9c14
```

Use the compatible Megatron, Transformer Engine, CUDA, PyTorch and Ray stack from this
checkout's [official setup](https://github.com/THUDM/slime/blob/18868a79b0b083a093798a94ed0f8cb4f71b9c14/docs/en/get_started/quick_start.md).
A plain `pip install slime` is not the NVIDIA training dependency installation.
Do **not** install this project's `requirements-sft.txt` into that environment: its
HF Trainer pins can conflict with Slime. Jinja2 and Transformers must be available.

Docker is not required. The upstream non-Docker path is:

```bash
# Linux NVIDIA training server; execute from the pinned Slime checkout.
# Upstream build script creates a micromamba environment and compiles dependencies.
# It also accesses /root/.cargo and modifies shell configuration; inspect it first.
BASE_DIR=/root SHELL=/bin/bash bash build_conda.sh
source ~/.bashrc
micromamba activate slime
export CUDA_HOME="$CONDA_PREFIX"
python -c 'import torch, ray, slime, megatron.core, transformer_engine; print(torch.cuda.device_count())'
```

This path requires suitable /root permissions, build tools, network access during setup,
and a compatible driver. The pinned script defaults its CUDA build architectures to
Hopper/Blackwell; adjust the upstream `TORCH_CUDA_ARCH_LIST` for your actual hardware
and check that its dependency stack supports that GPU. Slime's CUDA requirements are
separate from AppWorld's CPU-only installation. Existing compatible Slime environments
can be reused by specifying `SLIME_DIR`, `MEGATRON_DIR`, and `SLIME_PYTHON` below.

## 2. Replay official Train trajectories

Use the verified `experiment-outputs-0.1.1.bundle` with data version 0.1.0. Its experiment
name is **`react_gpt4o_train`**, with 90 execution logs before filtering. The stable SDK's
default downloader may still fetch the older 0.1.0 output bundle without Train logs.
The current upstream README's `simplified_..._train` name is not this bundle's directory.
Get the bundle from the [official S3 URL](https://s3.us-west-2.amazonaws.com/appworld.dev/experiment-outputs-0.1.1.bundle)
and unpack it with your installed SDK's `unpack_bundle` into `APPWORLD_ROOT/experiments/outputs`.
Expected SHA256:
`93cffe0a0b5946604a1b9b6c295bc75f03ad17e505aeef6f6788d9811a671950`.
Keep source trajectories and derived data local.

In the project directory, reuse the existing replay converter (launcher Python needs Jinja2):

```bash
python convert_official_sft.py \
  --appworld-python /opt/conda/envs/appworld/bin/python \
  --appworld-root /home/notebook/data/group/decision_model/code/appworld \
  --experiment react_gpt4o_train \
  --output-dir sft_data/official_gpt4o_v011
```

Only currently installed Train IDs are read. Every trajectory is replayed; only fresh
full-evaluation successes are exported. Real replay against your SDK remains necessary.

## 3. Prepare Slime data in the training environment

```bash
python prepare_slime_sft.py \
  --model /home/notebook/data/group/LLMs_data/Qwen3-4B-Instruct-2507 \
  --data sft_data/official_gpt4o_v011/train.jsonl \
  --output sft_data/official_gpt4o_v011/slime_train.jsonl \
  --max-length 8192
```

One example is a full inference history plus one new assistant code completion.
Tokenization uses the same `enable_thinking=False` template as inference and checks
that the inference prompt is an exact token prefix of the completed chat. Only the
last code completion and its end marker are supervised. Few-shot answers, earlier
assistant turns, task text and execution output are context, with zero loss. Overlength
examples are skipped whole, never truncated. Statistics are saved alongside the JSONL.

The custom rollout uses pretokenized `metadata.tokens` and `response_length`, checks the
model tokenizer/config signature, and returns Slime `Sample` objects with a response-only
mask. Slime constructs the causal full-sequence mask. The JSON `prompt` field is just
an ID for the data loader; model inputs come from the preserved token IDs, not this ID.
Do not enable `--apply-chat-template` or replace this adapter with the generic multi-turn
SFT rollout: that could reformat the already-tokenized sequence or supervise historical
assistant turns. No environment calls, generation, reward model, PPO or GAE run during SFT.

## 4. Train (single-node)

From the AppWorld agent repository:

```bash
export SLIME_DIR=/root/slime
export MEGATRON_DIR=/root/Megatron-LM
export SLIME_PYTHON=python  # Python in the activated Slime environment

NUM_GPUS=2 TP_SIZE=2 GLOBAL_BATCH_SIZE=16 bash run_sft_slime.sh
```

Default model/data paths match the commands above. Override `MODEL_PATH`, `DATA_PATH`,
`OUTPUT_DIR`, `NUM_GPUS`, `TP_SIZE`, `GLOBAL_BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, or
`SAVE_INTERVAL`, or `MAX_LENGTH` (must cover every prepared sample). Paths are visible to the Ray workers on the same host. Use absolute
paths for overrides. Launch without an unrelated Ray cluster (`unset RAY_ADDRESS` if
necessary); Ray can initialize locally. This script never kills other processes or
stops shared Ray clusters. It supports a fresh run into an empty output directory.

Defaults: full-parameter BF16, one epoch, learning rate 1e-5, distributed optimizer,
full activation recomputation, microbatch 1, native per-token SFT loss. RoPE is read
from your model's config instead of using the Qwen3 base model's default. The script
validates the Qwen3-4B architecture and visible GPU count before launch.

DP size is `NUM_GPUS / TP_SIZE`; the global batch must be divisible by DP size.
Unlike the old HF script, Slime takes a global batch directly; do not multiply it
by GPU count. Slime calculates batches per epoch as `floor(samples/global_batch)`;
the dataset cursor can cross epoch boundaries in later epochs. Use a smaller batch
or one dividing the sample count if the tail matters, and never set batch larger
than the prepared dataset.

The default single GPU configuration is intended for a high-memory GPU (start with
80 GB and measure). Multiple GPUs with TP distribute the model; DP with distributed
optimizer shards optimizer state. Exact memory depends on the CUDA kernels, lengths,
parallel configuration and checkpoint gathering. This is not a promise that 24 GB
GPUs can run full tuning. To reduce sequence length, regenerate data with a lower
`--max-length`, rather than editing or truncating token IDs.

Megatron checkpoints are saved under `OUTPUT_DIR/checkpoints`. `--save-hf` writes the
latest saved standalone HF model under `OUTPUT_DIR/hf`; use that directory as `--model`
for `run_agent.py`. No PEFT merging or external checkpoint conversion is required for
the pinned Slime revision. Run Dev evaluation in the existing AppWorld/LLM setup.

## Validation boundary

CPU tests check prompt/answer masking, adapter Sample fields, overlength rejection,
and tokenizer mismatch rejection. CLI help and shell syntax are checked. GPU Megatron
execution, Ray allocation, checkpoint loading/export and actual official-data replay
have **not** been run on the development Mac. The upstream pin is an integration
contract, not a claim of end-to-end GPU validation.

## Length, task and completion-action diagnostics

`prepare_slime_sft.py` now writes four local reports beside the output. Existing output
or report files are not overwritten; choose a new output basename to rerun:

```bash
python prepare_slime_sft.py \
  --model /home/notebook/data/group/LLMs_data/Qwen3-4B-Instruct-2507 \
  --data sft_data/official_gpt4o_v011/train.jsonl \
  --output sft_data/official_gpt4o_v011/slime_train_diagnosed.jsonl \
  --max-length 8192 \
  --anomaly-threshold 1000000
```

- `slime_train_diagnosed.stats.json`: total/prompt/completion token distributions for
  all input and retained samples (min/max/mean, interpolated P50/P75/P90/P95/P99,
  nonoverlapping length buckets); task coverage; completion-action retention; and
  the 20 longest samples with source line numbers. All lengths include the actual
  chat template; prompt includes history, and completion includes its end marker.
- `slime_train_diagnosed.lengths.csv`: one row per input example with lengths,
  task ID, step, retained flag, action presence and source line. Use this to compare
  8K/16K thresholds without conflating sample count with unique tasks.
- `slime_train_diagnosed.tasks.csv`: input/retained sample and completion-action
  counts, retention rates, and maximum length for **every input task**, including
  tasks with no retained samples. A null rate means no applicable denominator.
- `slime_train_diagnosed.anomalies.jsonl`: every example with total length at least
  the anomaly threshold. Records task ID, step, source line, source trajectory hash
  when available, each message's role/character count/independently encoded content
  token count, five largest messages, and the preceding assistant message index.

The default threshold is one million total tokens and is independent of the 8K/16K
training cutoff. Repeated long histories can create multiple anomaly rows from the
same task; anomaly count is not a count of independent root causes. Message token
counts are diagnostic content-only measurements, not an additive decomposition of
the full templated sequence. Source lines are one-based; message indices are zero-based.

To inspect a location without dumping a huge history:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('sft_data/official_gpt4o_v011/slime_train_diagnosed.anomalies.jsonl')
for line in p.open():
    r = json.loads(line)
    print(r['task_id'], 'step', r['step'], 'source_line', r['source_line'],
          'total_tokens', r['total_tokens'])
    for m in r['largest_messages'][:3]:
        print(' ', m)
PY
```

Locate the matching row in the original `train.jsonl`, then use its message index.
For a large user observation, `preceding_assistant_index` points to the code that
preceded that observation; this is a debugging clue, not proof of the root cause.
By default no raw text is copied to diagnostic reports. Optional
`--diagnostic-preview-chars 160` adds bounded message previews for local inspection.
These may contain simulated account data; keep derived reports local like the dataset.

Completion actions are detected with Python AST in the **current target code**:
exact `apis.supervisor.complete_task(...)` calls count, comments and strings do not.
Historical/few-shot calls do not enter the denominator. Sample retention means target
samples containing a call kept / such samples read. Call retention counts actual
syntactic call occurrences, and task retention counts tasks with at least one retained
action / tasks with at least one input action. Calls with explicit `status="fail"`
are broken out separately; other/dynamic status values and `**kwargs` are identified.
Aliases are not resolved, and a syntactic call (e.g. inside an unexecuted branch) is
not proof it ran or that the task succeeded.

Reports are also saved when all examples are overlength; the command exits nonzero
and leaves no training JSONL. The additional normal length statistics reuse the main
encoding pass. Only anomaly messages are individually re-encoded for localization.
A tokenizer warning before filtering can still appear for million-token input; these
examples are measured and then skipped, never sent to the training model by preparation.

## Build complete_8k / complete_16k / complete_32k

Use the **original replay-verified `train.jsonl`**, not the per-sample filtered Slime
JSONL. This preserves the denominator and lets the script reject whole trajectories.

```bash
python build_complete_sft.py \
  --model /home/notebook/data/group/LLMs_data/Qwen3-4B-Instruct-2507 \
  --data sft_data/official_gpt4o_v011/train.jsonl \
  --diagnostics-stats sft_data/official_gpt4o_v011/slime_train_diagnosed.stats.json \
  --output-dir sft_data/complete_versions
```

The optional `--diagnostics-stats` reuses the sibling `.lengths.csv`. It checks the
original input SHA256, tokenizer signature, task IDs, steps and line coverage.
Selected samples are re-tokenized and checked against cached lengths; excluded
million-token histories need not be tokenized again. Without this option, the first
pass measures every input sample. Use the same local tokenizer/model files as in
preprocessing. A cache mismatch aborts rather than silently using stale statistics.

Whole-task rules, applied independently at inclusive limits 8192, 16384 and 32768:

1. Any sample with total tokens **>= 1,000,000** excludes its task from all versions
   (configurable using `--anomaly-threshold`). Even early short samples are removed.
2. Steps must start at 1 and be contiguous without duplicates. Each later history
   must contain exactly the prior history, prior target assistant code and one user
   observation. Missing/mixed histories reject the whole task.
3. The final target code must contain an AST `apis.supervisor.complete_task(...)`
   call. This checks for an available ending, not runtime success; the input is
   required to come from the success-replay converter. Aliased calls are not resolved.
4. **Every** sample of the task must fit the version's length limit. No sample or
   observation is dropped, truncated or compressed to make a trajectory fit.

The task sets are nested: complete_8k is a subset of complete_16k, which is a subset
of complete_32k. The output directory must not already exist. Conversion stages all
files and publishes the directory only after successful completion.

```text
complete_versions/
  summary.json           # versions, task IDs, source/model signatures, anomaly task locations
  summary.csv            # tasks/samples/complete_task sample and call counts/supervised tokens
  task_audit.csv         # every task's retained/rejected status and reasons, per version
  complete_8k/
    train.jsonl          # unchanged source examples, full retained trajectories
    slime_train.jsonl    # tokens + final-completion-only mask metadata, ready for Slime
    stats.json
  complete_16k/          # same three files
  complete_32k/          # same three files
```

`complete_task_calls` counts syntactic calls in target code, `complete_task_samples`
counts examples containing them; neither counts historical/few-shot calls. A version
can be empty; its files and zero-count report are still generated, and it cannot be
used for training. Counts are measured from your actual data, not hardcoded to 26/47.
The script does not independently re-evaluate task success or prove that the final
call executed. Do not use arbitrary incomplete/failed rollout logs as input.

For example, train the 16K version directly without another preparation pass:

```bash
MAX_LENGTH=16384 \
DATA_PATH="$PWD/sft_data/complete_versions/complete_16k/slime_train.jsonl" \
OUTPUT_DIR="$PWD/sft_outputs/qwen3_4b_complete_16k" \
NUM_GPUS=2 TP_SIZE=2 GLOBAL_BATCH_SIZE=16 \
bash run_sft_full.sh
```

Use 8192/32768 and the corresponding directory for the other versions. Compare the
three `summary.csv` rows before allocating longer-context training; longer sequences
can require more GPU memory. Keep all variants of a task/scenario together if you
subsequently construct a holdout; these versions are training subsets, not new splits.
