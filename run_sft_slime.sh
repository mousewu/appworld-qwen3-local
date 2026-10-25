#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
SLIME_PYTHON="${SLIME_PYTHON:-python}"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
# Source directories are optional when the modules are already installed.
for source_dir in "${SLIME_DIR:-}" "${MEGATRON_DIR:-}"; do
  if [[ -n "$source_dir" ]]; then
    source_dir="$(cd "$source_dir" && pwd)"
    export PYTHONPATH="$source_dir:$PYTHONPATH"
  fi
done
SLIME_TRAIN_SCRIPT="$("$SLIME_PYTHON" "$PROJECT_DIR/slime_runtime.py")"
MODEL_PATH="${MODEL_PATH:-/home/notebook/data/group/LLMs_data/Qwen3-4B-Instruct-2507}"
DATA_PATH="${DATA_PATH:-$PROJECT_DIR/sft_data/official_gpt4o_v011/slime_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/sft_outputs/qwen3_4b_slime}"
NUM_GPUS="${NUM_GPUS:-1}"
TP_SIZE="${TP_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
export PYTHONUNBUFFERED=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Validate paths and geometry before allocating Ray actors. Read RoPE from the actual model.
MODEL_LIMITS="$("$SLIME_PYTHON" - "$MODEL_PATH" "$DATA_PATH" "$OUTPUT_DIR" "$NUM_GPUS" "$TP_SIZE" "$GLOBAL_BATCH_SIZE" "$MAX_LENGTH" <<'PY'
import json,sys
from pathlib import Path
import torch, ray, megatron.core, slime
model,data,output=map(Path,sys.argv[1:4])
gpus,tp,batch,max_length=map(int,sys.argv[4:])
if min(gpus,tp,batch) <= 0 or gpus % tp or batch % (gpus//tp):
    raise SystemExit('Require positive sizes, NUM_GPUS divisible by TP_SIZE, and batch divisible by DP size')
if torch.cuda.device_count() < gpus or not torch.cuda.is_bf16_supported():
    raise SystemExit('Insufficient visible CUDA GPUs or no BF16 support')
if not data.is_file(): raise SystemExit(f'Missing dataset: {data}')
if output.exists() and any(output.iterdir()): raise SystemExit('Choose an empty OUTPUT_DIR')
c=json.loads((model/'config.json').read_text())
expected={'model_type':'qwen3','num_hidden_layers':36,'hidden_size':2560,
          'intermediate_size':9728,'num_attention_heads':32,'num_key_value_heads':8,
          'head_dim':128,'vocab_size':151936}
for k,v in expected.items():
    if c.get(k)!=v: raise SystemExit(f'Unsupported Qwen3-4B config: {k}={c.get(k)!r}, expected {v!r}')
rows=0
for line in data.open():
    if not line.strip(): continue
    sample=json.loads(line)
    if len(sample['metadata']['tokens']) > max_length:
        raise SystemExit('Dataset exceeds MAX_LENGTH; regenerate with prepare_slime_sft.py')
    rows+=1
if rows < batch: raise SystemExit('Dataset smaller than global batch; reduce GLOBAL_BATCH_SIZE')
rope=c.get('rope_theta',c.get('rope_parameters',{}).get('rope_theta'))
if rope is None: raise SystemExit('Missing rope_theta')
max_positions=c['max_position_embeddings']
if not 0 < max_length <= max_positions: raise SystemExit('MAX_LENGTH exceeds model context')
print(rope, max_positions)
PY
)"
read -r MODEL_ARGS_ROTARY_BASE MODEL_MAX_POSITIONS <<< "$MODEL_LIMITS"
export MODEL_ARGS_ROTARY_BASE
source "$PROJECT_DIR/configs/qwen3_4b_slime.sh"
args=(
  --actor-num-nodes 1 --actor-num-gpus-per-node "$NUM_GPUS"
  --hf-checkpoint "$MODEL_PATH" --load "${INIT_CHECKPOINT:-$MODEL_PATH}"
  --save "$OUTPUT_DIR/checkpoints"
  --save-interval "${SAVE_INTERVAL:-100}"
  --rollout-function-path slime_sft.generate_rollout
  --prompt-data "$DATA_PATH" --input-key prompt --metadata-key metadata
  --rollout-shuffle --num-epoch "${EPOCHS:-1}"
  --rollout-batch-size "$GLOBAL_BATCH_SIZE" --global-batch-size "$GLOBAL_BATCH_SIZE"
  --n-samples-per-prompt 1
  --loss-type sft_loss --calculate-per-token-loss
  --disable-compute-advantages-and-returns --debug-train-only
  --tensor-model-parallel-size "$TP_SIZE" --pipeline-model-parallel-size 1
  --context-parallel-size 1 --expert-model-parallel-size 1
  --seq-length "$MAX_LENGTH" --max-position-embeddings "$MODEL_MAX_POSITIONS"
  --expert-tensor-parallel-size 1 --micro-batch-size 1
  --use-distributed-optimizer --bf16
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --optimizer adam --lr "${LEARNING_RATE:-1e-5}"
  --lr-decay-style cosine --min-lr 1e-6 --lr-warmup-fraction 0.03
  --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.95
  --attention-dropout 0.0 --hidden-dropout 0.0
  --attention-backend flash --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
)
if [[ "${SAVE_HF:-1}" == "1" ]]; then args+=(--save-hf "$OUTPUT_DIR/hf"); fi
if (( TP_SIZE > 1 )); then args+=(--sequence-parallel); fi
# Slime/Ray can initialize a local cluster. No shared Ray/Python processes are killed.
cd "$(dirname "$SLIME_TRAIN_SCRIPT")"
exec "$SLIME_PYTHON" "$SLIME_TRAIN_SCRIPT" "${MODEL_ARGS[@]}" "${args[@]}" "$@"
