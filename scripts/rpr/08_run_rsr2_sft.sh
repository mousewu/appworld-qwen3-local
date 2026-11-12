#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

print_stage "[08] Train the RSR-2 Actor from the RPR-1 Actor checkpoint"
require_file "$RSR2_SLIME_DATA"
require_file "$RPR1_RUN_DIR/checkpoints/latest_checkpointed_iteration.txt"
require_absent "$RSR2_SFT_RUN_DIR"

SAMPLES="$(wc -l < "$RSR2_SLIME_DATA" | tr -d ' ')"
[[ "$SAMPLES" -gt 0 ]] || die "RSR-2 dataset is empty"
(( RSR2_SFT_NUM_GPUS > 0 && RSR2_SFT_TP_SIZE > 0 )) || die "RSR-2 GPU/TP sizes must be positive"
(( RSR2_SFT_NUM_GPUS % RSR2_SFT_TP_SIZE == 0 )) || die "RSR-2 GPU count must be divisible by TP size"
SFT_DP=$((RSR2_SFT_NUM_GPUS / RSR2_SFT_TP_SIZE))
SFT_BATCH="$RSR2_SFT_GLOBAL_BATCH_SIZE"
(( SAMPLES < SFT_BATCH )) && SFT_BATCH="$SAMPLES"
SFT_BATCH=$((SFT_BATCH - SFT_BATCH % SFT_DP))
(( SFT_BATCH > 0 )) || die "RSR-2 samples cannot fill one data-parallel batch"
if [[ "$SFT_BATCH" != "$RSR2_SFT_GLOBAL_BATCH_SIZE" ]]; then
  echo "Adjusted RSR-2 SFT global batch for sample count/DP geometry: $SFT_BATCH"
fi

SLIME_DIR="$TERMINAL_RSI_DIR" \
SLIME_PYTHON="$SLIME_PYTHON" \
MODEL_PATH="$MODEL" \
INIT_CHECKPOINT="$RPR1_RUN_DIR/checkpoints" \
DATA_PATH="$RSR2_SLIME_DATA" \
OUTPUT_DIR="$RSR2_SFT_RUN_DIR" \
NUM_GPUS="$RSR2_SFT_NUM_GPUS" \
TP_SIZE="$RSR2_SFT_TP_SIZE" \
GLOBAL_BATCH_SIZE="$SFT_BATCH" \
MAX_LENGTH="$RSR2_MAX_LENGTH" \
EPOCHS="$RSR2_SFT_EPOCHS" \
LEARNING_RATE="$RSR2_SFT_LR" \
SAVE_INTERVAL="$RSR2_SFT_SAVE_INTERVAL" \
SAVE_HF="$RSR2_SFT_SAVE_HF" \
bash "$PROJECT_DIR/run_sft_slime.sh"

latest_checkpoint_iter "$RSR2_SFT_RUN_DIR/checkpoints"
