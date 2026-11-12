#!/usr/bin/env bash
# Copy to config.local.sh and edit machine/experiment-specific values.

SFT_CKPT_ROOT=/path/to/sft_outputs/qwen3_4b_slime/checkpoints
# Alternatively pin one exact iteration:
# SFT_ACTOR_ITER=/path/to/sft_outputs/qwen3_4b_slime/checkpoints/iter_0000123

EXPERIMENT_NAME=qwen3_4b_rpr_01
SWANLAB_MODE=cloud

# Four-GPU smoke test example:
# WARMUP_UPDATES=1
# RPR1_UPDATES=2
# RPR1_CRITIC_ONLY_STEPS=1
# RPR2_UPDATES=2
# RPR2_CRITIC_ONLY_STEPS=1
# RPR_BATCH_SIZE=4
# MAX_ENVS=4
# SERVER_CONCURRENCY=4
# RPR_GROUP_CAPACITY=8
# RPR_INFLIGHT_GROUPS=4
