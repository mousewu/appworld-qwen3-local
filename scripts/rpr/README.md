# RPR staged launch scripts

Each numbered script owns one reproducible stage of the AppWorld Terminal-RSI
experiment. Shared paths and hyperparameters live in `config.sh`; keep machine
specific values in the ignored `config.local.sh`.

## Configure

```bash
cd /home/notebook/data/group/decision_model/code/appworld-qwen3-local
cp scripts/rpr/config.local.example.sh scripts/rpr/config.local.sh
vim scripts/rpr/config.local.sh
```

At minimum set `SFT_CKPT_ROOT` or `SFT_ACTOR_ITER`. Change
`EXPERIMENT_NAME` for every independent run. Outputs are isolated under
`rpr_experiments/<EXPERIMENT_NAME>/` and stage scripts refuse to overwrite
existing artifacts.

## Run one stage at a time

```bash
bash scripts/rpr/00_preflight.sh
bash scripts/rpr/01_calibrate_rewards.sh
bash scripts/rpr/02_prepare_warmup_load.sh
bash scripts/rpr/03_dry_run_warmup.sh
bash scripts/rpr/04_run_warmup.sh
bash scripts/rpr/05_prepare_rpr1_load.sh
bash scripts/rpr/06_run_rpr1.sh
bash scripts/rpr/07_build_rsr2_data.sh
bash scripts/rpr/08_run_rsr2_sft.sh
bash scripts/rpr/09_prepare_rpr2_load.sh
bash scripts/rpr/10_run_rpr2.sh
bash scripts/rpr/11_status.sh
```

The preparation stages resolve the exact iteration recorded by each upstream
`latest_checkpointed_iteration.txt`; they do not silently select an unrelated
directory. `07_build_rsr2_data.sh` fresh-replays successful RPR-1 trajectories
before preparing Slime SFT rows. `09_prepare_rpr2_load.sh` deliberately pairs
the RSR-2 Actor with the unchanged RPR-1 Critic.

## Run the complete pipeline

```bash
bash scripts/rpr/run_all.sh
```

This is intended for a fresh `EXPERIMENT_NAME`. If a long stage is interrupted,
inspect it with `11_status.sh` and restart the appropriate stage with a new
experiment name or a deliberately prepared checkpoint root; existing results
are never deleted automatically.

For a four-GPU smoke test, copy the commented smoke overrides from
`config.local.example.sh`. For the eight-GPU topology set:

```bash
NUM_GPUS=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

Topology-aware defaults then become batch 32, `MAX_ENVS=48`, queue capacity
64 and 48 in-flight groups. Explicit values in `config.local.sh` still win.
