# AppWorld × Qwen3-4B Terminal-RSI RPR

这是一条与现有 GRPO 并存的 actor–critic PPO 路径。它使用 Terminal-RSI 仓库已经实现的
Critic、TB-GAE、跨 chunk continuation、value loss 和 actor/critic 双 checkpoint，不修改
Slime 源码；项目插件只负责本地 AppWorld rollout、FDR/SPR 和监控。

默认 4 卡拓扑：GPU 0–1 放 TP2 Actor 与 Critic，GPU 2–3 各运行一个 TP1 rollout engine。
Actor/Critic optimizer 使用 CPU offload。`run_rpr_8gpu.sh` 保持两张训练卡并将 rollout 扩到六张。

完整实验也可以使用 [`scripts/rpr/`](scripts/rpr/README.md) 中的编号脚本逐阶段运行。公共路径和
超参数集中在 `scripts/rpr/config.sh`，机器私有配置放在不会提交的 `config.local.sh`；
`run_all.sh` 可从预检一直串联到 RPR-2。

## 1. 环境和运行时

RPR 需要 Terminal-RSI 自带的 Slime 扩展，普通 Slime 的 GRPO 功能不能代替独立 Critic 和
TB-GAE。设置路径和两个 Python 环境：

```bash
export TERMINAL_RSI_DIR=/home/notebook/data/group/decision_model/code/Terminal-RSI
export SLIME_PYTHON=/opt/conda/bin/python

MODEL=/home/notebook/data/group/LLMs_data/Qwen3-4B-Instruct-2507
APPWORLD_PYTHON=/opt/conda/envs/appworld/bin/python
APPWORLD_ROOT=/home/notebook/data/group/decision_model/code/appworld
```

运行时预检：

```bash
$SLIME_PYTHON -c 'import torch, ray, megatron.core, transformer_engine, sglang; print(torch.cuda.device_count())'
test -f "$TERMINAL_RSI_DIR/slime_plugins/harbor/tb_gae.py"
```

项目不安装或改写 Slime/Megatron。`PYTHONPATH` 将本项目兼容插件放在前面，并通过 namespace
继续使用 Terminal-RSI 的 `slime_plugins.harbor.tb_gae` 和 `train_loss`。项目的同名 rollout 入口
在不改变 TB-GAE 校验路径的前提下，提供完整任务组的有界持续供给。

## 2. 冻结 AppWorld 奖励尺度

在训练前仅运行一次。脚本重置每个 Train 任务并读取 verifier 的检查总数，以任务模板 ID 为
family，写出 family 与全局 check-count p90。文件之后作为实验输入，不在训练中更新。

```bash
$SLIME_PYTHON calibrate_rpr_rewards.py \
  --appworld-python "$APPWORLD_PYTHON" \
  --appworld-root "$APPWORLD_ROOT" \
  --split train \
  --output rpr_data/train_reward_scales.json
```

RPR-1/Warm Critic 使用 FDR：有效检查报告为 `passed_checks / D_f`，无有效报告时退化为二元成功。
RPR-2 使用 SPR：`success + passed_checks / (passed_checks + D'_f)`。首轮跑通时 `D'_f` 使用同一份
冻结表；积累足够 RPR-1 轨迹后可以离线审计失败轨迹，再提供重新标定但仍冻结的 SPR 表。

## 3. Warm Critic

`--load-root` 是 SFT 的 Megatron checkpoint 根目录，里面必须有
`latest_checkpointed_iteration.txt`。它不能有同名 `_critic` sibling；Critic 从 Actor backbone
复制并新建 value head。Warm-up 使用同步 `train.py`，确保 Actor 完全冻结时最后一个 Critic
checkpoint 仍会保存。

先 dry-run 检查完整命令：

```bash
bash run_rpr_4gpu.sh \
  --phase warmup \
  --model "$MODEL" \
  --load-root /path/to/sft/checkpoints \
  --terminal-rsi-dir "$TERMINAL_RSI_DIR" \
  --appworld-python "$APPWORLD_PYTHON" \
  --appworld-root "$APPWORLD_ROOT" \
  --reward-scales rpr_data/train_reward_scales.json \
  --output-dir rpr_runs/warmup_dry \
  --updates 6 --dry-run
```

正式运行时换一个全新输出目录并删除 `--dry-run`。Warm-up 的 `--updates` 应至少覆盖一次任务池：
`ceil(train_tasks / batch_size)`；默认 batch=16，AppWorld 90 个 Train 任务对应 6 步。整个阶段
`num_critic_only_steps == updates`，只有 Critic 更新。输出在
`rpr_runs/warmup/checkpoints_critic/iter_*`。

## 4. RPR-1

把原 SFT Actor iteration 与 Warm Critic iteration 组合成 weight-only load root：

```bash
$SLIME_PYTHON prepare_rpr_load_root.py \
  --actor-iter /path/to/sft/checkpoints/iter_XXXXXXX \
  --critic-iter /path/to/rpr_runs/warmup/checkpoints_critic/iter_XXXXXXX \
  --output rpr_loadroots/rpr1
```

启动短跑。前 4 个 update 只校准继承的 Critic，之后 Actor/Critic 联合 PPO：

```bash
bash run_rpr_4gpu.sh \
  --phase rpr1 \
  --model "$MODEL" \
  --load-root "$PWD/rpr_loadroots/rpr1" \
  --terminal-rsi-dir "$TERMINAL_RSI_DIR" \
  --appworld-python "$APPWORLD_PYTHON" \
  --appworld-root "$APPWORLD_ROOT" \
  --reward-scales rpr_data/train_reward_scales.json \
  --output-dir rpr_runs/rpr1_short \
  --updates 12 --critic-only-steps 4 \
  --swanlab-mode cloud
```

重要默认值：

- Actor/Critic LR：`5e-7 / 1.5e-5`；
- PPO clip low/high：`0.3 / 5.0`，value clip：`4.0`；
- `gamma_turn=lambda_turn=1.0`；
- rollout context 16K，训练 chunk cap 8K，因此超过 8K 的完整轨迹会实际验证跨 chunk TB-GAE；
- 每任务一次独立 rollout，不做 GRPO 组内归一化；
- 只训练动作 token，观察 token 只作为条件；
- 每个有限时域 AppWorld episode 在最后动作处终止；单个 execute 超时重放耗尽时丢弃整个任务组并
  补充新任务，连续 3 个任务组都耗尽才终止训练，基础设施错误不会被伪装成奖励 0；
- 相同动作与异常默认重复 2 次便提前进入最终评估；无限 `while True` 异常重试代码不会提交给环境；
- 4 卡默认最多保留 32 个 pending/active/ready 任务组、并发 24 个，策略滞后最多 1 个 actor update；
  abort 最多重试 3 次，之后换一个新任务组。

## 5. 生成并训练 RSR-2 Actor

从 RPR-1 的逐步落盘轨迹中选择每个任务最短的一条成功轨迹，重新创建 AppWorld 世界、逐动作
重放并再次 `evaluate()`。只有仍然成功的轨迹会成为 SFT 数据：

```bash
$SLIME_PYTHON build_rsr2_sft.py \
  --episodes-dir rpr_runs/rpr1_short/episodes \
  --appworld-python "$APPWORLD_PYTHON" \
  --appworld-root "$APPWORLD_ROOT" \
  --output-dir rpr_data/rsr2_verified

$SLIME_PYTHON prepare_slime_sft.py \
  --model "$MODEL" \
  --data rpr_data/rsr2_verified/train.jsonl \
  --output rpr_data/rsr2_verified/slime_train.jsonl \
  --max-length 16384
```

Actor-only SFT 从 RPR-1 Actor checkpoint 初始化；RPR-1 Critic 不参与这一步：

```bash
SLIME_DIR="$TERMINAL_RSI_DIR" MODEL_PATH="$MODEL" \
INIT_CHECKPOINT=/path/to/rpr1/checkpoints \
DATA_PATH="$PWD/rpr_data/rsr2_verified/slime_train.jsonl" \
OUTPUT_DIR="$PWD/rpr_runs/rsr2_sft" \
NUM_GPUS=2 TP_SIZE=2 GLOBAL_BATCH_SIZE=8 MAX_LENGTH=16384 EPOCHS=1 \
bash run_sft_slime.sh
```

这一步复现“采用并重新验证成功轨迹”的 Actor 更新，不包含论文未开源的完整 self-rewrite 数据生成器。

## 6. RPR-2

组合 RSR-2 Actor 和原样保留的 RPR-1 Critic：

```bash
$SLIME_PYTHON prepare_rpr_load_root.py \
  --actor-iter /path/to/rsr2_sft/checkpoints/iter_XXXXXXX \
  --critic-iter /path/to/rpr1/checkpoints_critic/iter_XXXXXXX \
  --output rpr_loadroots/rpr2

bash run_rpr_4gpu.sh \
  --phase rpr2 \
  --model "$MODEL" \
  --load-root "$PWD/rpr_loadroots/rpr2" \
  --terminal-rsi-dir "$TERMINAL_RSI_DIR" \
  --appworld-python "$APPWORLD_PYTHON" \
  --appworld-root "$APPWORLD_ROOT" \
  --reward-scales rpr_data/train_reward_scales.json \
  --output-dir rpr_runs/rpr2_short \
  --updates 12 --critic-only-steps 4 \
  --swanlab-mode cloud
```

RPR-2 Actor LR 自动降为 `1e-7`，Critic LR 保持 `1.5e-5`，奖励切换为 SPR。

## 7. 迁移到 8 GPU

流程和 checkpoint 完全不变，只替换启动器：

```bash
bash run_rpr_8gpu.sh ...相同参数...
```

默认布局为 2 张 Actor/Critic TP2 + 6 张 rollout TP1，batch 从 16 提高到 32，`MAX_ENVS=48`。
第一次迁移建议仍运行 12 步，比较 `perf/rollout_time`、训练等待、GPU 平均利用率、CPU RAM 峰值和
AppWorld timeout，再决定是否增加 batch/环境并发。不要同时改变 LR、奖励和资源拓扑。

## 8. 验证与故障定位

```bash
$SLIME_PYTHON -m unittest test_rpr.py test_grpo.py test_grpo_recovery.py
```

SwanLab 会同步 Slime 原生 actor/critic/value/clip 指标、`appworld_rpr/*`、GPU/CPU/内存、环境事件、
完整日志和退出分类。每次运行还保存不可变的 `reward_scales.json`、实际角色 LR 配置、TB-GAE 配置
和完整命令。若进程异常结束，先检查 `exit_diagnosis.json`；如果训练在参数解析阶段终止，检查
`training.log` 中 Terminal-RSI TB-GAE preflight 的具体契约错误。

环境容错重点观察：

- `appworld_rpr/async/aborted_resampled_total` 与 `aborted_skipped_total`；
- `appworld_rpr/async/execute_timeout_skipped_total`、跳过率及 `termination/repeated_execution_error`；
- `appworld_rpr/unbounded_retry_rejections`。

可用 `RPR_GROUP_CAPACITY`、`RPR_INFLIGHT_GROUPS`、`RPR_MAX_POLICY_LAG`、
`RPR_GROUP_RETRIES`、`RPR_MAX_CONSECUTIVE_TIMEOUTS` 和
`REPEATED_EXECUTION_ERROR_LIMIT` 覆盖启动脚本默认值。
