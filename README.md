> **四卡异步 GRPO：见 [ASYNC_GRPO.md](ASYNC_GRPO.md)，入口 `bash run_grpo_async.sh`；包含 GPU 利用率与内存统计。**

> **同步 GRPO：见 [GRPO.md](GRPO.md)，入口 `bash run_grpo.sh`，支持项目端 SwanLab。**

> **SFT 当前入口已切换到 Slime 全参训练：见 [SLIME_SFT.md](SLIME_SFT.md)。**
> 下方 HF Trainer/LoRA 章节保留作历史替代方案。官方 0.1.1 输出包的 Train 目录为 `react_gpt4o_train`；旧版默认下载命令可能拿不到 Train 轨迹。

# 本地 Qwen3-4B 运行 AppWorld

基于 [AppWorld 官方 minimal_agent.ipynb](https://github.com/StonyBrookNLP/appworld/blob/main/notebooks/minimal_agent.ipynb) 实现的命令行 ReAct Agent。无需 Docker、真实 App 账户或云端模型 API。

官方 notebook 的任务提示、示范、逐轮执行和结束检测流程保留。上游示范中 `passwords` 未赋值及错误列表推导式已修正。提示来源的固定 commit 和原 notebook 校验值在 `upstream.json`。任务内容只填入既定消息，不重新解析任务里的角色标记。

## 1. 安装 AppWorld

推荐 Linux GPU 主机；环境执行不使用 GPU。模型与环境可以装在不同 Python 环境。

```bash
conda create -n appworld-local python=3.11 -y
conda activate appworld-local
pip install appworld
appworld install
mkdir -p "$HOME/appworld-data"
cd "$HOME/appworld-data"
export APPWORLD_ROOT="$PWD"
appworld download data
appworld verify tasks
python -c 'import sys; print(sys.executable)'
pip freeze > appworld-environment-lock.txt
```

记下打印的 Python 路径，后续传给 `--appworld-python`。`--appworld-root` 必须指向包含 `data/` 的目录。首次验证通过后固定 AppWorld 包和数据版本。安装需下载依赖/数据；全部准备好后，执行任务不访问真实 Amazon/Spotify。

AppWorld 会执行生成的 Python。子进程隔离用于避免状态和依赖冲突，**不是安全沙箱**。内置静态/动态保护保持开启；在专用低权限账户或实验虚拟机中运行，避免环境进程持有私人文件、SSH 密钥或云服务凭据。

## 2. 方式 A：直接加载本地权重（默认）

进入本项目目录。先按显卡/CUDA 安装匹配的 PyTorch，然后安装：

```bash
pip install -r requirements.txt
python run_agent.py \
  --model /absolute/path/to/Qwen3-4B \
  --appworld-root "$HOME/appworld-data" \
  --appworld-python /absolute/path/to/envs/appworld-local/bin/python \
  --split train --limit 1 --device cuda
```

模型目录必须已经完整下载，模型与 tokenizer 都使用 `local_files_only=True`，不会自动从网络拉权重。Qwen3 默认关闭 thinking；使用真实聊天模板和实际输入 token 长度检查上下文。默认贪心解码，每轮最多 1024 个输出 token，最多 50 轮。没有可用 CUDA 时可用 `--device mps` / `cpu`，速度与兼容性需在目标机器确认。

编码流程先 `apply_chat_template(tokenize=False)`，再以 `add_special_tokens=False` 编码，避免重复添加特殊 token；统一将 HF 编码字典、Tokenizers `Encoding`、单元素 batch 转为整数 ID 列表后再构造 LongTensor。较新 Transformers 使用 `dtype`，旧版本保留 `torch_dtype`。运行时打印 Transformers 版本和 tokenizer 类名；错误会保留完整 traceback 便于定位。

BF16/FP16 的 4B 模型权重约 8GB，另需 KV cache 和推理临时内存；优先使用 16–24GB 或更大显存，长历史可继续增加占用。这里是推理，未加载 Critic 或训练优化器。

## 3. 方式 B：连接本机 vLLM / SGLang

这条路径只要求运行脚本的 Python 安装 `requirements-api.txt`；模型服务与 AppWorld 可以各用自己的环境。不需要安装 OpenAI SDK或申请 API Key。

例如，在具备 vLLM 的环境启动（具体版本需与 CUDA/PyTorch 匹配）：

```bash
vllm serve /absolute/path/to/Qwen3-4B \
  --served-model-name qwen3-4b \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 32768 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

然后在客户端环境运行：

```bash
pip install -r requirements-api.txt
python run_agent.py --backend api \
  --base-url http://127.0.0.1:8000/v1 \
  --served-model-name qwen3-4b \
  --appworld-root "$HOME/appworld-data" \
  --appworld-python /absolute/path/to/envs/appworld-local/bin/python \
  --split train --limit 1
```

也可以连接本机 SGLang 的兼容接口；`--served-model-name` 必须与服务实际暴露的模型名一致。客户端每次请求发送 `chat_template_kwargs.enable_thinking=false`。只接受 loopback URL；不调用外部模型服务。API 模式由服务器负责上下文限制，`--context-length` 只用于直接加载模式；服务返回上下文错误时记为 error，不静默裁历史。

## 4. 更多任务与输出

### 遍历全部 Train，每任务一次

在已配置的服务器上可直接执行：

```bash
bash run_train_all.sh
# 查看每轮代码与观察（日志始终写入文件，verbose 只影响控制台）：
bash run_train_all.sh --verbose
```

脚本中的默认路径匹配当前服务器；其他机器可通过 `LLM_PYTHON`、`MODEL_PATH`、`APPWORLD_PYTHON`、`APPWORLD_ROOT` 环境变量覆盖。它使用 `--split train --all-tasks`，加载实际任务清单，**不抽样、不重复任务**。官方原始 Train 约 105 个任务，不会把重复执行算成 500 个独立任务。

任一接入模式均支持 `--all-tasks`，与 `--limit` 互斥；旧的 `--limit 0` 也继续有效。模型权重只加载一次，各任务使用独立 AppWorld 子进程。运行中每个任务完成后保存统计，普通单任务异常记录后继续下一个任务；不支持从中断位置自动续跑。

### GPU 利用率与并发

`run_train_all.sh` 默认 **4 个任务并发、最大推理 batch=4**。主进程的线程只调度 episode；真正的 AppWorld 在各自子进程中运行，一进程一个 world，不在同一进程共享世界状态。模型只加载一份，Transformers 推理由单独的调度线程统一合批，不启动四份模型。

```bash
# 默认 4 路，先测这一档
bash run_train_all.sh
# 显存充足时尝试 8 个任务、最大 batch=8
CONCURRENCY=8 INFERENCE_BATCH_SIZE=8 bash run_train_all.sh
# 显存不足/需要串行对照
CONCURRENCY=1 INFERENCE_BATCH_SIZE=1 bash run_train_all.sh
```

手动运行支持 `--concurrency 4 --inference-batch-size 4 --batch-wait-ms 20`。命令行单独运行仍默认 concurrency=1，启动脚本才默认 4。调度器最多等待 20ms 收集已就绪的推理请求，不会为了凑满 batch 长时间阻塞。批量输入左侧 padding，并使用 attention mask；输出剔除批量 padding，原始输入 token 日志不含 padding。单条上下文超限不影响其他请求；显存 OOM 会记录当前受影响任务的错误，不自动降低配置重试。

Transformers 是普通微批处理，同一批会等待最长输出结束，不具备 vLLM 的连续批处理。更高吞吐可使用本机 vLLM/SGLang 服务，加 `--backend api --concurrency 8`；客户端并发请求，合批由服务完成，`--inference-batch-size` 不作用于 API 模式。

新增汇总：`wall_seconds`（从评测开始计算，不包含模型加载）、`episodes_per_minute`、`completion_tokens_per_second`（包含环境等待的端到端吞吐）。Transformers 并发还记录 `inference.average_batch_size/max_batch_size/batches/service_seconds`。原 `seconds` 是各 episode 用时之和，并发时不能当成墙钟时间。每任务日志含模型等待时间、执行等待时间及生成 token 数。

先比较 concurrency=1/4/8 的吞吐与显存，不以 GPU 百分比作为唯一目标。长上下文批量会明显增加 KV cache 和 padding 成本，4 路也不保证适合所有 GPU；OOM 时降低并发和推理 batch。采样模式的随机数消耗顺序会随调度改变；即便贪心解码，批处理浮点差异也可能改变结果，因此比较模型成绩时固定并发、后端和采样配置。并发下不要开启大量 verbose 控制台输出，以免日志阻塞。

### 反馈与统计

语法/格式错误明确标记 `CODE NOT EXECUTED`，包含出错行，并说明本轮没有执行任何 API，不代表接口或文档不存在。实际环境反馈标记 `CODE WAS SUBMITTED`，日志同时保存原始环境输出及 `feedback_kind`。不会自动修改或执行语法错误代码。

`summary.json` 新增：计划/已完成/独立任务数、成功数、成功率、评测完成数、基础设施错误数、终止原因、平均轮数、成功任务平均轮数、格式错误数/比例、环境执行错误数/比例、认证错误数、累计用时。格式错误比例以模型输出轮数为分母；执行错误比例以实际提交执行次数为分母。认证错误基于环境输出中的 HTTP 401/403 状态标记，不把模型自述算作认证错误。

`results.csv` 保存每任务对应指标，方便后续表格分析。汇总成功率包含出错任务；`completed=false` 表示尚未处理完计划清单。检查通过数保留在任务结果中，但不是任务完成百分比。当前不输出二项置信区间，也不将 Train 成功率称为泛化成绩。

```bash
# 先查看当前 split 的实际任务 ID（在 AppWorld 环境中）
python -c 'from appworld import load_task_ids; print(load_task_ids("train"))'

# 指定任务，可重复 --task-id；任务必须属于 --split
python run_agent.py --model /path/to/Qwen3-4B \
  --appworld-root /path/to/appworld-root \
  --appworld-python /path/to/appworld-python \
  --task-id YOUR_TRAIN_TASK_ID --verbose
```

`--limit 0` 跑全部选中任务；默认只跑 1 个 train 任务。`--temperature 0.7 --top-p 0.8` 开启采样。每次运行使用唯一 experiment 名，避免重复 task 覆盖 AppWorld 输出。任务按配置并发执行，每个任务一个独立子进程；汇总按原始任务顺序排列。

结果写到 `runs/qwen3_local_<time>_<id>/`：

- `config.json`：配置、任务清单、上游来源，不保存 API Key。
- `<task_id>.jsonl`：初始提示、原始模型输出、实际执行代码、观察、生成元数据和评测结果。
- `summary.json`：每任务是否宣告完成、验证是否成功、轮数、结束原因和错误；成功率分母包括出错任务，另列 error 数。
- `results.csv`：每任务结果与错误类别统计。

Transformers 路径保存 prompt/completion token IDs；API 路径保存服务 usage。**这不是 PPO 训练器**，未导出完整采样 log_probs 或 token loss_mask，不能直接作为精确 PPO rollout 使用。普通成功轨迹 SFT 可基于日志进一步转换；训练采样需要另加对齐与概率记录。

`task_completed()` 只决定何时停止交互；成功来自终局 `world.evaluate().success`。私有评测反馈不送回 Agent。代码语法/格式错误返回公开纠正信息并消耗一次轮次；环境返回的运行异常交给模型自行修复。模型输出截断不会执行语法不完整的代码。Transformers 上下文不足时停止并评估当前状态，不偷偷删历史。环境进程硬超时默认 180 秒，模型 API 超时默认 300 秒。

测试集只用于最终评测；不要用其逐任务日志做提示调整或训练。当前不计算 SGC；需要对完整 scenario 的全部任务执行官方聚合评测才能报告 SGC。

## 5. CPU 验证与限制

AppWorld SDK 不同版本的构造参数存在差异。Worker 根据实际签名传入支持的可选参数；不支持的参数不传入，保留该 SDK 默认行为。支持的执行保护选项显式设为 True，不会为兼容而设为 False；不支持的保护选项也不声称已开启。每个任务日志记录 Python、AppWorld 版本、类所在文件及实际参数。

升级代码后，可先不加载模型，只检查环境是否能够初始化、执行和评测：

```bash
python run_agent.py --check-env \
  --appworld-python /absolute/path/to/appworld/bin/python \
  --appworld-root /absolute/path/to/appworld-root
```

检查会打印 `2` 对应的执行输出和验证状态；它不解决任务，`success=false` 正常。初始化与评测不报异常才代表检查通过。

```bash
python -m unittest -v test_agent.py
python run_agent.py --help
```

测试通过假的 AppWorld 子进程检查：日志不污染通信、角色边界、代码格式处理、宣告完成不等于成功、上下文结束仍执行评测。测试不下载模型、不安装 AppWorld，也不等于真实 4B + AppWorld 端到端成功。

首次真实运行应先完成 `appworld verify tasks`，再运行一个 train 任务，最后扩到更多任务。Linux/CUDA 的真实模型推理、AppWorld 数据兼容性及模型成功率需要在目标机器验证。

## 来源

- 官方最小 Agent：https://github.com/StonyBrookNLP/appworld/blob/main/notebooks/minimal_agent.ipynb
- AppWorld：https://github.com/StonyBrookNLP/appworld
- Qwen3-4B：https://huggingface.co/Qwen/Qwen3-4B

`prompt_template.txt` 改编自 AppWorld 官方 notebook，版权归原作者，遵循上游 Apache-2.0 许可；保留 `upstream.json` 以追踪来源。

## Official GPT-4o Train → Qwen3 LoRA SFT

The converter uses the **executed** Python in official `logs/environment_io.md`, not
Markdown/prose from teacher messages. It reuses this project's `prompt_messages`
and execution feedback, emits raw Python (the current prompt asks for no fences),
and obtains fresh observations by replaying each complete trajectory. Only trajectories
whose fresh `world.evaluate().success` is true are exported. Successful trajectories
can contain intermediate execution failures: these are preserved to teach recovery.
Invalid Python rejects the whole trajectory instead of silently deleting dependent steps.
No evaluator answers or test reports are placed in model inputs.

Official source: <https://github.com/StonyBrookNLP/appworld#-downloading-our-experiment-outputs>.
Parsing uses the installed SDK's `AppWorld.parse_environment_io_log(file_path=...)`.
Older SDKs without that method produce explicit audit errors; upgrade to a compatible
AppWorld release rather than guessing the log format.

### 1. Download in the AppWorld environment

```bash
/opt/conda/envs/appworld/bin/appworld download experiment-outputs \
  --root /home/notebook/data/group/decision_model/code/appworld
```

This official download also includes Dev/Test outputs. The converter only enumerates
`load_task_ids('train')` from your installed version and opens those task paths.
Do not point `--experiment` at Dev/Test experiments. Missing Train logs are audited,
never replaced with another split. Old bundles may have a different task list or layout;
inspect `audit.jsonl` if nothing is exported. Source data is not included in this repo.

### 2. Replay and convert (no GPU needed)

Run from this repository, with Jinja2 installed in the launcher Python:

```bash
/opt/conda/bin/python convert_official_sft.py \
  --appworld-python /opt/conda/envs/appworld/bin/python \
  --appworld-root /home/notebook/data/group/decision_model/code/appworld \
  --output-dir sft_data/official_gpt4o
```

Default experiment: `react_gpt4o_train`.
Use `--experiment` only for another official **Train** experiment directory.
Replay outputs use a unique `sft_replay/...` experiment; original logs are read-only.
The output directory must be empty. Outputs:

- `train.jsonl`: one history → next-code example per executed turn, with task/scenario IDs.
- `audit.jsonl`: each Train task's missing/error/failure/success status, source hash,
  fresh evaluation, and installed runtime metadata.
- `summary.json`: counts, source, replay name, seed, and current prompt hash.

Replay validates success on the installed SDK, not the historical GPT-4o score.
Version differences can cause rejection. The converter does not invoke another model
or repair teacher logic. It tries every installed Train ID; audit errors are not successes.

### 3. Train in the LLM environment

```bash
/opt/conda/bin/python -m pip install -r requirements-sft.txt
bash run_sft_lora.sh
```

Defaults match the server paths used in this project. Override `LLM_PYTHON`, `MODEL_PATH`,
`DATA_PATH`, `OUTPUT_DIR`, `MERGE_OUTPUT`, `MAX_LENGTH`, `EPOCHS`, `BATCH_SIZE`, or
`GRAD_ACCUM` as needed. Training requires a CUDA GPU supporting BF16. Start with one
24–48 GB GPU, batch size 1, gradient checkpointing, LoRA rank 16, length 8192;
actual memory depends on sequence lengths and installed kernels. If memory is tight,
reduce `MAX_LENGTH` to 4096 (more examples will be skipped). This is BF16 LoRA, not QLoRA.
No AppWorld installation is required in the LLM environment.

Each example supervises **only the final assistant completion and its template end marker**.
All earlier assistant turns (including few-shot examples), user messages, and observations
have label `-100`. Qwen chat formatting uses `enable_thinking=False` just like inference.
The script checks that the inference prompt token IDs are an exact prefix of the full
training sequence. Overlength examples are skipped whole, with counts saved in
`data_stats.json`; no left/right truncation or packing is performed. Padding is masked.

The output folder contains the PEFT adapter. The shell script also exports a merged
standalone model to `sft_outputs/qwen3_4b_merged`; pass this directory as `--model` to
`run_agent.py` or set `MODEL_PATH` when using `run_train_all.sh`. Export needs additional
disk space for the full 4B model. For adapter-only training, call `train_lora.py` directly
without `--merge-output`.

Start with one epoch and compare against the base model on Dev; do not use Dev/Test
trajectories as SFT targets. Do not randomly split individual turns: if holding out
Train data, group all three variants of a scenario together. Training loss alone is
not a measure of task success. Local validation here covers mocks and conversion/masking
logic; real official bundle replay and CUDA training must be verified on your server.

## Full-parameter SFT

`train_full.py` shares the LoRA pipeline's tokenization, prompt format, completion-only
loss masking, collator, and overlength filtering. It makes **all model parameters
trainable**, uses no PEFT adapter, and saves a standalone model directly. Defaults:
BF16, gradient checkpointing, learning rate `1e-5`, one epoch, cosine schedule,
3% warmup, weight decay 0.01, gradient clipping 1.0 (Trainer default).

```bash
# Same LLM environment and converted dataset as LoRA
/opt/conda/bin/python -m pip install -r requirements-sft.txt
bash run_sft_full_hf.sh
```

The default output is `sft_outputs/qwen3_4b_full`; use this directory as `--model`
with `run_agent.py`, or as `MODEL_PATH` with the evaluation shell script. There is
no merge step. Choose an empty output directory for each new run.

For multi-GPU DDP (model and optimizer replicated per GPU):

```bash
NPROC_PER_NODE=2 GRAD_ACCUM=8 bash run_sft_full_hf.sh
```

Effective batch size is `BATCH_SIZE × GRAD_ACCUM × NPROC_PER_NODE`. Thus the example
above keeps the default effective batch of 16. For sharded training, install DeepSpeed
in the LLM environment and enable the included ZeRO-3 configuration:

```bash
/opt/conda/bin/python -m pip install deepspeed
NPROC_PER_NODE=2 GRAD_ACCUM=8 DEEPSPEED_CONFIG=configs/zero3.json bash run_sft_full_hf.sh
```

ZeRO-3 partitions parameters, gradients, and optimizer states; this config does not
CPU-offload. Final model saving gathers weights, so allow memory headroom. DeepSpeed
requires a CUDA/toolchain-compatible installation. This is a single-node launcher;
DDP alone does not reduce per-GPU model/optimizer memory.

Full tuning consumes substantially more memory than LoRA. For 4B, BF16 weights and
BF16 gradients alone are about 16 GB combined; optimizer state, activations and save
buffers add more. Start on an 80 GB GPU or use multi-GPU ZeRO-3, then measure peak
memory on your actual lengths. A 24 GB GPU is not a supported expectation for this
full-training configuration. Override `MAX_LENGTH=4096` to reduce activations; skipped
sample counts are recorded. `LEARNING_RATE`, `EPOCHS`, `BATCH_SIZE`, `GRAD_ACCUM`,
`OUTPUT_DIR`, and the existing model/data path environment variables are configurable.
Extra CLI flags (e.g. `--warmup-ratio 0.05`) can be appended to the shell command.
Start with a small learning rate and compare Dev task success with the LoRA baseline.
Real CUDA/DDP/ZeRO-3 training has not been validated on the development machine.
