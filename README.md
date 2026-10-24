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

```bash
# 先查看当前 split 的实际任务 ID（在 AppWorld 环境中）
python -c 'from appworld import load_task_ids; print(load_task_ids("train"))'

# 指定任务，可重复 --task-id；任务必须属于 --split
python run_agent.py --model /path/to/Qwen3-4B \
  --appworld-root /path/to/appworld-root \
  --appworld-python /path/to/appworld-python \
  --task-id YOUR_TRAIN_TASK_ID --verbose
```

`--limit 0` 跑全部选中任务；默认只跑 1 个 train 任务。`--temperature 0.7 --top-p 0.8` 开启采样。每次运行使用唯一 experiment 名，避免重复 task 覆盖 AppWorld 输出。任务按顺序执行，每个任务一个独立子进程。

结果写到 `runs/qwen3_local_<time>_<id>/`：

- `config.json`：配置、任务清单、上游来源，不保存 API Key。
- `<task_id>.jsonl`：初始提示、原始模型输出、实际执行代码、观察、生成元数据和评测结果。
- `summary.json`：每任务是否宣告完成、验证是否成功、轮数、结束原因和错误；成功率分母包括出错任务，另列 error 数。

Transformers 路径保存 prompt/completion token IDs；API 路径保存服务 usage。**这不是 PPO 训练器**，未导出完整采样 log_probs 或 token loss_mask，不能直接作为精确 PPO rollout 使用。普通成功轨迹 SFT 可基于日志进一步转换；训练采样需要另加对齐与概率记录。

`task_completed()` 只决定何时停止交互；成功来自终局 `world.evaluate().success`。私有评测反馈不送回 Agent。代码语法/格式错误返回公开纠正信息并消耗一次轮次；环境返回的运行异常交给模型自行修复。模型输出截断不会执行语法不完整的代码。Transformers 上下文不足时停止并评估当前状态，不偷偷删历史。环境进程硬超时默认 180 秒，模型 API 超时默认 300 秒。

测试集只用于最终评测；不要用其逐任务日志做提示调整或训练。当前不计算 SGC；需要对完整 scenario 的全部任务执行官方聚合评测才能报告 SGC。

## 5. CPU 验证与限制

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
