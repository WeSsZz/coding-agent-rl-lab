# SWE-Gym 官方 OpenHands fork：RL 路线入口

当前主线是以 [SWE-Gym 的 OpenHands fork](https://github.com/SWE-Gym/OpenHands) 的真实代码交互和独立 verifier 为底座做在线 RL。固定提交为 `e644a2ca45c3623b27a7e6c169e3d479f0a87fbc`。旧自定义 shell/edit scaffold 和旧 `grpo_train` 不再用于当前路线；它们的成功率不能与官方 CodeActAgent 直接比较。

## 2026-09-28 实测状态

Qwen3-8B 已通过官方 fork 在三个固定 `train` 任务的容器内调用工具，干净 verifier 基线为 0/3；见 [`qwen3-openhands-gpu-pilot-20260928.md`](qwen3-openhands-gpu-pilot-20260928.md)。另一次 10 轮随机 rollout 记录了逐次生成 token ID、采样 logprob、模型可见 observation 和解析后的工具调用，并与 verifier 的 0 奖励拼接。该旧 trace 没有策略版本标识，是否含有可重放的 prompt token ID 尚未在本机核对，不能直接用于训练。**还没有 GPU 参数更新，也没有正奖励模型补丁的奖励闭环。**

## 新 RL 单步闭环

每次采样从同一不可变策略 checkpoint 启动 Qwen3 服务，并为 trace proxy 指定唯一 `--run-id`、固定 `--instance-id` 和相同的 `--policy-revision`。每次独立运行官方 `run_swegym_openhands.sh`，随后运行 `run_swegym_verifier.sh`，最后用 `join_swegym_rollout_reward.py` 将原始 trace、该次官方 output、该次独立 verifier 和官方 per-call completions 拼接。trace proxy 要求新的空 trace 路径，且请求 vLLM 0.11 返回 `prompt_token_ids`、生成 token IDs 与 logprobs；旧的无策略版本 trace 不能用于训练。

RL 采样须设置 `MODEL_TEMPERATURE=1`，并保证请求的 `top_p=1`、无采样惩罚；否则采样分布与训练器的原始 logprob 不一致，拼接和预检会拒绝。先前温度 0.7 的 pilot 只验证了采集，不用于训练。将至少两次**同任务、同一策略版本**的 `(trace.jsonl, reward.json)` 成对传给新训练入口：

```bash
PYTHONPATH=src python -m coding_agent_rl_lab.openhands_rl \
  --policy-revision qwen3-checkpoint-A \
  --rollout /path/run1.trace.jsonl /path/run1.reward.json \
  --rollout /path/run2.trace.jsonl /path/run2.reward.json
```

上面只做 CPU 预检。只有奖励同时含 0 和 1、每条轨迹的哈希/身份/响应与干净 verifier 一致，并且 GPU 上加载的权重确实是声明的同一个 checkpoint 时，才在 GPU 机器上加 `--train --model-path /root/Qwen3-8B --output-dir /path/new-output`。训练器逐次重放 vLLM 返回的精确 prompt token ID，仅给生成 token 计算一次组均值基线的 REINFORCE 梯度；工具 observation 不作为预测目标。它记录梯度范数、实际参数变化并保存 LoRA adapter。这个入口只做一次优化步的端到端 smoke，**不是可重复迭代的 GRPO/PPO 训练器**。每次更新后重新部署新权重并重新采样才可继续；不同策略版本的旧 trace 不能重复用于在线更新。

当前所有模型 rollout 奖励都是 0，因此严格预检会拒绝更新；不能通过注入 reference patch 或伪造正奖励来“跑通”真实 RL。GPU 显存和真正参数更新仍需实测，48GB 上不能假设与当前 vLLM 同卡并行。旧 `coding_agent_rl_lab.grpo_train` 默认拒绝执行，仅留 `CODING_AGENT_ALLOW_LEGACY_GRPO=1` 历史复现开关。

## 官方参数与本地单任务试跑

官方 [`rollout-swe-train-full.sh`](https://github.com/SWE-Gym/OpenHands/blob/e644a2ca45c3623b27a7e6c169e3d479f0a87fbc/scripts/rollout-swe-train-full.sh) 使用 `SWE-Gym/SWE-Gym` 的 `train` split、`CodeActAgent`、无 hint、`max_iter=50`、最多 3000 个任务、64 个 worker、默认 20 次 rollout。它指定私有 RemoteRuntime 服务和 API key；我们没有这项服务，所以在可控的 Ubuntu VM 上使用该 fork 官方支持的本地 `eventstream`/Docker runtime。任务、agent、提示词、容器镜像和迭代上限均保持 fork 原实现。

VM fork 目录为 `/home/wesz/swegym-openhands`，使用 Python 3.12、Poetry 锁文件及预缓存的 `getmoto__moto-7646` 实例镜像。先运行 `bash /home/wesz/run_swegym_openhands.sh --preflight`。接通模型 API 后，设置 `MODEL_ID` 为 vLLM 实际服务的模型名，`MODEL_API_BASE` 为 **VM 可达**的 OpenAI 兼容 `/v1` 地址，再运行 `bash /home/wesz/run_swegym_openhands.sh`。入口脚本源文件在 [`scripts/vm/run_swegym_openhands.sh`](../scripts/vm/run_swegym_openhands.sh)。它先只跑一个 train 任务、一个 worker、一次 rollout；扩展前需确认容器内交互、历史、补丁和独立 verifier 结果。

2026-09-27 环境记录：VM 的 `SWE-Gym/OpenHands` checkout 为上述固定提交；Python 3.12.14、Poetry 1.8.5 和 313 个锁定运行依赖已安装，`uv pip check` 通过。锁文件指向的 LiteLLM 1.52.5 wheel 已从 PyPI 下架（PyPI 返回 404），所以从 LiteLLM 官方 `v1.52.5` tag 对应提交 `33ceb7ca1ffab1401decca773253b58d5fc461a2` 构建了同版本 wheel；固定来源是 [LiteLLM 官方 tag](https://github.com/BerriAI/litellm/tree/v1.52.5)。`pylcs==0.1.1` 因 VM 没有系统编译器，使用校验过官方 SHA-256 的 Zig 0.13 编译。SWE-Bench 评测组固定在 fork 锁定的 `e440817248cf29bee71ca9ab3c3a762177ab8f7c`。

无模型检查已确认官方 `run_infer.py --help` 正常、313 个锁定依赖兼容、train split 共 2438 条且唯一选中 `getmoto__moto-7646`，其 base commit 为 `98ad497eb4de25943ceadabe7c2c5c73cc810e27`；对应 Docker 镜像已在 VM 缓存。no-model probe 已通过 runtime 启动、SWE-Gym 初始化和 `git rev-parse HEAD` 检查。VM 的 Docker bridge 无法让宿主机访问容器映射端口，因此 [`patch_swegym_host_network.py`](../scripts/vm/patch_swegym_host_network.py) 只将 fork 的 sandbox network 设置改为读取 `SWEGYM_USE_HOST_NETWORK`；启动入口为当前 VM 设置为 `true`。这是一项为本 VM 的 Docker 连通性做的兼容覆盖，task runtime 共享 VM 网络命名空间；fork 提交和 agent、prompt、任务、迭代参数不变。Docker 构建仍由 [`docker-build-host-network.sh`](../scripts/vm/docker-build-host-network.sh) 单独加 host network。

评测入口的 `output.jsonl` 保存最终补丁、事件历史和指标。模型应在真实实例容器内通过 OpenHands 工具读取、运行命令并编辑代码；运行结束后收集 `git diff`。任务通过与否须在干净评测容器中使用 fork 的 `eval_infer.sh` 判定，不能把“生成了补丁”当成通过，也不能将开发任务的结果当正式测试成绩。

## RL 接口边界

OpenHands 的推理历史能作为环境轨迹，但它默认输出不足以直接执行 GRPO：需在**当前策略**采样时记录每次生成的 token IDs、采样 logprobs、assistant/tool 边界及工具 observation mask，并将干净 verifier 的任务奖励和这次 rollout 精确关联。先用单任务检查这些字段和容器收尾，再实现采样适配器；不能从保存的文本补丁反推策略 logprobs，也不应把旧 gold-patch 合成样本混入这条路线。

无卡 AutoDL 容器保存模型和训练数据，但没有 Docker、CPU/内存不足以承载官方交互 runtime；Docker runtime 在 Ubuntu VM 上。Hub 当前不可达时，probe 从本地 Hugging Face 缓存读取 SWE-Gym train split。无卡预检通过只表示交互 runtime 和环境初始化可启动，**不代表模型 rollout、补丁质量、独立 verifier 或 RL 训练已经完成**。模型 API 需要有卡实例/其他推理服务，不能从无卡实例凭空验证。

2026-09-27 对新启动的 `weste:21248` 无卡实例做只读检查：AutoDL 面板报告 0.5 CPU 核、2 GB 内存、系统盘 30 GB（已用 19 GB）、数据盘 50 GB（已用 26 GB）；`/root/autodl-tmp/route-v2` 已存在（12 GB），Python 3.12.3，`pip check` 通过，Docker 和 GPU 均不可用。数据盘的 `Qwen2.5-Coder-7B-Instruct` 模型目录为 15 GB。该节点继续承担持久化模型/数据；执行 OpenHands 容器交互的 Ubuntu VM 已单独通过 no-model probe。
