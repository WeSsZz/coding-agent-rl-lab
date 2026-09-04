# Coding Agent RL Lab

一个与生产 Agent Runtime 解耦的、Verifier-first 的 Agentic RL 实验项目。

项目目标不是重新实现 Coding IDE 或通用 Agent Framework，而是建立一条可复现的学习闭环：

```text
Coding Task → Isolated Environment → Agent Rollout
            → Executable Verifier → Reward
            → Policy Update → Held-out Evaluation
```

## 当前阶段：M1.3 模型 rollout 与单步 GRPO 机制验证

当前版本已执行首个具有非零梯度的 fixture 单步 LoRA GRPO 更新，并完成固定 held-out 上的
基座/fixture-LoRA 对照；两者均为零成功，因此不宣称已经证明 Agentic RL 提升。真实 SWE-Gym
路径已打通 train-only gold SFT warm-start、分层训练奖励、远程 verifier、多轮工具 rollout 和
显存可控的 GRPO microbatch；当前真实任务仍是组内全零奖励，瓶颈已定位到编辑动作而非训练 OOM：

- `CodingTask`：issue、仓库快照、测试命令、split 与 provenance；
- `Trajectory`：每一步 action、observation、tool result 和版本信息；
- `RewardVector`：测试、回归、补丁、成本与安全违规；
- `PolicyManifest`：策略、模型、训练数据和版本；
- 受限代码环境：只允许列/搜/读文件、精确文本替换、已读源码的小范围行替换和受控测试；
- 可执行测试 Verifier，不使用 LLM judge 代替环境真实状态；
- `noop` 失败基线与 `reference` 基础设施自检策略；
- 多次 Trial、`pass@1`、`pass^3` 和 fully-reliable task rate；
- `CodingEnvironment` / `EnvironmentProvider` 协议，rollout 不再依赖具体环境实现；
- `LocalFixtureEnvironmentProvider`，完整保留 M0 的可信 fixture 流程；
- `SWEGymTaskAdapter`，校验官方 schema 并生成版本化 provenance 和实例镜像名；
- 公开 `CodingTask` 与私有 `DockerTaskSpec` 分离，不向 policy 暴露 gold patch 或 verifier test patch；
- `DockerSandboxProvider`，实现容器启动、base commit 校验、test patch 注入、fail-before、容器内工具、pass-after 与强制清理；
- Docker 默认无网络，并限制内存、CPU、PID、capability 和 privilege escalation。

`reference` 策略包含答案，只用于验证 environment/trajectory/verifier 管线，不能作为模型效果或训练基线。

SWE-Gym 原始记录不携带跨仓库通用的 `test_command`。adapter 因此要求调用方提供经过审计、按 repo/version 固定的 argv 命令，不会用猜测的 `pytest` 命令冒充官方 verifier。

## 快速运行

```bash
PYTHONPATH=src python -m coding_agent_rl_lab smoke
```

运行失败基线：

```bash
PYTHONPATH=src python -m coding_agent_rl_lab evaluate \
  --policy noop \
  --repetitions 3 \
  --output work/noop-report.json \
  --trajectories work/noop-trajectories.jsonl
```

运行参考管线自检：

```bash
PYTHONPATH=src python -m coding_agent_rl_lab evaluate \
  --policy reference \
  --repetitions 3 \
  --output work/reference-report.json \
  --trajectories work/reference-trajectories.jsonl
```

运行测试：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## 安全边界

本地环境只用于仓库中人工审核的微型 fixture：

- 不使用 shell；
- test command 以 argv 形式执行；
- 仅允许 Python 测试进程；
- 文件访问限制在临时 workspace 内；
- 限制步骤、输出大小和测试超时。

SWE-Gym 或其他外部任务只能通过 Docker/远程 sandbox provider 执行，不能直接在宿主机运行任意数据集命令。

Docker provider 不挂载宿主机工作目录或 Docker socket；所有文件修改均留在按 task 创建的临时容器中。它还会保护 test patch 涉及的文件，agent 修改这些文件会被记录为 hard violation。

默认自动化测试通过可注入的 runner 验证完整 fail-before/pass-after 生命周期。安装 Docker 的隔离 worker 还可以运行真实容器集成 smoke：

```bash
RUN_DOCKER_INTEGRATION=1 PYTHONPATH=src \
  python3 -m unittest discover -s tests -p 'test_docker_integration.py' -v
```

该 smoke 使用运行时生成的微型 Git 仓库，验证受限容器启动、test patch 注入、失败基线、代码修复、测试转为通过，以及容器和临时镜像清理。

首条官方 SWE-Gym instance smoke 使用 `getmoto__moto-7365`，测试命令来自固定的 SWE-Gym 环境常量提交。它自动下载官方 task row、拉取实例镜像、验证 fail-before，随后用 verifier 私有 gold patch 完成 pass-after 基础设施自检：

```bash
PYTHONPATH=src python3 -m coding_agent_rl_lab.swe_gym_smoke
```

gold/reference patch 不会进入 policy 可见的 `CodingTask`，该结果不能作为模型能力或训练效果。

## 首条无答案模型 rollout

模型策略通过 OpenAI-compatible Chat Completions API 连接 vLLM。模型服务运行在 AutoDL GPU，Docker 和 verifier 仍运行在 Ubuntu worker；任务容器保持 `--network none`。

模型每一步只能返回一个严格 JSON 动作。Trajectory v3 额外保存初始 verifier observation，并继续保存实际 prompt messages、模型原始输出、采样 seed、延迟、token usage、工具 observation 和协议错误，但不保存 API key。若 vLLM 启用了 API key，在 Ubuntu shell 中通过环境变量提供：

模型首步会直接看到真实的 fail-before verifier 输出，可据此读取精确测试路径；大型仓库的文件清单会明确标记截断，policy 可用受限的 `search_text` 动作按文件名或字面文本检索代码。搜索结果优先排列实现文件，再排列测试和文档；`read_file` 支持 20–400 行的局部范围读取，便于从搜索命中和 traceback 获取可精确替换的上下文。文件不存在、替换文本未匹配等普通工具错误会作为 observation 返回，循环保护器同时报告剩余步骤；相同失败动作、搜索或未变化文件读取不能原样重复。路径越界和修改 verifier-owned 测试等安全错误仍是立即终止的 hard violation。v12 prompt 还为全部历史 observation 设置共享字符预算，优先保留最近观察，要求为修改与复测预留步骤，并在补丁测试失败后优先处理新 traceback。

```bash
export CODING_AGENT_MODEL_API_KEY='<local-or-vllm-token>'
```

连接好指向 AutoDL 的本地 SSH 隧道后，采集首条不含 gold patch 的真实轨迹：

先准备固定任务元数据和 Docker 镜像。该命令会跳过已有镜像、失败自动重试，
并将去掉 gold patch 与 hints 的校验后元数据缓存在 `work/`：

```bash
PYTHONPATH=src python3 -m coding_agent_rl_lab.swe_gym_prepare --task-count 10
```

固定的 10 个 Moto 5.0 development 镜像已在 Ubuntu worker 准备完成，并划分为互斥的
6 个 train、2 个 regression 和 2 个 held-out 任务；同一命令还会写出
`work/swe-gym-grpo-prompts.jsonl`。使用 `--skip-images` 可只重新生成和检查 prompts。
`--task-set train|regression|held-out` 用于选择固定集合，重复提供 `--task-id` 可按稳定的
SWE-Gym instance id 精确准备一个或多个任务；`--task-id` 和 `--task-count` 不能同时使用。

```bash
PYTHONPATH=src python3 -m coding_agent_rl_lab.swe_gym_rollout \
  --model '<served-model-id>' \
  --api-base http://127.0.0.1:8000/v1 \
  --task-set train \
  --task-id getmoto__moto-7509 \
  --repetitions 5 \
  --seed 12345 \
  --test-timeout-seconds 180 \
  --resume
```

每个 repetition 都在全新的 Docker environment 中执行，并使用不重叠的确定性 seed 区间。结果写入 `work/swe-gym-model-report.json` 和 `work/swe-gym-model-trajectories.jsonl`；report 自动汇总 trial count、成功率、样本方差、`pass^3`、scalar reward 和逐任务可靠性。模型端点不可达或输出违反 JSON 协议时，rollout 仍会落盘，并以 hard violation 计为零 reward。

长任务会在每条完整 trajectory 后原子更新报告和 JSONL 检查点。报告中的
`planned_trial_count`、`completed_trial_count` 与 `run_complete` 可区分完整运行和中断运行；
中断不会再丢失此前已经完成的 trajectory。verifier 超时会终止当前 episode，避免模型
随后重复启动同一个长时间测试。再次执行相同模型、task、repetition、seed 和输出路径并
添加 `--resume` 时，会严格校验已有 trajectory 的 schema、policy manifest 与 seed，只运行
缺失的 trial；不兼容的检查点会直接拒绝恢复。

## Agentic RL episode 导出

Trajectory v3 需要先经过无泄漏筛选，不能直接冒充 GRPO 训练数据。导出器会排除
模型连接故障、reference/答案策略、异常 baseline 和敏感元数据；有效的模型失败会作为
零奖励 episode 保留，成功且无违规的 episode 才会标记为 SFT eligible：

```bash
PYTHONPATH=src python3 -m coding_agent_rl_lab.rl_dataset \
  --input work/swe-gym-model-trajectories.jsonl \
  --output work/rl-episodes-v1.jsonl \
  --report work/rl-episodes-v1-report.json
```

该步骤只建立训练数据边界，`training_performed` 仍为 `false`。

已保存的 trajectory 可在本地进行不含原始 prompt、代码或 observation 的失败归类：

```bash
PYTHONPATH=src python3 -m coding_agent_rl_lab.failure_analysis \
  --input work/swe-gym-model-trajectories.jsonl \
  --output work/swe-gym-failures.json
```

当前 taxonomy 区分上下文溢出、模型传输/协议错误、测试文件防篡改拒绝、其他非法动作、
verifier 超时、补丁未通过、编辑动作本身失败和未尝试编辑。旧 8K held-out 轨迹中，基座 8 条有 6 条属于
上下文溢出，fixture-LoRA 8 条有 7 条属于上下文溢出；因此这些结果主要用于发现运行问题，
不能当作干净的模型能力结论。16K `moto-7509` 运行消除了协议/上下文违规，4 条失败可进一步
分为 2 条补丁未通过和 2 条未产生补丁。

## Train-only SFT warm-start

基座策略在 6 个 train 任务上仍未产生非零 verifier reward，因此项目新增一个与 answer-free
trajectory 严格分离的监督 warm-start。构建器只接受固定 train split 中的 6 个 instance，
硬拒绝 regression/held-out，跳过测试文件、文件创建/删除、重命名和超过动作预算的 hunk。
输出中的每条样本都显式记录 `contains_answers=true` 与
`answer_source=official_swe_gym_gold_patch`，默认写入已被 Git 忽略的 `work/private/`：

```bash
PYTHONPATH=src python3 -m coding_agent_rl_lab.swe_gym_sft \
  --download-pinned-train
```

当前固定数据生成 204 条样本：51 个源码 hunk 各自生成 locate、inspect、edit、verify 四阶段
监督；另有 1 个超过 4096 字符动作上限的 hunk 被审计排除。该数据只能用于 train-only
warm-start，不能进入 held-out prompt、普通 trajectory 导出或能力报告。

SFT 入口默认只执行数据、prompt 版本、assistant JSON、tokenizer 长度和 CUDA preflight。
它把 conversational 数据转换为 prompt/completion，并只对 completion 计算 loss；任何超过
`--max-length` 的样本都会导致退出，不会静默截断 gold action：

```bash
PYTHONPATH=src /root/autodl-tmp/grpo-env/bin/python \
  -m coding_agent_rl_lab.sft_train \
  --model-path '<local-model-snapshot>' \
  --dataset work/private/swe-gym-train-gold-sft-v1.jsonl \
  --dataset-report work/private/swe-gym-train-gold-sft-v1-report.json
```

只有显式增加 `--train` 才会执行 Qwen2.5-Coder-7B 的 LoRA 更新。RTX 5090 上的
seed `61001` 已使用 `--example-limit 8 --max-steps 1` 完成机制 smoke：
实际序列为 994–1585 tokens，无截断，`train_loss=0.7067`、
`grad_norm=1.4811`，并保存 155 MB 的独立 LoRA adapter。这证明 answer-supervised
completion-only loss 路径可产生有效更新，但不代表任务能力已经提升。扩大训练后仍须在相同
v12 scaffold 和预算下先评估 train/regression，最后只做一次固定 held-out 对照。

项目选择 TRL 的 `GRPOTrainer.environment_factory` 作为首个单卡基线接口。现有
`GRPOCodingEnvironment` 将受限环境映射为具名工具，并只按最终 verifier 状态返回
二元奖励；它不改变 Docker 的断网、资源限制或测试文件防篡改边界。AutoDL 训练进程
通过仅绑定 loopback 的反向 SSH 隧道访问 Ubuntu Docker worker，bearer token 文件权限
为 `0600`。真实 SWE-Gym session 创建、baseline fail、文件读取和清理的跨主机 smoke 已通过。

GRPO 训练奖励与最终评测指标保持分离：评测仍只把“verifier 全通过、存在源码补丁且无违规”
记为严格成功；训练 worker 额外计算上限为 0.5 的确定性中间奖励。失败测试数量减少最多
贡献 0.25，没有新增失败贡献 0.10，语法有效的非空补丁在修改后主动运行 verifier
贡献 0.10 + 0.05。无补丁、超时或未验证的补丁仍为 0，安全/越界违规为 -1，
严格成功固定为 1。所有分量都来自 sandbox、补丁解析和 verifier 输出，不使用 LLM judge。

远程 worker 只监听 `127.0.0.1`，并要求至少 32 字符的 bearer token。令牌通过环境变量
或权限受限的 `--token-file` 提供，不能写入仓库、trajectory 或训练报告。

AutoDL 的推理镜像包含旧版 vLLM 0.11，而当前 TRL 的环境接口需要
`transformers>=5.2.0`。`scripts/setup_autodl_grpo_env.sh` 因此在数据盘创建隔离环境、复用
已有 PyTorch/CUDA，并只在该环境隐藏不兼容的 vLLM 包；原 vLLM 服务不受影响。当前首个
训练入口使用 Transformers generation 和 LoRA，不假装调用不兼容的 vLLM。

以下命令默认只执行 preflight：验证本地模型缓存、Qwen 工具模板、CUDA、prompt 数据和
远程 worker，不加载训练权重，也不更新参数：

```bash
PYTHONPATH=src /root/autodl-tmp/grpo-env/bin/python \
  -m coding_agent_rl_lab.grpo_train \
  --model-path '<local-model-snapshot>' \
  --prompt-rows work/swe-gym-grpo-prompts.jsonl \
  --worker-token-file /root/autodl-tmp/grpo-worker.token \
  --task-count 3
```

只有额外提供 `--train` 才会执行单卡 LoRA 权重更新。Qwen2.5-Coder-7B 在该项目的
审计 rollout 协议中稳定输出裸 `{"name": ..., "arguments": ...}` 动作；训练时必须显式
添加 `--bare-json-tool-calls`。该模式不改变 chat template，只为 Transformers 设置对应的
response parser，并在任何权重加载或更新前运行固定解析探针，探针失败时直接退出。

首个真实 SWE-Gym preflight 已全部通过。项目先用两个无答案可信 fixture 验证闭环：RTX 5090
上的 seed `41003` 单步实验得到 `reward=0.5`、`reward_std=0.7071`、`grad_norm=0.1464`，并保存
独立 LoRA adapter。固定 held-out 的基座/fixture-LoRA 对照仍是 0/8，不能视为能力提升。

随后只用六个 train task 的官方 gold patch 构造 204 条多轮、completion-only SFT 样本；动态工具
协议的 60-step adapter 最终 SFT loss 为 `0.0752`。真实 `moto-7509` 的 2-generation、4096-token
GRPO smoke 曾在 backward OOM；将 generation batch 保持为 2、训练 microbatch 降为 1 后，完整
1-step 在 249.9 秒内结束且不再 OOM。该步仍为 `reward=[0,0]`、`grad_norm=0`：脱敏审计显示
两条轨迹共 4 次 `replace_text`，其中 3 次无精确匹配、1 次参数错误，未产生补丁。环境因此新增
`replace_lines` 作为受控恢复路径：只允许修改已读取的源码、一次最多 80 行、编辑后行号立即失效，
且继续禁止修改 verifier-owned 测试。同 seed 复跑后两条轨迹都产生语法有效且已验证的源码补丁，
奖励由 `[0,0]` 提升为 `[0.15,0.15]`；更换为 seed `70002` 后得到 `[0,0.15]`，
`reward_std=0.1061`、`grad_norm=0.4935`、`train_loss=-0.8747`，首次完成真实 SWE-Gym 上的
有效单步 GRPO LoRA 更新。该结果只证明训练信号和参数更新链路成立；两个补丁均未减少失败测试，
下一阶段需在 train/regression 上验证更新方向，不能据此宣称任务成功率提升。

## SWE-Gym 数据边界

官方数据字段包括 `instance_id`、`problem_statement`、`repo`、`base_commit`、`version`、`test_patch`、`FAIL_TO_PASS` 和 `PASS_TO_PASS`。加载到本项目时应在独立的审计步骤补充：

```json
{"test_command":["python","-m","pytest","-q","<validated test target>"]}
```

`patch` 是 gold solution，不会被 adapter 放入 `CodingTask`；`test_patch` 和具体 verifier 测试集合只保存在 provider 私有 spec 中。

## 与 Durable Agent Runtime 的边界

两个仓库只共享版本化合同，不共享业务实现：

```text
durable-agent-runtime       coding-agent-rl-lab
---------------------       -------------------
可靠执行与恢复              rollout 与 policy optimization
工具权限和人工审批          coding environment 与 verifier
生产 Trace / Replay         reward 与训练实验
企业安全边界                held-out learning evaluation
```

未来可通过 `trajectory-v1.jsonl` 和 `policy-manifest-v1.json` 对接，但任何一方都不依赖另一方才能运行。

完整路线见 [`PROJECT_PLAN.md`](PROJECT_PLAN.md)。

