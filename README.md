# Coding Agent RL Lab

一个与生产 Agent Runtime 解耦的、Verifier-first 的 Agentic RL 实验项目。

项目目标不是重新实现 Coding IDE 或通用 Agent Framework，而是建立一条可复现的学习闭环：

```text
Coding Task → Isolated Environment → Agent Rollout
            → Executable Verifier → Reward
            → Policy Update → Held-out Evaluation
```

## 当前阶段：M0 环境与评测基线

当前版本不训练模型，也不宣称已经实现 Agentic RL。它先固定训练系统最容易被忽略的基础合同：

- `CodingTask`：issue、仓库快照、测试命令、split 与 provenance；
- `Trajectory`：每一步 action、observation、tool result 和版本信息；
- `RewardVector`：测试、回归、补丁、成本与安全违规；
- `PolicyManifest`：策略、模型、训练数据和版本；
- 受限本地代码环境：只允许列文件、读文件、精确文本替换和执行受控测试；
- 可执行测试 Verifier，不使用 LLM judge 代替环境真实状态；
- `noop` 失败基线与 `reference` 基础设施自检策略；
- 多次 Trial、`pass@1`、`pass^3` 和 fully-reliable task rate。

`reference` 策略包含答案，只用于验证 environment/trajectory/verifier 管线，不能作为模型效果或训练基线。

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

接入 SWE-Gym 或其他外部任务前必须增加 Docker/远程 sandbox provider，不能直接在宿主机执行任意数据集命令。

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

