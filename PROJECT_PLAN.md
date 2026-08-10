# Coding Agent RL Lab 路线图

## 目标

建立一个小而可信的 Coding Agentic RL 项目，证明从可执行任务、rollout、reward、策略更新到 held-out 评测的完整闭环。

## 非目标

- 不重新实现 OpenHands、Claude Code 或通用 Coding Agent；
- 不先爬取大规模 GitHub 数据；
- 不以 reference policy 的结果冒充模型结果；
- 不在没有真实权重/策略更新前宣称 RL 提升；
- 不把宿主机本地进程当成外部任务的安全 sandbox。

## 里程碑

### M0：Environment、Verifier 与 Rollout（当前）

- 稳定 Task/Trajectory/Reward/Policy 合同；
- 两个微型、可执行、可失败的 fixture；
- noop baseline 和 reference pipeline check；
- 多 Trial 可靠性报告；
- 本地安全限制与自动化测试。

### M1：SWE-Gym Adapter

- 读取 SWE-Gym task metadata；
- Docker sandbox provider；
- fail-before/pass-after 验证；
- 50 条 development、20 条 regression、20 条 held-out smoke subset；
- 接入一个现成轻量 Coding Agent。

### M2：Trajectory Dataset

- 采集成功和失败轨迹；
- 记录模型、prompt、tool、sandbox 和 verifier 版本；
- 去重、脱敏、许可与 provenance；
- failure taxonomy 与人工抽检。

### M3：Policy Optimization Baselines

- success filtering；
- rejection sampling；
- SFT/LoRA；
- 固定 held-out split 的训练前后比较；
- 至少三个 seed 或明确说明算力限制。

### M4：小规模 Agentic RL

- 接入 Agent Lightning 或 veRL；
- execution-verifiable reward；
- GRPO 小规模实验；
- reward hacking、安全违规和训练稳定性监控；
- 与 SFT/rejection sampling 做消融。

## 成功标准

项目只有满足以下条件才可以在名称之外声称完成 Agentic RL：

- 至少一次真实模型权重或可学习策略更新；
- 使用训练集之外的 held-out tasks；
- 同一 Agent scaffold、工具和测试预算下比较训练前后；
- 报告 resolve rate、回归率、成本、方差和失败类型；
- 提升不是由答案泄漏、测试泄漏或增加推理预算造成；
- 保存可复现配置、trajectory schema、policy manifest 和报告。

