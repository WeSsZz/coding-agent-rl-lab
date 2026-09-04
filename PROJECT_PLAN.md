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

### M0：Environment、Verifier 与 Rollout（已完成）

- 稳定 Task/Trajectory/Reward/Policy 合同；
- 两个微型、可执行、可失败的 fixture；
- noop baseline 和 reference pipeline check；
- 多 Trial 可靠性报告；
- 本地安全限制与自动化测试。

### M1：SWE-Gym Adapter（进行中）

#### M1.1：Environment Provider 边界（已完成）

- `CodingEnvironment` / `EnvironmentProvider` 协议；
- `RolloutCollector` 与 `LocalFixtureEnvironment` 解耦；
- 保留 M0 fixture provider 和全部既有行为；
- Docker sandbox 配置与 provider 骨架；
- Docker 骨架在没有真实生命周期实现时主动 fail-fast。

#### M1.2a：SWE-Gym Adapter 与 Docker 生命周期（已完成）

- 校验 SWE-Gym task metadata；
- 映射 instance image、base commit、显式 test command 与版本化 provenance；
- 将 policy 可见 task 与 verifier 私有 test patch/spec 分离；
- 实现 Docker container、workspace、执行与强制清理生命周期；
- 自动化验证 fail-before/pass-after、路径逃逸与测试文件防篡改；
- 在 Ubuntu 26.04 / Docker Engine 上完成真实容器集成 smoke：受限启动、
  fail-before、代码修改、pass-after、容器与临时镜像清理均通过。

#### M1.2b：首条真实任务（已完成）

- 使用已经验证的 Ubuntu Docker worker；
- 拉取并固定 `getmoto__moto-7365` 的 x86_64 instance image；
- 从固定 SWE-Gym 环境常量提交审计 `getmoto/moto@5.0` 的 test command；
- 官方 test patch 在受限、断网容器内稳定 fail-before；
- gold/reference patch 通过目标与回归测试，并仅标记为基础设施自检；
- 自动化 smoke 在退出时强制清理实例容器。

#### M1.3：首个模型 Agent rollout（进行中）

- 已实现兼容 vLLM/OpenAI Chat API 的轻量模型策略；
- 已定义严格 JSON 工具协议，并记录完整模型输入、输出、token usage 与协议错误；
- 已将 trajectory schema 升级到 v3，保存初始 verifier observation，模型连接密钥不进入 manifest 或轨迹；
- 已在相同 Docker worker 和工具预算下保存第一条无答案 Qwen2.5-Coder-7B trajectory；
- 已记录模型、prompt、采样参数、工具调用、reward 与失败类型，并区分基础设施错误与有效模型失败；
- 已冻结首版通用 Agent scaffold，并支持多 seed 重复采样和可靠性汇总；
- 支持同一任务多次独立采样，并自动汇总成功率、样本方差与可靠性；
- 已固定 10 条真实任务并完成全部 Docker image 准备，划分为互斥的 6 train、2 regression、2 held-out；
- rollout、prepare 与远程 worker 支持按固定 task set 或精确 instance id 选择任务；
- 长扫描按 trajectory 原子写入检查点，并显式记录计划/完成 trial 数与运行是否完整；
- 兼容检查点可用 `--resume` 恢复，并严格校验 schema、task、repetition、seed 与 policy manifest；
- verifier 超时现在会立即终止 episode，避免同一病态补丁重复启动长测试；
- v12 prompt 对历史 observation 使用共享预算，优先保留最近信息，避免上下文线性增长；
- `search_text` 按实现、测试、文档排序，`read_file` 支持 20–400 行局部读取，循环拒绝会报告剩余步骤；
- 已生成 10 条答案与 verifier 私有字段均不外泄的 GRPO prompt；
- 已完成 AutoDL → loopback SSH tunnel → Ubuntu Docker worker 的真实跨主机 smoke；
- 已在隔离环境验证 TRL 1.12 `environment_factory`、Qwen 工具模板、CUDA 与远程 baseline；
- 固定 held-out 的基座与 fixture-LoRA 对照均为 0/8，不声明能力提升；
- 训练课程首题 `getmoto__moto-7509` 的 4 个种子均为零奖励，虽有 2 条产生源码修改，
  仍无组内奖励差异，不足以启动真实任务 GRPO；
- v9 扫描其余 5 个 train 任务共 10 条均无补丁；v11 的 `moto-7509` 4 条中有 1 条形成补丁但
  verifier 未通过；v12 全部 6 个 train 任务各 1 条仍为零奖励，15 次局部读取未转化为补丁；
- 已建立与 answer-free trajectory 分离的 train-only gold SFT 数据边界，固定 6 个 train
  instance 生成 204 条四阶段工具监督样本，并硬拒绝 regression/held-out；
- 已实现默认 preflight、显式 `--train` 的 TRL 1.12 LoRA SFT 入口，启用 completion-only
  loss，拒绝静默截断答案动作，并记录数据哈希、token 长度与训练指标；
- 已在 RTX 5090 上完成 seed `61001` 的 8-example、1-step SFT smoke：
  `train_loss=0.7067`、`grad_norm=1.4811`，adapter 已保存且 GPU 已释放；
- 已建立两个不含答案的可信 fixture 课程，8 条随机基线采样为 4 成功、4 失败；
- 已为 Qwen v9 裸 JSON 动作增加显式、带前置探针的 TRL response parser；
- 已在 RTX 5090 上完成 seed `41003` 的首个有效单步 LoRA GRPO 更新：
  `reward_std=0.7071`、`grad_norm=0.1464`、`effective_update=true`；
- 已完成 fixture-LoRA 的固定 held-out 负向检查，但尚无真实 SWE-Gym adapter 可做训练前后比较。

#### M1.4 及后续

- 50 条 development、20 条 regression、20 条 held-out smoke subset；
- 已建立无原始内容泄漏的 failure taxonomy，并区分失败编辑动作与完全未尝试编辑；下一步增加人工抽检流程。

### M2：Trajectory Dataset

- 采集成功和失败轨迹；
- 记录模型、prompt、tool、sandbox 和 verifier 版本；
- 去重、脱敏、许可与 provenance；
- 扩展 failure taxonomy，并建立人工抽检。

### M3：Policy Optimization Baselines

- success filtering；
- rejection sampling；
- SFT/LoRA：train-only 数据、训练入口和 1-step GPU smoke 已完成，待扩大训练并做固定预算评测；
- 固定 held-out split 的训练前后比较；
- 至少三个 seed 或明确说明算力限制。

### M4：小规模 Agentic RL

- 已接入 TRL `GRPOTrainer.environment_factory`；首个单卡基线使用 Transformers generation + LoRA，
  升级到 TRL 支持的 vLLM 后再评估 colocate；
- 已验证 execution-verifiable reward 可产生非零组内优势与梯度；
- 已完成可信 fixture 上的单步 GRPO 机制实验；
- 已建立首个 6/2/2 SWE-Gym curriculum；基础策略扫描仍无非零 verifier reward。下一步先用
  train-only 审计轨迹做工具调用 warm-start（SFT/rejection sampling）或升级基础策略，获得
  非零组内优势后，再在固定 regression/held-out split 上做真实训练前后比较；
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

