# 已冻结路线索引：SWE-Gym + 自定义策略 + GRPO

> **状态：历史路线，非当前主线。** 当前主线在固定版本的 SWE-Gym OpenHands 交互与干净 verifier 上做在线 RL；66 条轨迹 SFT 仅作对照。当前入口见 [`swegym-openhands-rl-route.md`](swegym-openhands-rl-route.md)。旧 `grpo_train` 默认拒绝执行，只有显式设置 `CODING_AGENT_ALLOW_LEGACY_GRPO=1` 才允许历史复现；包级 `coding-agent-rl smoke/evaluate` 也需要 `CODING_AGENT_ALLOW_LEGACY_ROUTE=1`。不要把本页列出的旧数据、命令、配置或实验结果当作当前方案。

## 旧路线做过什么

旧路线从 SWE-Gym 读取真实代码修复任务，在自定义受限工具环境中收集策略 rollout，以 verifier 结果构造奖励，并探索 GRPO。另有 train-only 的官方 gold patch SFT warm-start：6 个固定 train task 中 51 个可用源码 hunk 各展开成 locate、inspect、edit、verify 四条监督样本，共 204 条。它们是答案可见的合成动作监督，**不是 204 条模型 rollout，也不是当前主线训练数据**。

## 应如何引用旧结果

- 只能表述为“旧路线历史实验”或“机制 smoke”。
- 旧 GRPO 单步更新只证明代码路径能产生更新，不证明任务能力提升。
- 旧的 204 条 gold-patch SFT 数据、旧七动作协议、旧 adapter、旧评测数字均与当前 OpenHands 轨迹管线隔离；不得当作新路线的训练输入、模型初始化或对照基线。
- SWE-Gym 的任务元数据、分割审计、测试解析、sandbox 资源限制等基础设施可作为实现参考，但复用前必须由当前路线重新验证。

## 旧文档和代码的处理

README 中 `LEGACY-SWE-GYM-GRPO-ROUTE` 标记之后的说明、旧阶段计划及实验记录继续保留，以便审计和历史复现。入口处的路线提示优先级高于这些旧文档中的“当前”“下一步”等原始措辞。旧代码保留在原位置，不能仅凭存在于仓库就推断它仍是主线入口。

当前权威路线与状态：[`swegym-openhands-rl-route.md`](swegym-openhands-rl-route.md)。
