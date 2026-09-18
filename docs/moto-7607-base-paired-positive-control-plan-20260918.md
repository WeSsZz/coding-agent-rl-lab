# moto-7607 干净 7B 底座配对与教师单步正对照（预注册）

本实验独立于 2026-09-17 的 D1 停止结论。既有 D1 结果保留，不用本轮结果回选 seed、checkpoint 或状态。本轮不训练、不做完整 rollout、不启动 GRPO。

## 冻结对象

- 状态：复用 D1 的 7 个真实编辑前状态，contexts SHA-256 为 `82a460a0405c37c7386feb20eecbe6f59540566bddb9334ab2a677bc7574be52`。
- 提示、工具 schema、chat template、最大 1024 新 tokens 和 greedy 解码保持一致。
- 模型条件：
  1. 干净 Qwen2.5-Coder-7B-Instruct，不加载任何项目 adapter；
  2. 同一底座在内存中合并固定 P2 step12 adapter，SHA-256 `9370e928903c5b59096a7add0bd04c55908574b9ee85ab155678eedfb6d376db`。
- seed：`123501`；greedy 下只用于固定所有辅助随机状态，两条件逐状态使用相同 seed。
- 记录模型 config、tokenizer config、special-token map、chat template 的路径/版本与 SHA-256；底座条件必须明确记录 `adapter_loaded=false`。

## C0：教师单步正对照

对每个状态执行两次全新仓库会话：

1. `teacher-immediate`：重放真实前缀，执行当前教师编辑，立即运行完整官方 verifier；
2. `teacher-suffix`：重放真实前缀，执行当前教师编辑，再执行该成功轨迹中当前编辑之后的固定教师后缀，直至官方 verifier。

立即 verifier 仅用于测量中间编辑本身是否足以减少失败，不作为教师动作正确性的唯一标准。正对照有效门槛是 7/7 教师目标动作可应用、7/7 固定教师后缀 strict success、无新增失败。若未通过，停止，不加载模型。

## C1：底座与 adapter 配对生成

每个模型条件、每个状态只生成一次完整下一步动作。保存：

- 原始生成 token ID；
- 含特殊 token 的完整解码文本；
- EOS、达到长度上限或其他终止原因；
- 每种 special token 的 ID、文本与出现次数；
- parser 原始结果、工具参数和执行 observation。

每个生成动作在两个全新仓库会话中执行：

1. `generated-immediate`：执行生成动作后立即运行 verifier；
2. `generated-with-teacher-suffix`：用生成动作替代当前教师编辑，再接相同固定教师后缀并运行 verifier。

若生成无法解析，则在后缀验证中明确记录“缺失当前动作”，不伪造编辑；仍执行剩余教师后缀，以验证缺失该步的最终影响。若生成过早终止会话或破坏后缀应用，也按实际结果记录。

## 判定

- `execution-semantic-success`：生成的是可应用编辑，且替代教师当前编辑后接固定教师后缀得到 strict success、无新增失败。
- 精确教师字符串一致仅为辅助指标；允许语义等价补丁。
- 特殊 token 退化单列：区分模型真实生成 special-token ID、parser 失败和显示/解码问题；不把三者合并。
- 逐状态比较底座、adapter、教师立即结果和教师后缀结果，不以单一总成功率替代。

## 后续门槛

- 底座语义表现优于 adapter：支持 adapter 退化假设；下一步审计监督数据、模板和 special-token 标签，不直接续训。
- 底座只消除 special-token 退化但语义修复仍为零：输出协议问题得到定位，局部语义能力仍是瓶颈。
- 两者都差：停止单题 checkpoint 工作，转向按题切分的 20–30 题局部编辑诊断集；之后才选择性比较 14B。
- 本实验无论结果如何都不授权 GRPO。

## 证据

使用新的远端目录，保存预注册、上下文引用、教师控制、两条件原始生成、token ID、工具轨迹、verifier 审计、模型/tokenizer 哈希、资源峰值、清理记录、SHA-256 清单和最终归档。不得覆盖 2026-09-17 证据。
