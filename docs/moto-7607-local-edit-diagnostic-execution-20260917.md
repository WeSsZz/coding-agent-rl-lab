# moto-7607 局部编辑诊断执行报告（2026-09-17）

## 结论

固定使用 P2 step12 adapter，在 7 个由真实 verifier-strict 教师轨迹冻结的编辑前状态上执行一次 greedy 自由动作生成。结果为 0/7 failure reduction、0/7 strict success，触发预注册停止门槛。

D2 阶段续跑、任何新 SFT 和 GRPO 均未启动。已有 P2 停止结论保持不变。

## 冻结协议

- 模型：Qwen2.5-Coder-7B-Instruct
- adapter：P2 step12
- adapter SHA-256：`9370e928903c5b59096a7add0bd04c55908574b9ee85ab155678eedfb6d376db`
- seed：`123401`
- 解码：greedy；每状态一个 assistant turn；最多 1024 新 tokens
- 来源：3 条已保存且完整 verifier strict 成功的真实恢复轨迹
- 编辑前状态：7
- 上下文 SHA-256：`82a460a0405c37c7386feb20eecbe6f59540566bddb9334ab2a677bc7574be52`
- 阶段状态已冻结但未执行：9；SHA-256 `9b1d28e30695c0b800fcf945a7ecb72bcb7ec11d507f26eb80aba364d854a926`

## D1 结果

- 可解析工具动作：4/7
- 实际编辑动作：4/7
- 成功应用补丁：4/7
- 编辑教师目标文件：3/7
- verifier-improving 编辑：0/7
- strict success：0/7
- 精确教师动作一致：0/7（仅辅助指标）
- 峰值显存：16,852,131,328 bytes，约 15.70 GiB
- adapter 运行前后哈希一致：是

失败归因为 3 个不可解析工具动作和 4 个“可应用但语义错误”的编辑。所有生成均在 1024-token 上限前结束，不是输出截断。

四个实际应用的错误编辑分别是：

1. 在 callback 文件的 import 区引入不存在/错误的导入，原失败未解决并新增 2 个失败。
2. 在 `state_task_service_aws_sdk.py` 中加入无关的 `ResourceCondition` 导入，编辑了错误文件，原失败保留。
3. 把 `callback_endpoint.wait(timeout=timeout_seconds)` 改成无超时等待，原失败保留。
4. 用 `replace_lines` 基本重写 callback 文件前 20 行，却没有加入所需 `time` import 或缺失 token 的处理逻辑，原失败保留。

三个不可解析响应均退化为反复生成 `<|im_start|>` / `<|endoftext|>` 等特殊 token；长度分别为 252、453、117 tokens。

## 基础设施审计

首次进程在模型加载前因系统旧版 PEFT 与 Transformers 不兼容而退出，显存仍为 1 MiB，没有产生 trial。失败目录和日志保留。随后使用上一轮已验证的隔离 PEFT 0.21 overlay，以完全相同的模型、adapter、上下文、seed 和预算执行 `retry1`；没有改变实验选择。

## 决策

证据把当前瓶颈定位到“已经提供正确真实执行状态时，仍不能稳定生成可解析且语义正确的下一步局部编辑”。现阶段扩大完整轨迹、训练搜索策略或启动 RL 都缺乏依据。

下一项应作为新的独立预注册实验：在同一批 7 个状态上补干净 7B instruct 底座对照，区分底座能力不足与 adapter 行为退化；随后才应把诊断扩展到按题切分的 20–30 道多题编辑集合。只有底座/adapter 对照和独立留出结果能证明某类小规模 recovery SFT 值得训练时，才进入新训练。

## 证据位置

- AutoDL 目录：`/root/autodl-tmp/moto-7607-local-edit-diagnostic-20260917`
- 完整 D1：`d1-p2-step12-seed123401-retry1/`
- 汇总：`d1-summary.md`、`d1-summary.json`
- 原始生成：`d1-p2-step12-seed123401-retry1/generations.jsonl`
- 工具轨迹：`d1-p2-step12-seed123401-retry1/tool-traces.jsonl`
- 每状态 verifier 审计：`d1-p2-step12-seed123401-retry1/*-reward-audit.jsonl`
- 首次基础设施失败：`d1-p2-step12-seed123401/` 与 `d1-run.log`
- 最终归档：`/root/autodl-tmp/moto-7607-local-edit-diagnostic-evidence-20260917-v2.tar.gz`
- 最终归档 SHA-256：`79d70924e95eafe92936328d877c05325f0599e1e9c45c38e9bea2f8c36911c7`
- 目录清单：`SHA256SUMS.txt`，37 个文件均已逐项校验
