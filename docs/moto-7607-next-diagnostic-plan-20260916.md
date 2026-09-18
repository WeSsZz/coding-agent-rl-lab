# moto-7607 下一轮：先验证学习链路，再决定扩展

> 已被 `moto-7607-next-diagnostic-plan-v2-20260916.md` 取代。此文件保留为历史方案，请执行 v2；v2 纳入现有 14B 冻结对照，不再默认先续训 7B。

本文件是待执行计划。本轮仅制定计划，不启动训练、评测或 GRPO。保持 48 GiB；保留现有代码、adapter 和原始证据。任何 GRPO 均须用户另行明确授权。

## 判断依据

上一轮真实恢复教师 3/3 通过，16 条混合样本共约 1,613 个估算监督 tokens，SFT 仅 4 steps/1 epoch。候选与 SFT60 在固定状态诊断均为 0/4 strict、0/4 failure reduction。这说明该候选没有达到使用门槛，不证明 SFT 或 7B 无法学习。三个诊断状态参与过训练，结果只支持训练状态拟合不足，不能当作泛化实验。

历史 20 条轨迹跨模型、adapter、协议和 seed，只用于发现失败类型，不能合并为某个模型的准确率。导航与编辑均有问题，不能仅凭路径到达率宣布唯一瓶颈。不同固定状态的奖励差异也不是同一 prompt 的 GRPO 组内方差。

## P0：训练—生成一致性审计（CPU 优先，禁止更新权重）

读取当前代码及安装版本，保存 git status/diff、源码哈希、输入与 adapter 哈希。不要覆盖旧实验；新增运行目录。

对同一真实状态、同一正确目标动作，截取实际 SFTTrainer 和 TRL rollout 的最终渲染文本、input_ids、attention_mask、labels/生成前缀。逐项核对：

1. 工具 schema 是否一致。现有 SFT 的 prepare_prompt_completion_rows 只传 prompt/completion，tokenizer preflight 未传 tools；rollout 使用 environment_factory 的工具 schema。先抓实际渲染，再判定是否构成偏移。
2. assistant 历史究竟以 bare JSON content 还是 tool_calls 渲染，tool role、generation marker、EOS、空 assistant 和最后 tool message 的处理是否一致。不要只测试一个 list_files parser 示例。
3. 检查 collator 实际 labels：历史（包括错误动作）、system/user、tool observation 必须全部 -100；正确下一动作和约定结束 token 必须被监督；逐行确认无全掩码、无截断。现有长度差只能作预检估计。
4. 冻结状态初始上下文来源。build_fixed_state_contexts 当前沿用旧 SFT 的 initial messages，实际 reset baseline 只存哈希；需要与正常 rollout 的 issue + baseline 拼接方式比较。真实 observation 允许做有文档的噪声规范化，禁止加入答案。
5. 统一 navigation-first policy。检查 12 条 recovery 与 4 条 audited 混合行是否使用同一 system/tool 协议；现有 answer-source 一致性校验并不能保证 prompt 一致。
6. 检查选样覆盖。上一轮每状态只取 4 个目标，部分状态的首个导航/read 动作未作为该状态的监督目标；列出完整恢复链中哪些转移只出现在 masked history。不要将此直接判定为唯一原因。
7. 核对 task.max_steps 与新工具调用预算：前缀重放消耗的工具步数是否影响续接上限；明确 iteration 与 tool call 的区别。核对成功 run_tests 后是否还要求模型调用已失效的 finish。

产出 protocol-audit.json、真实 labels 抽样、最小复现测试和问题清单。门槛：训练/生成差异均能解释，目标监督边界正确，工具 round-trip 可执行。发现实现错误时先做最小修复、版本化协议，双方 baseline/candidate 使用同一修复。无证据时不得宣称已找到根因。

## P1：现有权重是否学到了目标（只读 GPU，先不训练）

依次加载 SFT60 与现有 candidate，冻结参数，计算相同真实输入上的目标动作 NLL，按定位/read、撤销错误、核心语义 edit、verify 分组，不能仅报告平均 loss。优先看最早分歧动作及核心 patch；相似文本或 exact match 不能替代执行成功。

在相同渲染上对三个训练状态各做一次 greedy 下一动作生成，每个 adapter 共三次。保存原始 tokens、解析结果和相对教师的行为差异，最多执行一个生成动作验证可执行性。模型合法结束或无工具动作计模型失败；网络故障单列。

判断：NLL 未改善说明这次更新没有明显学到目标；NLL 改善但 greedy 动作错误，优先检查曝光覆盖与动作竞争；greedy 正确而随机 rollout 失败，先诊断采样/多步累积问题。任何一项都不能单独证明完整修复能力。

## P2：有条件的微型拟合实验（一个候选，上限 12 steps）

仅 P0 通过且 P1 给出明确不足时执行。此处是新一轮预先定义的学习能力测试，不恢复旧失败候选继续堆训练。

从稳定 SFT60 重新开始；固定基座、LoRA、bf16、batch=1、gradient accumulation=4、LR=2e-5。seed 在运行前查重并锁定。保留三条完整恢复链的全部有效下一动作（当前 18 条），优先保证每条链的首个动作、核心 edit、撤销错误和 verify 均被监督。数据只用 train，教师 observation 逐步真实执行且最终完整 verifier 通过。此次允许刻意拟合这三个状态，标为诊断 adapter，不宣称泛化或直接替换稳定 adapter。

只做一条预注册训练曲线，在 step 4/8/12 保存检查点。使用逐状态轮转并记录实际采样 ID；声明精确曝光次数，不以 optimizer step 代替数据覆盖率。各检查点只做固定教师 NLL；只有预先声明的最终检查点做三个状态 greedy 完整续接，避免反复采样选最优结果。若全部关键动作组 NLL 持续恶化、非有限梯度或 OOM，立即停止。

门槛：三个训练状态 greedy 续接全部 strict success，且无新增失败。3/3 只是学习链路通过，不是泛化成功。不通过则停止本轮训练扩展；检查最早未学会的动作，不扩大到 14B、不做 LR/seed 网格、不加 GRPO。

## P3：完整流程迁移（仅拟合门槛通过）

首先锁定四个未参与本轮拟合的 train session 状态。若没有足够近编辑状态，可从 train 任务新采真实 rollout；将采集预算单列，最多四条，结果不反馈训练。不能把三个训练状态重新命名为留出集；同题不同 session 也只称 session-disjoint 开发诊断。

固定候选与 SFT60 配对各四次完整官方 verifier 续接。通过要求 candidate 至少 2/4 strict 且高于 baseline，新增失败不多于 baseline。若只会拟合、不能迁移，在这里停止。

通过后才从原始 issue 启动端到端配对：预先锁定新 seed、每 adapter 四条、4096 completion tokens、最多 12 次工具 iteration、temperature=1、top_p=.95。不提供 gold 路径或教师上下文。0/4 failure reduction 就停止；有修复才用第二个新 seed 重复同一候选和协议。两个 seed 各至少一次 strict success、合计至少 3/8 且高于配对 baseline，才进入跨任务 smoke。

若条件编辑已能通过但端到端仍失败，下轮独立研究导航：从 issue/异常对象到具体类，再到已观察导入的父实现。用真实 train 检索轨迹教这些转移，保持固定检索预算，禁止硬编码 callback 路径。优先扩展多题真实成功链，而非同一补丁同义改写；这不属于本轮自动追加训练。

## P4：GRPO 前置条件

在原始 issue 的新冻结 seeds 上可复现完整修复，优于同协议 baseline，其他 train/开发回归任务无明显恶化；对同一 prompt 下多条真实 rollout 计算组内奖励方差。仅 0 与 0.03 的有效编辑奖励差异不足以进入 GRPO。全部 rollout 都失败或全部都成功且组内零方差时，不人为制造奖励差异。

条件满足也仅写单步 GRPO 提案，给出奖励版本、KL/reference 配置、显存与运行时间估算、停止条件，等待用户明确授权。

## 成本、证据与停止纪律

优先完成 P0，再按 P1→P2→P3 门槛推进。单卡 48 GiB；P0 CPU 为主，P1 不更新权重，P2 最多一个候选/12 steps。P0–P2 设置 GPU 活跃运行时间上限 20 分钟，超时先报告进度与新证据，不自动扩容；实际费用依实例费率另算。评测新增长上下文可能超过此前 19.6 GiB 峰值，不能以旧峰值保证新运行不会 OOM。

保存 planned/started/completed/invalid 及 attempt/session/state IDs，断连仅按预注册规则补齐、原日志保留；不要推断用户新消息必然导致远程中断。保存原始生成 token、解析结果、训练真实 labels、源码与权重哈希、补丁和完整测试结果。训练 loss 用于链路诊断，verifier strict success 才用于能力门槛。运行结束清理本轮进程/容器/隧道并核对 GPU；云实例计费状态另行告知。

执行者首先只交付 P0/P1 审计结论；是否启动 P2 必须由上述证据支持，不能因为还有预算就继续训练。
