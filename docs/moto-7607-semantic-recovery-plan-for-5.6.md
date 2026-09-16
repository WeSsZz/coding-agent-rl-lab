# 交给 5.6 的执行计划：moto-7607 真实状态语义恢复

## 任务目标与边界

在 48 GiB GPU 预算下，提高完整官方 moto-7607 verifier 上的实际修复率。先诊断、构建可执行的 train-only 语义恢复轨迹，再做一个小规模 SFT 候选及配对评测。按下面的阶段门槛推进，不默认执行所有阶段。没有可复现的 failure reduction 就停止 GPU 扩展；即使通过 GRPO 门槛，也只准备方案并告知用户，不启动 GRPO。

此文件是待执行计划，不代表下列实验已经完成。阈值是控制成本的工程筛选标准，不是统计显著性保证。用户当前要求的是计划；由用户派发本文件后，再执行其中实验。

## 已知事实与待验证假设

- 9/16 从真实 rollout 重建的 recovery 数据只有 2 条，最终监督目标都是打开 callback 文件；从稳定 SFT60 直接做 1 step 后，seed 121100 的完整 verifier 为 0/4 strict、0/4 failure reduction，3/4 新增失败，rewards=[0.03,0,0,0]。
- 0.03 来自有效、未恶化但未修复的补丁。组内 reward 方差不等于修复能力，也不足以支持 GRPO。
- 此轮没有同 seed 的未更新 SFT60 对照。不能把它与其他 seed、其他 adapter 或其他协议的历史结果混合，宣称 SFT 提升或下降。
- 14B parent-path v4 实验已在两个导航 seed 合计命中 4/8，但完整 verifier 没有 failure reduction。这支持“正确编辑也是瓶颈”，不证明 7B 的导航问题已经解决。
- 当前 sft_recovery.build_suggested_path_examples 用 gold 行的初始 system/user 加一个真实 search/observation 构建样本，未保留完整 session 前缀；它是局部恢复样本，不是完整真实轨迹回放。
- 当前 sft_cumulative 会拼接各 hunk 的既存 observation，并按完整 assistant 文本去重。编辑后再次读取同一范围可能是必要动作；旧 observation 是否仍与累计编辑后的状态一致，需要回放验证，不能先判定正确或错误。

工作假设：完整、状态一致、经 verifier 验证的“定位 → 阅读相关实现 → 最小编辑 → 验证”监督，比继续重复两条路径提示更有可能产生训练信号。该假设需由下面的对照验证。

## 先读这些文件

- docs/moto-7607-sft60-recovery-followup-20260916.md 和同名 JSON。
- docs/train-failure-reduction-search-20260914.md。
- docs/navigation-direction-check-20260915.md 和同名 JSON。
- src/coding_agent_rl_lab/sft_recovery.py、sft_cumulative.py、sft_train.py。
- src/coding_agent_rl_lab/grpo_remote.py、grpo_evaluate.py、grpo_train.py、reward_shaping.py。
- scripts/build_parent_path_sft.py。

开始前查阅适用 AGENTS.md，核对本地/Ubuntu/AutoDL 工作树和源码哈希。保留所有已有未提交改动。新增目录、文件及输出优先；需修改既有 builder 时先保留其原版本及旧数据可重现入口。AutoDL 副本可能没有 .git，使用源码 manifest 记录版本。

## 已核验的连接与证据

Windows → Ubuntu：wesz@192.168.137.130，私钥 C:\Users\Junwe\.ssh\id_ed25519_codex_vm。

Ubuntu → AutoDL：root@connect.bjb1.seetacloud.com:31719，私钥 /home/wesz/.ssh/id_ed25519_autodl。bjb2:32245 上次在 SSH banner 前重置。每次启动前重新验证连接、GPU 和证据内容，不盲用历史端口。

项目路径：Windows D:\Developer\coding-agent-rl-lab；Ubuntu /home/wesz/coding-agent-rl-lab；AutoDL /root/autodl-tmp/coding-agent-rl-lab。

AutoDL 训练环境 /root/autodl-tmp/grpo-env/bin/python。基座 Qwen2.5-Coder-7B-Instruct snapshot c03e6d358207e414f1eca0bb1891e29f1db0e242，路径见旧 training-report.json。

稳定 SFT60：/root/autodl-tmp/sft-grpo-dynamic-v2-60step-seed73001/final-adapter。
权重 SHA-256：55c947e02b404c3d150e3a2eabd4f55dd50c34641d8d68495f7fbb00c573081c。

最新证据目录：/root/autodl-tmp/moto-7607-sft60-recovery-followup-20260916。
归档：/root/autodl-tmp/moto-7607-sft60-recovery-evidence-20260916.tar.gz。
归档 SHA-256：95d46453ca623cd84fc5f8e6110e985de7d21b62b20a512b7e8d46a6f9853fd2。

其他输入：

- /root/autodl-tmp/train-full-20260914-sft-cumulative-from-sft60-7607-seed106100/tool-traces.jsonl。
- /root/autodl-tmp/train-full-20260914-sft-cumulative-from-sft60-7607-10step-seed107100/tool-traces.jsonl。
- /root/autodl-tmp/train-full-20260915-sft-recovery-7607-seed108100/。
- /root/autodl-tmp/full-parent-path-14b-v4-7607-seed116400/。
- /root/autodl-tmp/navigation-parent-path-evidence-20260916.tar.gz，SHA-256 5d65bdcce1ad82d1efd417bd07f49ac477c15b37585a76f18cf88419afb0b17c。
- AutoDL 项目 work/private/swe-gym-grpo-sft-7607-cumulative.jsonl 及配套报告；Ubuntu 官方缓存 work/swe-gym-development-rows.jsonl。

## 阶段 A：离线归因及 verifier 自检，暂不使用 GPU

1. 从上述已保存轨迹按 task_id + session_id 重建独立 session。核对 action 顺序、响应、最终 patch 和 reward audit 的映射；无法关联的记录单列为 incomplete，禁止把相邻 JSONL 行当同一 session。
2. 为每条轨迹输出：是否发出可解析工具调用、是否读到 callback、是否看到需要修改的完整函数、首次编辑位置、工具错误、最终修改、baseline/final failed test IDs、回归 test IDs、是否完成验证。gold 路径命中只供离线诊断，不作为部署时决策依据。
3. 以首个阻断点分类：导航未到达、阅读不足、编辑语义错误、编辑执行错误、未验证/未结束。允许多个附加标签，但首个阻断点互斥。比较 patch 与 train gold 时解释具体行为差异，不能只用文本 diff 距离判断。
4. 优先解释最新 3 条 1→3 failures：是语法/导入故障、共享前置条件破坏还是回调语义错误？从工具响应、patch 和失败报告取证。若轨迹没存 patch，重放已记录动作恢复，不凭印象推断。
5. 在干净 Docker 环境验证完整官方 baseline，再施加完整 train gold patch 验证。预期 baseline 为 1 fail + 2 pass，gold 为全通过；同时核对官方 test IDs、base commit、测试命令、测试文件哈希、容器镜像及资源限制。若不成立，先修复/说明环境问题，停止模型训练。
6. 保存 audit.json、简短诊断报告及 gold 回放证据。gold 和答案数据留在私有 train 目录，不能进入评测 prompt、模型搜索索引、held-out 数据或工具提示。

产出与门槛：可重建的 session、明确的主要失败阶段、可信 baseline/gold verifier。若真实状态不足以恢复，要报告缺失项，不伪造 observation。

## 阶段 B：构建真实状态的语义恢复样本

建议新增 sft_semantic_recovery.py 和对应测试/CLI，保留现有 sft_recovery 行为，避免旧实验失去可重复性。

1. 从 A 的 train sessions 选择 3～6 个有代表性的分歧状态：接近正确实现的导航前缀、已读相关实现但准备错误编辑的状态、已经写入错误补丁后的状态。状态必须真实存在；不要为凑样本增加同义改写。
2. 在干净 base commit 上逐步执行完整前缀，保留正确的会话历史、实际文件状态和实际工具响应。含错误编辑的状态需要在该状态上生成恢复动作，不能直接套用针对 base 文件的 gold 行号或替换串。
3. 在 train gold 监督下构造最短的正确续接：必要的 read/search → 可执行最小 edit → 完整 run_tests → finish。修复逻辑可以由教师编写，但每个工具响应必须实际执行取得；最终必须由完整官方 verifier 验证，无测试修改/安全违规才接纳。
4. 路径必须能由当前仓库、issue、已观察到的符号/导入推导。训练时允许 gold 提供编辑答案，但不要让依赖隐藏 gold 路径的定位捷径伪装成真实检索能力。
5. 每个前缀/下一动作构成 completion-only 样本。保留完整必要历史，仅对正确续接目标算 loss；历史中的错误动作只作为上下文。只接纳最后全通过的完整恢复轨迹，避免将普通失败 rollout 的动作当正确答案。
6. 预计得到 12～30 条有效动作样本，数量由状态覆盖决定。先覆盖关键编辑和验证状态，再补定位。所有样本保留 source trace SHA、session、action index、base commit、前缀重放结果、目标动作、最终 verifier 证据、teacher 来源。
7. train-only 在 builder 入口和 sft_train 加载处双重校验。显式允许真实的答案来源类型，必要时版本化 schema；不能把教师新构造的恢复 patch 全部冒充官方原始 gold patch。
8. 去重依据至少包含状态/历史和目标动作，不按工具调用文本全局去重。编辑后同样的 read_file 可以保留。排除重复/无效样本并记录原因；空集、跨 split、输出已存在时直接失败。
9. 做 tokenizer 预检，目标不截断；记录真实训练渲染的上下文长度和监督 token 数。CPU 重放每条完整恢复轨迹，确认工具行为、文件状态与样本 observation 一致。

关键测试：跨 session 不串接；非 train 拒绝；不存在的目标拒绝；不完整 observation 不伪造；错误 patch 后恢复成功；重复读取在状态变化后保留；工具响应/目标一致；输出覆盖拒绝。用小 fixture 覆盖通用逻辑，再用真实 7607 CPU 集成验证。

门槛：至少 3 个不同真实分歧状态的正确续接能稳定通过完整 verifier。若不足，停在数据/工具诊断阶段，不启动 SFT。

## 阶段 C：一个 7B 小规模 SFT 候选

以稳定 SFT60 为唯一初始 adapter，建立独立输出目录；不要沿着此前失败候选继续堆 step。固定当前工具协议，训练与评测一致。第一候选只改变监督数据；如必须改变协议，单列协议版本并给 baseline 使用同一协议。

- 默认 bf16、batch 1、gradient checkpointing，沿用 SFT60 LoRA 配置，不更换基座/量化方式/rank。
- 学习率 2e-5，训练 seed 122001（先检查是否使用过）。
- 约 75% 新语义恢复 + 25% 已审计 train 工具样本，按明确 ID 列表和权重确定性混合；恢复轨迹按 session 平衡，避免最长轨迹支配全部样本。
- 完整覆盖混合数据约一遍，gradient accumulation 4；optimizer steps=ceil(实际样本数/4)，上限 10。若数据规模导致一次覆盖超过上限，预先确定 session 平衡子集并报告，而非临时扩大训练。
- 在模型加载前固定随机种子；核对现有训练入口对 dropout/RNG 的实际处理。保存命令、超参数、数据及代码 SHA、输入输出权重 SHA、逐 step loss/grad norm、显存峰值、耗时与权重差异。
- loss 降低只是训练链路检查，能力判断取决于下面的冻结评测。

第一候选无有效更新、出现 OOM 或工具格式明显退化时停止并诊断。OOM 不自动租更大 GPU，也不默默改变量化后混合比较。

## 阶段 D：先诊断编辑，再做端到端配对对照

在评测开始前写冻结 manifest：模型/adapter、协议/工具、完整官方 verifier、预算、seed、planned 数量、停止条件。候选训练完成后不得根据某个 seed 的结果再训练并继续把它当盲测。

第一关：固定状态续接诊断（最多 8 条）。

- 从已有真实轨迹预先保留 4 个未用于构造训练样本的 session 状态，分组按 session 而非样本行；不足时明确标记“训练状态拟合检查”，不得称留出集。
- 未更新 SFT60 与候选各续接 4 次，使用同一组输入状态、seed、采样参数、工具预算及完整 verifier。
- 每条 continuation 从重放后的实际状态启动，传入真实完整历史，所有编辑均经过工具。保持与端到端一致的动作协议。
- 该指标是已给上下文条件下的编辑能力，不能合并到端到端成功率。门槛：候选至少 2/4 完整成功且高于 baseline；若 baseline 已达 3/4 或 4/4，说明当前诊断不支持“编辑是主瓶颈”，返回归因，不增加 SFT。
- 若候选 0/4，停止端到端扩展；检查数据/掩码/训练动作是否正确。至多允许修复一个有证据的实现错误后重跑，不做学习率/seed 网格搜索。

第二关：端到端新 seed 配对（首组 8 条）。

- seed 122100，SFT60 与候选各 4 条；从原始 issue 和正常 baseline observation 开始，不能提供目标文件、gold patch 或前面诊断中的人工上下文。
- 每条 4096 completion tokens、最多 12 次工具调用、temperature=1.0、top_p=0.95。与历史保持一致；若实现中“iteration”与工具调用数有区别，明确记录，双方一致。
- 完整官方 1 fail + 2 pass 的 verifier，conservative-v2 原样保留，测试超时和 Docker 资源双方一致。禁止 navigation-only 或只保留单个 failure 的 curriculum。
- 候选 0/4 failure reduction，停止第二 seed 和 GRPO；如果状态续接能修复、端到端失败，按轨迹确认是否导航/上下文获取失败，下轮再做独立导航实验。

第三关：确认性 seed（仅第二关有修复才执行，再 8 条）。

- 新 seed 122200，两种 adapter 各 4 条；复用冻结协议和同一候选，禁止看到结果后继续调参。
- 通过标准：候选两个 seed 各至少 1/4 完整成功、合计至少 3/8，成功数高于同 seed SFT60，新增失败 rollout 总数不高于 baseline。
- 未达标记为不稳定/无增益，停止扩大。8 条样本只作早期门槛，不宣称统计显著或跨任务泛化。

评测可靠性：保存 planned/started/completed/invalid 全部数量及归因；有效空动作/未修复应计失败；环境/网络故障单独标记、保留原日志，并按预先声明规则补齐，禁止悄悄排除模型失败。report 必须检查 optimizer_steps=0 与 adapter SHA 未变。

## 阶段 E：小规模跨任务检查与 GRPO 提案

仅 D 第三关通过才执行。先在预先锁定的其他 train 任务中选两个做配对 smoke（建议 7514、7646，每个 adapter 每题 2 条，共 8 条），判断是否破坏工具能力。随后以 regression 7385、7608 做一次锁定候选的配对检查（同样共 8 条），不把 regression 轨迹变成训练数据；一旦用于选择/诊断，就不能再称其为未见验证集。

held-out 任务保持封存；不因 moto-7607 是已训练题上的成功宣称泛化。若反复根据 regression 调参，必须在报告中注明开发集复用，最终泛化需要新的独立评测。

GRPO 提案至少要求：两个冻结新 seed 都有可复现完整修复、候选相对同 seed baseline 有增益、拟采用奖励下组内方差大于零，且没有观察到工具能力/回归恶化。若全部 rollout 都成功导致方差为零，也不强行制造方差。

即便通过，只交付一页单步 GRPO 方案及估算成本，等待用户明确启动授权。可讨论完整修复/失败数奖励如何避免让 0.03 的编辑奖励主导，但奖励调整应独立版本化评估，不能在本轮 SFT 对照中同时修改。

## 成本和停止纪律

- A/B 以现有轨迹及 CPU Docker 回放为主，先验证监督轨迹可执行，再占用 GPU。
- C 仅一个候选、最多 10 optimizer steps，沿用当前 48 GiB 实例。
- D 首关 8 次短续接；通过后首组端到端 8 次；有修复才再做确认性 8 次。上限 24 次生成，其中端到端最多 16 次。
- E 额外最多 16 次 rollout，仅确认性门槛通过才做。遇到明显回归可提前停止。
- 不承诺按 GPU 型号即可推断费用；记录实际运行分钟，若可获实例费率再计算。进程退出不等于停止云实例计费；及时告知用户可以关机，不自行关机或更改租赁。
- 基于新证据的一个修复可继续；没有新诊断的重复训练、反复换 seed 找阳性、扩大模型或多步 GRPO均停止。

## 交付和最终报告

新增可重放 builder、必要测试、冻结实验 manifest、私有数据 manifest、训练日志/adapter、配对 eval traces 与 reward audits。给出导航到达率、条件编辑成功率、端到端完整成功率、failure reduction、新增失败、planned 分母与有效工具轨迹数。不要把不同模型/协议/seed 的指标拼成一个提升数。

远端完整归档包含源码版本、必要输入或输入哈希及可达路径、训练命令、adapter、patch/测试证据、汇总和 SHA manifest；本地保存不含私有答案的 Markdown/JSON 报告。保留原始失败结果。最终报告明确当前最早失败阶段、通过/未通过的门槛、已花 GPU 分钟、是否还有计算进程及是否启动 GRPO。

执行完停止本轮 worker、隧道和临时容器，确认 GPU 空闲；保留证据与 adapter，不覆盖旧归档，不修改 unrelated 未提交代码。
