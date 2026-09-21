# 阶段 0：仓库审查与旧路线冻结（2026-09-22）

审查对象：`D:\Developer\coding-agent-rl-lab`，工作区**干净**（`git status --porcelain` 为空），
审查基线 `main@27b86c2a24c9dc9c1eed7d5dafee8f7e7eb5388d`。
本次审查**未修改任何历史代码、数据或实验结果**；新工作从 `agentic-rl-mainline` 分支开始，
分支点就是上面的 commit。

已读取：`README.md`、`PROJECT_PLAN.md`、`docs/` 全部路线与诊断文档、
`docs/resume/project-prompt-v3`、`src/coding_agent_rl_lab/` 全部 50 个模块（按层分工深读）、
`tests/`（44 个测试文件）、`scripts/`、`work/` 与 `remote-artifacts/` 的 manifest 与日志、
最近 30 个提交与 `git log -S` 溯源。

执行的验证命令（本节其余结论都以此为准）：

| 命令 | 结果 |
| --- | --- |
| `python -m pytest tests -q` | **335 passed, 1 failed, 1 skipped**（唯一失败见 §5.1） |
| `ssh wesz@192.168.137.130 "docker images"` | Docker 29.7.2；**10 个固定 SWE-Gym 镜像全部在位**（7365/7385/7393/7446/7509/7514/7537/7607/7608/7646），各 4.23 GB |
| 工作区体积 | `remote-artifacts/` 106.3 MB、`work/` 122.2 MB（两者均被 gitignore，**未入库**） |

---

## 1. 当前真正跑通了什么（有直接证据）

按证据强度排序，只列能从产物反查的部分。

### 1.1 真实代码仓库里的多轮工具交互 —— 已跑通

- 真实 SWE-Gym 实例、真实 instance 镜像、真实 base commit 校验、真实 test patch 注入、
  真实 fail-before，全部在 Ubuntu VM 的 Docker 上跑过：
  `work/private/gold-selfcheck-20260922.json`（3/3 verified：7365 baseline `1 failed/1 passed` →
  gold 后 `2 passed`；7514 `3 failed/7 passed` → `10 passed`；7608 `2 failed/3 passed` → `5 passed`）。
- 参考修复**通过本项目的受限编辑工具**（而不是 `git apply`）也能走完，3/3 reachable：
  `work/private/repair-through-tools-20260922.json`。这条证明的是"工具约束没有把任务变成不可执行"，
  属于基础设施自检，不是模型能力。
- 跨主机模型 rollout：`remote-artifacts/swe-gym-abc-diagnostic-20260922-trajectories.jsonl`
  18 条、**0 条基础设施失败**、0 条 `policy_transport_error`、1 条 `policy_protocol_error`。
- 工具面在 `docker_environment.py` 里是 7 个动作（`list_files / search_text / read_file /
  replace_text / replace_lines / run_tests / finish`），全部 argv 形式、无 shell，
  `--network none`、`--cap-drop ALL`、`--pids-limit`、内存与 CPU 上限、测试文件防篡改。

### 1.2 轨迹与 provenance —— 已跑通且质量较高

- `Trajectory` schema v3 逐步保存：真实 prompt messages（`policy_input`）、模型原始输出、
  seed、延迟、token usage、工具 observation、violation；**不保存 API key**。
- 长扫描按 trajectory 原子写检查点，报告区分 `planned_trial_count / completed_trial_count /
  run_complete`；`--resume` 会逐字校验 schema、task、repetition、seed 与 policy manifest，
  不兼容直接拒绝。
- 这是一条**胜过多数自建项目**的能力，应当原样保留。

### 1.3 执行反馈可作为奖励 —— 已跑通

- 奖励全部来自 sandbox / 补丁解析 / verifier 输出，**没有 LLM judge**。
- 两个版本化奖励（`reward_shaping.py:13`）：`legacy-v1`（上限 0.50）、`conservative-v2`（上限 0.38）。
- 训练奖励与评测指标分离：评测只认"verifier 全通过 ∧ 有源码补丁 ∧ 无违规"。

### 1.4 参数更新链路 —— 只到"单步机制成立"

- 真实 SWE-Gym 任务上的单步 LoRA GRPO：`reward=[0,0.15]`、`reward_std=0.1061`、
  `grad_norm=0.4935`、`train_loss=-0.8747`
  （`remote-artifacts/grpo-shaped-line-edit-sft60-7509-mb1-2x4096-seed70002-training-report.json`）。
- 可信 fixture 上的单步 LoRA GRPO：`reward=0.5`、`reward_std=0.7071`、`grad_norm=0.1464`。
- SFT：completion-only、拒绝静默截断，60-step adapter 已保存，报告里有 adapter 权重前后 SHA
  与 `adapter_weights_changed`。
- **三个 GRPO 报告全部是 `optimizer_steps: 1`。仓库里不存在任何多步 GRPO 记录。**

### 1.5 明确的负面/中性结果（同样有价值，不要丢）

- 固定 held-out 上：基座与 fixture-LoRA 对照**均为 0/8**。
- v9/v11/v12 的 train 任务扫描：基座在 6 个 train 任务上零 verifier 奖励。
- `moto-7514` train curriculum 上曾出现 1/4 严格成功（失败数 1→0），但独立种子的完整任务
  仍是 0/4，且 3 条补丁都引入新增失败。

---

## 2. 已有最新诊断回答了什么

最新一轮是 A/B/C 模型瓶颈诊断（18 条）+ 第一次关键失败审计。它回答的是
"14B 基座在这些开发题上卡在哪里"，**不是**"GRPO 有没有用"。

### 2.1 它给出的结论

- 三臂 strict 全 0/18；但**第一次关键失败审计把"0"拆开了**，终止原因分布为：
  `no_edit_attempt 7`、`semantic_incomplete 7`、`recovery_loop 3`、`protocol_error 1`。
  也就是说这 18 条并不是同一个失败。
- 真正属于语义证据的只有 7 条，其中干净的两条是：
  `7514 A r1` 写对了 gold 附近的内容却**多写一份到 `responses.py`，把 7 个原本通过的测试全部打破**；
  `7514 A/C r2` 读过目标区域后选择 `errors='ignore'` **压症状**。
- 环境与评测本身**没有**失败：18/18 完成、0 基础设施失败、三题 fail-before + gold pass-after 全部自检通过。

### 2.2 它同时推翻了自己的一半结论（这点必须继承）

- 条件 C 的窗口对 3 题中的 2 题**没有覆盖修复区域**（7365 漏 2 个 hunk，7514 的 `models.py`
  漏掉主 hunk）。所以"C 给了上下文仍 0/6 ⇒ 语义瓶颈"**不成立**，只能下调为待验证。
- `replace_lines` 只按**文件**放行、不按**行范围**放行 → C 的 45 次编辑里 11 次是盲改，
  失败却被记成语法错误。
- 读失败会永久毒化该动作（9 步 / 2 条）。

### 2.3 因此"下一步该做什么"已经被上一轮写清楚了

审计文档 §7 定的第三道门槛是：**用 v4 辅助信息（sha256 `59bead89…`）+ 新的编辑约束重跑 18 条**，
并按终止原因分布判断，而不是只看 0 strict。这道门槛**尚未执行**，产物 `v4` 已生成、未使用。
这是旧路线**唯一**有明确停止规则的未完成实验。

---

## 3. 哪些代码可复用（按目标六层边界归位）

| 目标层 | 现有实现 | 判定 | 依据 |
| --- | --- | --- | --- |
| **environment** | `docker_environment.py`（容器生命周期、base commit 校验、test patch 注入、fail-before、清理）、`swe_gym.py`（row→task 映射、镜像名、已审计 test command） | **可复用（需适配）** | 不 import `model_policy.py` / `rollout.py`；但 `step()` 的 7 动作契约是本 scaffold 的，不是环境本身 |
| **environment** | `environment.py` 的 verifier 输出解析、失败节点比对、`graded_targets` | **可复用** | 纯 verifier 侧逻辑，假设 pytest `FAILED <node>` 摘要格式 |
| **agent/rollout** | `grpo_remote.py` 的 HTTP worker（`POST /v1/sessions`、`/actions`、`/finalize`，≥32 字符 bearer token、仅 loopback） | **可复用（价值最高）** | **与 agent scaffold 无关**；任何新 scaffold 都可以把它当"sandbox as a service" |
| **agent/rollout** | `rollout.py` 的轨迹落盘、原子检查点、严格 resume | **可复用** | schema v3 JSONL 自洽 |
| **data** | `rl_dataset.py` 的无泄漏筛选（排除连接故障、reference 策略、敏感元数据） | **可复用** | 筛选规则与模型无关 |
| **training** | `sft_train.py`（completion-only、无静默截断、报告含权重 SHA） | **可复用** | 薄封装，未复制框架源码 |
| **evaluation** | `rollout.py` 的成功口径、`failure_analysis.py` 的失败分类 | **可复用（需适配）** | 口径可保留，taxonomy 需要按新 scaffold 重定义 |
| **manifests** | `PolicyManifest`、`CodingTask.provenance`、`scripts/write_sha256_manifest.py` | **可复用** | 自洽且版本化 |
| **environment** | `scripts/check_pinned_gold.py`、`check_repair_through_tools.py`、`sync_and_validate_vm.ps1` | **可复用** | 是环境可信度的直接证据，必须保留 |

---

## 4. 哪些属于旧自定义路线（不继承、但保留）

这一节是本次审查最重要的判断。以下内容**不是可复用的基础设施，而是上一轮的自定义实验设计**。
按用户要求，它们必须保留可追溯历史，但**不得**进入新主线，也不得在同一命令下被悄悄替换。

| 旧自定义项 | 位置 | 为什么不能继承 |
| --- | --- | --- |
| 严格 JSON 单动作协议 `{"kind":…,"arguments":…}` | `model_policy.py:406-502`、`PROMPT_VERSION` `:22` | 协议、prompt、parser、训练数据四方耦合；换了 scaffold 就全部作废 |
| prompt 迭代线 v12→v15→v20→v24/v25 | `model_policy.py`、`swe_gym_sft.py` | 上一轮已证明它优化的是"搜到正确字面量"，而把文件与源码直接给足后仍 0/6 |
| 从 gold hunk 合成的四阶段 locate/inspect/edit/verify SFT 数据（204 条） | `swe_gym_sft.py` | **用户明令不再使用**："不要继续从 gold hunk 生成虚构搜索和读取历史" |
| bare-JSON response parser 与其探针、`sitecustomize` 屏蔽 vLLM | `grpo_train.py:18-28,402-476`、`scripts/grpo_sitecustomize.py` | 让该解释器**永远无法使用 vLLM**，是旧路线的绕行方案，不是资产 |
| 中间奖励 / 奖励形状工程（`legacy-v1`、`conservative-v2`） | `reward_shaping.py` | 新主线以真实修复成功为主要奖励；部分奖励需要有来源与防投机检查之后才谈 |
| 导航提示注入（`OBJECT_UNDER_FAILURE`、`SUGGESTED_PATH`、`PARENT_IMPLEMENTATION_PATH`） | `grpo_remote.py:44-80` | 与旧 prompt 线绑定的提示工程 |
| 7 个动作的循环守卫 / `premature_finish_refusal` | `environment.py:925-975, 1004-1130` | 守卫规则本身是可分析资产，但它是为旧协议的失败模式定制的 |
| TRL `GRPOTrainer.environment_factory` 单卡路线 | `grpo_train.py` | 只跑过 1 个 optimizer step，多步稳定性完全未验证 |

**旧入口保持可追溯**：上述命令与文件一律不改名、不删除；新主线使用新的入口名，
README 会明确写出"哪个入口属于哪条路线"。

---

## 5. 本次审查新发现的缺陷（旧路线当前状态并不干净）

### 5.1 `main@27b86c2` 的测试是红的（1 项），且是"半提交状态"

`tests/test_model_policy.py:91` 断言 `prompt_version == "coding-tools-json-v25"`，
而 `model_policy.py:22` 仍是 `"coding-tools-json-v24"`。
`git log -S` 显示：**最后一次提交改了测试，却没改源码**（同一个 commit 还给该测试文件加了 UTF-8 BOM）。

### 5.2 更严重：两个不同的 prompt 文本共用了同一个版本号

`git diff 6b24a43 HEAD -- src/coding_agent_rl_lab/model_policy.py` 显示，最后一提交在
`_SYSTEM_PROMPT` 里**新增了一条 `replace_lines` 只能改已读行范围的规则**，但版本号仍是 `v24`。
而 A/B/C 诊断（manifest 记录运行在 HEAD `6b24a43`）用的是**加这条规则之前**的文本，也记作 `v24`。

> 结论：`coding-tools-json-v24` 这个标识符现在**不能唯一确定 prompt 文本**。
> 所有历史轨迹的 `prompt_version` 字段因此不能单独用来判断模型当时看到的是什么。

这条直接违反用户"旧入口保持可追溯、避免同一标识对应不同实现"的要求，必须修。

### 5.3 依赖没有任何声明，且存在未记录的 overlay

- `pyproject.toml:10` 是 `dependencies = []`；仓库里**没有** `requirements*.txt` / lock / `environment*.yml`。
- 唯一 pin 在 `scripts/setup_autodl_grpo_env.sh`：`trl==1.12.0`、`transformers>=5.2.0`、
  `datasets>=4.7.0`；`torch` / `peft` 来自基础镜像的 `--system-site-packages`，**未固定**。
- 记录显示系统 `peft 0.17.1` 与 `transformers 5.16.1` 不兼容，靠一个**没写进仓库的**
  `/root/autodl-tmp/peft-0.21.0-overlay` 绕过。
- 换句话说：**旧路线当前不可复现**——换台机器装不出同样的环境。

### 5.4 观测保真度上的一个已知但有记录的限制

环境写盘的 observation 与策略实际读到的 prompt 不是同一个字符串：
`build_action_messages` 会做后缀截断 + 8000 字符共享历史预算。这不是缺陷（`policy_input`
逐字保存了真实 prompt），但解读"模型看到了什么"时必须读 `policy_input` 而不是 `observation`。

### 5.5 上游事实核查（本次新查，影响阶段 1）

- `OpenHands/OpenHands` 的默认分支**已经是一个 TypeScript/React 应用**（`src/`、`package.json`、
  `electron/`、`vite.config.ts`），旧的 Python `openhands/` 包在默认分支上**已不存在**。
  SWE-Gym 论文使用的 Python OpenHands scaffold 与今天的默认分支不是同一个东西。
- `verl-project/verl` 的 `recipe/` 是一个指向 `verl-project/verl-recipe` 的 **git submodule**，
  而 `verl-recipe` 的目录列表里**没有任何 SWE / SWE-Gym / Docker 环境相关 recipe**
  （只有 dapo、retool、spin、langgraph_agent、nemo_gym 等）。
- 因此"OpenHands + veRL 的现成连接方式"在本项目可查证的范围内**不存在**。
  详见阶段 1 报告。

---

## 6. 哪些结论仍无证据（必须写进任何汇报的"未知"栏）

1. **GRPO 相对 SFT 有增益** —— 无证据。三个 GRPO 报告都是 1 个 optimizer step，
   固定 held-out 上基座与 LoRA 都是 0/8。
2. **多步 GRPO 稳定** —— 无证据。从未跑过 >1 step。
3. **旧奖励形状能带来能力提升** —— 无证据。配对实验反而显示有效补丁增加的同时新增失败也增加。
4. **14B 基座"语义修复能力不足"** —— **不成立**（被自己的审计推翻）。
   干净证据只有 7 条 `semantic_incomplete`，且窗口缺陷尚未用 v4 重跑。
5. **SFT 有没有让模型获得真实修复行为** —— 只有 train curriculum 上的 1/4，
   独立种子完整任务是 0/4，且引入新增失败。不能算。
6. **跨仓库泛化** —— 无证据。全部 10 个任务都是 `getmoto/moto@5.0` 同一仓库。
7. **成本/预算** —— 仓库内**没有**任何 GPU 费率记录，只有时长上限。
8. **真实成功轨迹 SFT warm-start** —— 未做。现有 SFT 数据是 gold-hunk 派生的，
   不是模型自主成功轨迹；这正是用户要求补上的核心缺口。

---

## 7. 旧路线冻结记录

冻结基线：`main@27b86c2a24c9dc9c1eed7d5dafee8f7e7eb5388d`（工作区干净，无未提交改动，无 stash）。
新工作分支：`agentic-rl-mainline`（分支点即上述 commit）。

已固定的标识（本次实测）：

| 项 | 值 |
| --- | --- |
| 旧学生模型 | `Qwen/Qwen2.5-Coder-7B-Instruct`（训练）、`Qwen2.5-Coder-14B-Instruct`（诊断推理） |
| 旧 prompt 标识 | `coding-tools-json-v24`（**已知歧义，见 §5.2**） |
| 旧 GRPO 实现 | `trl==1.12.0` `GRPOTrainer.environment_factory`，`use_vllm=False` |
| 旧 GRPO 记录步数 | 1 optimizer step（×3 次实验） |
| 训练数据（gold 派生） | `work/private/swe-gym-train-gold-sft-v1.jsonl`，sha256 `40e6a0d5d2451d3d…` |
| 参考补丁 | `work/private/pinned-gold-patches.json`，sha256 `e9db5177b0d7772f…` |
| 诊断辅助信息 v4（未使用） | `work/private/abc-diagnostic-oracle-20260922-v4.jsonl`，sha256 `59bead8955c074b9…` |
| A/B/C 轨迹 | `remote-artifacts/swe-gym-abc-diagnostic-20260922-trajectories.jsonl`，sha256 `736eee8ecc576070…`，18 条 |
| A/B/C 报告 | `remote-artifacts/swe-gym-abc-diagnostic-20260922-report.json`，sha256 `86e40cd10ecd04fb…` |
| A/B/C 冻结 manifest | `remote-artifacts/swe-gym-abc-diagnostic-20260922-manifest.json`，sha256 `44b33af0a7c77cfd…` |
| SFT60 adapter 配置 | `remote-artifacts/sft-warm-start-v24-60step-adapter-config.json`，sha256 `0cdec8f0a7f486d4…` |
| 本机已有源码快照 | `snapshot-0038db4.tar`，sha256 `cc1cbd9239c0fbc2…`，附 15581 行逐文件 sha256 清单 |
| 固定 SWE-Gym 行缓存 | `work/private/swe-gym-train-rows.jsonl`，sha256 `6eb7cc4ad70f55f2…` |
| 环境镜像 | 10 个 `xingyaoww/sweb.eval.x86_64.getmoto_s_moto-*:latest`，均在 VM 上，各 4.23 GB |

> 说明：`work/` 与 `remote-artifacts/` 都被 `.gitignore` 忽略（`.gitignore:8-9`），
> 所以上述哈希是**离线核验**的唯一入口，必须随报告交付。
> 本次**没有**删除或改写其中任何文件。

---

## 8. 阶段 0 结论

**可以进入阶段 1。** 理由：

- 环境层、verifier、轨迹与 provenance 三块是本项目真实、可复用的资产，
  且已经通过真实 SWE-Gym 镜像和跨主机 rollout 验证；
- 旧自定义路线（协议 + prompt 线 + gold 派生监督 + 形状奖励）已经被上一轮自己的诊断判定为
  "瞄准的不是瓶颈"，且与用户要求的新主线冲突，应当冻结而非继续；
- 新的核心缺口被明确为：**真实成功轨迹的 SFT warm-start** 与 **多步、可归因的 GRPO**，
  两者都需要先在阶段 1 确定训练架构。

进入阶段 1 前需要先处理的、可在本地完成的三件事（均不消耗 GPU）：

1. 修掉 §5.2 的 prompt 版本歧义，把 `v24`（历史文本）与 `v25`（当前文本）区分开，
   并让 HEAD 的测试变绿；
2. 把依赖从"基础镜像 + 未记录 overlay"变成**显式声明的文件**（阶段 3 的一部分）；
3. 在新分支上建立 `manifests/`，把 §7 的冻结记录机器可读地写下来。

`main` 分支本身**不改动**，保持为旧路线的可追溯入口。
