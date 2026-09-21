# 阶段 1：训练架构选型（2026-09-22）

目标：在**单张 96GB GPU、首次 2–4 小时**的约束下，确定一条"真实 Docker 代码环境 + 多轮工具交互 +
执行反馈驱动 GRPO"的可复现主线，并且**不自行发明工具协议、训练数据格式、奖励体系或优化算法**。

本报告的每条外部事实都标注了来源与 commit；未验证的一律写"未验证"。

---

## 0. 结论先行

| 项 | 选择 |
| --- | --- |
| 训练框架 | **veRL**，release tag `v0.9.1` = commit `1876b06d0a3e4e71e06230be10af14492ca8a75b` |
| 多轮接入方式 | **veRL 原生 `AgentLoopBase`（token-in / token-out）+ 自写 `BaseTool` 包一层已有 Docker worker** |
| Agent scaffold | **不上完整 OpenHands**；采用 SWE-agent / OpenHands-v0.54 风格的**函数调用工具协议**（数据侧对齐，见阶段 4） |
| Docker 环境 | **保留本项目已有实现**（`docker_environment.py` + `grpo_remote.py` HTTP worker），veRL 只通过工具调用它 |
| 奖励 | **真实测试执行结果**（已有 verifier），二元为主 |
| 学生模型 | `Qwen/Qwen3-8B`，non-thinking，LoRA（fsdp/vLLM 路线） |
| 被否决 | 完整 OpenHands scaffold + veRL 自建桥接；Lego-RL；继续扩 TRL `environment_factory` 路线 |

**一句话理由**：veRL 是唯一把"多轮 agent 的 token 流"当成一等公民（框架自己持有 `prompt_ids` /
`response_ids` / `response_mask`）的成熟框架；而 OpenHands 与 veRL 之间**不存在**官方或已验证的适配，
自己搭桥会同时违反"不一边自建框架一边声称复现成熟方案"。所以选 veRL 的原生 AgentLoop，
只把 Docker 环境做成工具，这是**最小、可审计**的桥接。

---

## 1. 三个候选的核实结果

### 候选 A：OpenHands scaffold + veRL（用户点名优先核实）

**核实结论：不存在可用的现成适配。**

- `All-Hands-AI/OpenHands` 已重定向到 `OpenHands/OpenHands`（同一 repo id `771302083`），
  默认分支 `main` 的**主要语言是 TypeScript**，`contents/openhands` 返回 **404**、
  `contents/evaluation/benchmarks/swe_bench` 返回 **404**。
  也就是说：**SWE-Gym 论文用的那套 Python OpenHands 已经不在默认分支上了。**
- 论文级 scaffold 的冻结副本仍在：`SWE-Gym/OpenHands`（MIT，Python，
  `openhands/` 包完整，含 `evaluation/swe_bench/`），但**最后推送 2025-02-20**，
  且是 **CodeActAgent 2.1/2.2（函数调用之前）**的文本动作格式。
- 现代的 Python agent 是 `OpenHands/software-agent-sdk`（MIT，活跃），
  提供 `LLM/Agent/Conversation/Tool`，但**它自己的 LLM 是否接受 OpenAI 兼容 `base_url` 未验证**——
  这是能否接进自建推理端的前提。
- veRL 侧：`recipe/` 是 git submodule（`verl-project/verl-recipe`），
  **`verl-recipe@main` 的目录里没有任何 SWE / SWE-Gym / Docker recipe**。
- 唯一的第三方 OpenHands×veRL 集成是 **Lego-RL**（`LegoX/Lego-RL`，Apache-2.0，arXiv 2608.17393）：
  它把 Claude Code / OpenHands / OpenCode 接进 veRL，并用**进程内代理**在生成时刻记录
  token id、mask 和 logprob——这正是正确的做法。但它的 README 明确写 **"Needs 8× A100/H100-class GPUs"**，
  且依赖额外的 Harbor sandbox 与 K8s/Docker 编排。**预算内不可行。**

→ **裁定**：把 Lego-RL 当作**设计参考**（证明"进程内代理记录 token/mask/logprob"是可行且必要的），
不把它当作可运行方案。完整 OpenHands 接入需要自建大型桥接层，按用户规则应当转而寻找
已有的成熟 SWE agent RL 实现——也就是下面的候选 B。

### 候选 B：veRL 原生 AgentLoop + 把已有 Docker worker 做成工具（**选定**）

- `verl/experimental/agent_loop/agent_loop.py`（`v0.9.1` 与 `main` 文本一致）定义
  `AgentLoopBase.run(sampling_params, **kwargs) -> AgentLoopOutput`，
  `AgentLoopOutput` 字段为 `prompt_ids / response_ids / response_mask / response_logprobs /
  reward_score / num_turns / metrics / extra_fields`。`response_mask` 的语义在代码注释里写明：
  **1 = 模型生成 token，0 = 工具返回 token**。
- 注册方式：`@register("name")` 进 `_agent_loop_registry`；自定义 loop 从
  `actor_rollout_ref.rollout.agent.agent_loop_config_path` 以 hydra `_target_` 加载；
  逐样本选择来自 dataset 列 `agent_name`。打开它需要的开关（来自上游 recipe 的启动脚本）：
  `rollout.name=vllm`、`rollout.mode=async`、`agent.agent_loop_config_path=…`、
  `data.return_raw_chat=true`、`trainer.use_legacy_worker_impl=disable`。
- 工具接口 `verl/tools/base_tool.py`：`create / execute / calc_reward / release`（全 async），
  `execute` 返回 `(ToolResponse, reward, metrics)`。**没有一等的远程 HTTP/MCP 工具**
  （`ToolType` 只有一个成员，注释写着 MCP 已移除），但官方明确指引：
  "每个 trajectory 需要建立和拆除的状态（sandbox VM、临时目录、DB 行）请用 `BaseTool`"
  —— 容器正是这种情况。
- **veRL 核心不含任何 SWE-bench / SWE-Gym / Docker 集成**；`verl/tools/` 只有
  `base_tool.py / function_tool.py / schemas.py / tool_registry.py`。

→ 需要写的适配是**一个工具文件 + 一个 loop 配置**，这是可审计的小改动，不是自建框架。

### 候选 C：上游 `recipe/swe_agent`（SWE-agent CLI + ModelProxy + Docker，**作为参考实现**）

这是本次核实中最重要的发现：在 submodule 的 pinned commit
`verl-recipe@e7f889574b8301cc0f0fc1d57c6d67f31ffeb689`（= veRL v0.9.1 的 gitlink）上，
存在一个**完整的、Docker 沙箱化的 SWE-agent GRPO recipe**：

```
recipe/swe_agent/{swe_agent_loop.py, model_proxy.py, subprocess_runner.py,
                  reward.py, patch_extractor.py, config.py, trajectory.py,
                  config/swe_agent_config.yaml, prepare/, docker/, example/}
```

它的数据流是：`SWEAgentLoop.run()` → 起一个 `ModelProxy` HTTP 服务 →
在 Docker 容器里跑 `sweagent run` 子进程 → 子进程把 OpenAI Chat 请求打到代理 →
代理转给 veRL 的 vLLM 做 on-policy 生成 → 结束后用 patch 与参考 patch 比对算奖励 → 返回 `AgentLoopOutput`。

**采用它的什么，不采用它的什么**（本节在读完其源码后**修正过一次**，见下方"修正"）：

| 采用 | 不采用 | 理由 |
| --- | --- | --- |
| token-in / token-out 的生成路径 | 子进程 + 文本 HTTP 代理这一层 | 见"修正"：它的 id 确实来自进程内 `server_manager.generate`，但文本往返带来的对齐校验是**失败即整条作废**的，代价高且不必要——我们的工具是自己的代码，可以直接持有 token 流 |
| "RL 训练中绝不允许改写历史 observation"这条明确警告 | 它的合成数据集与 patch-diff 奖励 | **它的奖励是把生成补丁与参考补丁做文本相似度，完全不读 pytest 结果**；这与本项目"执行反馈驱动"的要求直接冲突 |
| 它的容器生命周期与配置分层思路 | 它的硬件档位与 token 预算 | 启动脚本把 `infer_tp`/`train_sp` 都设成 `GPUS_PER_NODE`，token 预算按 SP=8 标定；单卡不是被禁止，是**没被标定过** |
| "每条轨迹的 alignment 失败必须显式处理"这条设计 | 它的失败处理方式 | 对齐失败时它返回 **1 个 pad token 且 mask=1**，也就是说坏轨迹会以退化样本进入训练，而不是被丢弃 |

> **修正（2026-09-22，读完源码后）**：我最初写的是"它必须重新分词去还原 `response_ids`"——**这条不准确，予以更正**。
> 实际机制是：vLLM 的 `output.token_ids` / `output.log_probs` 由**进程内**的 `server_manager.generate` 返回，
> 直接以 mask=1 追加进训练序列；子进程只传文本，而 `TrajectoryReconstructor` 用**重渲染 chat template 后逐 token 相等**
> 来校验"文本与 id 一致"，不一致就整条作废。所以它是 token-in/token-out **加上一道 fail-closed 的对齐闸门**，
> 不是"靠重新分词还原"。这个区别影响我们参考它的深度：我们沿用其"生成路径直取 id"的部分，
> 但**不沿用**其"回合间必须靠重渲染校验"的脆弱性——我们的 loop 自己持有 `prompt_ids`，tools 只回文本、
> 由 loop 编码一次，不需要把整段历史重渲染来对账。

> **上游文档原文警告（必须遵守）**：`last_n_observations` / `closed_window` 这类"动态改写历史
> observation"的 history processor **不能**用于 RL，因为回滚修改历史会让 rollout 与 PPO 更新阶段的
> token 序列不对齐，导致 **KL 爆炸、不收敛**。要控上下文只能用 `max_observation_length`
> （**进入历史之前**截断）。

**读源码得到的三个单卡约束**（直接进阶段 5 的调参清单）：

1. 启动脚本里 `infer_tp=$GPUS_PER_NODE`、`train_sp=$GPUS_PER_NODE`，并把
   `actor_max_token_len_per_gpu=2560` / `log_prob_max_token_len_per_gpu=5120` 按 **SP=8**
   标定（注释原文 "With SP=8: effective limit = 2560*8 = 20480 tokens/micro-batch"）。
   单卡时 SP 塌回 1，动态 batch 预算掉 8 倍，而 `max_prompt_length + max_response_length = 12288`，
   于是微批退化到近乎一条序列、激活与 logit 显存尖峰。**必须重新标定，不能只改 `GPUS_PER_NODE=1`。**
2. 它默认同时驻留 FSDP actor + vLLM + **KL reference**，且 `param_offload=false` /
   `optimizer_offload=false`。8B × 3 放一张 96GB 卡是**算得出来的 OOM**，必须先开 offload 或减配。
3. 沙箱内存默认 `--memory=8g`、每 worker 最多 32 并发，而我们的 VM 只有 **5GB 总内存**。
   → **rollout 农场必须放在 GPU 机器上，不能放在这台 VM 上**；VM 只适合串行验证。


---

## 2. 用户指定问题的逐条回答（以选定方案为准）

| 问题 | 答案 | 依据 |
| --- | --- | --- |
| 模型调用由谁负责 | **veRL** 的 `LLMServerClient` → 异步 vLLM server。loop 只调 `generate(prompt_ids=…)`，不自己发 HTTP | `verl/experimental/agent_loop/agent_loop.py` |
| Agent 工具循环由谁负责 | **我们自己写的一个 `AgentLoopBase` 子类**（注册名如 `swe_docker_tool`），循环体是"生成 → 解析工具调用 → 执行 → 追加 observation"。工具集合复用 veRL 的 `ToolAgentLoop` 语义 | `tool_agent_loop.py` + `base_tool.py` |
| Docker 环境由谁创建和释放 | **本项目已有代码**：`docker_environment.py` 建/删容器，`grpo_remote.py` 暴露 HTTP worker；veRL 工具在其 `create()` / `release()` 里调用 `/v1/sessions` 与 `DELETE`。veRL 的 `_call_tool` 用 `finally: await tool.release(...)` 保证每条轨迹都回收 | `env` review §2；`BaseTool` 生命周期 |
| 如何保存模型实际看到的 observation | 工具把容器 stdout（已按字符上限截断）作为 `ToolResponse.text` 返回；loop 把它**编码一次**并追加，同时把 `response_mask` 追加 0。完整逐轮消息另存一份 trajectory JSONL 用于审计 | veRL mask 语义 + 本项目 `Trajectory` |
| 如何得到生成 token、采样 logprob 和 response mask | **框架原生**：vLLM 返回 `output.token_ids` 与 `output.log_probs`；loop 用 `ct_merge_assistant_token` / `ct_merge_context_msg` 把助手 token 与工具 token 追加，mask 由 `continuous_token.align_response_metadata` 保持位置对齐；训练侧 `rollout.calculate_log_probs: True` 默认开启 | `agent_loop.py`、`verl/utils/tokenizer/continuous_token.py` |
| 如何把完整多轮轨迹送入 GRPO | `AgentLoopOutput` 直接给出 `prompt_ids/response_ids/response_mask` → `AgentLoopWorker._agent_loop_postprocess` 拼成 `input_ids` 与 `response_mask` 张量 → `verl/workers/utils/losses.py::ppo_loss` 里 `log_prob` 由**训练引擎重新前向计算**，`old_log_probs` 来自 rollout，`response_mask` 同时作为损失掩码 | `losses.py`（已读取） |
| 如何把更新权重同步回 rollout | veRL 的 `AgentLoopManager` 在 rollout 阶段 `wake_up` 推理实例，做 FSDP→vLLM 权重同步（LoRA 下支持 `layered_summon=True` 降低显存峰值）；rollout 结束再 `sleep` 释放 KV cache | `agent_loop.html` 架构节；`ppo_lora.rst` |
| 是否存在已验证的现成适配 | **OpenHands×veRL 不存在**。存在的是 SWE-agent×veRL（上游 recipe，8 卡档位）与 OpenHands×veRL 的第三方 Lego-RL（8 卡档位）。两者都不能按原样在单卡跑 | 见 §1 |
| 单卡 LoRA / 多轮 / Qwen3-8B 支持情况 | 框架都支持，但**组合未验证**：`examples/tuning/lora/run_qwen3_8b_fsdp.sh` 默认 `NGPUS_PER_NODE=2` 且 `ROLLOUT_TP=2`，单卡必须同时改这两个；官方 device tuning 只记录到 **≤7B GRPO-LoRA 的 1×H100 最小值**，8B agentic 单卡**无文档** | 同上 |

---

## 3. 选定版本的固定信息

| 组件 | 版本 / commit | 来源 |
| --- | --- | --- |
| veRL | tag `v0.9.1`，commit `1876b06d0a3e4e71e06230be10af14492ca8a75b` | GitHub tags API |
| veRL 依赖（该 tag 的 `pyproject.toml`） | `torch==2.11.0` / `torchvision==0.26.0` / `torchaudio==2.11.0`、CUDA **cu130**、`vllm==0.24.0`、`sglang==0.5.12`、`transformers==5.9.0`、`ray[default]>=2.41.0`、`tensordict>=0.8.0,<=0.10.0,!=0.9.0`、`flash-attn==2.8.3`、python `>=3.10,<3.13` | 同上 |
| `recipe/swe_agent`（只作参考） | submodule gitlink `e7f889574b8301cc0f0fc1d57c6d67f31ffeb689`；其中 `REQUIRED_VERL.txt` 另指 `verl@bcb638649a50e58494a8ddd92085ad1174f674b8`（main，torch 2.13.0 / vllm 0.29.0 / transformers 5.12.1） | 该目录的 `README.md`、`REQUIRED_VERL.txt` |
| 学生模型 | `Qwen/Qwen3-8B`，non-thinking，LoRA（`target_modules=all-linear`） | 本项目决策 |
| Docker worker | 保留本项目实现；VM `wesz@192.168.137.130`，Docker 29.7.2，8 vCPU / 5 GB RAM / 75 GB 空闲 | 本次实测 |

> **注意版本冲突**：`recipe/swe_agent` 的 `REQUIRED_VERL.txt` 要求的 veRL commit 与 v0.9.1 的
> gitlink **不是同一个**。如果将来采用那个 recipe，必须整包按它的 commit 对齐；
> **不可**把它的 recipe 文件拷到 v0.9.1 上混用。

---

## 4. 模块职责与数据流（目标架构）

```text
                 ┌────────────────────── veRL (v0.9.1) ──────────────────────┐
                 │  RayPPOTrainer → AgentLoopManager → AgentLoopWorker        │
                 │      │                                    │                │
                 │      │ wake_up / sleep（FSDP ⇄ vLLM 权重同步）             │
                 │      ▼                                    ▼                │
                 │  Async vLLM server  ◄──── generate(prompt_ids) ────┐      │
                 │                                                     │      │
                 │                              ┌──────────────────────┴───┐  │
                 │                              │  SWEDockerAgentLoop       │  │
                 │                              │  (我们的一个文件)          │  │
                 │                              └───────┬──────────────────┘  │
                 │                                      │ BaseTool.execute    │
                 │                        ┌─────────────▼──────────────┐     │
                 │                        │ SWEGymDockerTool (一个文件) │     │
                 │                        └─────────────┬──────────────┘     │
                 └──────────────────────────────────────┼────────────────────┘
                                                        │ HTTP (loopback + bearer)
                 ┌──────────────────────────────────────▼────────────────────┐
                 │  本项目已有：grpo_remote.py HTTP worker                    │
                 │    POST /v1/sessions          → docker_environment.reset   │
                 │    POST /v1/sessions/{id}/actions → .step（7 个工具动作）   │
                 │    POST /v1/sessions/{id}/finalize → verifier 结果 + 奖励   │
                 │    DELETE /v1/sessions/{id}      → .close（强制删容器）     │
                 └──────────────────────────────────────┬────────────────────┘
                                                        │ docker exec（argv、无 shell）
                                       真实 SWE-Gym 实例容器（--network none）
```

六层职责与现有代码的映射：

| 层 | 目标职责 | 复用来源 | 处置 |
| --- | --- | --- | --- |
| environment | SWE-Gym 实例、Docker 生命周期、测试执行、补丁与结果 | `docker_environment.py`、`swe_gym.py`、`verifier.py` | **复用**（工具面按新协议改造） |
| agent/rollout | 成熟 scaffold、工具、真实轨迹、终止与预算 | 新增 `verl_agent/` 薄层；轨迹落盘复用 `rollout.py` 的 schema | **新增小文件 + 复用持久化** |
| data | 教师/官方轨迹导入、筛选、split、SFT 渲染 | `rl_dataset.py` 筛选复用；渲染**替换** | 见阶段 4 |
| training | SFT、GRPO 配置与薄启动入口 | veRL 的 `main_ppo` + 我们的 hydra override 文件 | **不复制框架源码** |
| evaluation | 固定任务集、统一预算、逐题结果、失败类型、成本 | `rollout.py` 口径 + `failure_analysis.py` | **复用并扩展** |
| manifests | 代码/模型/数据/环境/奖励/运行版本 | 新增 `manifests/`（阶段 0 已建） | **已建立** |

**目录策略**：不做全仓库搬迁。新代码集中在
`src/coding_agent_rl_lab/verl_agent/`（loop、tool、配置）与 `manifests/`、`envs/`、`experiments/`，
旧模块原地保留并且**入口名不变**。

---

## 5. 尚未验证的兼容点（阻塞项按优先级）

| # | 未验证项 | 为什么重要 | 怎么验证（不需要 GPU 的先做） |
| --- | --- | --- | --- |
| 1 | **veRL v0.9.1 在单卡上同时跑 FSDP actor + 异步 vLLM + 8B LoRA 的显存是否够** | 直接决定方案是否成立 | 只能在租卡后验证；先准备 `ROLLOUT_TP=1`、`NGPUS_PER_NODE=1`、`layered_summon=True`、`gpu_memory_utilization` 的扫描脚本 |
| 2 | **`AgentLoopWorker` 的并发与我们的 Docker worker 是否能匹配** | worker 是同步 HTTP 服务；veRL 侧是 asyncio。若工具调用阻塞事件循环，吞吐会崩 | 在 VM 上用假 loop 跑并发工具调用压测（纯 CPU） |
| 3 | **VM 只有 8 vCPU / 5 GB RAM** | 每个容器限 2GB；并发 4–8 个容器就可能打爆内存，直接限制 GRPO 每步耗时 | 在 VM 上实测并发 1/2/4/8 的容器内存与耗时曲线 |
| 4 | **Qwen3-8B 的 chat template 与函数调用格式** | 决定工具协议能否直接用；non-thinking 的 `enable_thinking=False` 是否被模板支持 | 只用 tokenizer 文件（几 MB）即可预检；在 VM 上下载 tokenizer |
| 5 | **`data.return_raw_chat=true` 之下数据集需要哪些字段** | loop 依赖 `raw_prompt`；字段错就直接跑不起来 | 读 `RLHFDataset` 代码 + 用最小 parquet 在 CPU 上跑 `AgentLoopWorker` 的单测 |
| 6 | **上游 recipe 的 token 还原方式** | 决定我们是"参考它"还是"避开它" | 读 `swe_agent_loop.py` 与 `trajectory.py`（下一步） |
| 7 | **SWE-Gym 其他仓库镜像的 test command** | 现表只有 `getmoto/moto@5.0`；多仓库 split 需要逐个审计 | 在 VM 上拉 1–2 个新仓库镜像跑 baseline（纯 CPU） |
| 8 | **HuggingFace 直连在本机与 VM 都不可达** | 模型/数据集下载通道 | **已解决（2026-09-22 实测）**：`huggingface.co` curl 超时（000），但 **`hf-mirror.com` 可用**（HTTP 206），`modelscope.cn` 与清华 PyPI 镜像也可达。已用 hf-mirror 成功取到 `Qwen/Qwen3-8B` 的 `config.json` 与 `tokenizer_config.json`、`nebius/SWE-rebench-openhands-trajectories` 的 `README.md`/`tools.json` 与数据集 API。数据集与模型获取走 `HF_ENDPOINT=https://hf-mirror.com`。**租卡机是否同样可达 hf-mirror 必须开机后第一件事确认。** |

---

## 6. 最小可运行路径（按顺序，每步都有明确的"过/不过"）

**阶段 5 的 2–4 小时只做这七件事，不追求训练效果：**

1. 环境起来：装好 pinned veRL 环境，`nvidia-smi` + `python -c "import verl"` 通过；
2. **模型加载 + 一次真实工具调用**：`Qwen3-8B` 载入 vLLM，loop 成功调用一次 `/v1/sessions` 并拿到真实 observation；
3. **代表性长度的 SFT forward/backward**：用约 8K / 16K / 24K **总轨迹**长度各跑一次，记录峰值显存；
   （长度必须来自真实代码交互，不许用只有几句话的样例冒充）
4. **完整多轮 rollout**：一条完整 episode 跑通，落盘 trajectory，确认 `response_mask` 与工具 observation 对齐；
5. **同题多轨迹分组**：`rollout.n=4`，确认同组来自同一策略版本；
6. **一次真实 GRPO 更新**：1 个 optimizer step 跑完，记录 reward 分布、有效组比例、grad_norm、KL；
7. **权重同步 + 保存/导出/加载**：更新后重新采样一次（证明权重同步到了 vLLM），并保存 LoRA 后重新加载。

**通过的判据**：七步全部有日志与产物；任何一步 OOM/超时就**记录边界**并停止，不做无记录的降级。

---

## 7. 预算与风险（在开机前必须先说清楚）

已知的**实测**吞吐事实（来自旧路线，可用于估算）：

- A/B/C 诊断：18 条轨迹 × 最多 24 步，**23 分钟**（vLLM 加载另约 4 分钟），即约 **77 秒/轨迹**。
- GRPO 一步 = `任务数 × group size` 条轨迹。若 6 题 × group 4 = 24 条 ≈ **31 分钟/步**（串行）。
- VM 只有 8 vCPU / 5 GB RAM，**并发是主要瓶颈**，不是 GPU。

因此对 2–4 小时首次验证的**诚实预期**：能覆盖上述七步 + 最多个位数 optimizer step，
**不可能**得到任何能力提升结论。

风险与缓解：

| 风险 | 缓解 |
| --- | --- |
| veRL + vLLM + flash-attn 安装吃掉 1 小时以上 | 优先用官方预构建镜像；把安装脚本与依赖清单在**开机前**准备好并离线校验 |
| 单卡 8B + vLLM + FSDP 显存不足 | 先跑 LoRA + `layered_summon` + 低 `gpu_memory_utilization`；准备好"只跑 rollout 不跑训练"的降级路径用于诊断 |
| 租卡机访问不了 HuggingFace | 开机前确认；必要时把模型权重预先下载/上传 |
| 时间为零导致"什么都没验证" | 把七步做成脚本，每步独立可跑、独立落盘，任何中断都保留已完成证据 |

**费用**：截至目前项目内**没有任何费率记录**，此前约 ¥6.98/小时只是参考报价。
按用户要求，开机前必须给出**具体实例、实时价格、最长运行时间与总费用上限**并取得授权；
本报告不预设价格。

---

## 8. 下一步（阶段 2 / 3 的本地可逆准备，不消耗 GPU）

1. 读上游 `swe_agent_loop.py` / `trajectory.py` / `model_proxy.py`，把它的 token 还原方式查清楚
   （决定参考深度，见 §5 #6）；
2. 建立 `envs/`：把 veRL 的 pinned 依赖写成显式文件（替换掉旧路线"基础镜像 + 未记录 overlay"的做法）；
3. 建立 `src/coding_agent_rl_lab/verl_agent/`：只写 loop 与 tool 两个文件 + 配置，并配单测
   （用假 vLLM server 与假 worker，纯 CPU 可测）；
4. 在 VM 上做 §5 #2、#3、#7：并发压测、容器内存曲线、新仓库 test command 审计；
5. 建立 `manifests/route-v2-*.json`：固定模型 ID/revision、tokenizer、chat template、`enable_thinking`、
   数据哈希、环境镜像摘要。

阶段 5 的授权请求会在上述 1–5 完成后一次性提出。
