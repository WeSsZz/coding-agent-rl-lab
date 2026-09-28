> SUPERSEDED FOR EXECUTION: follow [next-session.md](next-session.md) and manifests/route-v2-model.json. Historical Qwen3 defaults, automatic model cleanup and untimed paid setup below must NOT be executed.

# 阶段 5：短租 GPU 可行性验证——计划与授权请求（2026-09-22）

本轮**不训练、不追求效果**。目标只有一件事：在真实租来的单卡上，按顺序验证"这条链路到底能不能跑"，
并把每一步的实测数字留下来。任何一步过不去，就记录边界并停止，不做无记录的降级。

在此之前**没有开过机、没有充过值、没有调用过任何付费 API**。

---

## 1. 为什么要先跑这一段，而不是先写桥接代码

阶段 1 选定的是"veRL 原生 AgentLoop + 把已有 Docker worker 做成工具"。但**我们自己的适配代码还不该先写**，
理由是：

- 上游 `recipe/swe_agent` 的启动脚本把 `infer_tp` / `train_sp` / token 预算都按 **SP=8** 标定，
  单卡时 `ulysses_sequence_parallel_size` 塌回 1，动态 batch 预算掉 8 倍；
- 它默认同时驻留 **FSDP actor + vLLM + KL reference**，且 `param_offload=false`，
  8B×3 放一张 96GB 卡是**算得出来的 OOM**；
- 也就是说"单卡能不能装下"本身就是一个未验证的假设，而它**决定了适配层要不要按 offload/分片重写**。

所以顺序是：**先用 veRL 自带的 agentic 示例把框架跑起来并量出显存**，再按实测结果写我们的工具适配。
反过来做，等于在一个可能装不下的地基上浇混凝土。

---

## 2. 开机前必须已经就位的东西（本地已完成）

| 项 | 状态 |
| --- | --- |
| 固定 veRL 版本与依赖 | `envs/verl-v0.9.1.pins.txt`（来自 v0.9.1 自带 `pyproject.toml`，文件 sha256 已记录） |
| 固定模型与 chat template | `envs/qwen3-8b-chat-template.jinja` + manifest（sha256 `a55ee1b1…`，LF，来源 tokenizer_config sha256 `d5d09f07…`） |
| 固定工具协议 | `envs/openhands-v0.54-tools.json`（sha256 `d13c473d…`）+ `manifests/route-v2-tool-protocol.json` |
| 任务划分 | `manifests/route-v2-task-split.json`（13 train / 10 dev / 8 final-test，3 个仓库，全部已审计 test command） |
| Docker 环境与 verifier | 已在 VM 上实测通过（3 仓库的 fail-before + 失败节点精确匹配） |
| 环境预检脚本 | `scripts/preflight_gpu_env.py`（驱动 CUDA、显存、内存、磁盘、HF 可达性、包版本；**已在本机跑通**） |
| 模板渲染预检 | `scripts/preflight_chat_template.py`（19 项断言全过，无需 GPU/transformers） |

---

## 3. 七步验证（按顺序，每步独立落盘）

| # | 步骤 | 通过的判据 | 预计耗时 |
| --- | --- | --- | --- |
| 1 | 环境起来 | `preflight_gpu_env.py` 无 FAIL；`uv sync --extra fsdp --extra vllm` 成功；`import verl, vllm, torch` 且 `torch.cuda.is_available()` | 30–60 min（**最不可控**） |
| 2 | 模型加载 + 一次真实工具调用 | Qwen3-8B 载入 vLLM；AgentLoop 成功调用我们的 worker 并拿到真实容器 observation | 15 min |
| 3 | 代表性长度的 SFT forward/backward | 约 8K / 16K / 24K **总轨迹**各跑一次，记录 `torch.cuda.max_memory_allocated/reserved` 与整卡峰值 | 30 min |
| 4 | 完整多轮 rollout | 一条完整 episode 跑通并落盘；确认 `response_mask` 里工具 token 为 0、模型 token 为 1 | 20 min |
| 5 | 同题多轨迹分组 | `rollout.n=4`，同组来自同一策略版本；记录有效组 / 全同组数量 | 30 min |
| 6 | 一次真实 GRPO 更新 | 1 个 optimizer step 跑完；记录 reward 分布、grad_norm、KL、clip fraction、entropy | 30 min |
| 7 | 权重同步 + 保存/导出/加载 | 更新后重新采样证明权重同步到 vLLM；LoRA 保存后能重新加载 | 20 min |

**总计约 3–4 小时**（不含失败重试）。步骤 3 的长度必须来自**真实代码交互**的轨迹，
不允许用只有几句话的样例冒充 16K/24K。

---

## 4. 记录项（每一步都要写下来，缺一项就不算完成）

- GPU 型号、**实际总显存**、驱动版本、驱动报告的 CUDA 版本；
- 每个阶段的 `torch.cuda.max_memory_allocated` / `max_memory_reserved` 与 `nvidia-smi` 整卡峰值；
- CPU 核数、内存、磁盘余量；
- rollout / 容器测试 / 训练 / 权重同步**各自的耗时**；
- token 数、并发数、有效组数；
- **OOM 与失败的配置**（这比成功的配置更有价值）；
- 实际在线分钟数与费用。

---

## 5. 授权请求（需要用户确认后才能开机）

**这是本阶段唯一需要用户决定的事。** 我无法从本机查询 AutoDL 的实时价格，所以需要用户确认下列三项：

| 项 | 我的提案 | 需要用户确认 |
| --- | --- | --- |
| 实例 | 单张 **RTX PRO 6000 96GB**（若该档位不可用，第二选择是单张 A100/H100 80GB；两者都够跑 8B LoRA 的显存验证） | 具体型号与是否可租 |
| 价格 | 项目内**没有任何费率记录**；此前 ¥6.98/小时只是参考 | **实际单价** |
| 最长运行时间 | **4 小时**（步骤 1 若超过 60 分钟仍未起来，按失败处理并停止） | 是否同意 |
| **总费用上限** | **¥40**（≈4 小时 × ¥7 + 余量）。到点无条件关机 | 是否同意这个上限 |

**硬性停止条件**（不需要再问，我会直接执行）：

1. 步骤 1 超过 60 分钟未起来 → 停止，写出失败记录；
2. 步骤 3 在 8K 就 OOM 且关掉 offload 后仍 OOM → 停止，记录"单卡 8B agentic 不可行"的边界；
3. 任何一步出现无法归因的基础设施错误且重试 2 次仍失败 → 停止；
4. 到 4 小时或 ¥40 → 停止并关机，**不自动续租**。

**不做的事**：不在这一轮启动正式 SFT 或正式 GRPO，不用参考补丁冒充能力，不在无记录的情况下缩短正式任务。

---

## 6. 开机后第一件事（顺序不能变）

1. `nvidia-smi` + `python scripts/preflight_gpu_env.py --json preflight.json`
   → **确认驱动 CUDA ≥ 13**。pinned 栈是 **cu130**；如果驱动只到 12.x，必须**先**决定
   "升级驱动"还是"换 veRL 版本"，不能带着不兼容往下走；
2. 确认能访问 `hf-mirror.com`；不能的话模型权重必须预先上传；
3. 确认 Docker 可用且能拉取任务镜像（本机与 VM 的 Docker Hub 都被 DNS 污染，
   租卡机大概率也需要 mirror 方案）。

这三件事都在**花大钱之前**，且都不需要训练。
