# 阶段 5：GPU 会话运行手册（2026-09-22）

**目的**：把付费时间只花在"验证"上，不花在"试错"上。开机后照着这份文档执行，每一步都有判据。

实例配置已由用户提供并记录在 `manifests/route-v2-gpu-instance.json`：
RTX PRO 6000 96GB ×1 / 22 vCPU / 110 GB 内存 / 系统盘 30 GB + 数据盘 81 GB /
**CUDA 13** / ¥6.98 或 ¥7.35 每小时。**CUDA 13 与 pin 的 cu130 栈匹配**，不需要改驱动或换 veRL 版本。

---

## 0. 用户需要提供的一件事

启动实例后，把面板上的 SSH 命令给我，形如：

```
ssh -p <port> root@connect.bjb1.seetacloud.com
```

`connect.bjb1.seetacloud.com` 在 worker 上可能被 DNS 污染，所以我还需要**面板上显示的真实 IP**
（上次是 `106.120.183.117`）。给我端口就够了，IP 我可以自己解析并验证。

**在此之前我不会开机，也不会产生任何费用。**

---

## 1. 我会先做（不计入 4 小时预算，属于准备）

| 步骤 | 命令 | 判据 |
| --- | --- | --- |
| 连通性 | `scripts/gpu/connect_probe.sh <port>` | 端口开放 + SSH banner 正常 |
| 传输代码 | Windows → worker → 实例（`scripts/gpu/deploy_bundle.sh`） | 实例上 `check_shell_syntax.sh` 通过 |
| 预检 | `python3 scripts/preflight_gpu_env.py` | 无 FAIL；记录 GPU 型号/显存/驱动/CUDA |
| 清点 | `bash scripts/gpu/inventory_disk.sh` | 拿到真实占用清单 |
| 瘦身 | `bash scripts/gpu/slim_disk.sh`（**先 dry-run，给你看，再 `--apply`**） | 释放 Qwen2.5 权重与缓存 |

**瘦身是前置条件**：数据盘 81 GB，而 Qwen3-8B 要 16 GB、veRL 的 venv 要 25–30 GB。
如果克隆把约 40 GB 的 Qwen2.5 权重带过来，不删就装不下。

---

## 2. 一次性 bring-up（也放在计时之前）

```bash
bash scripts/gpu/setup_instance.sh --slim
```

它会依次做：预检 → 清点 → 瘦身 → **在 pinned commit `1876b06d…` 上 clone veRL** →
`uv run --frozen --all-packages --extra vllm --extra fsdp` 物化环境 →
下载 Qwen3-8B 并用 `envs/` 里已固定的 `config.json` / `tokenizer_config.json` 的 sha256 校验。

**关键发现（读上游 `install.rst` 得到，不是猜的）**：veRL v0.9.1 **没有安装步骤**——
`uv run --frozen --extra vllm --extra fsdp` 直接从仓库里已提交的 `uv.lock` 物化 `.venv`。
`--frozen` 表示永不重新解析依赖，所以这比手写 pip 安装更可复现。

**这一步最耗时**（torch 2.11 + vllm 0.24 + flash-attn 的轮子，约 10–30 分钟），
所以我把它放在计时之前，并记录实际秒数。

模型问题的答案：**Qwen3-8B 没有预下载**。旧实例只有 route v1 的 Qwen2.5-Coder 快照。
下载走 `HF_ENDPOINT=https://hf-mirror.com`（`huggingface.co` 在 worker 上实测不可达）。

---

## 3. 计时会话：七步（预算 4 小时 / ¥40 上限）

| # | 步骤 | 通过的判据 | 预计 |
| --- | --- | --- | --- |
| 1 | 环境起来 | `import verl, vllm, torch` 且 `torch.cuda.is_available()` | 已在上一步完成 |
| 2 | 模型加载 + 一次真实工具调用 | Qwen3-8B 载入 vLLM；经过隧道调用 worker 的 `/v1/sessions` 拿到真实 observation | 15 min |
| 3 | 代表性长度 SFT forward/backward | 8K / 16K / 24K **总轨迹**各一次，记录 allocated/reserved/整卡峰值 | 30 min |
| 4 | 完整多轮 rollout | 一条完整 episode 落盘；`response_mask` 中工具 token 为 0、模型 token 为 1 | 20 min |
| 5 | 同题多轨迹分组 | `rollout.n=4`；记录有效组 / 全同组 | 30 min |
| 6 | 一次真实 GRPO 更新 | 1 个 optimizer step；reward 分布、grad_norm、KL、clip fraction、entropy | 30 min |
| 7 | 权重同步 + 保存/加载 | 更新后重新采样证明同步生效；LoRA 保存后可重新加载 | 20 min |

每一步独立落盘；OOM 或超时**记录边界后停止**，不做无记录的降级。

---

## 4. 本轮已确定的架构约束

**真实仓库执行留在 worker（VM），不放在 GPU 机。**
因为 AutoDL 实例本身就是容器，在它里面再跑 Docker 起任务沙箱不可靠。
项目已有的形态就是为此设计的：训练进程通过 loopback SSH 隧道 + bearer token 调用 worker 的 HTTP API。

**同时更正一个我先前的错误判断**：我以为 worker 的 5 GB 内存会把并发限制在 2 个容器左右
（因为每个容器用 `--memory 2g` 启动）。实测（`scripts/vm/measure_container_concurrency.sh`）：

| 并发 | 失败 | 单容器峰值 RSS | 宿主可用内存变化 |
| --- | --- | --- | --- |
| 1 / 2 / 3 / 4 / 6 | 0 | 约 80–81 MB | 基本不变 |

`--memory 2g` 是**上限不是预留**，所以内存不是瓶颈。
**仍未测**：真实多步 rollout（容器长时间存活、跑完整 pytest 套件）下的吞吐。

---

## 5. 到点就停

4 小时或 ¥40，**以先到者为准**，无条件下机关机，**不自动续租**。
额外停止条件见 `manifests/route-v2-gpu-instance.json` 的 `budget.stop_conditions`。
