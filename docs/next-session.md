# 下一次会话：官方 SWE-Gym OpenHands 交互与 RL 准备

> 2026-09-28 路线更新：当前模型为 Qwen3-8B，先用 SWE-Gym 官方 OpenHands fork 完成真实任务交互与干净 verifier，再建立基线、接入 RL。执行入口为 [`swegym-openhands-rl-route.md`](swegym-openhands-rl-route.md) 和 [`qwen3-gpu-handoff.md`](qwen3-gpu-handoff.md)。下方 Qwen2.5-Coder SFT 内容仅是历史记录，不再充当执行指令。

> 下方内容是旧阶段记录。旧 SWE-Gym/GRPO 路线已冻结；README 后续旧章节、旧 PROJECT_PLAN 条目及其报告只作历史材料。不要使用旧 204 条 gold-patch 合成样本、旧策略协议、adapter 或结果作为当前数据/基线。细节见 [`legacy-swe-gym-grpo-route.md`](legacy-swe-gym-grpo-route.md)。

以下历史实验使用 `manifests/route-v2-model.json` 中的 Qwen2.5-Coder-7B-Instruct；当前模型以 `manifests/route-v3-qwen3-model.json` 为准。此前 phase5 手册和下方旧命令均不适用于当前执行。

## 2026-09-23 48GB GPU 实测结果

### 后续精度审计与真实容器验证

- 已纠正重载 loss 差异的归因：Trainer 包装后的 forward 启用 BF16 autocast，直接重载后的 forward 原先没有。并非已证实的随机 BF16 归约误差。
- 保存的 32K adapter 上三次重复对照：原 Trainer 包装 loss 均为 0.7661163806915283；不使用 autocast 的重载 loss 均为 0.7669700980186462；恢复 BF16 autocast 后三次均与原值完全相同。
- 训练入口现在显式统一两侧 autocast，并恢复 1e-4 容差。重新执行完整 32,766-token smoke 成功，重载前后 loss 均为 0.7662872672080994，196 个 adapter 张量更新且重载逐张量一致。
- 新报告位于 GPU 实例 `work/gpu-smoke-32k-precision-fixed/smoke-report.json`；原始报告保留，不覆写。
- Linux VM Docker 29.7.2 可用。在独立目录 `/home/wesz/route-v2-validation-20260923`，使用当前源码和冻结开发任务 getmoto__moto-7365，校验 base_commit、题目、测试及 gold patch 哈希后，原始代码测试失败、参考补丁后通过，容器清理成功。这仅是基础设施验证，不是模型成绩，也未验收新工具协议循环或独立干净 checkout verifier。
- 下一步仍需实现并验证与训练数据一致的工具调用执行循环、独立干净 checkout 判定及模型 rollout，再进行 Base/SFT 对比。VM 原项目未被覆盖。

- 单卡 RTX 4090 48GB，PyTorch 2.8.0+cu128、Transformers 4.57.1、PEFT 0.17.1。
- 标准交叉熵在 22,843-token 完整轨迹上因物化完整词表 logits 耗尽显存；不得通过截断轨迹掩盖该问题。
- 训练入口现固定使用 Liger Kernel 0.8.3 的融合线性层交叉熵，依赖见 `envs/trajectory-gpu.requirements.txt`。
- 22,843-token smoke 通过，训练 loss 0.920600，峰值分配显存 27,385,847,296 bytes。
- 最长 32,766-token smoke 通过，训练 loss 0.766566，峰值分配显存 32,587,005,952 bytes。
- 两次均有 196 个 adapter 张量更新；最终 adapter 保存后逐张量重载完全一致。原先放宽到 1e-3 的 loss 判据已被上面的精度修复和 1e-4 严格重跑取代。
- 最终报告位于远端 `work/gpu-smoke-48g-32k/smoke-report.json`；adapter 位于同目录的 `adapter/`。
- 本地正式测试范围：426 passed / 1 skipped。尚未执行正式 SFT 或 Base/SFT 任务成功率评测。

## 已完成的本地准备

- 安装/下载入口读取统一模型清单，固定 revision，不自动选取任意已有模型。
- 校验 config、tokenizer config、全部权重分片的结构和上游文件内容哈希。
- 禁止安装时自动清理；保留模型、adapter、Hugging Face 和 ModelScope 缓存。
- 关键预检失败立即停止；CUDA 不可用不能报准备成功。
- 安装进程截止时间从平台计费起点算起，包含安装和下载。

## 2026-09-22 本地继续准备的实际结果

### 最新：完整短轨迹与单步训练入口已准备

- 来源固定为 nebius/SWE-rebench-openhands-trajectories @ `35455389ab51bf5e2306bfd436ef72d0f98bf882`，CC-BY-4.0。沿用原始 OpenHands 工具定义、系统消息和全部 observations。
- 首个 Parquet 行组 4,096 行，范围读取压缩数据约 127 MB；按序列化文本长度选每个任务最短成功尝试，保留前 128 个不同任务。不是随机/代表性抽样。
- 2 条与评测仓库重叠被排除。126 条候选中，完整序列 <=16K 为 0、<=24K 为 4、<=32K 为 65。
- 固定模型配置 `max_position_embeddings=32768`，未修改 RoPE 或扩大原生上下文。选择 32K 是数据容量选择，不代表 96GB GPU 已实测可训。
- 最终 65 条来自 65 个任务、62 个仓库；长度 21,449–32,766；总 token 1,884,388，其中监督 token 533,137。
- `work/route-v2/sft-short32k/train.jsonl` 与同目录 `manifest.json` 已导出；可追踪摘要在 `manifests/route-v2-sft-short32k.json`。
- 每条轨迹一个完整序列；仅助手内容/动作与结束标记参与监督。padding、system、user、tool 内容均不参与 loss。拒收整条超长轨迹，不保留其早期回合来补数量。
- 使用真实 tokenizer 的 offset mapping 标定监督区间；拒绝边界跨 token。对一条 22,592 token 真实轨迹与原逐前缀 token 校验实现逐项比较，input_ids、mask、labels 完全一致。
- 默认数据检查入口 `python -m coding_agent_rl_lab.trajectory_train` 已在这 65 条真实数据上通过；不加载权重、不训练。
- 本次最终全量测试：420 passed / 1 skipped（4 条既有 collection warning）；新增 Bash 入口语法检查通过。
- GPU 单步入口已在单卡 48GB RTX 4090 上实测通过最长 32,766-token 真实序列：LoRA r16/alpha32、BF16、gradient checkpointing、SDPA、Liger 融合交叉熵、AdamW、batch1、一步更新；检查权重确实变化且有限、保存 adapter、逐张量重载、比较 loss 并报告峰值显存。此入口仅做 smoke，不执行正式 SFT。
- 新执行层/干净 checkout verifier 仍未验收；上游成功标签没有本地复现。Formal SFT、Base/SFT 任务评测、GRPO 均未执行。

复现本次 CPU 数据准备（已有固定 tokenizer 目录）：

```powershell
$env:PYTHONPATH='src'
python scripts/fetch_trajectory_pilot.py --output work/route-v2/short-source-new --short-successes 128
python -m coding_agent_rl_lab.trajectory_sft --input work/route-v2/short-source-new/sample.jsonl --output work/route-v2/short-prepared-new
work/route-v2/cpu-venv/Scripts/python.exe scripts/preflight_trajectory_tokens.py --prepared work/route-v2/short-prepared-new --tokenizer work/route-v2/tokenizer --output work/route-v2/short-preflight-new.json --max-length 32768 --complete-only
work/route-v2/cpu-venv/Scripts/python.exe scripts/build_trajectory_training_data.py --prepared work/route-v2/short-prepared-new --preflight work/route-v2/short-preflight-new.json --tokenizer work/route-v2/tokenizer --output work/route-v2/sft-new
python -m coding_agent_rl_lab.trajectory_train --data work/route-v2/sft-new --output work/route-v2/data-check-new
```

实例到位后，只先验证依赖和一步训练（项目已安装，权重/数据已传输且模型校验通过）：

```bash
export SESSION_STARTED_AT='<actual-provider-billing-start-unix-seconds>'
bash scripts/gpu/run_trajectory_sft_smoke.sh \
  --data work/route-v2/sft-short32k \
  --output work/route-v2/gpu-smoke \
  --model-path '<verified-model-directory>' \
  --inventory '<pinned-model-inventory.json>'
```

便宜 GPU 的第一步使用完整轨迹上限，避免直接用最长 32K 样本撞显存：

```bash
export SESSION_STARTED_AT='<actual-provider-billing-start-unix-seconds>'
bash scripts/gpu/run_trajectory_sft_smoke.sh \
  --data work/route-v2/sft-short32k \
  --output work/route-v2/gpu-smoke-24k \
  --model-path '<verified-model-directory>' \
  --inventory '<pinned-model-inventory.json>' \
  --max-smoke-tokens 24576
```

`--max-smoke-tokens` 只从已有完整轨迹中选择不超过上限的最长样本；没有符合条件的完整轨迹就失败，
不会截断、拼接或补造样本。通过 24K 后再移除该参数验证 32K。该 smoke 仍不是正式 SFT，
也不能把 loss 或显存通过解释为模型能力提升。

该步骤仍可能因训练依赖兼容性或 32K 显存失败，须读取错误和测量后调整；不能直接改成截断训练。
timeout 覆盖模型校验/加载、训练、保存和重载，沿用同一计费起点；**不负责云实例关机**。

### 以下为先前首 32 条试样的历史检查

- 新增 `src/coding_agent_rl_lab/trajectory_sft.py`，保持旧协议入口独立。
- 已从固定版本真实数据取首 32 条，15 条上游成功、17 条失败；这是格式试样，不是正式训练集。
- 15 条成功来自 15 个任务，共 859 个助手回合。参数 JSON 反序列化、JSON Schema、调用/返回配对、工具定义一致性检查通过。
- 排除所有开发/最终测试仓库（moto、conan、pydantic）；原数据缺少 base_commit，因此采用仓库级隔离。
- 实际使用 transformers 5.9.0、固定 Qwen2.5 tokenizer，在 CPU 检查了所有 859 个回合的完整历史。
- 16,384 token 上限下：169 个回合通过，690 个超长；15 条轨迹中 **0 条完整轨迹能全部保留**。无 token 前缀不一致错误。通过项的监督 token 总量 17,769。
- **不得直接拿 169 个早期回合开训并称为成功轨迹 SFT**。它会丢掉后期修复和结束行为。下一步先筛完整短轨迹或验证成熟 scaffold 原有的历史管理机制，不能随意截断 observation。
- 证据：`manifests/route-v2-trajectory-pilot.json`、`work/route-v2/prepared-validated/report.json`、`work/route-v2/token-preflight.json`。
- 全量测试 411 passed / 1 skipped；GPU forward/backward、运行时回放与正式训练未执行。

可复用：任务元数据、分割清单、测试结果解析、日志/产物哈希、模型检查。旧 Docker 资源隔离可复用，但新 shell/edit 执行与干净 checkout 上的判定还需接通和验证。
不能直接复用：旧 204 条合成训练集、旧七动作策略入口、旧 adapter 与旧评测数字（不能作为新框架的对照）。

CPU 复现（PowerShell，所有路径在项目目录）：

```powershell
$env:PYTHONPATH='src'
python -m coding_agent_rl_lab.trajectory_sft --input work/route-v2/source/sample.jsonl --output work/route-v2/prepared-new
work/route-v2/cpu-venv/Scripts/python.exe scripts/preflight_trajectory_tokens.py --prepared work/route-v2/prepared-new --tokenizer work/route-v2/tokenizer --output work/route-v2/token-preflight-new.json
```

隔离 CPU 环境依赖记录在 `envs/trajectory-cpu.requirements.txt`；真实模型权重没有下载。
`scripts/fetch_trajectory_pilot.py` 提供有大小上限的重取入口（需要 pyarrow/pandas/fsspec/aiohttp）；本轮实际取样使用等价的范围读取脚本，新增下载入口尚未端到端复跑。
本地试样位于 gitignored 的 work；版本和哈希已记入清单。Parquet 采用范围读取，未下载整文件验证 LFS 哈希，不宣称独立验证源文件完整性。

## 还没有完成的 SFT 前提

已有上面 65 条通过 CPU 检查的完整真实轨迹，但未完成 GPU 和运行时验收；没有新工具执行层。
旧 `sft_train.py` 绑定旧协议/固定任务，不能直接作为新 OpenHands 轨迹的训练入口。
依赖预检通过不等于 SFT 已准备完成；不得用旧 204 条合成样本填补数据缺口。

正式 SFT 前需要：

1. 获取固定来源的小批真实成功轨迹，核对许可、成功标签、工具格式。
2. 核查 shell 持久状态、文件编辑、输出截断、结束动作是否与运行环境一致。
3. 排除开发/最终测试任务与近重复，保存接受/拒绝清单和数据哈希。
4. `function.arguments` 反序列化为对象，保留真实 observation，不伪造工具历史。
5. 用真实 tokenizer 检查长度与 loss mask；环境输出不得参与策略监督，超长不能静默截断。
6. 验证对应 SFT 入口、LoRA 配置、checkpoint 导出和再加载；先单 batch 后正式训练。

此前 web 工具获取数据卡失败；本次使用固定 revision 的镜像通过 shell 下载成功，已经做了上述试样检查。
已有单步验证候选数据，仍不能宣称正式训练和评测链路就绪。

## 开机前的本地检查

```powershell
python scripts/model_contract.py --field id
python scripts/preflight_chat_template.py
python -m pytest tests/test_model_contract.py tests/test_preflight_chat_template.py tests/test_verify_route_manifest.py -q
```

## 实例启动后的命令结构（Linux/GPU 尚待实测）

先执行 `scripts/gpu/inventory_disk.sh`、`scripts/gpu/find_local_model.sh` 清点已有权重。
`LOADABLE` 仅表示候选，不能证明分片完整或模型版本正确。

```bash
# 使用实际平台计费起点和清点后明确选择的模型目录。
export SESSION_STARTED_AT='<provider-billing-start-unix-seconds>'
export MODEL_DIR='<existing-Qwen2.5-Coder-7B-Instruct-snapshot>'
bash scripts/gpu/setup_instance.sh --skip-model
```

`--skip-model` 禁止下载；已有模型验证失败就停止，不自动降级/更换。
确实没有权重时，先确认下载耗时和剩余预算，再决定是否去掉开关。
不要传 `--slim`，该入口已经拒绝自动删除。

校验读取约 15GB 权重，耗时也计费。上游 inventory 记录固定 revision，镜像不构成独立签名证明。
成功检查 `setup-report.json`、`model-verification.json` 和 `verified-model-path.txt`；
失败检查 `setup-failure.txt`、`setup.log`。

## 预算必须从平台计费起算

历史记录为 4 小时 / 40 元；启动前核对当前授权和实际费率，不自动扩展。
连接、传输、安装、下载、校验、训练全部计入同一预算。
`setup_instance.sh` 要求计费起点，并以 `timeout` 限制其进程到四小时截止。
**停止进程不等于实例关机，也不会自动停止计费。** 必须确认平台定时关机或实际执行人工关机。
后续 SFT/GRPO 继续沿用同一个截止时间，不能每个脚本重新获得四小时。

## GPU 到位后的验收顺序

数据和训练入口通过验收后，用真实轨迹做短、中、长序列 forward/backward，记录 allocated/reserved、
整卡显存、CPU 内存和耗时。不要通过重复填充文本冒充长交互轨迹。
然后做预算内 SFT、保存并重载 adapter、真实工具 smoke 和开发集检查。
未验证容量前不承诺训练规模，未完成数据对齐前不启动正式 SFT。
