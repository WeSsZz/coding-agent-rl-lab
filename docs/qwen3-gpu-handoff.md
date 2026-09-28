# Qwen3-8B 无卡核对与挂卡交接

> 2026-09-28 GPU 收尾更新：已完成官方 fork 的真实模型交互与两项固定 `train` 任务基线，结果与明日恢复点见 [`qwen3-openhands-gpu-pilot-20260928.md`](qwen3-openhands-gpu-pilot-20260928.md)。本页以下“尚无模型 rollout”等语句保留为挂卡前的历史状态，不代表当前状态。按用户要求，7607、vLLM 与 SSH 隧道已停止；**尚未通过 AutoDL 控制台确认云实例停机**。

此交接针对 AutoDL 数据盘 `/root/autodl-tmp/route-v2` 和系统盘模型 `/root/Qwen3-8B`。模型是固定 revision `b968826d9c46dd6066d109eabc6255188de91218`，全部七个 tokenizer/权重文件已与上游 SHA256 清单核对。无卡阶段已生成 66 条完整、长度不超过 32768 token 的非思考轨迹，保存在 `work/route-v3-qwen3/sft-short32k`。数据与模型分别由 `manifests/route-v3-qwen3-sft-short32k.json`、`manifests/route-v3-qwen3-model.json` 固定。

无卡实例已准备隔离环境 `.venv-qwen3`：torch 2.8.0/cu128、Transformers 4.57.1、PEFT 0.17.1、Accelerate 1.11.0、Liger 0.8.3、datasets 3.6.0、vLLM 0.11.0。`pip check`、模块导入、tokenizer、数据预检均通过。运行报告为 `work/qwen3-runtime-verification.json`。当前容器没有 GPU 驱动库（`libcuda.so.1` 是空占位文件），所以 vLLM CLI 和 CUDA 执行必须等 GPU 挂载后验收；这不影响已完成的 CPU 环境检查。

2026-09-28 重新连接本次无卡实例后的只读核对：`/root/Qwen3-8B` 含五个 safetensors 分片及 tokenizer 文件；`work/qwen3-8b-system-verification.json` 存在，记录上述固定 revision 且 `verified=true`。`work/qwen3-runtime-verification.json` 存在，记录 `model_verified=true`、66 条数据的预检、`cuda_visible=false` 和 GPU runtime 待验证。`.venv-qwen3/bin/python -m pip check` 返回 `No broken requirements found`。`work/route-v3-qwen3/sft-short32k/train.jsonl` 实数 66 行，当前 SHA256 为 `656df5e40f1a457fc5f66412ed64a7e780463ebfed9421f253671cf1b61b9848`，与 manifest 一致。此轮没有重新计算模型权重的全量 SHA256；报告来自此前核对。`nvidia-smi -L` 没有列出 GPU。

## 挂卡后先跑真实任务

选单张 48GB 显卡，启动后记录平台实际计费起点（Unix 秒），进入项目目录。下列入口都使用同一计费起点和四小时截止时间；超时停止进程不代表实例自动关机。

```bash
cd /root/autodl-tmp/route-v2
export SESSION_STARTED_AT=<平台实际计费起点Unix秒>
nvidia-smi
.venv-qwen3/bin/python -m pip check
bash scripts/gpu/run_qwen3_server.sh base
```

服务使用非思考模板 `envs/qwen3-8b-nonthinking-rollout.jinja`，默认只监听 AutoDL 本机 8000 端口。**真实 SWE-Gym Docker 环境在 Ubuntu VM `wesz@192.168.137.130` 上，不在 AutoDL 容器中。** VM 已通过无模型的 runtime 启动检查，但尚无模型 rollout。先从 VM 开 SSH 隧道，用 `/v1/models` 确认 `qwen3-base`，再运行固定的 `train` 任务 `getmoto__moto-7646`。模型名由 `envs/qwen3-serving.env` 固定；正式 OpenHands 入口为 `scripts/vm/run_swegym_openhands.sh`。

```bash
# VM 第一终端：使用届时的 AutoDL SSH 地址与端口，以交互密码认证；保持隧道运行。
ssh -N -L 18000:127.0.0.1:8000 -p <AutoDL-SSH端口> root@<AutoDL-SSH主机>
# VM 第二终端：
cd ~/coding-agent-rl-lab
curl -s http://127.0.0.1:18000/v1/models
export SESSION_STARTED_AT=<同一平台计费起点Unix秒>
export MODEL_ID=qwen3-base
export MODEL_API_BASE=http://127.0.0.1:18000/v1
bash scripts/vm/run_swegym_openhands.sh
```

检查 `output.jsonl` 的容器内读文件、命令、编辑、补丁和工具事件，再用 fork 的 `eval_infer.sh` 在干净环境判定；若只完成启动或生成文本，不能记作模型任务结果。单任务闭环稳定后再扩展固定任务组，记录耗时、工具调用和失败层级。

## SFT 对照实验

四小时 GPU 窗口如有余量，可先做 Qwen3 的 24K/32K SFT smoke。检查 loss、显存、adapter 保存与重载后，才考虑 66 条正式 SFT 和 base/paired 对照。paired 服务将 LoRA 注册为 `qwen3-sft`；`scripts/vm/run_qwen3_eval.sh` 检查 `/v1/models` 后使用同一别名。此评测入口不是官方 fork 的任务完成率基线。

```bash
bash scripts/gpu/run_qwen3_training.sh smoke-24k
bash scripts/gpu/run_qwen3_training.sh smoke-32k
bash scripts/gpu/run_qwen3_training.sh formal
```

任何 CUDA OOM 或模板失配都须先诊断，不能静默截断数据。正式训练和 RL 仍取决于前置检查；Qwen2.5 的旧 adapter 和报告不能混入 Qwen3 对照。收尾需检查并关闭 AutoDL 实例，进程超时不会自动关机。
