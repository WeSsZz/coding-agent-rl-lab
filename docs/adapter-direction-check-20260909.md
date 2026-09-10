# 单步 GRPO adapter 更新方向验证

## 本轮收尾状态（用户要求在额度耗尽前暂停）

评测已停止，GPU 使用显存已确认归零；本轮 9011 专用 worker、隧道及临时容器已清理。
没有启动多步 GRPO，没有运行 held-out。代码修改尚未提交。

SFT 完成 16/16 条；GRPO 完成前三个 train 任务的 6/16 条，第 4 题中断且不计入结果。
只比较双方均完成的前三题：严格成功均为 0/6，有效测试改善均为 0/6；
有效补丁数从 1 增至 3，平均 shaped reward 从 0.025 增至 0.075，
但 GRPO 组出现 1 条引入新失败的补丁。GRPO 的 regression 尚未运行，不能宣称泛化改善。
当前证据不足以支持扩大到多步 GRPO；先完成剩余配对评测，再决定奖励或课程调整。

原始报告和逐题 audit 已同时备份在 AutoDL、Ubuntu `work/` 和本地 `remote-artifacts/`。
本地目录为 `direction-eval-20260909-sft/`、`direction-eval-20260909-grpo/`；
配对部分结果另见 `adapter-direction-check-20260909-partial.json`。
权重差异相对 L2 范数为 0.00019136，确认 GRPO adapter 实际发生更新。

### 下次恢复

1. 沿用 Windows 私钥 `C:\Users\Junwe\.ssh\id_ed25519_codex_vm` 连接 `wesz@192.168.137.130`；
   Ubuntu 用 `/home/wesz/.ssh/id_ed25519_autodl` 连接当前 AutoDL
   `root@connect.bjb1.seetacloud.com:18235`，如实例更新则替换地址和端口。
2. 在 Ubuntu 重新启动 9011 worker，使用下文固定的八个任务和原 token 文件；
   `work/direction_tunnel.py` 为带实际心跳的反向隧道脚本，先验证 GPU 端 HTTP 可达。
3. AutoDL 执行 `bash /root/autodl-tmp/coding-agent-rl-lab/work/run_direction_check.sh`。
   脚本含 `--resume`，已完成任务不会重跑。GRPO 从 `moto-7646` 恢复，剩余
   `7646、7446、7607、7385、7608` 五题，共 10 条。
4. 两组全部完成后，使用本地 `scripts/summarize_adapter_direction.py` 重算最终结果。

注意：当前运行所用 worker 仍是修复前源码；恢复本轮对照时不要中途切换奖励实现。
本地 `reward_shaping.py` 已修复未知计数审计，待本轮完整比较结束后再统一部署。

## 预先固定的比较

比较 `sft-grpo-dynamic-v2-60step-seed73001/final-adapter` 与在它之上更新的
`grpo-shaped-line-edit-sft60-7509-mb1-2x4096-seed70002/final-adapter`。
后者此前只证明存在非零梯度，不能证明严格任务成功率提高。

固定全部 6 个 train 和 2 个 regression 任务，每个 adapter 每题采样 2 条，合计 32 条。
按 train curriculum 顺序，再按 regression 顺序，每题 seed 为 `81000 + 100 * index`。
两个 adapter 的任务、seed、采样 batch 和预算保持相同；不使用 held-out 做选择。

使用已安装的 TRL 1.12 `GRPOTrainer.evaluate()`，沿用训练时的 Transformers generation、
动态工具 schema、bare JSON parser、`replace_lines` 和远程 Docker worker。
两者都处于 eval 模式、冻结所有参数，不调用 `train()`，结束时验证 optimizer step 为 0
且 adapter 权重文件 SHA-256 未变。此处比较的是训练协议内的更新方向，
不与历史 vLLM/v12 rollout 的分数直接拼接。

- 模型：固定 Qwen2.5-Coder-7B-Instruct snapshot `c03e6d358207e414f1eca0bb1891e29f1db0e242`。
- 每条 completion 总预算 4096 tokens，最多 8 次 tool calling iteration。
- temperature 1.0，top-p 0.95，generation batch 2。
- Docker：4 GiB RAM、2 CPU、512 PID、无网络；每次 verifier 超时 300 秒。
- 两个 adapter 在同一新 GPU 实例执行；硬件变更不与旧 RTX 5090 的耗时作效果比较。
- 每个任务单独保存 reward audit 和报告检查点；缺少任一 verifier audit 时拒绝汇总为完整结果。

## 判断原则

主要检查严格成功与失败测试的实际改善；同时检查新增失败、安全违规、有效补丁和编辑动作。
shaped reward 的 0.15 等编辑奖励只能作为辅助信号，不能替代测试改善。

每题只有两条样本，这是小规模方向筛查，不足以证明统计显著提升。
若没有严格成功或可执行测试改善，或者 regression 恶化，不直接扩大到多步 GRPO；
先根据失败类型修正奖励、训练课程或工具使用，再设计新的 train/regression 对照。
只有存在可解释的测试改善且没有观察到 regression 恶化时，才考虑有步数上限的后续试验。

## 可复现入口

```bash
PYTHONPATH=src /root/autodl-tmp/grpo-env/bin/python \
  -m coding_agent_rl_lab.grpo_evaluate \
  --model-path '<fixed-model-snapshot>' \
  --adapter-path '<adapter>/final-adapter' \
  --prompt-rows work/swe-gym-grpo-prompts.jsonl \
  --worker-token-file /root/autodl-tmp/grpo-worker-shaped.token \
  --worker-base-url http://127.0.0.1:9011 \
  --output-dir '<new-output-directory>' \
  --seed 81000 --num-generations 2
```

默认要求输出目录不存在。提供 `--resume` 时严格校验 adapter、prompt 哈希、seed 和预算，
只跳过审计条数完整且 task/split/seed 匹配的任务；未完成任务的旧 audit 单独保留，不纳入汇总。
SSH 使用 Windows → Ubuntu → AutoDL 的既有公钥链；不保存或传递密码。

## 运行中发现的问题

首次运行在第 5 题时反向 SSH 隧道断开。前 4 题检查点完整保留，清理该中断题的两个
临时容器后，启用自动重连隧道并恢复。连接故障样本不计作模型失败。

旧 reward audit 在 `final_failure_count=null` 时错误地将其当作零计算
`resolved_failure_count`，并可能错误地标记无新增失败。该情况的实际 shaped reward 仍为零。
本地修复未知计数处理，并从原始 audit 的非空基线/最终计数、有效补丁和无违规条件重算
测试改善。此次运行中的 worker 保持原版本，以确保两个 adapter 条件一致；原始 audit
保留作为证据，旧 `resolved_failure_count` 字段不能单独用于改善判断。
