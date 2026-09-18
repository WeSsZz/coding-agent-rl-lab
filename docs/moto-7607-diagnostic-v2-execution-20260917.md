# moto-7607 P0/P1/P2 诊断执行汇总（2026-09-17）

## 结论

本轮按预注册门槛停止在 P2。修复训练/rollout 协议差异后，目标动作 NLL 明显下降，但最终 adapter 在三个训练状态仍为 0/3 strict、0/3 failure reduction，因此不进入 P3，不追加 SFT，不启动 GRPO。

## P0

- 修复后 P0 gate：`True`。
- 监督 tokens：1613；全 mask、边界错位、截断均为 0。
- SFT 与 rollout 的工具 schema 和历史结构化 tool_calls 渲染哈希一致。
- 完整教师链 18 个动作；旧混合数据只选了 12 个，漏掉的 6 个主要是 search/read。

## P1 冻结对照

- A_7b_sft60: target NLL=0.572087, token accuracy=0.870361, fixed-state strict=0/4。
- B_7b_semantic_old: target NLL=0.547179, token accuracy=0.870501, fixed-state strict=0/4。
- C_14b_parent_path: target NLL=0.766203, token accuracy=0.849162, fixed-state strict=0/4。

## P2 唯一诊断 SFT

- 12 steps，耗时 149.147s，峰值 21976684544 bytes，train loss 0.416952。
- NLL 曲线：0.572087（稳定 A）→ 0.362836 → 0.304592 → 0.290596。
- 最终训练状态：strict 0/3，failure reduction 0/3，new failures 1。

## 停止决定

NLL 改善只证明监督信号被学到一部分，没有转化为完整行为恢复。继续增加 steps、换 seed 或直接进入 GRPO 都没有当前证据支持。下一轮应重新设计多题语义编辑数据和动作级评估，而不是在这三个状态上继续拟合。

所有 adapter 路径与 SHA-256、逐状态轨迹、verifier 结果和资源数据见 `summary.json`。

远端证据目录为 `/root/autodl-tmp/moto-7607-next-diagnostic-v2-20260916`；归档为
`/root/autodl-tmp/moto-7607-next-diagnostic-v2-evidence-20260917.tar.gz`（264 KiB，SHA-256
`7a4a5c255238d59c1b48b8fe0c2bb02a3f06740a5df7024f9b2cac6788ab7c77`）。训练输出与四个
checkpoint 保存在 `/root/autodl-tmp/sft60-protocol-v2-full18-12step-seed123201`，未复制进小型证据归档；
其路径和权重哈希已写入 JSON 汇总。
