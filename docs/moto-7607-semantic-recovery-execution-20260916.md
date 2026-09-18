# moto-7607 语义恢复执行结果（2026-09-16）

结论：按冻结门槛停止。真实状态恢复数据和 4-step SFT 训练链路都有效，但新 candidate 在第一关固定状态配对中仍为 0/4 strict、0/4 failure reduction，与 SFT60 相同。因此未启动 seed 122100 端到端评测、确认性 seed、跨任务扩展或 GRPO。

## A：真实轨迹归因与 verifier 自检

- 20 个 rollout session 全部按完整 `action_kinds + action_outcomes` 唯一关联到 reward audit；0 个 incomplete。
- 首个阻断点：18 次未到 callback、1 次读到 callback 后未形成编辑、1 次读到后编辑了错误的父实现。
- callback 到达率 2/20；看到完整 `_wait_for_task_token` 也是 2/20；真实 rollout 的 strict success 与 failure reduction 均为 0/20。
- seed 121100 的三个 1→3 回归来自无关模块宽替换：两个 `IndentationError`、一个 `SyntaxError`，均破坏共享导入/解析前置条件，并非正确 callback 修复后的回归。
- 独立受限 Docker 自检固定在 commit `ca24f65…`、镜像 `sha256:796db3…`：baseline 1 fail + 2 pass；完整 train gold 后 3/3 pass，strict reward=1，无新增失败。

## B：可执行语义恢复数据

- 从三个真实分歧状态重放前缀并取得实际工具响应；三条教师恢复都通过完整官方 verifier，strict rewards=`[1,1,1]`。
- 生成 18 条 completion-only 动作样本。训练子集按 session 平衡为每状态 4 条，共 12 条 recovery；再混入 4 条 audited train tool 样本，得到严格 75/25 的 16 条混合数据。
- tokenizer preflight：完整长度 1,291–7,245 tokens，最大监督长度 269，总监督 1,613；`max_length=16384` 下 0 截断。

## C：唯一最小 SFT candidate

- 初始 adapter：稳定 SFT60，权重 SHA-256 `55c947e02b404c3d150e3a2eabd4f55dd50c34641d8d68495f7fbb00c573081c`。
- 配置：seed 122001，LR 2e-5，batch 1，gradient accumulation 4，4 optimizer steps，完整覆盖 16 条数据一遍。
- 训练耗时 42.375 秒，峰值 CUDA memory 21,065,972,736 bytes；每步 grad norm 非零，train loss 0.4465。
- 新 adapter 权重 SHA-256 `312c607bc73643182d2fbdffe216875ab5b306d5dadfe1c91ec1f458072cdfc6`，确认发生有效更新。

## D 第一关：冻结固定状态配对

历史里没有四个未用于训练且接近正确编辑的 session，故按计划明确降级为“三个训练状态拟合检查 + 一个未用 session 导航状态”，不能称为四个留出编辑样本。双方使用同一 contexts、seed 122050、temperature 1.0、top_p 0.95、最多 8 次新工具调用和完整 verifier。

- SFT60：0/4 strict，0/4 failure reduction，2/4 建补丁，1/4 patch valid，1/4 新增失败，mean reward 0.0075。
- Candidate：0/4 strict，0/4 failure reduction，2/4 建补丁，1/4 patch valid，1/4 新增失败，mean reward 0.0075。
- 双方 adapter 在评测前后哈希不变，optimizer steps=0。
- Candidate 有两个明确记录的无效尝试：用户新消息中断了第三状态的部分 session；首次 resume 前反向隧道掉线、在创建 session 前连接被拒。严格 resume 校验原 manifest 后只补缺失状态，最终有效分母仍为预先计划的 4/4；原始无效轨迹未删除。

行为上，candidate 没有把已监督的恢复串起来：第一个近编辑状态没有可解析工具调用；第二个状态做了可解析但不修复故障的 callback 大范围替换；错误补丁状态仍回到 AWS SDK 的错误区域并造成 1→3；未用 session 继续使用不存在的 `src/moto/...` 路径。训练更新没有转化为可复现修复。

## 停止决定

文件门槛要求 candidate 至少 2/4 strict 且高于 baseline；实际为 0/4 且完全没有 failure reduction。继续端到端 rollout、换 seed 或再次训练都缺乏依据，所以停止。GRPO 未启动，也没有达到提出 GRPO 的 failure-reduction 前提。

完整私有证据目录：`/root/autodl-tmp/moto-7607-semantic-recovery-20260916`。归档为 `/root/autodl-tmp/moto-7607-semantic-recovery-evidence-20260916.tar.gz`（SHA-256 `f9fb9a2450d20fd210d9dc0ad87cd3f1a1de02e579839a092627e0b1c8b18734`，145 MiB）。candidate 保留在 `/root/autodl-tmp/sft60-semantic-recovery-seed122001/final-adapter`。
