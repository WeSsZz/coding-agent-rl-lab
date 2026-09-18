# moto-7607 干净 7B 配对与教师正对照执行报告（2026-09-18）

## 结论

教师正对照证明，7 个状态中只有 3 个教师当前编辑在“立即运行 verifier”时能 strict success；但 7 个教师编辑接固定教师后缀后全部 strict success、没有新增失败。此前仅以单步 failure reduction 判断中间编辑的口径不充分，现已修正。

干净 7B 底座与 P2 step12 adapter 在教师后缀语义判定上均为 0/7。adapter 没有表现出相对底座的语义退化或提升，但明显改善了工具协议格式：底座 0/7 可解析，adapter 4/7 可解析且可应用。

特殊 token 退化不是 parser 或显示层伪影。原始生成 token ID 本身包含大量 `151643=<|endoftext|>` 和 `151644=<|im_start|>`，并通常以 `151645=<|im_end|>` 主动终止。干净底座为 6/7 特殊 token 退化，adapter 为 3/7；因此现有证据不支持“adapter 导致特殊 token 退化”，反而说明 adapter 部分改善了格式生成。

训练、完整 rollout 与 GRPO 继续暂停。

## 冻结协议

- 7 个上下文 SHA-256：`82a460a0405c37c7386feb20eecbe6f59540566bddb9334ab2a677bc7574be52`
- 教师轨迹 SHA-256：`c69473b09f1c2ad27451c311ec8515b319099f2067b4b8c17234ae79cca347a7`
- adapter SHA-256：`9370e928903c5b59096a7add0bd04c55908574b9ee85ab155678eedfb6d376db`
- seed：`123501`
- greedy；每条件每状态一次生成；最多 1024 新 tokens
- 底座条件记录为 `adapter_loaded=false` 且模型没有 `peft_config`
- 同一模型进程先运行底座，再在内存合并 adapter；共享 tokenizer、chat template、工具 schema 和执行环境

## 教师正对照

- 教师目标编辑应用成功：7/7
- 教师当前编辑立即 strict success：3/7
- 教师当前编辑立即 failure reduction：3/7
- 教师编辑接固定教师后缀 strict success：7/7
- 教师后缀新增失败：0

这说明前四个中间编辑即使正确，单独执行后也不应被要求立即修复最终测试；教师后缀替换判定才是本轮的主要语义指标。

## 配对结果

### 干净 7B instruct 底座

- 可解析工具动作：0/7
- 可应用编辑：0/7
- 教师后缀语义成功：0/7
- 特殊 token 退化：6/7
- 达到 1024 token 上限：1/7

六个特殊 token 响应的原始 ID 中，约 66%–70% 是 `<|endoftext|>` / `<|im_start|>`，最后由 `<|im_end|>` 正常 EOS。剩余一个状态先产生少量特殊 token，随后重复生成 `s` 直至 1024 token 上限。

### P2 step12 adapter

- 可解析工具动作：4/7
- 可应用编辑：4/7
- 教师后缀语义成功：0/7
- 特殊 token 退化：3/7
- 达到 1024 token 上限：0/7
- adapter 文件运行前后哈希一致

四个可应用动作与上一轮 D1 的错误模式一致；把它们替代教师当前编辑后继续固定教师后缀，最终仍为 0/4 strict，因此可确定不是“正确中间编辑但即时 verifier 尚未改善”。

## 版本与哈希

- PyTorch：`2.8.0+cu128`
- Transformers：`5.16.1`
- tokenizer：`Qwen2Tokenizer`，vocab 151665
- chat template SHA-256：`cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`
- model config SHA-256：`c0242402ad6a13b331ea320feea8c7e3776ffb7a4eff0757b9cd667e116d9a28`
- tokenizer config SHA-256：`959e7f1d9a1b7641a6d6ce05ca97b75c7894fcb66cbe5a040406458fb1128ee4`
- tools schema SHA-256：`e990f5f5f02759c1c30654fad4306341dc2744be71ab0f530587d08b44637f00`
- 原始 report SHA-256：`0e790140765d61f822fde90ee09d67dccef02f31c3b2cdb18de19c391b240078`
- CUDA allocator 峰值：16,851,879,936 bytes，约 15.69 GiB

## 决策与后续

1. 不把特殊 token 问题归因于 adapter；它在纯底座中更严重。
2. 不把 0/7 语义成功单独用于模型选型；七个同题状态只用于故障定位。
3. 下一主阶段应构建按题划分的 20–30 题局部编辑诊断集，每题保留教师单步和可执行教师后缀，训练集与留出集不得共享同题状态。
4. 在扩大数据前可做一个不训练的协议自检矩阵：Qwen 原生 tool-call template 与当前 bare-JSON 协议、结构化历史与无历史最小提示。目标只是定位特殊 token 的共同模板原因，不回选本轮结果。
5. 多题集建立后再选择性比较 14B；没有留出题真实修复前不启动新 SFT。任何 GRPO 仍需用户另行明确授权。

## 证据位置

- AutoDL：`/root/autodl-tmp/moto-7607-base-paired-positive-control-20260918`
- 原始报告：`paired-seed123501/report.json`
- 底座原始生成/token ID：`paired-seed123501/clean-base-generations.jsonl`
- adapter 原始生成/token ID：`paired-seed123501/p2-step12-adapter-generations.jsonl`
- 完整工具轨迹：`paired-seed123501/tool-traces.jsonl`
- 每状态 verifier 审计：`paired-seed123501/*-reward-audit.jsonl`
- 汇总：`summary.json`、`summary.md`
- 最终归档：`/root/autodl-tmp/moto-7607-base-paired-positive-control-evidence-20260918.tar.gz`
- 最终归档 SHA-256：`a697de5b61e544481fd18e4772945dfe18fb6396a7971cf4568a708e2ea7e89e`
- 目录清单：`SHA256SUMS.txt`，67 个文件
