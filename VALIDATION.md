# Validation Status

## 已验证

- 26 个本地单元测试通过；
- 两个 repository-owned fixture 在修复前失败、参考补丁后通过；
- noop 3×2 trials 为 0/6，reference pipeline check 为 6/6；
- 合法临时目录与 macOS canonical path 兼容；
- `../` 路径逃逸会成为 hard violation；
- SWE-Gym/SWE-bench JSONL 字段和 prediction 格式有回归测试；
- Docker 命令构建不使用 shell，默认禁网、drop capabilities、只读 rootfs，并限制内存、CPU 和 PID；
- 模型配置不在 repr 中暴露 API key，模型失败会转成失败 Trial。
- trajectory 包含 task 与 execution provenance，报告可检测 split 泄漏和重复内容；
- reference 答案轨迹即使通过测试也不会计入 training-eligible 数据。

## 尚未验证

- 当前开发机器未安装 Docker，未执行真实 SWE-Gym/SWE-bench 容器；
- 当前环境没有 `CODING_AGENT_API_KEY`，未执行真实模型 rollout；
- 未下载 SWE-Gym 训练集，未产生训练或 held-out 效果数字；
- 未执行 SFT、rejection sampling 或 GRPO；
- `training_performed` 必须继续保持 `false`。

## 下一验收门槛

1. 安装并验证 Docker；
2. 选择 5 条 SWE-Gym smoke subset，固定实例 ID 和镜像版本；
3. 验证 baseline fail-before、gold pass-after；
4. 使用不含答案的模型 Agent 生成 prediction JSONL；
5. 用官方 SWE-bench harness 输出 resolved 结果；
6. 扩展到 50/20/20 split 后再进入轨迹筛选和训练。
