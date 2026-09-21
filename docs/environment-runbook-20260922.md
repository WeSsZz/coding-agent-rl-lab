# 环境运行手册（2026-09-22）

这份文档只写**实测过**的事实与命令。任何没跑过的都标注"待验证"。
目的：换一台机器或换一个人时，不必重新踩一遍今天的坑。

---

## 1. 机器与角色

| 机器 | 角色 | 实测配置 |
| --- | --- | --- |
| Windows 工作站 | 编辑面；跑单元测试 | 仓库 `D:\Developer\coding-agent-rl-lab`；pytest 可用；**无 GPU**、**无外网直连** |
| Ubuntu VM `wesz@192.168.137.130` | **权威测试环境 + Docker worker**；执行真实 SWE-Gym 容器 | Ubuntu 26.04 / 内核 7.0.0-31 / Docker **29.7.2** / **8 vCPU** / **5 GB RAM** / 根分区 97 GB（已用 23 GB，余 70 GB）/ **无 GPU** |
| AutoDL GPU 实例 | 模型推理与训练 | **当前未开机**；历史记录为 RTX 4090 48G / RTX 5090 / RTX 3090 等，**无费率记录** |

VM 的 5 GB 内存是**硬约束**：`docker run` 默认给每个任务容器 `--memory 2g`，所以并发容器数一旦上去，
内存就是先于 CPU 与磁盘的瓶颈。**rollout 农场不能建在这台 VM 上**，它只适合串行验证。

访问方式（PowerShell 侧）：

```powershell
$key = "$env:USERPROFILE\.ssh\id_ed25519_codex_vm"
ssh -i $key -o BatchMode=yes -o StrictHostKeyChecking=no wesz@192.168.137.130 "命令"
```

`~/.ssh/config` 在 Windows 侧**不存在**，必须显式 `-i`。

---

## 2. Docker 镜像：DNS 被污染，必须走镜像站

### 症状

```
Error response from daemon: failed to resolve reference "docker.io/xingyaoww/...":
  dial tcp 162.125.18.133:443: connect: connection refused
```

### 根因（实测）

`registry-1.docker.io` 在这张网络上被解析到 **Facebook/Instagram 的地址段**：

```
registry-1.docker.io   2a03:2880:f134:183:face:b00c:0:25de
auth.docker.io         2a03:2880:f10c:83:face:b00c:0:25de
```

`2a03:2880:...:face:b00c` 是 Facebook 的知名前缀，`31.13.88.26` / `199.59.148.206`
也是 Meta 的地址。也就是说这不是"网络不通"，而是**DNS 应答被替换**，换 DNS 服务器也没用
（223.5.5.5 / 8.8.8.8 / 1.1.1.1 都返回同一批错误地址）。

### 实测可用的镜像站

| 镜像 | `GET /v2/` | 能否拉到 `xingyaoww/*` |
| --- | --- | --- |
| `docker.m.daocloud.io` | 401（存活） | **不能**：`docker.io/xingyaoww/* is forbidden` |
| `dockerproxy.net` | 200 | **能**（本次 3 个镜像全部成功） |
| `docker.1panel.live` | 200 | 未测 |
| `hub-mirror.c.163.com` | 连不上 | — |
| `mirror.ccs.tencentyun.com` | 连不上 | — |

### 解决办法（**不需要 root**）

daemon 级 mirror 需要改 `/etc/docker/daemon.json` 并重启 Docker，而该账号**没有免密 sudo**
（`sudo: A terminal is required to authenticate`）。因此采用显式拉取 + 重新打标签：

```bash
docker pull dockerproxy.net/xingyaoww/sweb.eval.x86_64.<instance>:latest
docker tag  dockerproxy.net/xingyaoww/sweb.eval.x86_64.<instance>:latest \
            xingyaoww/sweb.eval.x86_64.<instance>:latest
```

这样镜像名与 `swe_gym.py` 生成的名字完全一致，环境代码不需要任何改动。
脚本：`scripts/vm/pull_task_images.sh`（按 daocloud → dockerproxy → 1panel 顺序重试）。

> **可信度限制（必须随结果一起报告）**：经社区镜像拉到的镜像**没有与 Docker Hub 官方 digest 比对过**。
> 下面记录的是本次实际拿到的 digest，用于复现，不等于经过来源验证。

| 镜像 | 增量磁盘 | digest（前 19 位） |
| --- | --- | --- |
| `...conan-io_s_conan-10975:latest` | 与下面两个共享层 | `sha256:f3d35940cf1a189f` |
| `...conan-io_s_conan-13757:latest` | 与上面共享层 | `sha256:786530bdf0314269` |
| `...pydantic_s_pydantic-8947:latest` | 与上面共享层 | `sha256:00dc41009288facc` |
| 10 个 `...getmoto_s_moto-*:latest` | 合计约 11 GB（共享层） | 见 `work/private/gold-selfcheck-20260922.json` |

实测：拉 3 个跨 2 个仓库的新镜像，根分区从 **19 GB 用到 23 GB**（+4 GB），
即单个新仓库的**边际**成本约 1–2 GB，远低于 `docker images` 报的 4.23 GB/镜像
（那是含共享基础层的完整大小）。**按边际成本规划，70 GB 余量足够几十个实例。**

---

## 3. 模型与数据下载：HF 直连不通，`hf-mirror.com` 可用

| 主机 | 结果 |
| --- | --- |
| `huggingface.co` | **不可达**（curl 超时，HTTP 000），本机与 VM 都是 |
| `datasets-server.huggingface.co` | 不可达 |
| **`hf-mirror.com`** | **可用**（HTTP 206/200） |
| `modelscope.cn` | 可达（302） |
| `pypi.tuna.tsinghua.edu.cn/simple/` | 可用 |
| `github.com` / `raw.githubusercontent.com` / `api.github.com` | 可用 |
| `arxiv.org` | 不可达 |

已用 `hf-mirror.com` 实际取到：

- `Qwen/Qwen3-8B` 的 `config.json`、`tokenizer_config.json`
- `nebius/SWE-rebench-openhands-trajectories` 的 `README.md`、`tools.json`、数据集 API
- `SWE-Gym/SWE-Gym` 的数据集 API（`license: mit`，`sha bb94ed9e…`，2438 行）

**但 `SWE-Gym/SWE-Gym` 的 parquet 通过 hf-mirror 的实际下载本次超时**（`resolve` 会跳到
另一个 CDN）；本机已有该分片副本，所以不影响当前工作，但**若要在新机器上重建数据集，
必须先解决这条下载路径**。

给 `huggingface_hub` 用的设置是 `HF_ENDPOINT=https://hf-mirror.com`。

---

## 4. 已知陷阱（都会伪装成别的问题）

| 陷阱 | 表现 | 规避 |
| --- | --- | --- |
| PowerShell → ssh → sh 的引号链 | 远端把 `\|`、`(`、`$` 当 shell 元字符；报 "command not found" 或语法错 | **凡是非平凡命令一律写成本地 `.sh` 文件，`scp` 过去，`sed -i 's/\r$//'` 去 CR，再 `bash` 跑** |
| Windows 写出的脚本带 CRLF | `bash: $'\r': command not found` | 同上，`sed -i 's/\r$//'` |
| 远端命令的 stderr 被 PowerShell 当成失败 | 实际通过的测试被报成失败 | `sync_and_validate_vm.ps1` 已改为在**远端**合并 `2>&1`；见该脚本注释 |
| 没有免密 sudo | `sudo: A terminal is required to authenticate` | 不要依赖改 `/etc`；用显式拉取 + retag 之类的用户态办法 |
| VM 无 pyarrow | 依赖分片的脚本在 VM 上不可用 | 在 Windows 侧导出 JSONL（`scripts/export_task_rows.py`）再送过去 |
| VM 的 Python 是 3.14 | 某些旧包轮子缺失 | 不要假设 VM 的 python 版本与训练机一致 |

---

## 5. 已实测可用的命令

```powershell
# 权威测试（Windows 是编辑面，VM 才是权威）
& "$PWD\scripts\sync_and_validate_vm.ps1" -TimeoutSeconds 1800

# 在解包后的当前代码上跑任意命令（注意：不含 .git，也不含 work/）
& "$PWD\scripts\sync_and_validate_vm.ps1" -TimeoutSeconds 5400 -RemoteCommand "<命令>"
```

```bash
# 在 VM 上跑镜像准备（无 GPU，串行）
bash scripts/vm/pull_task_images.sh
# 诊断 registry 可达性
bash scripts/vm/diagnose_registry.sh
# 审计一个仓库的 pytest 调用（需要先把行数据 export 过去）
bash scripts/vm/run_test_command_audit.sh
```

`-RemoteCommand` 在**解包的临时树**里执行，所以输入输出要用绝对路径
（例如 `/home/wesz/coding-agent-rl-lab/work/private/...`），否则跑完就被清理。

---

## 6. 待验证

- [ ] 租卡机能否访问 `hf-mirror.com`（**开机后第一件事**；不能的话必须预置权重）
- [ ] 租卡机能否访问 Docker Hub 或是否需要同样的 mirror 方案
- [ ] VM 上并发 2 / 4 个任务容器的内存与 CPU 实测曲线（决定 rollout 吞吐）
- [ ] `xingyaoww/*` 各仓库镜像的官方 digest（用于把上面的可信度限制补掉）
