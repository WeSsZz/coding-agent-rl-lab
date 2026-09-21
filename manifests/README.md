# Manifests

本目录是**机器可读**的版本记录。规则只有三条：

1. 每条记录必须能被 `python scripts/verify_route_manifest.py <manifest>` **重新计算并核对**；
   核不出来的字段不许写进来。
2. 路径分两类：**仓库内**的用 `repo_path` + `git_blob`（`git rev-parse <rev>:<path>`），
   **仓库外**的用 `path` + `sha256`。被 `.gitignore` 忽略的 `work/` 与 `remote-artifacts/`
   一律属于后者，因此 `sha256` 是它们唯一的离线核验入口。
3. 任何"结论"必须写在 `evidence` 字段里并指向一个具体文件或命令；没有证据的写进 `unproven`。

`route-v1-freeze.json` 冻结的是旧的自建 scaffold 路线（`main@27b86c2`）。
它**不是**新主线的配置；新主线的 manifest 会另开文件，文件名带 `route-v2-` 前缀。
