# A/B/C 第一次关键失败审计（2026-09-22，不消耗 GPU）

对被审计的 18 条轨迹，**不统计循环产生的重复错误**，而是每条只找第一次关键失败：第一次被拒前
模型实际看到了什么、最后一条指令是什么；第一次不可解析编辑错在哪；每个已应用补丁为什么没解决
剩余失败；C 窗口是否覆盖关键区域。

**结论：三处问题来自实验输入与编辑协议，必须先修测量；但同时也存在无法用测量解释的语义失败。
因此现在是"先修测量并重跑"，还不是"优先比较更强模型"。**

---

## 1. 第一次被拒前，模型看到了什么、最后一条指令是什么

18 条轨迹的第一次拒绝，按原因分：

| 第一次拒绝的原因 | 条数 | 出现在第几步 |
| --- | --- | --- |
| `do not reread an unchanged file`（重复读取已成功读过的文件） | **10** | 1–5，其中 8 条在 2–3 |
| `read_file failed: Traceback ... No such file`（模型猜的路径不存在） | 3 | 1–4 |
| `replace_text requires exactly one match, found 0/2` | 2 | 6 |
| `do not repeat a search_text query that already returned a result` | 2 | 4、9 |
| 编辑不可解析 | 1 | 5 |

**10/18 条的第一次失败就是"你已经读过这个文件"，发生在模型对代码做任何实质尝试之前。**

模型在这之前看到的是：一条成功的带行号 `read_file` 输出，末尾是 harness 自己写的续读指令，例如

```
[read_file lines 1-200: file has 446 lines; continue with read_file start_line=201 end_line=400]
```

它随后用**同样的参数**再读一次同一个文件，于是被拒。被拒后 harness 给出的"最后一条指令"是

```
Tool error: do not reread an unchanged file; use search_text or inspect another file
You have 22 tool steps left; switch to a different tool or target, and prioritize an
evidence-backed source edit when enough context is available.
```

逐条证据（第一次拒绝的步骤与它回答的上一个观察）：

| 题 | 条件 | 次 | 步 | 动作 | 上一个观察 | 拒绝首行 |
| --- | --- | --- | --- | --- | --- | --- |
| 7365 | A | 1 | 6 | replace_text | read dynamo_type.py → `201:` | requires exactly one match |
| 7365 | A | 2 | 4 | search_text | read \_\_init\_\_.py → `1: import copy` | do not repeat the same search |
| 7365 | B | 1 | 2 | read_file | read dynamo_type.py → `1: import base64` | do not reread an unchanged file |
| 7365 | B | 2 | 2 | read_file | read dynamo_type.py → `1: import base64` | do not reread an unchanged file |
| 7365 | C | 1 | 2 | read_file | read dynamo_type.py → `1: import base64` | do not reread an unchanged file |
| 7365 | C | 2 | 5 | read_file | read dynamo_type.py → `81: def __ne__` | do not reread an unchanged file |
| 7514 | A | 1 | 5 | replace_text | read models.py → `201:` | unparseable (IndentationError) |
| 7514 | A | 2 | 6 | replace_text | read models.py → `1: import base64` | requires exactly one match |
| 7514 | B | 1 | 4 | read_file | read responses.py → `1: import io` | do not reread an unchanged file |
| 7514 | B | 2 | 2 | read_file | read models.py → `1: import base64` | do not reread an unchanged file |
| 7514 | C | 1 | 3 | read_file | search `select_object_content` | do not reread an unchanged file |
| 7514 | C | 2 | 2 | read_file | read models.py → `1: import base64` | do not reread an unchanged file |
| 7608 | A | 1 | 9 | search_text | read eval_component.py → `1: import abc` | do not repeat the same search |
| 7608 | A | 2 | 4 | read_file | search `send_task_success` | read_file failed: Traceback |
| 7608 | B | 1 | 3 | read_file | read custom_error_name.py → `1: from typing` | do not reread an unchanged file |
| 7608 | B | 2 | 3 | read_file | read custom_error_name.py → `1: from typing` | read_file failed: Traceback |
| 7608 | C | 1 | 1 | read_file | （初始观察） | read_file failed: Traceback |
| 7608 | C | 2 | 2 | read_file | read execute_state.py → `1: import abc` | do not reread an unchanged file |

三条 7608 的"路径不存在"分别是 `moto/stepfunctions/backend.py`、
`.../component/exec/execute_state.py`、`.../component/models.py`：模型在猜路径。

## 2. 第一次不可解析编辑：缩进、范围、陈旧行号还是修改方案

把这个问题压成单一标签是不诚实的：同一次编辑通常**两个原因同时成立**。所以用两个独立轴来记
（`scripts/audit_first_failures.py`）：

* `first_line_unindented`：替换文本的**第一行从第 0 列开始**，这是 `IndentationError` 的直接原因；
* `target_range`：模型此前是否**被展示过**它改写的那些行（`read` / `partly_read` / `never_read` /
  `file_never_read` / `unknown`）。`read_file` 只对**文件**放行，不对行范围放行，所以 `never_read`
  意味着缩进只能靠猜。

5 条轨迹产生了第一次不可解析编辑，自动分类结果：

| `first_line_unindented` | `target_range` | 条数 |
| --- | --- | --- |
| 是 | `never_read` | 2（7514 A r1 第 5 步、7514 C r1 第 5 步） |
| 是 | `unknown`（`old` 已被自己上一步改掉，底本里找不到） | 1（7514 C r2 第 6 步） |
| 否 | `read` | 2（7365 A r2 第 6 步、7608 C r2 第 6 步） |

逐条人工判读（自动分类**测不出过度缩进**，只能测"首行在第 0 列"，所以这一列需要人读）：

| 题/条件/次 | 步 | 动作与目标 | 直接原因 | 证据 |
| --- | --- | --- | --- | --- |
| 7514 A r1 | 5 | `replace_text`，把 `query_input = key.value.decode("utf-8")` 换成 11 行 `if/elif/else` | **缩进（首行丢缩进）+ 该行从未读过** | 新块首行无缩进、后续行有缩进 → IndentationError；第 7 步用 `replace_lines` 提交**同样内容**（整块带缩进）就应用成功 |
| 7514 C r1 | 5 | 同上（同 seed 的另一条件） | **缩进 + 未读** | 同上 |
| 7514 C r2 | 6 | `replace_text`，把 decode 行换成 `if key.value is not None: ...` | **缩进（首行丢缩进）** | `old` 是它自己在第 3 步写进去的文本，底本里已不存在 |
| 7608 C r2 | 6 | `replace_text`，把 `pass` 换成 14 行实现 | **缩进（**过度**缩进）** | 该 `pass` 位于 4 空格层，替换体用 8 空格；目标区间此前读过 |
| 7365 A r2 | 6、7 | `replace_lines` 改 `moto/dynamodb/models/__init__.py` 497-498、486-507 | **臆测行号 + 选错文件** | 该文件**不是**修复涉及的文件；此前只读过 1-196 行 |

结论：**第一次不可解析编辑里没有一例是"修改方案写错但语法正确"** —— 语法正确的错误方案会应用
成功，落在第 3 节。也就是说，这个阶段卡住模型的是"缩进/上下文"与"在没读过的行上动手"，
而不是"想法本身对错"。真正暴露语义问题的，是那些**语法通过并应用**的补丁。

## 3. 每个已应用补丁，为什么没有解决剩余失败

7 条轨迹产生了已应用补丁：

| 题/条件/次 | 已应用编辑 | 结果 | 为什么没解决 |
| --- | --- | --- | --- |
| 7514 A r1 | `models.py:2873` 插入 GZIP/BZIP2 解压分支（**与 gold hunk 2874-2926 内容基本一致**）+ 把一份 GZIP 分支重复加到 `responses.py:299-302` | f2p 0/3、**p2p 回归 7/7** | **第二处放错层**：解压属于 model 层，重复实现进 response 层后把 7 个原本通过的测试全打破 |
| 7514 A r2 | `models.py` 单行：`decode("utf-8", errors='ignore')` | **f2p 2/3** | **压症状而非实现行为**：剩下 `test_select_unknown_key` 要求键不存在时报 `MissingKey`，这个编辑只是让解码不再抛异常 |
| 7514 C r2 | 同上（同 seed 的另一条件） | **f2p 2/3** | 同上 |
| 7365 C r2 | `dynamo_type.py:102-105` 改写为 decimal 相加，重复 3 次 | f2p 0/1、**p2p 回归 1** | **hunk 内语义错**：目标节点是 `test_update_item_add_float`，改法不对，且顺手破坏一个原本通过的节点 |
| 7608 C r1 | `models.py:641-649` 写入 `send_task_failure/send_task_heartbeat/send_task_success`，**函数体全是 `pass`** | f2p 0/2 | **占位实现**：有签名没有行为 |
| 7608 C r2 | `models.py` 写入 14 行 `CallbackOutcomeFailure` 真实实现 | f2p 0/2 | **只做了 4 个必需文件中的 1 个**；它在第 6 步对 `parser/models.py` 的尝试因缩进被拒后再没重试 |
| 7608 A r2 | `eval_component.py:68-71` 两次改写 | f2p 0/2 | **文件不在修复范围内**（该文件不是 gold 触碰的 4 个文件之一） |

**关键一条**：7514 A r1 说明模型**能**在正确文件、正确位置、以正确内容写出修复（它与 gold hunk
几乎一致），然后在第二个文件里多写了一份，把 7 个通过测试变成失败。这不是定位问题，也不是
"写不出补丁"，而是**完整性与作用域判断**问题。

## 4. C 窗口是否覆盖关键区域

**没有。这是我这一轮自己的实验输入缺陷。**

| 题 | 文件 | gold hunk（新侧） | 实际窗口 | 覆盖 |
| --- | --- | --- | --- | --- |
| 7365 | `moto/dynamodb/models/dynamo_type.py` | 1-6、**100-113**、**390-396** | 1-80 | **漏掉 2 个 hunk** |
| 7514 | `moto/s3/models.py`（2947 行） | 1-9、14-20、**2861-2867**、**2874-2926** | 1-80 | **漏掉主 hunk**（失败测试对应的函数在 2861 附近） |
| 7514 | `moto/s3/responses.py` | 2291-2299 | 2271-2319 | 完整 |
| 7514 | `moto/s3/select_object_content.py` | 49-56 | 29-59 | 完整 |
| 7608 | 4 个文件 | 638-644 / 13-19 / 106-112 / 192-199,203-210 | 618-664 / 1-20 / 86-132 / 172-230 | 全部完整 |

原因：旧窗口规则取"第一个 hunk 到最后一个 hunk 的并集"，超过 80 行就**从尾部截断**。7355 与
7514 的 hunk 相距 2800 行，于是窗口锚在文件开头的 import 上，把主 hunk 整段丢掉。

**这一条对上一轮结论的影响必须说清楚**：C 条件本意是"位置和代码都给"，但对 3 题中的 2 题，
它实际只给了 import 区。轨迹显示模型多数时候靠自己的 `read_file` 补读到了正确区域
（7514 A r1 读了 2860-3000；A r2 读到 hunk4 的 20/73 行；C r2 读到 hunk4 的部分），
因此这个缺陷**没有单独决定结果**，但它使 C 臂不能被读成"上下文已给足"。

### 4.1 已修：窗口规则改为"绝不丢 hunk"

`scripts/build_diagnostic_oracle.py` 的 `render_window` 现在按 hunk 生成区间、重叠即合并，
**只让步 padding、绝不让步 hunk**；hunk 自身超过上限时保留并标记 `truncated`。
重新生成 `work/private/abc-diagnostic-oracle-20260922-v2.jsonl`
（sha256 `7368ecd5ca676a78bb5745b9c9a0efa6e407bd485777e4c5dda26541d9c2150d`）：

| 题 | 窗口行数（v1 → v2） | 未覆盖 hunk |
| --- | --- | --- |
| 7365 | 80 → 77 | 2 → **0** |
| 7514 | 160 → 156 | 2 → **0** |
| 7608 | 173 → 173 | 0 → **0** |

窗口更小且完整（因为被浪费的是 padding）。校验脚本 `work/verify_oracle_text.py` 也补上了
"每个 hunk 必须落在某个已展示区间内"的检查 —— 旧版只校验"渲染是否忠实底本"，
**忠实地渲染错误的区域仍然是错误的区域**，这正是它没抓到这个 bug 的原因。
现在对 v1 产物运行会明确失败：

```
FAILED
 - getmoto__moto-7365: dynamo_type.py does not show repair lines 100-113 (14 lines)
 - getmoto__moto-7365: dynamo_type.py does not show repair lines 390-396 (7 lines)
 - getmoto__moto-7514: models.py does not show repair lines 2861-2867 (7 lines)
 - getmoto__moto-7514: models.py does not show repair lines 2874-2926 (53 lines)
```

## 5. 另外两处协议缺陷（尚未修）

### 5.1 `replace_lines` 只按**文件**放行，不按**行范围**放行

README 对该工具的契约是"只允许修改**已读取的源码**"，实现检查的是"这个文件读过至少一次"。
后果：模型可以对一个自己从未看过的行区间发 `replace_lines`，失败时表现为**语法错误**，
从而把"上下文缺失"误记成"语法/语义错误"。

量化（把 45 + 14 次编辑尝试按目标行是否落在此前成功读过的区间里分类）：

| 条件 | 应用 | `unparseable` 但**行范围读过** | `unparseable` 且**行范围从未读过** | 未读先改 | 文本不匹配 |
| --- | --- | --- | --- | --- | --- |
| A（14 次） | 6 | 0 | **3** | 2 | 3 |
| C（45 次） | 6 | **8** | **11** | 10 | 4 |

即 C 的 45 次编辑尝试里 **11 次是在没读过的行号上盲改**（例如 7514 C r1 对 `models.py:2872-2873`
连改 6 次，而它从未读过 2861-2926 这一段）。修法：`replace_lines` 的目标区间必须已被展示过，
否则返回"你还没有读过 2872-2873 行，先读它"，而不是让编辑在语法检查处失败。
这**不改变任务难度**（这些编辑本来就被拒），只是把失败归因摆正。

### 5.2 读失败会永久毒化该动作；缺文件返回 Python traceback

- `ActionLoopGuard._base_rejection` 只检查"同一动作是否出现过"，不检查上一次是否**成功**。
  于是一次 `FileNotFoundError` 之后，同一路径再读会被答成"do not reread an unchanged file"
  —— 告诉模型它已经拿到了从未拿到的内容。实测影响 **9 步 / 2 条轨迹**（7608 B r2 第 5、9、17、
  22、24 步；7608 C r1 第 3、10、15、20 步）。修法是 3 行：只有成功读过的动作才算"已读"。
- 读取不存在的文件时，观察是 **10 行 Python traceback**（`FileNotFoundError` + `pathlib` 栈），
  共 4 次。应换成一行"文件不存在，可用 search_text/list_files 找正确路径"。

## 6. 对上一轮结论的修正

| 上一轮的说法 | 审计后 |
| --- | --- |
| B 命名文件后 6/6 沦为拒绝循环 | **成立且更强**：18 条里 10 条的第一次失败就是重复读取守卫，B 的 6 条全在第 2-4 步命中 |
| C 给了位置和源码仍 0/6，说明语义修复是瓶颈 | **需要下调**：3 题中 2 题的窗口没给到关键区域；且 C 的 45 次编辑里 11 次是盲改。语义瓶颈的**干净证据**只来自：8 次"读过行范围却仍不可解析"的编辑、7514 A r1（写对一处 + 多写一处导致 7 个回归）、7514 A/C r2（读过区域后选择压症状） |
| 环境/评测缺陷本轮基本排除 | **修正**：评测本身没有失败（18/18、0 基础设施失败、三题 gold 自检通过），但**实验输入与编辑协议有 3 处缺陷**，其中 1 处（C 窗口）已经实际让 2 题的条件不成立 |
| 建议只跑 C×8 再判断 | **改为**：先修 5.1/5.2，用 v2 窗口重跑 A/B/C，再决定是否比较更强模型 |

## 7. 下一步（仍不需要 GPU 的部分）

1. **修 5.1**（两个环境实现 + 单元测试 + VM Docker 集成测试）：`replace_lines` 目标区间必须已被读过。
2. **修 5.2**（`ActionLoopGuard` 3 行 + 读失败的一行式错误 + 测试）。
3. 用 v2 窗口重跑 A/B/C（18 条，约 25 分钟 GPU，`--resume` 不能复用旧条件），
   并把"窗口是否覆盖 hunk""编辑目标行是否读过"作为**运行前**的断言写进 manifest。
4. 只有当"窗口完整 + 只能改读过的行"下 C 仍然 0 strict，且失败发生在模型已读过目标区域之后，
   才把结论定为"14B 基座语义修复不足"，届时再比较更强模型才有依据。

## 8. 复现材料

- **已入库**：`scripts/audit_first_failures.py`（本次审计的工具，输出 §1/§2/§3 的全部逐条数据）
  与 `tests/test_audit_first_failures.py`（钉住两个分类轴）。
  运行：`python scripts/audit_first_failures.py --trajectories <run>/trajectories.jsonl
  --gold-patches work/private/pinned-gold-patches.json --output <run>/first-failures.json
  --text <run>/first-failures.txt`
- 本次产出的逐条审计数据：`work/abc-diagnostic-20260922/first-failures.json` / `.txt`
- 审计输入：`work/abc-diagnostic-20260922/trajectories.jsonl`（sha256 `736eee8e…`）
- 修正后的辅助信息：`work/private/abc-diagnostic-oracle-20260922-v2.jsonl`（sha256 `7368ecd5…`）
- 覆盖性校验：`work/verify_oracle_text.py`（现在会检查 hunk 覆盖率，对 v1 产物明确失败）；
  "绝不丢 hunk"这一条由 `tests/test_build_diagnostic_oracle.py::test_a_far_apart_second_hunk_is_never_dropped`
  钉住
- 本地一次性分析脚本（`work/` 被 gitignore）：`audit_c_window_coverage.py`、`audit_read_refusals.py`、
  `audit_edit_targets.py`、`audit_applied_bodies.py`、`audit_read_coverage.py`、
  `audit_poisoned_reads.py`、`audit_edit_locations.py`、`audit_first_refusal_table.py`
