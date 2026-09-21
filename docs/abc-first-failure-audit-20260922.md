# A/B/C 第一次关键失败审计（2026-09-22，不消耗 GPU）

对被审计的 18 条轨迹，**不统计循环产生的重复错误**，而是每条只找第一次关键失败：第一次被拒前
模型实际看到了什么、最后一条指令是什么；第一次不可解析编辑错在哪；每个已应用补丁为什么没解决
剩余失败；C 窗口是否覆盖关键区域。

**结论：三处问题来自实验输入与编辑协议，都必须先修；同时也存在无法用测量解释的语义失败。
但"0 strict"本身不是结论——它必须按每条轨迹的实际终止原因拆开读。因此下一步是"修完测量并重跑"，
而不是"优先比较更强模型"，也不是"判定 14B 不值得训练"。**

---

## 0. 两处更正（对上一版审计的修正）

**更正一：「B 的结论更强了」不准确。**
18 条里 10 条的第一次失败是重复读取，这加强的是**整个交互流程存在读取停滞**的证据，
**不是 B 独有的证据**（A 也有，C 也有）。而且必须把三种情况分开，它们的责任方不同：

| 情况 | 判据 | 责任方 | 本轮实测 |
| --- | --- | --- | --- |
| 模型用**完全相同的参数**重读 | 动作与上一次相同 | 模型的动作选择 | **10/18 条**都属于这一类（`start_line=201` 的续读提示就在上一条观察里，模型没有用它） |
| 模型读**不同区间**仍被拒 | 动作参数不同却被拒 | 守卫实现缺陷 | **0 例**（守卫按完整动作比较，不同区间不会被拒；已加测试钉住） |
| **读取失败**却被记为已读 | 失败动作进入了"已读"记录 | 确定的状态记录缺陷 | **9 步 / 2 条**（7608 B r2、7608 C r1） |

**更正二：「停止所有 SFT」不是本审计能推出的结论。**
即使修正后 C 仍为 0，能说的也只是：

> 当前 14B 基座在这些开发题、此协议与预算下没有观察到完整成功，且失败证据指向下列具体能力缺口。

剩余失败仍可能来自编辑格式、动作重复、输出截断或预算耗尽。**基座不会做，恰恰可能是需要监督
训练的**理由；是否值得训练要有另外的证据。因此每个 0 都必须按终止原因拆开（见 §6），
而不是统一归因为"语义不足"。

同样，**现在就暂停扩大 GRPO**：不是因为它已被证明无效，而是**现有证据不足以支持扩大**；
"14B 不值得训练"也**不能**提前判定。

## 0.1 辅助信息的呈现方式必须改（混杂因素）

上一轮的 B/C 辅助文本被**作为一条单独的 user 消息、在每一步的最后重新发出**，而且正文以祈使句结尾
（"Read them yourself before editing…"、"Verify anything you rely on with your own tools."）。
这使"这条信息可用"和"你刚被要求去读"混在一起，也无法排除它加重了 B 的反复读取。
已改为：辅助信息进入**任务 payload 的独立字段** `diagnostic_auxiliary_input`
（放在 `baseline` 之后、`initial_observation`/`history` 之前，因此**不是最后读到的东西**），
正文改为纯陈述、去掉全部祈使句，provenance 由 payload 字段自己声明。详见 §5.3。

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

## 5. 另外两处协议缺陷

### 5.1 `replace_lines` 只按**文件**放行，不按**行范围**放行（已修）

README 对该工具的契约是"只允许修改**已读取的源码**"，实现检查的是"这个文件读过至少一次"。
后果：模型可以对一个自己从未看过的行区间发 `replace_lines`，失败时表现为**语法错误**，
从而把"上下文缺失"误记成"语法/语义错误"。

量化（把 45 + 14 次编辑尝试按目标行是否落在此前成功读过的区间里分类）：

| 条件 | 应用 | `unparseable` 但**行范围读过** | `unparseable` 且**行范围从未读过** | 未读先改 | 文本不匹配 |
| --- | --- | --- | --- | --- | --- |
| A（14 次） | 6 | 0 | **3** | 2 | 3 |
| C（45 次） | 6 | **8** | **11** | 10 | 4 |

修法（已完成）：环境记录每次 `read_file` **实际展示**的行区间（从观察里的行号解析，
因为字符预算会截断），`replace_lines` 的目标区间必须落在某个已展示区间内，否则返回
"replace_lines A-B is outside every range you have read of <file>. You have read lines X-Y."
一次成功编辑后该文件的已读区间**全部失效**（行号已移动），必须重读。
本地与 Docker 两个实现行为一致，并有各自的测试。

### 5.2 读失败会永久毒化该动作；缺文件返回 Python traceback（部分已修）

- `ActionLoopGuard._base_rejection` 原本只检查"同一动作是否出现过"，不检查上一次是否**成功**。
  于是一次 `FileNotFoundError` 之后，同路径再读会被答成"do not reread an unchanged file"
  —— 告诉模型它已经拿到了从未拿到的内容。**已修**：只有成功返回内容的读取才使重复读取无意义；
  失败后的立即重试仍由"不要重复上一个出错动作"这条规则兜住。
- 读取不存在的文件时观察是 **10 行 Python traceback**，共 4 次。**尚未修**：应换成一行
  "文件不存在"提示。

### 5.3 已修：C 窗口的坐标、覆盖、呈现（Gate 1）

| 项 | 修前 | 修后 |
| --- | --- | --- |
| 坐标 | 用 hunk 的**新侧** `+c,d`（渲染的却是修复前的文件） | 改用**旧侧** `-a,b`；7365 第三个 hunk 就是 old 385-396 vs new 390-396 |
| 覆盖 | 取首尾 hunk 并集，超限从尾部截断 → 7365 漏 2/3 个 hunk、7514 漏掉主 hunk | 每个 hunk 一个区间、重叠合并、只让 padding、绝不让 hunk；manifest 记录 `hunks_not_covered` |
| 校验 | 只校验"渲染是否忠实底本" | 增加"每个 hunk 必须落在已展示区间内"；对旧产物会明确 FAILED |
| 呈现 | 每步最后追加一条带祈使句的 user 消息 | 任务 payload 的独立字段 `diagnostic_auxiliary_input`，纯陈述、非末尾 |

新产物 `work/private/abc-diagnostic-oracle-20260922-v4.jsonl`
（sha256 `59bead8955c074b9dd02f6f129bb0c2cb49febf36242e96b9eb0c7f6a65a8418`），
三题 0 个未覆盖 hunk，共 77 / 160 / 172 行。

## 6. 对上一轮结论的修正

| 上一轮的说法 | 审计后 |
| --- | --- |
| B 命名文件后 6/6 沦为拒绝循环 | **更正**：18 条里 10 条的第一次失败是重复读取守卫，这是**整个交互流程**的读取停滞问题，不是 B 独有；且其中 10 条都是模型用相同参数重读（模型的动作选择），0 例是"不同区间被拒"（守卫缺陷），9 步是"失败读取被记为已读"（状态记录缺陷）。不能把它当作 B 的专属证据 |
| C 给了位置和源码仍 0/6，说明语义修复是瓶颈 | **需要下调**：3 题中 2 题的窗口没给到关键区域（坐标用错侧 + 尾部截断）；C 的 45 次编辑里 11 次是盲改。语义瓶颈的**干净证据**只来自：8 次"读过行范围却仍不可解析"的编辑、7514 A r1（写对一处 + 多写一处导致 7 个回归）、7514 A/C r2（读过区域后选择压症状） |
| 环境/评测缺陷本轮基本排除 | **修正**：评测本身没有失败（18/18、0 基础设施失败、三题 gold 自检通过），但**实验输入与编辑协议有 3 处缺陷**，其中 1 处（C 窗口）已经实际让 2 题的条件不成立 |
| 建议只跑 C×8 再判断 | **改为**：先过 Gate 1/2（均已通过），用 v4 辅助信息重跑 A/B/C，再按终止原因分布决定下一步 |
| （新增）0 strict ⇒ 语义不足 ⇒ 停止 SFT | **不成立**：见 §0 更正二。旧轨迹的终止原因分布本身就分成 7 / 3 / 7 / 1，不应用一个 0 统一归因 |

## 6. Gate 2：参考修复能否通过**真实编辑工具**完成（已通过，3/3）

`scripts/check_repair_through_tools.py` 在受限容器里用 `read_file → replace_text/replace_lines →
run_tests` 走完整套修复，要求官方 test command 最终通过。这是基础设施自检（补丁就是答案），
但它回答了一个 `git apply` 回答不了的问题：**新的编辑约束有没有让任务变得不可执行。**

结果：**3/3 reachable**，`changed_files` 恰好是各题的实现文件，0 违规。
证据 `work/private/repair-through-tools-20260922.json`。

过程中确认了三件**可教会的具体行为**（都是模型在上一轮里做错的）：

| 事实 | 证据 |
| --- | --- |
| 单靠 `replace_lines` **不能**表达参考修复 | 7365 三个 hunk 全部由 `replace_text` 完成；7514 需要 5 次 `replace_text` + 2 次 `replace_lines`；7608 需要 2 次 `replace_text` + 3 次 `replace_lines` |
| 从上到下改 hunk 会**行号错位** | 7365 第 2 个 hunk 插入 5 行后，第 3 个 hunk 的旧坐标 385-396 已挪到 390-401；按原坐标 `replace_lines` 就会 `unmatched '}'`。自检改为**每文件自下而上**应用后通过 |
| 「整条语句」往往超过 `replace_lines` 的 80 行上限 | 7608 `responses.py` 第 202 行的 hunk 所属语句是 **11-231 行（221 行）**，且 221 行文本在一次 `read_file` 里会被字符预算截掉 42 行；该 hunk 只能靠 `replace_text` 自己的文本完成 |

这三条正好对应上一轮 A/B/C 里观察到的失败形态：模型大量使用 `replace_lines`、按从上到下的顺序改、
并且反复在"未读区间"上动手。

## 7. 下一步：三道门槛（前两道已过，第三道待跑）

1. **CPU 测量验收 —— 已通过。** 窗口旧侧坐标、全 hunk 覆盖、失败读取不计入已读、同文件不同区间
   允许、编辑后权限失效、本地与 Docker 行为一致（VM 337 tests OK / 4 skipped，Windows 336 / 1），
   辅助信息改为固定背景字段且不再每步重申。唯一的遗留项是 §5.2 的 traceback 提示（尚未修）。
2. **真实编辑工具正向自检 —— 已通过（3/3）。** 见 §6。
3. **冻结后重跑完整 18 条 —— 待跑（约 25 分钟 GPU）。** 要求：
   - 不与旧轨迹合并；A 也要重跑，作为新环境下的基线；
   - 不顺手改训练与奖励；
   - 用 v4 辅助信息（sha256 `59bead89…`）与新的编辑约束；
   - **按每条轨迹的实际终止原因分别判断**，而不是只看 0 strict。

终止原因分类器已加入 `scripts/audit_first_failures.py`（`termination` 字段，判据见其 docstring）：
`protocol_error` / `no_edit_attempt` / `recovery_loop` / `edit_construction` / `semantic_incomplete`。
它在**旧**轨迹上的分布是 —— recovery_loop 3、no_edit_attempt 7、protocol_error 1、
semantic_incomplete 7 —— 也就是说旧协议不但每一条都是 0，而且**卡住的地方并不相同**：
7 条从头到尾没编辑过，3 条把预算耗在拒绝循环上，只有 7 条真正走到了"改完仍然失败"。
这 7 条才是语义证据所在，而重跑要回答的正是：修掉测量之后，这个分布往哪边移。

**什么结果才允许下结论**：若重跑后仍 0 strict，但 `no_edit_attempt` 与 `recovery_loop` 基本消失、
`semantic_incomplete` 明显上升且失败发生在模型已读过目标区域之后，才能说"14B 基座的主要瓶颈是
语义修复"；届时比较更强模型或设计"编辑质量"监督数据才有依据。反之，若 `edit_construction`
仍占多数，要修的是工具与提示，不是模型。

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
