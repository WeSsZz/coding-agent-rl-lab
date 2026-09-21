# Data-source feasibility survey: SWE-Gym SFT warm-start + GRPO with a container scaffold

> **Provenance (added when this was filed into the repository).** This survey was produced on
> 2026-09-22 as part of phase 1 architecture selection. It is a *survey*, not a result: every row
> count and licence below is a claim about an upstream artifact, and anything marked UNKNOWN is
> genuinely unverified. Dataset cards were read through a Gitee mirror of the HuggingFace dataset
> org, **not** through `huggingface.co` — that host is unreachable from both this Windows checkout
> and the Ubuntu VM (verified). Treat any HuggingFace row count here as "one mirror's reading of a
> card" until it is re-checked from a host that can reach HuggingFace. GitHub facts (repo state,
> licences, file contents, issue threads) were read directly from `api.github.com` /
> `raw.githubusercontent.com`.

## 0. Evidence base and one hard limitation

**`huggingface.co` and `datasets-server.huggingface.co` are UNREACHABLE from this
environment.** Every `web_fetch` to them failed (`TypeError: fetch failed`), and a direct
`Invoke-WebRequest` from pwsh timed out. `arxiv.org` is also unreachable. Only
`api.github.com`, `raw.githubusercontent.com`, `github.com` and `gitee.com` responded.

I recovered HF dataset cards through the Gitee mirror of the HF dataset org:
`https://raw.giteeusercontent.com/hf-datasets/<dataset-name>/raw/main/README.md`.
That mirror covers some but **not all** datasets. Where a card could not be mirrored I say so
explicitly. **I could not use the datasets-server API at all**, so no `/info`, `/splits`,
`/rows` or `/first-rows` output is reported; row counts below come from dataset cards and
from code that consumes the datasets.

## 1. SWE-Gym official assets

Repo (all from the GitHub API): `SWE-Gym/SWE-Gym`, **Apache-2.0** (LICENSE blob fetched,
11,357 B), default branch `main`, HEAD
`b681068ca20628c6987b7416cc4cf03f06b77ba5` (2025-07-29, "Merge pull request #19 from
dreasysnail/patch-1 / Update README.md"), `pushed_at` 2025-07-29T17:38:25Z, 742 stars,
primary language Jupyter Notebook. **Dormant since July 2025.**

The full recursive tree (fetched) shows the repo ships **no dataset files, no evaluation
harness and no Dockerfile/build script**. It contains only `README.md`, `docs/OpenHands.md`
(5,648 B), `docs/MoatlessTools.md`, `assets/` (incl. `assets/paper.pdf`), and `scripts/`
(`moatless-verifier/`, `openhands-verifier/`, `modal_misc/`, `serving/`,
`training/openhands/{train_torchtune_full.py, train_unsloth_qwen25coder_32b_verifier.py,
configs/policy/*.yaml, configs/verifiers/*.yaml}`).

### Datasets verified from the actual cards (via the Gitee mirror)

| id | split / config | rows | size |
|---|---|---|---|
| `SWE-Gym/SWE-Gym` | `train` / `default` | **2438** | download 43,644,473 B (~41.6 MB); dataset 213,621,640 B (~204 MB) |
| `SWE-Gym/SWE-Gym-Raw` | `train` / `default` | **64,689** | download 728,489,913 B; dataset 2,500,852,534 B (~2.33 GB) |

`SWE-Gym/SWE-Gym` fields: `instance_id`, `hints_text`, `patch`, `test_patch`, `created_at`,
`problem_statement`, `repo`, `base_commit`, `version`, `PASS_TO_PASS` (sequence<string>),
`FAIL_TO_PASS` (sequence<string>). Note there is **no `difficulty` field** on HF even though
Harbor's adapter reads one with a default.

**Licence caveat:** the `SWE-Gym/SWE-Gym` card frontmatter declares **no `license:` key**, and
neither does `SWE-Gym-Raw`. The GitHub repo is Apache-2.0; the HF artifacts do not state a
licence.

`SWE-Gym/SWE-Gym-Raw` fields add `repo`, `pull_number`, `issue_numbers`, with
`PASS_TO_PASS`/`FAIL_TO_PASS` as `null`. The card states most instances "doesn't have
associated python environment configured and is not validated with SWE-Bench verification
process".

`SWE-Gym/SWE-Gym-Lite`: the id is verified by consumers (Harbor:
`DATASET_MAP = {"lite": "SWE-Gym/SWE-Gym-Lite", "full": "SWE-Gym/SWE-Gym"}`, both loaded via
`load_dataset(hf_name)["train"]`), but **its card is not mirrored → row count unverified**.
Discrepancy to resolve: SWE-Gym's README says "Lite split of 234 instances"; Harbor's adapter
says **230** tasks.

### OpenHands rollout trajectories: yes, but names are only partly verified

SWE-Gym's own `docs/OpenHands.md` (fetched) says: "Download **OpenHands SFT trajectory
dataset** from huggingface (https://huggingface.co/datasets/**SWE-Gym/OpenHands-SFT-Trajectories**)",
and for the reward model "**SWE-Gym/OpenHands-Verifier-Trajectories**". Nebius's dataset card
additionally references **`SWE-Gym/OpenHands-Sampled-Trajectories`**.

I could **not** fetch any of these three cards (HF blocked; no Gitee mirror) → **fields, exact
row counts, splits and licence are UNKNOWN.** The only third-party numbers I found
(Nebius's cross-dataset table) for `SWE-Gym/OpenHands-Sampled-Trajectories`: **6,055
trajectories total, 491 successful**, 11 repos, OpenHands scaffold, gpt-4o-2024-08-06 +
claude-3-5-sonnet-20241022, function calling yes, max 50 turns, avg 18.9 turns. 491 matches the
paper's "less than 500 agent-environment interaction trajectories", which is a good coherence
check but is not a verified row count.

**Rollout output format** (verified from the `SWE-Gym/OpenHands` fork): the official script
`scripts/rollout-swe-train-full.sh` runs
`run_infer.sh <MODEL> HEAD CodeActAgent 3000 50 64 SWE-Gym/SWE-Gym train <N_RUNS>` with
`RUNTIME=remote`, and produces
`evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/CodeActAgent/<model>_maxiter_50_N_v2.2-no-hint-<exp>-run_N/output.jsonl`.
So the paper's training split for rollouts is `SWE-Gym/SWE-Gym`, split `train`, agent
`CodeActAgent`, `max_iter=50`.

**How trajectories became SFT data — and why this is now a risk.** In issue #9 (verified;
closed 2025-02-25) maintainers answer: set `log_completions = true`, then "use info in
`llm completion` and eval results to construct the dataset", and "we use Openhands version
**CodeActAgent 2.1**, you can use the folk here https://github.com/SWE-Gym/OpenHands". The
conversion script was linked as `convert_data.ipynb` under
`All-Hands-AI/OpenHands` branch **`xw/swegym`**. **That branch no longer exists** —
`/repos/OpenHands/OpenHands/branches/xw/swegym` returns 404 "Branch not found" and the raw
paths 404. **No official trajectory→SFT conversion script survives on GitHub.** A 2025-05-23
comment on the same issue notes the script kept the complete trajectory but dropped anything
over 128k tokens, and asks whether history trimming matched inference-time behaviour.

## 2. Other real OpenHands multi-turn trajectory datasets

### `nebius/SWE-rebench-openhands-trajectories` — card VERIFIED (highest-value find)
* Licence **`cc-by-4.0`**; tags code/synthetic/tools/agents/software; size 10K<n<100K.
* Scaffold **OpenHands v0.54.0**; generator `Qwen/Qwen3-Coder-480B-A35B-Instruct`; tasks from
  `nebius/SWE-rebench`.
* **67,074 trajectories total, 32,161 successful**; 1,823 repos; max 100 turns; avg 64.3 turns.
* Fields: `trajectory_id` (str), `instance_id` (str), `repo` (str), `trajectory` (list of chat
  messages), `model_patch` (unified diff), `exit_status` (`'submit'` or an OpenHands error
  string), `resolved` (int 0/1), `gen_tests_correct` (int|null), `pred_passes_gen_test`
  (int|null).
* Per-message fields by role: `system` → role, content; `assistant` → role, content,
  **`tool_calls`**; `user` → role, content; `tool` → role, content, `name`, `tool_call_id`.
* **Assistant tool calls AND environment observations (role `tool`) are both present.**
* `tool_calls[i].function.arguments` is stored as a **JSON string** and must be deserialized
  before chat templates are applied (explicit warning in the card).
* **No token ids and no logprobs** anywhere in the schema — confirmed absent.

### `nebius/swe-agent-trajectories` — card VERIFIED
80,036 rows, `cc-by-4.0`, **SWE-agent** scaffold (not OpenHands). Fields: `instance_id`,
`model_name`, `target` (bool), `trajectory` (list of `{cutoff_date, mask, role,
system_prompt, text}`), `exit_status`, `generated_patch`, `eval_logs`. Note the per-message
`mask` flag (useful) and a Llama-3.1 licence notice.

### SWE-Gym trajectory datasets
Three ids verified by reference (above); **cards not retrievable**, so schema/licence unknown.

### Published by OpenHands / All-Hands-AI
I found **no** All-Hands-AI-published agent-trajectory dataset I could verify. The old
OpenHands SWE-Bench README instead points at `git clone
https://huggingface.co/spaces/OpenHands/evaluation` (an HF *space* of `output.jsonl` archives).
Contents unverified because HF is blocked.

## 3. OpenHands code status

* **`All-Hands-AI/OpenHands` redirects to `OpenHands/OpenHands`** (same repo id 771302083) —
  the rename is real. MIT, 88,712 stars, default `main`, `pushed_at` 2026-09-21, size
  436,803 KB, **primary language TypeScript**, homepage openhands.dev.
* `contents/openhands` on default branch → **404 Not Found**. `contents/evaluation/benchmarks/swe_bench`
  → **404 Not Found**. **The old Python `openhands/` package and the Python SWE-Bench harness
  are gone from `main`**, replaced by the TypeScript app.
* **`OpenHands/software-agent-sdk`** — MIT, **Python**, 1,147 stars, created 2025-08-23,
  pushed 2026-09-21. README (fetched) states it provides "Python, TypeScript, and REST APIs for
  building agents that work with code"; agents can use the local machine or "run inside
  ephemeral workspaces (e.g. in Docker or Kubernetes) using the Agent Server". Example:
  `from openhands.sdk import LLM, Agent, Conversation, Tool`, plus `TerminalTool`,
  `FileEditorTool`, `TaskTrackerTool`; `Conversation(agent=agent, workspace=cwd)`,
  `.send_message(...)`, `.run()`. **So yes — a programmatically drivable Python agent with
  container execution and a terminal tool exists.** Caveat: I verified `LLM(model=...,
  api_key=...)` only; an explicit OpenAI-compatible **`base_url`** argument is **not verified**
  for the SDK (the old Python OpenHands did support it via `config.toml`).
* **`OpenHands/codescout`** — exists, described on its GitHub page as "Repo for collaboration on
  OSS agentic search". Verified from the HTML page only; the API JSON fetch failed repeatedly,
  so **licence and language are unverified**.
* **`SWE-Gym/OpenHands`** — fork, **Python, MIT**, 5 stars, default `main`, created
  2024-12-23, **last pushed 2025-02-20** (frozen). The `openhands/` package is intact:
  `agenthub, controller, core, events, linter, llm, memory, resolver, runtime, security,
  server, storage, utils`. Also `evaluation/swe_bench/{run_infer.py, eval_infer.py, prompt.py,
  README.md, scripts/}` and `scripts/rollout-swe-train-{full,lite}.sh`,
  `eval-swetrain-{full,lite}-rollout.sh`. **This is the frozen Python scaffold that reproduces
  the SWE-Gym paper (CodeActAgent v2.1/2.2, Python).**

## 4. Environment images

**Convention VERIFIED twice, exactly.** Harbor's `get_image_names()` builds
`xingyaoww/sweb.eval.x86_64.{instance_id.replace("__","_s_")}`, lowercased — so
`conan-io__conan-14760` → `xingyaoww/sweb.eval.x86_64.conan-io_s_conan-14760`. The OpenHands
SWE-Bench README gives the same rule for `django_django-11011` →
`sweb.eval.x86_64.django_s_django-11011`, pulled automatically by `run_infer.sh` when
`USE_INSTANCE_IMAGE=true` (default). SWE-Gym's README points at the `xingyaoww/sweb.eval.x86_64`
prefix on Docker Hub.

**Count:** 2,438 full instances ⇒ up to 2,438 images, minus **38 that were never published**.
Harbor's oracle validation (verified) reports swegym full = 2,438 tasks / 2,396 passed / 38
errors / mean reward 0.983, and swegym-lite = 230/230 passed / 1.000; all 38 errors are missing
DockerHub images, and oracle pass rate on available images is 2,396/2,396 = 100%. Missing-image
tasks by repo: pandas 15, dvc 5, conan 4, dask 4, modin 3, hydra 2, pydantic 2, bokeh 1, monai 1.
The upstream issue **SWE-Gym/SWE-Gym#23 is still OPEN** (updated 2026-05-11); the reporter counted
only 71 conan-io images against 75 dataset instances.

**Size:** Harbor states SWE-Gym images are "large (several GB each)". The OpenHands SWE-Bench
README states "at least 200-500GB" for instance-level images, ~200 GB to pull all
instance-level images, or ~50 GB for env-level images. **There is no documented per-image size
for SWE-Gym specifically** — "several GB each" is the only figure I verified.

**Disk implication (derived, not documented):** 2,438 × ~2–5 GB ≈ **5–12 TB** for the full
instance set. That is not viable on one 96 GB workstation. Plan for pull-on-demand plus a
bounded local cache (start with SWE-Gym-Lite's ~230 images) and `docker system prune`.

## 5. Verdict

### What can serve as a REAL SFT warm-start source

1. **`SWE-Gym/OpenHands-SFT-Trajectories`** — the on-distribution choice: real OpenHands
   rollouts on SWE-Gym tasks, documented by SWE-Gym's own instructions. **UNKNOWN: row count
   (likely ~491 successful), fields, licence, and whether it is function-calling or
   CodeActAgent-2.1 text-action format.** Highest-priority verification item.
2. **`nebius/SWE-rebench-openhands-trajectories`** — the best *verified* large source:
   **32,161 successful** OpenHands-v0.54 function-calling trajectories with real tool calls and
   real environment observations, under permissive `cc-by-4.0`. Off-distribution in task ids
   (SWE-rebench, not SWE-Gym) but excellent for teaching tool-use format to a
   container-executing scaffold. Use a filtered successful subset.
3. **`SWE-Gym/OpenHands-Sampled-Trajectories`** — small, real, on-distribution; verify the card.
4. **`nebius/swe-agent-trajectories`** (80,036) — large, but SWE-agent scaffold; only worth it if
   you accept tool-name remapping.
5. **`SWE-Gym/SWE-Gym-Raw`** (64,689) — problems only, no verified environments; at most for
   prompt/curriculum diversity.

Gold-hunk-derived synthetic SFT is not needed: enough real tool-use data exists.

### What must be converted

* **Deserialize** `tool_calls[].function.arguments` from JSON strings (explicit warning in the
  Nebius card) so chat templates render training and inference identically.
* **Tool-name mapping.** Source trajectories use the source scaffold's function names; they must
  be renamed onto your scaffold's tool schema. *I did not verify the exact OpenHands v0.54 or
  CodeActAgent-2.1 tool names in this survey* — inspect real rows first. Note the versions
  differ in kind: OpenHands **v0.54** (nebius) is natively function-calling, whereas the
  SWE-Gym fork is **CodeActAgent 2.1** (pre-function-calling action parsing), so the SWE-Gym
  data likely needs a larger structural rewrite.
* **Role-stream normalisation:** `system/assistant/user/tool` → your template. If your model's
  template has no tool role, tool observations must be re-rendered as user text.
* **Loss masking must be built by you.** `nebius/swe-agent-trajectories` has a `mask` field;
  the OpenHands-format datasets expose only roles, so assistant-only masking is on you.
* **Reimplement the deleted conversion script** (`convert_data.ipynb`): inputs are the OpenHands
  `output.jsonl` plus `log_completions=true` LLM-completion records plus eval results (issue #9).
* **No token ids / logprobs exist** in any of these datasets — confirmed for the two cards I
  could read. They are therefore **SFT-only**; GRPO must generate on-policy rollouts in your own
  containers.

### Unknowns that gate the plan

(i) the four SWE-Gym dataset cards (`SWE-Gym-Lite`, `OpenHands-SFT-Trajectories`,
`OpenHands-Verifier-Trajectories`, `OpenHands-Sampled-Trajectories`) — must be checked from an
HF-connected host (or the datasets-server API); (ii) whether `software-agent-sdk`'s `LLM`
accepts an OpenAI-compatible `base_url`; (iii) exact source tool names/action formats.

### Recommended plan

Verify the cards first, then take SFT data from `SWE-Gym/OpenHands-SFT-Trajectories` (filtered to
successful) plus a mapped slice of `nebius/SWE-rebench-openhands-trajectories`; run GRPO on
**SWE-Gym-Lite train (230 tasks)** before the full 2,438, excluding the 38 image-less instances,
with pull-on-demand images. With one 96 GB GPU the binding constraints are rollout throughput
and image I/O — not data: 50–100 max iterations with 19–64 average turns per trajectory will
dominate wall-clock, and a co-resident vLLM rollout + trainer must fit alongside it.

## URLs actually fetched (200 unless noted)

GitHub API: `api.github.com/repos/SWE-Gym/SWE-Gym`,
`.../branches/main` (404), `.../commits?per_page=2`,
`.../git/trees/afe7b20f6085e75450802de1dbadf635343ec6e8?recursive=1`,
`.../issues/9`, `.../issues/9/comments`, `.../issues/23`;
`api.github.com/repos/All-Hands-AI/OpenHands` (redirect → OpenHands/OpenHands, id 771302083);
`api.github.com/repos/OpenHands/OpenHands`, `.../contents/openhands` (404),
`.../contents/evaluation/benchmarks/swe_bench` (404),
`.../branches/xw/swegym` (404);
`api.github.com/repos/OpenHands/software-agent-sdk`, `.../contents/README.md`;
`api.github.com/repos/OpenHands/codescout` (fetch failed; `github.com/OpenHands/codescout` HTML 200);
`api.github.com/repos/SWE-Gym/OpenHands`, `.../contents/openhands`,
`.../contents/evaluation/swe_bench`, `.../contents/evaluation/swe_bench/scripts`, `.../contents/scripts`;
`api.github.com/repos/SWE-rebench/SWE-rebench-V2/contents/README.md`;
`api.github.com/search/repositories?q=swe-rebench+in:name`;
`api.github.com/orgs/nebius/repos` (nebius/SWE-rebench → 404; nebius/swe-rebench → 404).

Raw GitHub: `raw.githubusercontent.com/SWE-Gym/SWE-Gym/main/README.md`, `/LICENSE`,
`/docs/OpenHands.md`; `raw.githubusercontent.com/SWE-Gym/SWE-Bench-Fork/main/README.md`;
`raw.githubusercontent.com/SWE-Gym/OpenHands/main/evaluation/swe_bench/README.md`,
`/scripts/rollout-swe-train-full.sh`;
`raw.githubusercontent.com/harbor-framework/harbor/main/adapters/swegym/{README.md,adapter.py,utils.py}`;
`raw.githubusercontent.com/OpenHands/OpenHands/xw/swegym/...` (404).

Gitee HF mirror: `raw.giteeusercontent.com/hf-datasets/SWE-Gym/raw/main/README.md`,
`.../SWE-Gym-Raw/raw/main/README.md`,
`.../SWE-agent-trajectories/raw/main/README.md`,
`.../SWE-rebench-openhands-trajectories/raw/main/README.md`;
404s for `OpenHands-SFT-Trajectories`, `OpenHands-Sampled-Trajectories`,
`OpenHands-Verifier-Trajectories`, `SWE-Gym-Lite`, `SWE-Gym-lite`,
`SWE-Gym-OpenHands-SFT-Trajectories`.

Failed/unreachable: `datasets-server.huggingface.co/{info,splits,rows,valid}` (all),
`huggingface.co/**` (all), `hf-mirror.com`, `r.jina.ai`, `arxiv.org/abs/2412.21139`,
`raw.githubusercontent.com/nexustech101/.../data_engineering_sources.md` (timeout).
