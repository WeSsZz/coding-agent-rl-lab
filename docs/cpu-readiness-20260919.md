# Local and Ubuntu CPU readiness check (2026-09-19)

## Decision

The local and Ubuntu worker paths are ready for the next GPU-side preflight. No model service,
model weights, CUDA workload, SFT update, or GRPO update was started during this check.

The GPU run must still begin without `--train` and pass the model/tokenizer/CUDA/TRL preflight
before any SFT update is authorized.

## Local validation

- Environment: Windows, Python 3.12.7.
- Reference pipeline: 2 tasks x 3 repetitions, 6/6 strict successes, no violations.
- Unit suite after fixes: 183 tests passed; the real Docker integration test was skipped locally by
  its explicit `RUN_DOCKER_INTEGRATION=1` guard.
- `git diff --check`: passed.

Two validation defects were corrected:

1. `grpo_evaluate.validate_resume` used direct indexing for newly added navigation fields. A legacy
   manifest could therefore raise `KeyError` instead of being compared and rejected as an
   incompatible resume. Both manifests now use optional lookup, and a regression test verifies that
   a legacy manifest is rejected when the expected navigation mode is explicit.
2. The Docker integration smoke depended on downloading and installing `git` from Debian on every
   run and matched a stale baseline message. It now accepts an explicit cached base image that
   already contains Git, while preserving the original default, and matches the stable
   `Tests failed` observation.

## Ubuntu validation

- Environment: Ubuntu kernel 7.0.0-31, Python 3.14.4, Docker Engine 29.7.2.
- Validation used an isolated `/tmp` copy of the current local source. The existing experimental
  worktree under `/home/wesz/coding-agent-rl-lab` was not modified.
- Source archive SHA-256 used for the main remote run:
  `8e1e5c7a79cbb23d7fb82484471e5db01badafb412dc7d1d14fe8e6524314ed0`.
- All 182 non-integration tests passed on Ubuntu. The Docker integration test passed separately
  with the cached `xingyaoww/sweb.eval.x86_64.getmoto_s_moto-7365:latest` base image.
- Official `getmoto__moto-7365` SWE-Gym smoke passed: pinned row and image identity, restricted
  container startup, fail-before, verifier-only gold patch, pass-after, and cleanup.
- A process-level fixture worker smoke passed on loopback with authentication, baseline failure,
  file listing, finalize, reward, and session cleanup.
- A process-level SWE-Gym worker smoke also passed on loopback for `getmoto__moto-7365`, including
  HTTP worker -> Docker environment -> verifier -> finalize and cleanup.
- No validation worker, sandbox container, or temporary smoke image remained after the run. The
  isolated `/tmp` validation directory was removed.

The first Docker integration attempt failed before the test environment was built because the
temporary build container could not resolve `deb.debian.org`. The same integration test then passed
against the already cached, Git-containing SWE-Gym image, removing external package mirrors from
the validation path.

## Rebuilt SFT data

The six pinned train tasks were downloaded again from the official rows and rebuilt with the current
code. The result contains 204 examples: 51 each for locate, inspect, edit, and verify. One oversized
target action was excluded by the existing audit rule. Regression and held-out tasks are absent.

The current source prompt is `coding-tools-json-v14`; the converted dynamic-tool dataset uses
`grpo-tools-dynamic-bare-json-v2` and `grpo-bare-json`. Compared with the older local v12 dataset,
all 204 example IDs, stages, and target actions are identical. The 204 message payloads changed only
because the current system prompt documents the bounded `replace_lines` recovery path.

Files retained under ignored `work/private/`:

- `validation-gold-sft-v1.jsonl`:
  `0f2fdb505ad9d1b60acfe6c9c3e0cacdfa36a2dce58491c7f94a8d45b93fd837`
- `validation-gold-sft-v1-report.json`:
  `7e606d9f3e278ba74326012fa0ac3a2d37767e2c0bd2ba26bfbd6bba226eeb43`
- `validation-grpo-sft-v2.jsonl`:
  `03581d42354c9a4f9dc5309366b0c1554e4f4fb979d835092f098fa775ef04cb`
- `validation-grpo-sft-v2-report.json`:
  `07c24d6c46e4a3863ae8cf32b5123e3773eb849bf092771d181d09cbc3eff6fb`

Use the rebuilt v14 files for the next preflight. Do not silently fall back to the older canonical
files whose source report is `coding-tools-json-v12`.

## GPU-side entry gate

After the GPU instance is started, first run `sft_train` without `--train` against
`work/private/validation-grpo-sft-v2.jsonl` and its report. This remaining preflight must verify the
exact local model snapshot, tokenizer/chat template rendering, dynamic tool JSON schema, token
lengths without truncation, installed Torch/Transformers/TRL versions, and CUDA visibility. Only a
successful recorded preflight permits a later command with `--train`.
