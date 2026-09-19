# Harness observations, loop recovery and failure taxonomy (2026-09-19)

## Decision

The execution verifier stays the ground truth. This change is about the two things around it:
what the policy actually sees, and whether a failed trial is attributed to the policy, the
harness, or the served model configuration. No learned scorer was added and no scoring rule was
relaxed.

The starting evidence was the stored rollouts: every base-policy train trial was reported as
`no_patch`, and the held-out batch that produced `0/8` contained six trials that never reached
the model at all. Both symptoms were measurement artefacts.

## What changed

1. `read_file` returns numbered lines (`N: text`) plus a self-describing footer
   (`render_numbered_window` in `src/coding_agent_rl_lab/environment.py`, `_READ_FILE_SCRIPT` in
   `src/coding_agent_rl_lab/docker_environment.py`). The window is 200 lines and 8000 characters;
   when it cuts, the footer names the next `start_line`. A file is never silently shown in part,
   and the policy no longer has to re-count lines to build a `replace_lines` range.
2. `search_text` ranks implementation files ahead of tests and prose, caps matches per file and
   caps the total, and trims from the top so the best-ranked matches survive the prompt budget.
   A query with no literal hit is retried once with its longest token, labelled
   `Longest token in the query: <token>`, before falling back to `SUGGESTED_PATH`.
3. Loop refusals escalate. After the first refusal the guard repeats the rule; after that it
   states what the policy is failing to track (files already read, queries already used, whether
   any patch exists) and tells it to edit a file it already read or call `finish`.
   `CodingEnvironment.loop_rejections` exposes the count and `RewardVector.loop_rejections`
   stores it per trajectory.
4. Docker observations use a head-and-tail bounded renderer (`_bounded_observation`) instead of
   `stdout[-max_output_chars:]`, which used to drop exactly the head, so the highest-ranked
   matches and the `read_file` footer were the first things lost.
5. The failure taxonomy splits the old `no_patch` catch-all into `loop` and `no_edit_attempt`,
   and classifies `infra_error` / `context_window_exceeded` before any behaviour-based category
   (`src/coding_agent_rl_lab/failure_analysis.py`). The report gains `infra_error_count`,
   `policy_failure_count` and per-trial `loop_rejections` / `infra_error`.
   `policy_transport_error` and `context_window_exceeded` stay in `FAILURE_CATEGORIES` so older
   reports remain readable.
6. `PROMPT_VERSION` moves to `coding-tools-json-v15`
   (`src/coding_agent_rl_lab/model_policy.py`): the system prompt documents the numbered read,
   the ranked and capped search with its token fallback, and the escalating refusal, and the
   hard-coded "12-step episode, edit by step 8" instruction is replaced by a budget rule.
7. `OpenAICompatiblePolicyConfig.context_window_tokens` adds a preflight. When it is set to the
   served `--max-model-len`, an oversized prompt is refused before the request is sent and is
   recorded as a transport violation, so a context overflow is an `infra_error` instead of a
   policy failure. `swe_gym_rollout.py` exposes it as `--context-window-tokens`.
8. `swe_gym_rollout.py --max-steps` defaults to 24 instead of 12. Seven of twelve steps were
   loop refusals in the stored runs, so the old budget could not reach a patch.

## Evidence

Re-running the stored checkpoints through the new classifier separates configuration failures
from policy failures:

- `swe-gym-train-all-6x1-base-v12-seed57000`: 6 trials, 0 success, 0 infra, `loop 6/6`.
- `swe-gym-train-remaining-5x2-v9-seed54000`: 10 trials, 0 success, 0 infra, `loop 10/10`.
- `swe-gym-held-out-base-seed51000`: 8 trials, 0 success, **6 infra** (all
  `context_window_exceeded`), 2 `protected_test_edit`. Only two held-out trials are attributable
  to the policy at all, so the stored `0/8` is not a policy measurement.
- `swe-gym-train-7509-base-16k-seed52000`: 4 trials, 0 success, 0 infra,
  `patch_failed_verifier 2`, `loop 1`, `edit_action_failed 1`. With a 16k window the policy does
  reach edits and fails at the patch, which is the signal worth optimising.

## Compatibility

The trajectory schema stays at version 3. `RewardVector.loop_rejections` defaults to `0`, so
every stored `remote-artifacts/` and `work/private/` trajectory still loads; for those rows the
classifier falls back to the refusal wording in the observations. `RewardVector`, `scalar` and
`mean_scalar_reward` semantics are unchanged, so existing `pass_at_1` numbers stay comparable.
The v15 prompt bump does invalidate the rebuilt v14 SFT artifacts for any comparison that looks
at prompt text.

## Validation

- Windows, Python 3.12.7: 217 passed, 1 skipped (the guarded Docker integration test).
- Ubuntu VM (`wesz@192.168.137.130`, Python 3.14.4), isolated `/tmp` copy of the current source:
  217 passed, 1 skipped.
- Ubuntu VM real Docker integration test against the cached
  `xingyaoww/sweb.eval.x86_64.getmoto_s_moto-7365:latest` base image: passed. The test now also
  asserts a numbered `read_file` and a ranked `search_text` inside the container, so the two
  rewritten Docker scripts are exercised in the real image rather than only on the host.
- `git diff --check` is clean; temporary sync bundles and their remote `/tmp` copies were
  removed and no container or image was left behind.

## Not in this change

- No learned outcome verifier, and no change to what counts as success.
- The served `--max-model-len` is a launch setting outside this repository. The client now
  refuses an oversized prompt instead of recording an HTTP 400.
- Reward arithmetic, the graded-node contract and the step-observation budget
  (`max_observation_chars` / `max_history_chars`) are unchanged.
