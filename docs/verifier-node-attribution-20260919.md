# Verifier node attribution (2026-09-19)

## Decision

The execution verifier keeps deciding success. This change only makes its result attributable
to the declared `FAIL_TO_PASS` / `PASS_TO_PASS` nodes, so a rollout report can separate "did
not fix the target" from "broke a regression test" from "never produced a patch". No learned
outcome verifier was added, and no scoring rule was relaxed.

## What changed

1. `RewardVector.regression_free` is no longer a copy of `tests_passed`
   (`src/coding_agent_rl_lab/rollout.py`). When the task declares graded nodes it is derived
   from them: `PASS_TO_PASS` failures and ungraded failures both clear it. Fixture tasks that
   declare no nodes keep the previous `final.passed` behaviour.
2. `VerifierBreakdown` (`src/coding_agent_rl_lab/contracts.py`) records
   `fail_to_pass_total`, `fail_to_pass_resolved`, `pass_to_pass_total`,
   `pass_to_pass_regressed`, `failed_nodes`, `ungraded_failed_nodes` and `collection_error`.
   Unknown counts stay `None`: a run that produced no usable per-test summary never becomes a
   zero.
3. `verifier_breakdown()` and `verifier_collection_error()`
   (`src/coding_agent_rl_lab/reward_shaping.py`) compute that mapping. `build_training_reward`
   accepts the declared targets and attaches the breakdown to every training reward audit. The
   `conservative-v2` arithmetic is unchanged; only its collection-error regex was factored into
   the shared helper.
4. `SWEGymTaskAdapter` refuses a `test_command` that does not score exactly the declared
   targets (`src/coding_agent_rl_lab/swe_gym.py`). A command that dropped `PASS_TO_PASS` nodes
   silently removed regression coverage, and a command that added nodes graded tests the reward
   contract never declared. Both now fail before a container starts.
5. The rollout path records the `conservative-v2` training reward per trajectory and
   `mean_training_reward` per report (`src/coding_agent_rl_lab/rollout.py`), so the rollout and
   GRPO scales can be compared instead of silently differing. `RewardVector` and
   `mean_scalar_reward` are untouched, so existing `pass_at_1` numbers stay comparable.
6. The failure taxonomy report carries the per-trial node attribution plus a
   `graded_failure_attribution` block (`src/coding_agent_rl_lab/failure_analysis.py`). The
   categories themselves are unchanged. 19 new tests cover the breakdown, the adapter guard and
   the rollout path.

## Compatibility

The trajectory schema stays at version 3. `verifier` and `training_reward` are optional, so
every stored `remote-artifacts/` and `work/private/` trajectory still loads; those rows report
unknown node attribution instead of fabricated zeros. `git diff --check` is clean.

## Validation

- Windows, Python 3.12.7: 202 tests passed, 1 skipped (the guarded Docker integration test).
- Ubuntu VM (`wesz@192.168.137.130`, Python 3.14.4), isolated `/tmp` copy of the current source:
  202 tests passed, 1 skipped.
- Ubuntu VM real Docker integration test against the cached
  `xingyaoww/sweb.eval.x86_64.getmoto_s_moto-7365:latest` base image: passed.
- `python -m coding_agent_rl_lab.failure_analysis` on the stored
  `swe-gym-train-all-6x1-base-v12-seed57000` trajectories still reports `no_patch 6/6` with zero
  declared targets, confirming the backward-compatible read path.
- Temporary sync bundles and their remote `/tmp` copies were removed; no run left a container or
  image behind.

## Not in this change

- No learned outcome scorer, and no change to what counts as success.
- The failure classifier still merges "no edit attempted" with "no patch produced"; the
  loop/infrastructure split is separate harness work.
- The prompt, tool protocol, step budget and observation formatting are untouched.
