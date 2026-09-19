# Harness navigation, finish refusal and edit integrity (2026-09-19)

## Decision

Keep the execution verifier as the only ground truth and keep changing only what the policy
sees. This round followed one held-out task (`getmoto__moto-7393`, 3 repetitions, 14B at
temperature 0.8, 24 steps, 32k context) through six prompt versions. Each version moved the
stored failure category, and each move named the next defect:

| prompt | categories | steps | changed files | what the trajectories show |
| --- | --- | --- | --- | --- |
| `v15` | `no_edit_attempt` 3/3 | 4/5/4 | 0/0/0 | read the test, search the route string, `finish` |
| `v16` | `protected_test_edit` 3/3 | 21/6/14 | 0/0/0 | the refusal removed the give-up, so the policy edited the only file it had read, which was the verifier-owned test |
| `v17` | `patch_failed_verifier` 3/3 | 9/8/7 | 1/1/1 | `IMPLEMENTATION_CANDIDATE` lines sent it to the implementation, and it patched the wrong one |
| `v18` | `patch_failed_verifier` 2, `protected_test_edit` 1 | 10/6/6 | 1/0/1 | edit integrity plus a legible verifier output; one trial still edited the test file |
| `v19` | `patch_failed_verifier` 2, `edit_action_failed` 1 | 10/24/16 | 1/0/1 | the failure summary names the failing node, the exception and the frame literals |
| `v20` | `patch_failed_verifier` 1, `protected_test_edit` 1, `edit_action_failed` 1 | 10/6/24 | 1/0/0 | adds the failing statement with its file and line |

`v16` is the load-bearing result. Refusing a `finish` that has no edit behind it is what made
the missing navigation visible: before it, the policy ended the episode before the harness could
show that it had no path from the failing test to the implementation.

The last three versions do not raise the success rate on this task - `pass_at_1` is `0.0` in
every row - and they were not expected to. They move the failure from "the episode ended without
an edit" to "the episode edited the wrong place and said why", which is the state a training run
can learn from. The task's real fix is the missing `/moto-api/config` route: the graded failure
is a `404` whose body is `Not yet implemented`, produced by
`moto/core/botocore_stubber.py:116`, while every arm patched `moto/core/config.py` - the module
the failing test imports. The harness can hand the policy the literal; selecting the module that
produces it is policy work.

## What changed

1. `finish` is refused while the verifier fails and no source file has been edited
   (`premature_finish_refusal` in `src/coding_agent_rl_lab/environment.py`, both step
   implementations). Both environments now run the verifier as part of `finish` and answer a
   give-up with the failure itself, so the refusal carries evidence instead of hiding the
   episode's real state. An unrepaired failure is still valid evidence and is reused rather than
   paid for twice, and a verifier timeout is allowed through exactly as `run_tests` terminates
   on it.
2. `search_text` names the implementation a test-only result exercises. When every match is in a
   test or documentation file, the result ends with `IMPLEMENTATION_CANDIDATE:<path>` lines
   resolved from those files' imports (`imported_module_paths`,
   `implementation_candidate_lines`), deepest module first. In `v16` the policy guessed
   `moto/server/__init__.py`; in `v17` it read `moto/core/config.py` on the step after the search.
3. The loop-recovery directive no longer sends the policy into a verifier-owned file. The guard
   takes an `is_read_only` predicate, and the directive separates implementation files already
   read from test-only reading.
4. An edit that would leave the edited Python file unparseable is refused, not applied, and the
   refusal names the line (`python_edit_syntax_error`, mirrored inside the Docker
   `_REPLACE_TEXT_SCRIPT` and `_REPLACE_LINES_SCRIPT`). Two `v17` trials replaced a line whose
   indentation did not belong to its block, the container then imported a broken module, and
   pytest answered `found no collectors` - a state the policy could not see or repair.
5. The verifier observation shows both captured streams and leads with the actionable part
   (`verifier_output_streams`). `stderr` used to win outright, which discarded the import
   traceback that `stdout` carried, and `failure_summary` now names the failing node, the last
   `E` line, and the short string values in the failing frame - the `s = 'Not yet implemented'`
   that the `v17`/`v18` policies needed and never searched for.
6. `PROMPT_VERSION` moves to `coding-tools-json-v19`
   (`src/coding_agent_rl_lab/model_policy.py`): the prompt documents the candidate lines, the
   refusal, the unparseable-edit refusal, the collection-error meaning, the failure summary, and
   states that editing a verifier-owned file is a hard violation that ends the episode.
7. `failure_summary` also lifts the failing statement with its file and line from pytest's `>`
   frame marker, and `v20` documents that line. On this task it reads
   `[failing statement] tests/test_core/test_config.py:20: assert resp.json()["batch"] == ...`,
   which is the request assertion rather than the config literal the policy kept editing.
8. A failed `replace_text` whose text is nowhere in the file now says to run `search_text` with
   it, because the file that produces a runtime string is not necessarily the file the policy has
   open. This is the third arm's dominant mistake: the policy had `Not yet implemented` in the
   observation and used it as a patch target instead of a search term.

## Evidence

All rows are the same task, seeds (`140001`, `150001`, `160001`), model, temperature and step
budget; `remote-artifacts/swe-gym-v1?-held-out-14b-32k-t08-trajectories.jsonl` holds the raw
trajectories and `swe-gym-v1?-held-out-14b-32k-t08-report.json` the reports.

- `v15`: 3/3 `no_edit_attempt`, 0 changed files, steps 4/5/4.
- `v16`: 3/3 `protected_test_edit`, 0 changed files, steps 21/6/14, loop refusals 15/1/9.
- `v17`: 3/3 `patch_failed_verifier`, 1 changed file each (`moto/core/config.py`), steps 9/8/7.
- `v18`: `patch_failed_verifier` 2, `protected_test_edit` 1, changed files 1/0/1, steps 10/6/6.
- `v19`: `patch_failed_verifier` 2, `edit_action_failed` 1, changed files 1/0/1, steps 10/24/16.
- `v20`: `patch_failed_verifier` 1, `protected_test_edit` 1, `edit_action_failed` 1, changed files
  1/0/0, steps 10/6/24.

The policies now reach and edit the implementation instead of ending the episode or touching the
test file, and the remaining gap is patch content: every `v17`/`v18` patch changed
`moto/core/config.py`, while the graded failure comes from the `/moto-api/config` handler that
answers `Not yet implemented`.

## Validation

- Windows: `python -m pytest tests -q` -> 236 passed, 1 skipped.
- Ubuntu VM (authoritative): `scripts/sync_and_validate_vm.ps1 -DockerIntegration` -> 236 tests
  OK, 1 skipped, and the real-Docker integration test passes, including the finish refusal and
  the exact-text repair hint inside the container.

## Not in this change

No scoring rule was relaxed, no learned scorer was added, and the verifier-owned test boundary is
still a terminating violation. `RewardVector`, `schema_version` 3 and the trajectory/report
schemas are unchanged, so older artifacts still load.

## Next

- A 10-task, 1-repetition sweep of the current prompt is running on the GPU host
  (`work/swe-gym-v20-all-14b-32k-t08-*`, seed `142001`) to see whether the navigation fix
  generalises beyond the held-out task.
- Re-run the 7B arm, whose `v15` failure was a degenerate repeated search, against the current
  prompt.
- Rebuild the SFT dataset on the GPU host so its `prompt_version` matches the current prompt.
- Train on these trajectories: every arm now reaches a source edit or a diagnosable refusal, so
  the stored rollouts carry the target-selection and repair signal that the prompt alone did not
  supply.
