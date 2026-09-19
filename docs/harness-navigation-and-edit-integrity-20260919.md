# Harness navigation, finish refusal and edit integrity (2026-09-19)

## Decision

Keep the execution verifier as the only ground truth and keep changing only what the policy
sees. This round followed one held-out task (`getmoto__moto-7393`, 3 repetitions, 14B at
temperature 0.8, 24 steps, 32k context) through eight prompt versions. Each version moved the
stored failure category, and each move named the next defect:

| prompt | categories | steps | changed files | what the trajectories show |
| --- | --- | --- | --- | --- |
| `v15` | `no_edit_attempt` 3/3 | 4/5/4 | 0/0/0 | read the test, search the route string, `finish` |
| `v16` | `protected_test_edit` 3/3 | 21/6/14 | 0/0/0 | the refusal removed the give-up, so the policy edited the only file it had read, which was the verifier-owned test |
| `v17` | `patch_failed_verifier` 3/3 | 9/8/7 | 1/1/1 | `IMPLEMENTATION_CANDIDATE` lines sent it to the implementation, and it patched the wrong one |
| `v18` | `patch_failed_verifier` 2, `protected_test_edit` 1 | 10/6/6 | 1/0/1 | edit integrity plus a legible verifier output; one trial still edited the test file |
| `v19` | `patch_failed_verifier` 2, `edit_action_failed` 1 | 10/24/16 | 1/0/1 | the failure summary names the failing node, the exception and the frame literals |
| `v20` | `patch_failed_verifier` 1, `protected_test_edit` 1, `edit_action_failed` 1 | 10/6/24 | 1/0/0 | adds the failing statement with its file and line |
| `v21` | `patch_failed_verifier` 1, `edit_action_failed` 2 | 10/24/24 | 1/0/0 | the search that found only the test now leads with the route table; trials 1-2 read `moto/moto_api/_internal/urls.py` and `responses.py`, and trial 1 edited the handler |
| `v22` | (running) | - | - | the failure summary stops reporting captured-log lines as failing nodes and names the exception the code under test logged |

`v16` is the load-bearing result. Refusing a `finish` that has no edit behind it is what made
the missing navigation visible: before it, the policy ended the episode before the harness could
show that it had no path from the failing test to the implementation.

The versions up to `v20` do not raise the success rate on this task - `pass_at_1` is `0.0` in
every row - and they were not expected to. They move the failure from "the episode ended without
an edit" to "the episode edited the wrong place and said why", which is the state a training run
can learn from. The task's real fix is the missing `/moto-api/config` route: the graded failure
is a `404` whose body is `Not yet implemented`, produced by
`moto/core/botocore_stubber.py:116`, while `v15`-`v20` patched `moto/core/config.py` - the module
the failing test imports.

`v21` answers that with the one search the policy could not get out of the harness: the query
`moto-api/config` exists verbatim only in the test, so `search_text` now re-queries its longest
path segment and prints the implementation matches for it. On the real container image that is
`moto/moto_api/_internal/urls.py:10`, the route table that has no `config` entry. Both trials that
searched the path read the route table and then the handler, and one patched the handler, so the
policy now reaches the file the fix belongs in. What is left is the change itself - register the
route and implement the two handlers - which is policy work rather than navigation.

A 10-task sweep of `v20` over the development set (`pass_at_1` `0.1`) shows the same split: one
task solved end to end, three patches in the right implementation, and four episodes that never
edited anything because they burned the step budget on refusals. That last group is why `v22`
exists.

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
9. `search_text` re-queries the longest path segment of a query that no implementation file
   spells out (`related_query_lines`, mirrored inside `_SEARCH_TEXT_SCRIPT` in
   `docker_environment.py`), and prints
   those matches ahead of the import-derived candidates. Files whose path carries the segment come
   first and ties break towards the file with the most matches, because path order alone puts
   `moto/core/...` ahead of `moto/moto_api/...`. The segment has to match at most
   `RELATED_QUERY_FILE_LIMIT` files, so a generic word such as `config` cannot displace the file
   the policy needs.
10. The navigation hints are budgeted inside the same character limit as the matches
    (`_remaining_search_chars`). They are appended after the trim, and a caller that truncates an
    observation from the front would otherwise keep the hints and drop the highest-ranked matches.
11. The loop-recovery directive no longer says the edit belongs in the module the test imports,
    and `IMPLEMENTATION_CANDIDATE` lines are described in the prompt as the matched test's
    dependencies rather than as proof that the failing behavior lives there. The directive now
    names the value the failure quotes, which is what `v21` made discoverable.
12. `failure_summary` no longer reads pytest's captured-log lines as failing nodes
    (`failing_nodes`): a logged `ERROR <logger>:<file>:<line> <message>` line carries a logger
    name, so only tokens that name a path or a node id count as failures. `[last error]` takes the
    exception rather than the assertion diff under it (`raised_line`), which had reported
    `+ States.Runtime` as the error. A new `[logged errors]` line names the module and line of the
    exception the code under test logged, because a run can end on an assertion about a wrong
    value while the cause only ever appears in the log.

13. `PROMPT_VERSION` moves to `coding-tools-json-v22`: the prompt documents the related-query
    lines, describes `IMPLEMENTATION_CANDIDATE` as the matched test's dependencies rather than as
    the home of the failing behavior, and names `[logged errors]` as the module and line an
    exception was logged from.

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
- `v21`: `patch_failed_verifier` 1, `edit_action_failed` 2, changed files 1/0/0, steps 10/24/24,
  loop refusals 2/9/11. Trial `140001` patched `moto/moto_api/_internal/responses.py`; trial
  `150001` read `urls.py`, tried to rewrite the `url_paths` dict and had the edit refused as
  unparseable, then repeated it until the guard stopped it.

Development set, 10 tasks, 1 repetition, seed `142001`, same model and budget
(`remote-artifacts/swe-gym-v20-all-14b-32k-t08-trajectories.jsonl`):

| task | category | steps | changed files |
| --- | --- | --- | --- |
| `getmoto__moto-7365` | `success` | 21 | `moto/dynamodb/models/dynamo_type.py` |
| `getmoto__moto-7514` | `policy_protocol_error` | 12 | `moto/s3/models.py` |
| `getmoto__moto-7646` | `loop` | 24 | - |
| `getmoto__moto-7446` | `loop` | 24 | - |
| `getmoto__moto-7607` | `patch_failed_verifier` | 14 | `.../state_task/service/resource.py` |
| `getmoto__moto-7509` | `edit_action_failed` | 24 | - |
| `getmoto__moto-7385` | `patch_failed_verifier` | 23 | `moto/ce/models.py` |
| `getmoto__moto-7608` | `edit_action_failed` | 24 | - |
| `getmoto__moto-7393` | `protected_test_edit` | 6 | - |
| `getmoto__moto-7537` | `patch_failed_verifier` | 11 | `moto/s3/models.py` |

`getmoto__moto-7608` is the clearest case for `v22`. Its initial observation reported four
`[failing tests]`, three of which were pytest's captured-log `ERROR` lines rather than failing
nodes, and `[last error] + States.Runtime`, which is a line of the assertion diff. The policy then
searched the whole logged string twice, had both repeats refused, and spent all 24 steps without
an edit. Its real cause - `AttributeError: 'NoneType' object has no attribute 'startswith'` in
`moto/stepfunctions/parser/asl/component/eval_component.py:60` - was in the observation, but under
the misleading head. The container check for `v21` ran the shipped search script against
`xingyaoww/sweb.eval.x86_64.getmoto_s_moto-7393:latest` with the verifier's test patch applied,
because that test is what puts the path in the repository at all:

```
docker run --rm -v /tmp/swe7393:/in -w /testbed \
  xingyaoww/sweb.eval.x86_64.getmoto_s_moto-7393:latest \
  bash -c "cp /in/test_config.py tests/test_core/test_config.py && python /in/search.py /moto-api/config"
```

The result leads with the route table rather than with the module named after the failing test:

```
tests/test_core/test_config.py:19:    resp = requests.get(f"http://{base_url}/moto-api/config")
No implementation file contains "/moto-api/config". Shorter query "moto-api" matches implementation files:
moto/moto_api/_internal/urls.py:10:    "{0}/moto-api/$": response_instance.dashboard,
moto/moto_api/_internal/urls.py:11:    "{0}/moto-api/data.json": response_instance.model_data,
...
```

## Validation

- Windows: `python -m pytest tests -q` -> 248 passed, 1 skipped.
- Ubuntu VM (authoritative): `scripts/sync_and_validate_vm.ps1 -DockerIntegration` -> 248 tests
  OK, 1 skipped, and the real-Docker integration test passes, including the finish refusal, the
  exact-text repair hint and the route re-query inside the container.
- The route re-query was also run against the cached task image directly (the `docker run` above),
  because the unit tests only prove the ranking against a fixture.
- Source snapshots: `python scripts/source_snapshot.py create` writes a 146-file archive plus a
  `sha256sum` manifest for this revision; `sha256sum -c` and
  `python3 scripts/source_snapshot.py verify --tree` both pass on the unpacked copy on the VM.

## Not in this change

No scoring rule was relaxed, no learned scorer was added, and the verifier-owned test boundary is
still a terminating violation. `RewardVector`, `schema_version` 3 and the trajectory/report
schemas are unchanged, so older artifacts still load.

## Next

- The `v22` held-out arm and its 10-task sweep are running on the GPU host
  (`work/swe-gym-v22-*`, seeds `140001` and `142001`) to test whether the summary fix converts
  the four `no edit` development tasks into edits.
- The `v21` evidence names the next harness defect: trial `150001` rewrote the middle of
  `url_paths` and had the edit refused as unparseable, with the syntax error on a line *outside*
  the range it replaced. A refusal that reports the enclosing block's line range would turn that
  refusal into a repair.
- Re-run the 7B arm, whose `v15` failure was a degenerate repeated search, against the current
  prompt.
- Rebuild the SFT dataset on the GPU host so its `prompt_version` matches the current prompt.
- Train on these trajectories: every arm now reaches a source edit or a diagnosable refusal, so
  the stored rollouts carry the target-selection and repair signal that the prompt alone did not
  supply.
