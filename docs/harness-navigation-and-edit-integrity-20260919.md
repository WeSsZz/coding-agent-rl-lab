# Harness navigation, finish refusal and edit integrity (2026-09-19, updated 2026-09-20)

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
| `v22` | `patch_failed_verifier` 1, `edit_action_failed` 2 | 13/24/24 | 1/0/0 | the summary names the logged exception, and both stalled trials now spend their budget rewriting the middle of the `url_paths` dict instead of editing the test |
| `v23` | `patch_failed_verifier` 3/3 | 10/8/7 | 1/1/1 | the span hint is used: trial `150001` handed `replace_lines` lines 9-30 on the very next step and the edit applied, so the retry loop is gone; all three trials then called `finish` at step 10/8/7 of 24 with the error unrepaired |
| `v24` | `patch_failed_verifier` 3/3 | 24/24/24 | 2/1/1 | the second give-up is gone: every refused `finish` is followed by more reading or another edit, and trial `140001` patches a second file at step 20, but the run still ends on a failing verifier |

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

14. An unparseable edit that replaced part of a wider statement now names the statement's lines
    (`enclosed_statement_span` and `replaced_line_span` in
    `src/coding_agent_rl_lab/environment.py`, mirrored inside `_REPLACE_TEXT_SCRIPT` and
    `_REPLACE_LINES_SCRIPT` in `docker_environment.py`), so the refusal reads
    `The statement you replaced lines 9-14 of spans lines 9-31: give replace_lines that whole
    range.` Trials `150001` and `160001` of `v22` both replaced the first five entries of the
    `url_paths` dict, which runs to line 31; the container then reported the syntax error on line
    26, a line the policy had not touched, and the policy retried the identical edit until the
    step cap. The hint is suppressed when the replaced range is exactly one statement, because
    then the range is not wrong and pointing at the enclosing block would send the policy off to
    rewrite code that was never the problem.
15. The loop-recovery directive names the implementation files the episode's own results pointed
    at and the policy never read (`unread_evidence_paths`): `IMPLEMENTATION_CANDIDATE`,
    `SUGGESTED_PATH` and `PATH_MATCH` lines, plus the `<path>.py:<line>:` lines its own searches
    printed, minus everything read-only or already read. A policy stalled on refusals is usually
    holding its next move in an observation it already has, and the refusal now names it.
16. `PROMPT_VERSION` moves to `coding-tools-json-v23`: the prompt documents both the
    `spans lines X-Y` repair and the unread-file list.
17. `finish` is refused while a patch is applied and the verifier still fails
    (`premature_finish_refusal` takes the remaining budget, both environments pass
    `task.max_steps - self.steps`). `v23` moved the give-up rather than removing it: the guard
    that refuses a `finish` with no edit behind it made every trial edit, and the policy then
    read the failure the refusal had handed it, wrote nothing, and called `finish` again at step
    10, 8 and 7 of 24. A patched failure is now refused too, until one step remains, because a
    repair costs an edit plus the test run that confirms it. The message points at the error the
    observation already quotes and reports the remaining budget. This changes how long an
    episode runs, not how it is scored: `finish` still ends the episode as soon as the verifier
    passes, `RewardVector` and `schema_version` are untouched, and the final result is still the
    verifier's.
18. `PROMPT_VERSION` moves to `coding-tools-json-v24`: the prompt says a refused `finish`
    returns a failure to repair rather than an episode to end, in both the unpatched and the
    patched case.
19. An episode whose policy cannot reach the model now stops instead of spending its budget on
    the policy's own fallback (`POLICY_FALLBACK_VIOLATIONS` in
    `src/coding_agent_rl_lab/contracts.py`, checked in `RolloutCollector.collect`).
    `next_action` answers an unreachable endpoint or an unparseable response with a `finish`
    fallback plus a violation, and stepping that fallback executes a decision the policy never
    made. The first `v24` arm lost its SSH tunnel mid-run, and every trial after that sat at 24
    steps with 23 loop refusals, one per fabricated `finish`, while the endpoint refused every
    request: a two-hour arm that only looked invalid afterwards. The episode now ends on the
    fallback, keeps the violation, and still lets the verifier grade the container. The run ends
    with it - `collect_trajectories_incrementally` raises once a trial records
    `policy_transport_error`, with the checkpoint already written so `--resume` can finish the
    arm - and `main` probes `GET {api_base}/models` before the first trial
    (`require_served_model` in `swe_gym_rollout.py`), so a dead endpoint or a model id the
    server does not serve costs one request instead of an arm. This changes no prompt text and
    no scoring: `PROMPT_VERSION` stays at `v24` because the policy sees none of it.
20. `finish` re-runs the verifier only when a file has been edited since the last run
    (`_edited_since_verification` in both environments, set by a successful edit and cleared by
    every verifier run). The `v24` held-out trials call `finish` three to five times each as a
    "is it done yet" probe, and every probe paid for a pytest run inside the container because
    the guard reran whenever any edit existed. Nothing but an edit can change a failure, so the
    probe is now answered from the failure the last run produced. The refusal text and the step
    count are unchanged, which is why this does not move `PROMPT_VERSION`: the policy sees the
    same observation one container run sooner.
21. The budget cutoff in item 17 now applies to the unpatched refusal as well. A `finish` on
    the last step used to be refused in a state where the episode ends either way, so the
    terminal action of a give-up episode was a refusal instead of the verifier result it had
    already asked for.

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
- `v22`: `patch_failed_verifier` 1, `edit_action_failed` 2, changed files 1/0/0, steps 13/24/24,
  loop refusals 3/5/7. Trial `140001` reached the handler again; trials `150001` and `160001`
  both read the route table after the related-query hint, both replaced lines 9-14 of the
  `url_paths` dict, and neither could see that error line 26 was outside the range it had
  replaced. That is what `v23` answers.
- `v23`: `patch_failed_verifier` 3/3, changed files 1/1/1, steps 10/8/7, loop refusals 0/0/1.
  The retry loop is gone. Trial `150001` read the route table, had lines 9-14 refused as
  unparseable, and the *next* step replaced lines 9-30 - the range the refusal named - which
  applied. All three trials then stopped with the step budget more than half unspent: `140001`
  finished after `moto/core/config.py`, `150001` finished with `urls.py` routing to a
  `MotoAPIResponse.config` handler that does not exist (`AttributeError: 'MotoAPIResponse'
  object has no attribute 'config'`), and `160001` finished after editing
  `moto/moto_api/_internal/models.py`. That second give-up is what `v24` answers.

The first `v24` arm is not evidence: its SSH tunnel to the model host died about two minutes
into the held-out run, so trial `140001` finished normally, trials `150001` and `160001` spent
every remaining step on refusals against the policy's `finish` fallback, and all ten development
trials ended at 24 steps with no file changed and one `policy_transport_error` each
(`work/swe-gym-v24-invalid-transport-*`). That arm is what produced item 19: the endpoint
failure is now caught before the first trial and stops the run instead of being recorded as
twenty-three refusals per episode.

The `v24` rerun on a live endpoint (`work/swe-gym-v24-held-out-14b-32k-t08-trajectories.jsonl`)
answers item 17 directly: `patch_failed_verifier` 3/3, changed files 2/1/1, steps 24/24/24, loop
refusals 5/4/5. No trial gives up early any more, and every refused `finish` is answered with
more reading or another edit rather than a repeat - trial `140001` reads, is refused at step 10
and again at 13 and 16, and patches `moto/core/config.py` at step 20. The task is still unsolved,
so the remaining gap is the fix itself, not the harness's willingness to accept a stop.

The development sweep is where `v24` looks worse, and the honest reading is that it is not
evidence either way: 3 of 10 trials reach a source edit against 7 of 10 under `v23`, but the two
sweeps are not paired even though they share seeds. One prompt sentence differs, that changes the
first sampled action, and at temperature 0.8 every later step diverges from there - `getmoto__moto-7365`
alone has been `success` (`v20`), `patch_failed_verifier` (`v22`) and `protected_test_edit`
(`v23`) at seed `142001`. The `v24` sweep also spends its steps differently rather than better:
`getmoto__moto-7446` probes with `finish` four times and never edits, and `getmoto__moto-7537`
ends on a verifier timeout after 25 minutes. Separating a prompt change from sampling noise needs
several repetitions per task, which is why the arms that follow a change now run three or more.

Two engine facts from this session are worth recording. vLLM is deterministic for a repeated
request within one engine instance (same prompt and seed, identical output), which is what makes
seed-paired arms meaningful; and the first `v24` arm ran against the engine that had just been
restarted with the same model and `--max-model-len 32768`, so its invalidity comes from the dead
tunnel and not from the restart.

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
  the misleading head.

  The same 10 tasks under `v22` (`work/v22-all.jsonl`) hold `pass_at_1` at `0.0` - the one task
  `v20` solved, `getmoto__moto-7365`, now patches and fails the verifier - but the group that
  never edits anything shrinks from four tasks to two. `getmoto__moto-7446` (5 steps,
  `moto/emr/models.py`) and `getmoto__moto-7509` (24 steps, `moto/ec2/models/vpcs.py`) now make a
  source edit, which leaves `getmoto__moto-7646` and `getmoto__moto-7608` as the remaining
  `edit_action_failed` pair, the two the `v22` summary targets. Eight of ten development tasks
  now reach a source edit.

  Under `v23` (`work/swe-gym-v23-all-14b-32k-t08-trajectories.jsonl`) `pass_at_1` stays at
  `0.0`, and the same two tasks (`getmoto__moto-7646`, `getmoto__moto-7608`) are still
  `edit_action_failed` on 16 and 15 refusals. `getmoto__moto-7365`, the one task `v20` solved,
  now edits the verifier-owned test at step 4 and terminates as `protected_test_edit`. Every
  other task reaches a source edit - and four of them end on an accepted `finish` while the
  verifier still fails: `getmoto__moto-7514` at step 9 of 24, `getmoto__moto-7537` at 11,
  `getmoto__moto-7607` at 14 and `getmoto__moto-7393` at 20. Those four are the second give-up
  that `v24` refuses.

  The container check for `v21` ran the shipped search script against
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

- Windows: `python -m pytest tests -q` -> 257 passed, 1 skipped.
- Ubuntu VM (authoritative): `scripts/sync_and_validate_vm.ps1 -DockerIntegration` -> 258 tests
  OK, 1 skipped, and the real-Docker integration test passes, including the finish refusal, the
  exact-text repair hint, the route re-query inside the container, and the patched `finish`
  refusal that reuses the recorded failure instead of re-running pytest.
- The route re-query was also run against the cached task image directly (the `docker run` above),
  because the unit tests only prove the ranking against a fixture.
- Source snapshots: `python scripts/source_snapshot.py create` writes a 146-file archive plus a
  `sha256sum` manifest for this revision; `sha256sum -c` and
  `python3 scripts/source_snapshot.py verify --tree` both pass on the unpacked copy on the VM.

### Model host operations

The rollout host is reached from the VM only, and the two processes that make it reachable die
independently of the instance, which stays up and keeps billing:

```
# on the model host (RTX 4090, vLLM 0.11.0), after ssh -p <port> root@connect.bjb1.seetacloud.com
setsid nohup /root/miniconda3/bin/vllm serve /root/autodl-tmp/qwen2.5-coder-14b-modelscope \
  --served-model-name Qwen2.5-Coder-14B-Instruct --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.85 \
  > /root/autodl-tmp/vllm-14b-32k-20260920.log 2>&1 &

# on the VM: a tunnel that reconnects, because a one-shot `ssh -L` does not
setsid nohup sh scripts/autodl_tunnel.sh 10283 > /dev/null 2>&1 < /dev/null &
curl -s http://127.0.0.1:8000/v1/models
```

## Not in this change

No scoring rule was relaxed, no learned scorer was added, and the verifier-owned test boundary is
still a terminating violation. `RewardVector`, `schema_version` 3 and the trajectory/report
schemas are unchanged, so older artifacts still load.

## Next

- Settle the `v24` finish refusal against the `v23` one on a single variable: the held-out task,
  eight repetitions, same prompt, with `premature_finish_refusal` reverted in one of the two
  trees. The held-out arm showed the refusal producing a second edit; the development sweep
  showed it producing probes instead. Both are single-repetition samples of a temperature-0.8
  policy, and the earlier sweep tables cannot separate the two effects because a one-sentence
  prompt change moves the first sampled action.
- `getmoto__moto-7646` and `getmoto__moto-7608` have now failed to edit under `v22`, `v23` and
  `v24` - 14 and 11 loop refusals in the last one. Read their refusal directives next: the
  directive already names the files the results pointed at and the policy never opened, so if
  these two still do not edit, the missing evidence is something neither the summary nor the
  directive carries.
- Re-run the 7B arm, whose `v15` failure was a degenerate repeated search, against the current
  prompt.
- Rebuild the SFT dataset on the GPU host so its `prompt_version` matches the current prompt.
- Train on these trajectories: every arm now reaches a source edit or a diagnosable refusal, so
  the stored rollouts carry the target-selection and repair signal that the prompt alone did not
  supply.
