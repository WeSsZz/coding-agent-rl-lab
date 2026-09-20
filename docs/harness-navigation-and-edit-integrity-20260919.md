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

### The paired refusal arm

Item 17 wanted the two effects separated, because the `v24` refusal changes the prompt by one
sentence and at temperature `0.8` that is enough to move the first sampled action. This arm
changes nothing but `premature_finish_refusal`: `armA` is a copy of the current tree with the
function reverted to the `v23` form (a patched failure may finish immediately) and `armB` is the
current tree. Both run the held-out task `getmoto__moto-7393` with `--repetitions 8 --seed 140001
--temperature 0.8 --max-steps 24 --context-window-tokens 32768` on the same live engine instance.
Raw rows: `remote-artifacts/swe-gym-ab-finish-v23-held-out-14b-32k-t08-trajectories.jsonl`
(`armA`) and `remote-artifacts/swe-gym-ab-finish-v24-held-out-14b-32k-t08-trajectories.jsonl`
(`armB`), written on the VM as `work/swe-gym-ab-finish-v23-trajectories.jsonl` and
`work/swe-gym-ab-finish-v24-trajectories.jsonl`; refusals are `finish` actions whose observation
is a `Tool error`.

| seed | v23 steps | v23 edited | v23 refusals | v23 loops | v24 steps | v24 edited | v24 refusals | v24 loops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `140001` | 10 | `responses.py` | 0 | 1 | 24 | `config.py`, `responses.py` | 4 | 5 |
| `150001` | 8 | `urls.py` | 0 | 0 | 24 | `urls.py` | 5 | 6 |
| `160001` | 12 | `config.py` | 1 | 3 | 24 | `models.py` | 6 | 2 |
| `170001` | 24 | - | 5 | 7 | 24 | - | 5 | 7 |
| `180001` | 19 | `config.py` | 2 | 4 | 6 | `config.py` | 0 | 0 |
| `190001` | 13 | `responses.py` | 0 | 0 | 23 | `responses.py` | 1 | 1 |
| `200001` | 9 | `config.py` | 0 | 1 | 24 | `config.py`, `urls.py` | 1 | 3 |
| `210001` | 24 | - | 6 | 13 | 24 | - | 6 | 12 |
| **total** | **119** | **6 trials** | **14** | **29** | **173** | **6 trials** | **28** | **36** |

Six of eight `v23` trials stop with the error unrepaired at step 10, 8, 12, 19, 13 and 9 of 24;
two `v24` trials do, at 6 and 23. Six trials in each arm still reach a source edit, and the
refusal is answered every time with another action rather than with a repeat, so per step the
loop rate is flat (`0.24` against `0.21`). No trial in either arm passes the verifier, so the
refusal buys attempts and not success - and it does buy attempts: the two `v24` trials that edit
a second file (seed `140001` and `200001`) have no counterpart in `v23`, and the `v24` arm runs
14 pytest verifications against 7.

The reports, part of the same two files, put that in numbers: `pass_at_1` and `mean_scalar_reward`
are `0.0` in both arms because a strict success never happened, while the shaped
`mean_training_reward` that training would actually see improves from `-0.235` to `-0.1062` and
the violation count falls from 2 to 1.

Where the attempts go is the same in both arms, and the gold patch of this task shows why none
of them can pass. The accepted fix is three edits - a `"{0}/moto-api/config"` entry in
`moto/moto_api/_internal/urls.py`, a `config` handler on `MotoAPIResponse`, and
`get_config`/`set_config` on `MotoAPIBackend` reading and writing
`moto.core.config.default_user_config` - and every arm reads at least one of those three files.
What it applies instead:

- `moto/core/config.py`, the default value, in three trials of each arm (`160001`, `180001`,
  `200001` under `v23`, `140001`, `180001`, `200001` under `v24`). The assertion the test opens
  with is made true by shipping the value the test expects, not by serving the endpoint, and the
  routes that would make the rest of the test pass are never added. This is the shortcut a
  reward-shaped policy finds first.
- The route table in three trials (`150001` both arms, `200001` under `v24`), which under `v23`
  reaches a handler that does not exist: `AttributeError: 'MotoAPIResponse' object has no
  attribute 'config'` is what ends that trial.
- `MotoAPIResponse` itself in four trials, at line 120 (`v24` `190001`) or 199-207 (`190001` in
  both arms), which is far from the route table that would have to name it.
- `moto/moto_api/_internal/models.py` in one trial, six times (`v24` `160001`) - the third file of
  the gold patch - without the route that would call it.

Two trials per arm (`170001`, `210001`) spend the budget on edits that are refused and end with
no changed file at all. Three of the sixteen trials end on the same shortcut taken one step
further: `v23` `160001` and `200001` and `v24` `180001` replace the verifier-owned
`tests/test_core/test_config.py`, which is the one edit the harness terminates on. `v24` does not
remove that temptation, it only puts the default-value edit in front of it - `160001` and
`200001` reach the test file at steps 12 and 9 with nothing else changed, `180001` at step 6
after editing the default value first.

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

- Keep the `v24` refusal. The paired arm above shows it buying the work it was meant to buy - 2 of
  8 trials stop before the budget against 6 of 8, with the same 6 of 8 reaching a source edit, 28
  refusals answered by another action instead of a repeat, and a flat loop rate per step - and
  shows that it cannot buy `pass_at_1` on this task, because the remaining gap is *where* the
  policy edits. The two shapes that gap takes are the default value in `moto/core/config.py` (6 of
  the 16 trials) and the verifier-owned test file (3 of 16, and the only terminating violation).
- `getmoto__moto-7646` and `getmoto__moto-7608` have now failed to edit under `v22`, `v23` and
  `v24` - 14 and 11 loop refusals in the last one. Read their refusal directives next: the
  directive already names the files the results pointed at and the policy never opened, so if
  these two still do not edit, the missing evidence is something neither the summary nor the
  directive carries.
- Re-run the 7B arm, whose `v15` failure was a degenerate repeated search, against the current
  prompt.
- Train on the `v24` warm start and then on these trajectories. The dataset is rebuilt at
  `prompt_version=coding-tools-json-v24` (204 examples, 51 per stage, one oversized hunk skipped)
  and the published-shard fallback means it rebuilds on a machine whose network cannot reach the
  rows API, which is both machines here. Every arm now reaches a source edit or a diagnosable
  refusal, so the stored rollouts carry the target-selection and repair signal that the prompt
  alone did not supply; a warm start that has seen the shape of a route-and-handler fix is the
  cheapest test of the bullet above.

## Why the `sftv24` arm lost 4 of 8 trials to `policy_protocol_error` (2026-09-20)

The warm start removed the premature `finish` and cost 4 of 8 trials to protocol errors. Dumping
the failing trials names the cause, and it is not the token budget, the checkpoint, or the data
mix that the pause note proposed to choose between.

**What the four trials actually emitted.** Every one of them ended on a `replace_text` whose `old`
was the *verbatim numbered `read_file` output*, including the observation's own trailing
`[read_file lines A-B: file has N lines; continue with read_file start_line=...]` marker:

```
{"kind":"replace_text","arguments":{"path":"moto/autoscaling/models.py",
 "old":"et_tracking_config\n107:         self.step_adjustments = step_adjustments\n108: ...
   [read_file lines 104-281: character budget reached; file has 1701 lines; continue with ...]\n",
 "new":"..."}}
```

That action is unusable twice over. `replace_text` matches `old` byte for byte against the file, so
a numbered copy can never match - `model_policy.py` rule 493 already tells the policy to copy
`old` from the numbered output, and `docker_environment.py` already refuses a numbered `old` with a
directive naming the `replace_lines` range. And the block is roughly the whole file: seed `140001`
aimed one `replace_text` at `moto/autoscaling/models.py`, 1701 lines, slice 104-281 alone.

**It is truncation, but the budget is not the lever.** The four raws end mid-string with two
unclosed braces, so they were cut off at `max_tokens`. Raising the budget only moves the cut:

| arm (same task, same 8 seeds) | `max_tokens` | pass | tests | trials with a source edit | protocol errors | longest raw | `mean_training_reward` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base model, `v24` prompt | 1024 | 0/8 | 0/8 | 6/8 | 0 | 792 chars | -0.1062 |
| `sftv24` adapter | 1024 | 0/8 | 0/8 | 1/8 | 4/8 | 3639 chars | -0.4963 |
| `sftv24` adapter | 4096 | 0/8 | 0/8 | 2/8 | 2/8 | 14070 chars | -0.2425 |

At 4096 the two surviving failures emit 13976 and 14070 characters - still cut off, and still
aimed at whole files. The base model never crosses 792 characters on the same task, so this is the
adapter's learned output shape, not a ceiling the task forces. 4096 is also the practical maximum
for this server, and the limit is the context, not the GPU: the prompt grows to ~17464 and then
~22313 tokens, so `max_tokens=16384` and `max_tokens=12000` were refused by the context preflight
after one trial, and `max_tokens=8192` - which does fit - died on a request timeout because the
completion took longer to stream than the transport allows. Raising the budget is therefore not an
available lever at all.

**The training data taught the right shape, against the wrong observation format.** All 51
`replace_text` targets in `swe-gym-train-gold-sft-v24.jsonl` are clean unnumbered fragments -
`old` median 509 chars, max 993, and none of them carry observation line numbers. The observations
are present too, as `history` entries inside the single user payload, and the `search_text`
observation is already in the live format (`moto/acm/models.py:409:self._certificates: ...`, which
is exactly what `_collect_search_matches` renders). The defect is the `read_file` observation: the
builder passed the bare hunk text (`swe_gym_sft.py`, `TrajectoryStep(2, read_action,
hunk.old_text, ...)`), so the adapter's context showed unnumbered source - while the live tool
returns numbered lines plus a `[read_file lines A-B: ...]` footer, and `render_numbered_window`
documents that numbering as the thing that "lets a policy derive `replace_text` and
`replace_lines` arguments from an observation". Across the whole dataset that is 510 observations,
**0** of them numbered, **0** carrying the footer.

An `edit` example whose history shows bare source next to a target `old` of that same bare source
teaches the copy, not the derivation. Fine-tuning then amplifies it: the adapter copies whatever
the observation contains, and at inference the observation contains line numbers. The base model
never learned the copy, which is why it stays at 792 characters and still reaches 6 of 8 edits.

### The fix works: rebuild the dataset and the adapter recovers

`swe_gym_sft.py` now renders the `edit` stage's `read_file` history entry with
`render_numbered_window`, the same function the live tool uses. The rebuilt dataset
(`work/private/swe-gym-train-gold-sft-v25.jsonl`, `sha256 40e6a0d5...`, 204 examples) differs from
`v24` in exactly one respect: **102 of 102** `read_file` history observations carry line numbers,
against **0 of 102** before. The target actions, stage counts, example ids and prompt sizes are
byte-identical, so nothing about what the model must predict changed - only what it is shown.

A 60-step warm start on the same base, same seed `62001`, same hyper-parameters, then the same
8-seed held-out arm at `--max_tokens 4096`:

| arm | pass | tests | seeds with a changed file | applied edits | refused edits | protocol errors | longest raw | `mean_training_reward` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base model, `v24` prompt | 0/8 | 0/8 | 6/8 | - | - | 0 | 792 ch | -0.1062 |
| `sftv24` adapter (bare read observation) | 0/8 | 0/8 | 2/8 | 7 | 27 | 2 | 14070 ch | -0.2425 |
| `sftv25` adapter (numbered read observation) | 0/8 | 0/8 | **7/8** | **26** | **15** | **1** | 14425 ch | **-0.1062** |

The copy defect was the whole regression. Applied edits go from 7 to 26, seeds that reach a source
edit from 2 of 8 to 7 of 8, and `mean_training_reward` recovers from `-0.2425` to `-0.1062` - exactly
the base model's number, so the warm start no longer costs anything on this task. The one
surviving protocol error (`170001`) still hits the truncation path.

`pass_at_1` stays 0. The remaining gap is no longer "can it produce a valid edit" but "does it edit
the right thing": the trials now land in the wrong file, and `170001` is the only seed still
losing an episode to the protocol rather than to its choice of target.

**Harness fixes landed with this analysis** (`model_policy.py`, `swe_gym_sft.py`,
`tests/test_model_policy.py`, `tests/test_swe_gym_sft.py`):

- The dataset builder renders the `edit` stage's `read_file` history entry with
  `render_numbered_window`, the same function the live tool uses, so the numbering conversion is
  in the training distribution instead of being a format the adapter meets for the first time at
  inference. Two tests pin it: the observation must be numbered, and the target `old` must not
  appear in the history verbatim.
- A protocol retry now advances the parent seed by `RETRY_SEED_STRIDE`. Under a fixed seed a
  completion cut off at `max_tokens` is deterministic, so both attempts replayed the identical
  truncation - which is why every failing trial records the same error string twice. `attempt_seed`
  is recorded, so a replayed attempt is visible in the trajectory.
- `finish_reason="length"` is now reported as truncation at `max_tokens` rather than collapsed into
  "model content is not valid JSON", and the retry message asks for a smaller edit instead of a
  generic re-ask.
- A refused episode keeps the failed response's `finish_reason` and `usage`. They were dropped on
  the failure path, which is why the arm above had to be diagnosed from raw text length.

**Next.** The numbering fix is verified, so stop looking at the warm start and look at target
selection: 7 of 8 trials now apply a real edit and 0 of 8 pass the verifier, which is a
"wrong file / wrong hunk" gap rather than a protocol one. Read the verifier output of the `sftv25`
trials in `remote-artifacts/swe-gym-sftv25-maxtok4096-held-out-14b-32k-t08-trajectories.jsonl`
next, and decide whether the missing evidence is in the failure summary or in the prompt's
locate-to-edit transition. Do not spend GPU on checkpoint-20 or a protocol-heavy data mix.

### What the `sftv25` arm actually gets wrong (2026-09-20, second pass)

Running the repo's own taxonomy over the arm
(`python -m coding_agent_rl_lab.failure_analysis`) gives 1 `policy_protocol_error`,
1 `edit_action_failed` and **6 `patch_failed_verifier`**, with `fail_to_pass_resolved = 0` on every
trial and the same verifier node failing in all of them
(`tests/test_core/test_config.py::test_change_configuration_using_api`).

**First finding: the environment accepted edits that changed nothing.** The model emits
`replace_text` with `new` byte-identical to `old`, and `LocalFixtureEnvironment` only checked that
`old` occurred exactly once - so `content.replace(old, new, 1)` wrote the file back unchanged,
answered `Updated <path>.`, and set `_edited_since_verification = True`. A no-op was
indistinguishable from a repair, both to the policy and to `changed_files()`:

| seed | edits reported applied | of which changed nothing | actually changed the file |
| --- | --- | --- | --- |
| 140001 | 3 | 3 | 0 |
| 150001 | 6 | 4 | 2 |
| 160001 | 4 | 4 | 0 |
| 170001 | 2 | 1 | 1 |
| 190001 | 3 | 0 | 3 |
| 200001 | 3 | 2 | 1 |
| 210001 | 5 | 5 | 0 |

26 "applied" edits were really 19 no-ops and **7 real changes**. Both environments now refuse a
no-op `replace_text` (`old == new`) and a `replace_lines` whose range is rewritten with its own
text, naming the refusal instead of claiming the file was updated.

**Second finding: none of those 7 real edits lands in a file the gold patch touches.** The fix for
`getmoto__moto-7393` is `moto/moto_api/_internal/{models,responses,urls}.py` - a new
`GET/POST /moto-api/config` handler on the dashboard API that reads and writes
`moto.core.config.default_user_config`. What the policy edited instead:

| real edits | path | gold? |
| --- | --- | --- |
| 3 | `moto/core/models.py` | no |
| 2 | `moto/config/urls.py` | no |
| 1 | `moto/server.py` | no |
| 1 | `moto/core/decorator.py` | no |

`moto/moto_api` does not appear in a single observation across all eight trials. The failure text
leads with a test file called `test_config.py` and a route `/moto-api/config`, and the policy reads
"config" as the service directory `moto/config/` - a real, differently-shaped service that also has
`urls.py`, `models.py` and `responses.py`. Its first query was `moto/config/server.py`, which
misses and returns eight `SUGGESTED_PATH` lines that are all under `moto/config/` or `moto/`, so the
very first observation confirms the wrong hypothesis. The search tool can reach the right file, but
only for the right query:

```
query 'moto/config/server.py'  -> SUGGESTED_PATH:moto/config/urls.py, ... (no moto/moto_api)
query 'moto-api/config'        -> Shorter query "moto-api" matches implementation files:
                                  moto/moto_api/_internal/urls.py:2: "{0}/moto-api/": ...
query 'Not yet implemented'    -> moto/moto_api/_internal/urls.py:11: return "Not yet implemented"
```

**Both of those working queries were already in the prompt.** The `swe_gym_rollout` initial
observation is 6794 characters and carries `[failing statement]`, `[last error]` and
`[string values in the failing frame] s = 'Not yet implemented'` - the literal that appears only in
the gold file - and the `v24` system prompt says to use those literals. The policy searched a
guessed filename instead.

**Why it does not use them: the training observations are a different format.** The builder's
`_training_initial_observation` is 172 characters and contains only
`Baseline verifier result:\nTests failed (exit=1).\nFailing tests:\n<node>` - it omits
`[failing statement]`, `[last error]` and `[string values in the failing frame]` entirely, and no
`search_text` training observation carries `SUGGESTED_PATH`, `PATH_MATCH` or the
`Shorter query "<segment>"` fallback. So the adapter was trained on 204 locate examples that say
*which test failed* and never on the evidence that says *where the value comes from* - the same
class of defect as the unnumbered `read_file` observation, one layer up. `search_text` history
entries match the live `path:line:text` shape and the `read_file` entries are now faithful; the
initial observation is the remaining synthetic placeholder.

**Next.** Extend the fidelity fix to the initial observation, and make the search hint reachable
for a query that misses: when a query has no exact match, its longest path-like segment is a
candidate (`moto-api` above), and today that fallback only runs for a query that matched tests or
documentation. Then re-run: the arm has 7 real edits in 8 trials against 24 steps of budget, so the
cost of a wrong first query is most of the episode. Re-running the same arm with the no-op refusal
in place is the cheapest control, because it turns 19 wasted steps into refusals that the loop
guard can direct.

**Determinism is not what a fixed seed suggests.** Two arms were run over the same eight seeds with
byte-identical source, and only **2 of 8** trials reproduced: the other six differ in action
sequence, step count and changed-file count by one to two edits. Seeded vLLM sampling is therefore
not repeatable at this batch shape, so a single trial's `changed_files` or step count is not
evidence, and every claim above rests on arm-level totals (2 of 8 against 7 of 8 seeds reaching a
change, 7 against 26 applied edits) rather than on any one seed. Any A/B run on this stack needs
the whole arm, and a paired per-seed reading is only meaningful where the gap exceeds that noise.

### The no-op refusal is correct but does not move the task

The two arms above were followed by a third over the same eight seeds with the no-op refusal live.
19 no-ops that had been reported as `Updated ...` become 21 explicit refusals, and the count of
edits that genuinely change a file rises from 7 to 9:

| arm | tests passed | seeds reaching a change | real edits | no-ops accepted | no-ops refused | `mean_training_reward` |
| --- | --- | --- | --- | --- | --- | --- |
| `sftv25`, no refusal | 0/8 | 7 | 7 | 19 | 0 | -0.1062 |
| `sftv25`, refusal live | 0/8 | 6 | 9 | 0 | 21 | -0.1100 |
| `sftv24` (before either fix) | 0/8 | 2 | 2 | 5 | 0 | -0.2425 |

The reward is unchanged inside the noise above, and the guard does what it was written to do: the
policy can no longer spend a step on an edit that changes nothing and be told it made progress.
What it also shows is that the failure is no longer reachable from the harness side - 0 of the 9
real edits lands in `moto/moto_api/_internal/`, in this arm or the previous one. The two levers
that remain are the training-observation fidelity above and the initial search hint; neither is a
scoring or environment rule, and both change only what the policy sees.

### The search retry was retrying the query, and fixing that does not help this task

`search_repository` promised a missed query one retry and picked its longest whitespace token, so a
path-like query was retried with itself: `moto/config/server.py` is one token, it does not exist, and
the second attempt found the same nothing. The model was then left with eight `SUGGESTED_PATH` lines
from `difflib`, all under `moto/config/` or `moto/`, which is exactly the query it used on
`getmoto__moto-7393` and exactly the confirmation of the wrong hypothesis. Both searches now walk the
query's own segments with the basename first, and both share one matcher so a retry cannot re-enter
the fallback and recurse.

The retry is verified working - `search_repository(root, "moto/dynamodb/models.py")` now answers
`PATH_MATCH:moto/dynamodb/models/__init__.py` where it used to offer a fuzzy suggestion - and it
does not help this task, for a reason worth recording. In the real repository there is no
`moto/server.py` at that path, so the basename of the failing query is not a file either:

| arm (same task, same 8 seeds, `--max_tokens 4096`) | tests passed | seeds reaching a change | real edits | no-ops refused | steps naming `moto/moto_api` | `mean_training_reward` |
| --- | --- | --- | --- | --- | --- | --- |
| `sftv24` | 0/8 | 2 | 2 | 0 | **0** | -0.2425 |
| `sftv25` (dataset fix) | 0/8 | 7 | 7 | 0 | **0** | -0.1062 |
| `sftv25` + no-op refusal | 0/8 | 6 | 9 | 21 | **0** | -0.1100 |
| `sftv25` + refusal + search retry | 0/8 | 1 | 3 | 17 | **0** | -0.2500 |

**No arm ever sees the gold package, and the reason is the queries themselves.** Every `search_text`
the policy issues across all four arms is a guessed file path - `moto/config/server.py`,
`moto/api/server.py`, `moto/core/server.py`, `moto/batch/server.py`, `moto/s3/server.py`. It never
searches either literal that reaches the answer, and both were in its prompt:

```
query 'Not yet implemented'    -> moto/moto_api/_internal/urls.py:11   (the failure frame quotes it)
query 'moto-api/config'        -> moto/moto_api/_internal/urls.py:2    (the failing test requests it)
```

The retry fix cannot substitute for a query the policy never makes, and the no-op refusal converts
19 silent no-ops into 21 honest refusals without giving the policy anything to do instead. Patch
selection is the whole remaining gap on this task, and it is a training-data problem: the builder's
`_training_initial_observation` is 172 characters of test name while the live observation is 6794
characters carrying `[failing statement]`, `[last error]` and
`[string values in the failing frame] s = 'Not yet implemented'`. The adapter has never been shown
that a failure names a searchable literal, so it does what its 204 locate examples taught - name a
file. Extending the fidelity fix to the initial observation is the next change, and it must not
simply copy the live failure text in: the frame literals are free at train time and are the signal
being graded at eval time. Build them from the test patch and the gold patch the row already
carries, so the training prompt shows a failure of the same *shape* without shipping the answer.

### Teaching the failure literal moves the search and still misses the file

That change was made in two steps, and the second one is the informative one.

**Step one: give the observation the live shape.** `_training_initial_observation` was 172
characters of test name. It now also carries
`[failing statement] <path>:<line>: <assert ...>`, lifted from the test patch the verifier will run
(`_test_patch_assertion`), so 204 of 204 examples have the shape instead of 0. Every extractable
statement is correct for the pinned rows, and for `getmoto__moto-7393` the extractor independently
lands on `tests/test_core/test_config.py:20: assert resp.json()["batch"] == {"use_docker": True}` -
the exact line and text the live verifier reported.

**It changed nothing.** The arm trained on it (`sftv26`) searched `moto/config/server.py`,
`moto/api/server.py`, `moto/core/server.py` and `moto/batch/models.py`, exactly as before, and never
searched one failure literal:

| arm (`--max_tokens 4096`, same 8 seeds) | tests passed | seeds reaching a change | real edits | `search_text` steps using a failure literal | steps naming `moto/moto_api` | `mean_training_reward` |
| --- | --- | --- | --- | --- | --- | --- |
| `sftv25` | 0/8 | 7 | 26 | 0 | 0 | -0.1062 |
| `sftv26` (failing statement in the observation) | 0/8 | 4 | 8 | **0** | 0 | -0.2350 |
| `sftv27` (locate target is a failure literal) | 0/8 | 6 | 16 | **21** | 0 | -0.2350 |

**Step two: teach it with the action, not the observation.** The locate stage's target was
`{"kind":"search_text","arguments":{"query":"moto/acm/models.py"}}` - the gold file's path, a query
no policy can derive and one the search already answers. The observation above it described the
failure, and the target told the policy to ignore that and name a file, which is what a warm start
copies. `_search_literal` now derives the taught query from the failing assertion
(`use_docker` out of `assert resp.json()["batch"] == {"use_docker": True}`), and the arm's search
behaviour flips: **0 to 21** `search_text` calls whose query is a failure literal, with
`RequestsJSONDecodeError`, `logger.warning`, `Service` and `msg` replacing the guessed paths.

**And it still never reaches `moto/moto_api`.** The policy now searches the wrong *part* of the
failure. For `getmoto__moto-7393` its prompt offers four evidence lines and it picks from the third:

```
[failing tests]                      tests/test_core/test_config.py::test_change_configuration_using_api
[failing statement]                  tests/test_core/test_config.py:20: assert resp.json()["batch"] == {"use_docker": True}
[last error]                         requests.exceptions.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
[string values in the failing frame] s = 'Not yet implemented'
```

It searches `RequestsJSONDecodeError` - the exception class - while `moto-api` (from the route the
test requests) and `'Not yet implemented'` (the server's own response text, defined only in the gold
file) both sit in the same prompt unsearched. The training rows cannot demonstrate those two: the
builder never runs the verifier, so `[last error]` and `[string values in the failing frame]` have no
honest source, and the only failure evidence a row really has is the assertion. The taught literal is
therefore always an assertion literal, and the assertion's own values - `use_docker`, not `moto-api` -
are what the policy learned to prefer.

So the next lever is to give the taught literal a source that includes the failure's *runtime* values
without inventing them: run the baseline verifier for each pinned train row at dataset build time and
store its real `failure_summary`. That is one test run per row for six rows, it is the same evidence
the live prompt carries, and it removes the last synthetic stand-in in the `locate` stage.

**That lever is a data-coverage decision, and the split cannot supply it.** A route extractor was
written and tried: it pulls `moto-api/config` out of the held-out test patch, which is exactly the
literal the policy fails to search, and it returns `None` for **all six** pinned train rows because
none of their tests speaks HTTP - they are DynamoDB, S3, EMR and stepfunctions tests whose failures
name their own values rather than a runtime response. So a route-aware `locate` target is a no-op for
every training example, and it was reverted rather than shipped as a fix that changes nothing. The
same constraint applies to `[string values in the failing frame]`, whose values only a verifier run
produces.

The remaining gap is therefore not a prompt or an environment rule: the held-out task is reached by
searching a route or a runtime response value, and the six pinned train tasks contain no example of
either. Closing it means either admitting route-shaped or HTTP-shaped tasks to the pinned train split,
or running the baseline verifier per train row at build time to harvest real failure summaries - both
of which are dataset decisions rather than another harness tweak.

**The second of those turned out to be free.** The archived train-task rollouts already hold the
verifier's own output for all six pinned train tasks, truncated at ~4000 characters, which is enough
to keep `[last error]`, `[logged errors]` and the frame's string values even though the
`[failing statement]` marker is cut off. `work/harvest_train_failures.py` extracts them through
`failure_summary` - the environment's own function - into
`work/private/swe-gym-train-failure-lines.json`, and the builder now carries them into the
observation and prefers them when it picks the literal the `locate` stage teaches. Every pinned row
moves from an assertion's own operand to a value a real run produced:

| task | taught query | source line |
| --- | --- | --- |
| `getmoto__moto-7509` | `InvalidServiceName` | `[last error]` |
| `getmoto__moto-7365` | `t911877` | `[string values in the failing frame]` |
| `getmoto__moto-7514` | `select_query` | `[string values in the failing frame]` |
| `getmoto__moto-7646` | `UpdateItem` | `[last error]` |
| `getmoto__moto-7607` | `States.Runtime` | `[logged errors]` |
| `getmoto__moto-7446` | `cfnTask2` | assertion (no runtime line survives the truncation) |

`_search_literal` was rewritten for that text: a quoted value wins outright, because a value a
failure reports or compares against is a string the source has to contain, and a bare candidate must
carry an underscore or an inner capital to count as a name. `com.amazonaws.us-west-1.config`,
`Not yet implemented` and `InvalidServiceName` survive; `Expecting`, `IndexError`, `ClientError` and
`botocore.exceptions` do not. Dataset `v28` (204 examples, stages unchanged, only the `locate` query
differs) is built and inspected; **the arm that would measure it has not been run, because the GPU
instance is stopped** - that measurement is the next action.
