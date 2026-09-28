# Qwen3-8B + SWE-Gym OpenHands GPU pilot (2026-09-28)

## Environment and timing

- AutoDL reports one RTX 4090 with 49,140 MiB total and 48,509 MiB free at start. Torch 2.8.0+cu128 sees CUDA and 50,864,390,144 device bytes.
- Qwen3-8B is served locally through vLLM 0.11.0 as `qwen3-base`, BF16, 32,768-token context, 0.85 GPU memory utilization. The Ubuntu VM reaches `/v1/models` through an SSH tunnel on localhost port 18000.
- The provider's exact billing start was not available when the run began. Scripts use 2026-09-28 00:00 Asia/Hong_Kong (Unix 1790524800) as an earlier, conservative four-hour deadline. This is an operational limit, not a claim about provider billing.
- The official SWE-Gym OpenHands fork is pinned to `e644a2ca45c3623b27a7e6c169e3d479f0a87fbc`, with `CodeActAgent` v2.2, one worker, one rollout, 50 maximum iterations, no hint, `SWE-Gym/SWE-Gym` `train`, and task `getmoto__moto-7646` at base commit `98ad497eb4de25943ceadabe7c2c5c73cc810e27`.

## Diagnostic rollout

The first model-driven run started an actual task container and produced parsed `str_replace_editor` and `execute_bash` calls. OpenHands did not send `enable_thinking=false`, so the original service template let Qwen3 emit long `<think>` text. With no per-response output cap, one request continued generating while GPU utilization reached 98%. This run was stopped at the timeout supervisor; its partial log remains at `/home/wesz/swegym-qwen3-base-20260928.log`. It is excluded from task-success counts. The task container was removed after interruption.

The service template now defaults to non-thinking when the request omits the flag, and the OpenHands config caps each response at 2,048 tokens. The revised template was rendered with the pinned tokenizer and ends the assistant prefix with an empty `<think></think>` block. Its SHA256 is `0389654058fcc7dddb8b2faaf20ac27a3cd81f6abfe014a116fff4ef2e8e1214`.

## Fixed task result

The fresh non-thinking run completed in 141 seconds. Official output:

`/home/wesz/swegym-openhands/evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/CodeActAgent/qwen3-base_maxiter_50_N_v2.2-no-hint-swegym-local-pilot-nonthinking-run_1/output.jsonl`

- 46 model completions; 1,040,520 prompt tokens and 2,644 generated tokens in aggregate. Maximum single prompt: 30,699 tokens; no completion ended with `finish_reason=length`.
- Parsed calls: 45 `str_replace_editor` views of `moto/dynamodb/models/table.py`, plus one `execute_bash` directory listing. No edit call was recorded.
- The final `git_patch` is empty. This is a model decision/workflow failure after real container interaction, not a task success or a Docker launch failure.
- The agent ended at its 50-iteration limit (`Agent reached maximum iteration in headless mode`), without making an edit.
- The official `output.jsonl` contains 14 action/observation pairs plus the user message. Event IDs jump from 0 to 67; this output does not preserve the full 46-call interaction. The per-request completion files remain under the same output directory's `llm_completions/` folder, but they do not supply rollout-time sampled token logprobs.
- The official `eval_infer.sh` report classifies the prediction as `empty_patch_instances=1`, `resolved_instances=0`, `error_instances=0`. Its report is `/home/wesz/swegym-openhands/qwen3-base_maxiter_50_N_v2.2-no-hint-swegym-local-pilot-nonthinking-run_1.20260927_172219.json`. Because the patch was empty, the harness skipped test-container execution; this is an empty-patch verdict, not a failing-test verdict.

## Verifier control and next gate

The first verifier invocation lacked Poetry in `PATH`; the retry used `/home/wesz/swegym-bin:/home/wesz/.local/bin` and produced the empty-patch report above. The fork's `eval_infer.py` was then run with the cached instance image and host networking. On the model output it reported `empty_generation=true`, `resolved=false`, `error_eval=false`; no test container was started. This is the appropriate reward verdict for an empty patch.

A separate positive control uses the cached train row's 746-byte reference patch (SHA256 `8af6687c1796c475e34cacad8519e1a42f0c17c28800fe0911a2b439991db1f6`) only as a verifier input. It is not part of any model prompt or model score. The clean verifier applied it to `moto/dynamodb/parsing/executors.py`, passed the target FAIL_TO_PASS test and two PASS_TO_PASS tests, and reported `resolved=true` with no error. The control result is `/home/wesz/swegym-openhands/evaluation/evaluation_outputs/controls/qwen3-7646-reference-20260928.swebench_eval.jsonl`.

## Baseline extension and stop point

The next fixed `train` tasks were `getmoto__moto-7365` at `7f6c9cb1deafb280fe7fcc7551c38e397f11a706` and `getmoto__moto-7607` at `ca24f65f2ff415a7eb7eddcea3bd8173e5d5f920`; both instance images passed preflight. The first 7365 attempt never reached model interaction: its new OpenHands runtime image lacked Playwright Chromium and failed on browser startup even with browsing disabled. That attempt is excluded from the model baseline. The runtime build helper now installs the browser's OS dependencies; a local overlay provides the Chromium binary.

The 7365 image was repaired by copying the exact Chromium revision from the previously working local 7646 runtime image and installing its OS dependencies. The rerun reached the task container and made four parsed `str_replace_editor` calls, each viewing the repository directory. The returned listing named only that directory; the model repeated the same call until OpenHands reported `Agent got stuck in a loop`. No edit was made and the patch was empty. Runtime was ready, so this is a model/tool-use loop rather than an infrastructure failure. The clean verifier reported `empty_generation=true`, `resolved=false`, `error_eval=false`. Rollout log: `/home/wesz/swegym-qwen3-7365-browser-fixed-20260928.log`; output: `/home/wesz/swegym-openhands/evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/CodeActAgent/qwen3-base_maxiter_50_N_v2.2-no-hint-swegym-baseline-7365-browser-fixed-run_1/output.jsonl`.

The 7607 process was started but stopped on the user's request before task/model interaction. It is excluded from the baseline. Its partial log is `/home/wesz/swegym-qwen3-7607-20260928.log`.

At the stop point, the fixed two-task model baseline is 0/2 resolved: 7646 exhausted 50 iterations after repeated file views, and 7365 hit loop detection after repeated directory views. Both have empty patches, with clean verifier empty-generation verdicts. This is too small to estimate general capability, but it does identify a model/tool-use bottleneck. The reference-patch positive control confirms that the verifier can produce a resolved result. Neither 7607 nor the infrastructure-only failed attempts enter the denominator.

On the user's request to stop, the 7607 run, vLLM service, and SSH tunnel were stopped. The VM had no remaining OpenHands process or task container. AutoDL cloud-instance power state has **not** been verified or changed through the provider control plane; stopping vLLM does not stop billing. Shut the instance down in the AutoDL console, then verify its stopped state. Tomorrow, recheck the instance, model files, service, tunnel, and runtime images before resuming 7607. Do not start RL or SFT from this two-task result; token/logprob capture and exact reward association are still absent.

Before RL, capture the complete action/observation stream, per-generation token IDs and sampled logprobs, and an exact rollout-to-verifier reward key. Do not infer these from the truncated final history or from text completions. Formal SFT and RL remain gated on separate smoke and memory measurements.

## Later GPU resume: third baseline task and trace pilot

A later AutoDL connection exposed one NVIDIA vGPU-48GB (49,140 MiB total). CUDA worked with torch 2.8.0+cu128; `pip check` passed. The Qwen3 model directory, prior verification reports, `.venv-qwen3`, and all 66 training rows remained present. The data SHA256 still matched the manifest (`656df5e40f1a457fc5f66412ed64a7e780463ebfed9421f253671cf1b61b9848`). Model weight hashes were not recomputed. The provider billing start was not supplied, so this session used a separate conservative process deadline of 2026-09-28 08:04:33 UTC, not a claimed billing deadline.

The third fixed task, `getmoto__moto-7607`, needed a new runtime image. Its first startup lacked Chromium and never reached the model; it is excluded. A local overlay copied the exact browser revision from the working 7646 image without another browser download. On the repaired run, the runtime became ready and the model issued seven editor `view` calls: first directories, then Step Functions files, then repeated views of `execute_state.py`. OpenHands stopped the repeated action/observation loop. The patch was empty. The clean verifier reported `empty_generation=true`, `resolved=false`, `error_eval=false`. Rollout log: `/home/wesz/swegym-qwen3-7607-browser-fixed-20260928.log`; output: `/home/wesz/swegym-openhands/evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/CodeActAgent/qwen3-base_maxiter_50_N_v2.2-no-hint-swegym-baseline-7607-browser-fixed-run_1/output.jsonl`. The fixed three-task model baseline is therefore **0/3 resolved**, excluding infrastructure-only attempts.

A separate **10-iteration stochastic data-capture pilot** on 7646 used temperature 0.7 and did not enter the baseline denominator. `scripts/vm/trace_openai_rollout.py` recorded the raw OpenAI-compatible requests and responses through a VM-local proxy, requesting vLLM `logprobs`, `return_token_ids`, and `return_tokens_as_token_ids`. All 10 responses succeeded and each had aligned generated token IDs and sampled logprobs: 540 tokens total. The raw requests retain nine model-visible observation turns. The pinned fork's per-call completion files matched all 10 response IDs and supplied parsed tool boundaries (one `execute_bash`, nine `str_replace_editor` calls). `scripts/vm/join_swegym_rollout_reward.py` checked contiguous trace sequence, response IDs, token/logprob alignment, instance identity, patch equality, and the verifier result before assigning reward 0. The model again made no edit; verifier returned `empty_generation=true`, `resolved=false`, `error_eval=false`.

Pilot trace: `/home/wesz/swegym-traces/qwen3-7646-trace-20260928T0743Z.jsonl`; joined reward manifest: `/home/wesz/swegym-traces/qwen3-7646-trace-20260928T0743Z.reward.json`. This demonstrates one data/reward association, **not** a tested RL update. The proxy records model-visible observations, but full OpenHands event-stream capture, reward behavior on a nonempty model patch, training-memory measurement, and a policy update remain unverified. Formal 66-row SFT and RL training remain gated.

At this resume's close, the vLLM process occupied 42,651 MiB of the 49,140 MiB reported by `nvidia-smi`; this rules out assuming concurrent training fits, but says nothing conclusive about training after stopping inference. No SFT or RL optimizer step was attempted. The VM had about 8.8 GiB free after the new runtime build, so further image builds require disk planning. All OpenHands jobs, verifier jobs, the trace proxy, vLLM, and the SSH tunnel were stopped. AutoDL cloud-instance shutdown still requires provider-console confirmation; stopping these processes does not stop billing.
