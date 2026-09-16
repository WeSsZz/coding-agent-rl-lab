# Navigation direction check (2026-09-15)

This check isolates source-file discovery from patch correctness. The worker runs the normal baseline
test once to construct the task observation, then disables later verifier actions. A rollout counts
as a navigation hit when it reads an official gold source file within its first eight tool calls.

## Navigation v1

The first dataset contained 40 train-only examples from five tasks: 26 direct gold locate/inspect
examples and 14 prefixes recovered from real tool traces. Ten optimizer steps continued the stable
SFT60 adapter (`train_loss=0.2726`, final `grad_norm=0.8882`).

At the same seeds and four rollouts per task, SFT60 hit 8/16 gold files while navigation v1 hit 7/16.
The per-task change was 7365 2/4 to 1/4, 7514 4/4 to 4/4, 7646 2/4 to 2/4, and 7607 0/4 to 0/4.
The adapter therefore failed the navigation gate and was not sent to full verifier evaluation.

## Navigation v2

The direct locate examples were removed because they teach the model to emit a complete gold path
that is not derivable from the task input. V2 instead adds trace bridges when a file observation
imports the gold module. Its 41 examples contain 18 inspect actions, 21 trace-following actions, and
2 bridge actions. For 7607, one bridge follows a search suggestion into
`state_task_service_aws_sdk.py`, whose import then exposes the callback service target.

Ten independent optimizer steps from SFT60 produced an effective update (`train_loss=0.1948`, final
`grad_norm=0.6449`). A gated same-seed check on 7607 still hit 0/4 gold files. The first actions
reproduced the earlier paths (`execute_state.py`, unrelated X-Ray or DynamoDB files, and a misspelled
`parsing/eval_component.py` path), so the bridge state was not reached reliably. No additional tasks,
full verifier run, or GRPO update were started.

## Interim decision

The reward is not implicated by this check. Both navigation datasets updated the adapter but did not
improve the policy's initial search distribution. More task-specific SFT on the current 7B model is
stopped.

## Frozen 7B/14B capacity control

Bare Qwen2.5-Coder-7B-Instruct and bare Qwen2.5-Coder-14B-Instruct were evaluated on 7607 with the
same prompt, seed 110400, four rollouts, and eight-call navigation-only budget. Both models hit the
official gold source file 0/4 times. Each model produced tool actions in only two of four rollouts.
The 7B runs searched mainly around `execute_state.py` and test text; one attempted to edit a
protected test file. The 14B runs were somewhat more coherent: one searched for
`waitForTaskToken`, and another read and patched the adjacent
`state_task/service/resource.py`. It still did not reach the official target and therefore failed
the predeclared navigation gate. The full verifier and GRPO were not started.

## Capacity-control decision

The larger frozen model improved the quality of an individual search path but did not improve the
measured success rate or the frequency of usable tool trajectories. Model capacity alone is not the
main blocker in this setup. The next experiment should change the navigation policy or tool
protocol so the model first maps symbols and imports before editing; another task-specific adapter
or GRPO run is not justified by the current evidence.

The remote evidence archive is `/root/autodl-tmp/navigation-evidence-20260915.tar.gz` with SHA-256
`652e4324db1a94cddda1abe148f089d242e34815064fbc3264ce48a06f8ec5f7`.
The capacity-control evidence archive is
`/root/autodl-tmp/navigation-capacity-evidence-20260915.tar.gz` with SHA-256
`2c67f2eb1a2a413b6b87c307f66f16d05e85c6e6856a844beaacddf40ca8dc36`.

## 14B protocol and SFT follow-up (2026-09-16)

The follow-up used an RTX PRO 6000 Blackwell Server Edition with 97,887 MiB VRAM. The v3 client
adds `OBJECT_UNDER_FAILURE` and `EXCEPTION_CLASS` fields extracted from the baseline failure, makes
the first response a tool call, and asks the model to follow the failing object through implementation
files before editing. Frozen 14B checks of protocol v1, v2, and v3 all hit the official 7607 source
file 0/4 times at seed 110400. V3 did improve the intermediate route: three rollouts produced tool
actions and reached `state_task_service_aws_sdk.py`, but none reached
`state_task_service_callback.py`.

A fresh 14B LoRA adapter was then trained from the base model on the 41 navigation-v2 examples.
The run used rank 16 on all linear layers, ten optimizer steps, learning rate `1e-4`, and seed
113001. It completed an effective update with `train_loss=0.4669`, final loss `0.3246`, and final
`grad_norm=0.3338`.

At seed 110400 the adapter hit the official target in 1/4 planned rollouts; three rollouts produced
tool trajectories, and the hit occurred at the third action. At independent seed 114400 it hit 0/4;
again three rollouts produced tool trajectories. The combined result is therefore 1/8 planned
rollouts with 6/8 usable tool trajectories. It fails the predeclared gate requiring at least one hit
at each seed and at least 3/8 overall. Full-verifier scouting and GRPO were not started.

The first recount incorrectly treated navigation `target_action` bridge reads as final gold paths.
Navigation report schema 2 instead uses `source_path` for generated navigation-SFT rows and the
original inspect action path for inspect rows. The corrected results above match direct trace review.

The current evidence says the 14B model can follow the first symbol-to-file hop, while the policy
does not reliably continue from the AWS SDK service into the callback implementation. Another GRPO
run is not justified until that second-hop behavior is taught or exposed directly and passes the
same two-seed navigation gate. The data disk has 106 GiB free, so no capacity expansion is needed.

The complete follow-up archive, including the adapter, gold data, protocol checks, and both adapter
evaluations, is `/root/autodl-tmp/navigation-14b-protocol-sft-evidence-20260916.tar.gz` with SHA-256
`13a3ad5646c8b288b25dc4348568e1c07ccc4905534386997cd4c1051bad85a7`.

## Parent-path protocol follow-up (2026-09-16)

Protocol v4 derives a local parent implementation path from real imports and class inheritance in a
`read_file` observation. For example, reading `StateTaskServiceAwsSdk(StateTaskServiceCallback)`
now exposes the callback module as `PARENT_IMPLEMENTATION_PATH`. This transformation uses source
already returned by the tool and does not consult the gold patch. Relevant local and remote tests
pass.

With the existing 14B navigation adapter, v4 reached the 7607 callback file in 3/4 planned rollouts
at seed 110400 but 0/4 at seed 114400. Six real successful parent-path transitions from the first
run were extracted as train-only completion examples. A two-step continuation at learning rate
`5e-5` covered these examples once and produced an effective update (`train_loss=0.04364`, final
`loss=0.02906`, final `grad_norm=0.07533`). The resulting adapter reached the callback in 3/4 and
1/4 planned rollouts at the two seeds, passing the navigation gate with 4/8 combined hits.

The complete-verifier result did not pass the reward gate. On 7607, two of four rollouts produced
verified patches, but both introduced new failures; all four rewards were zero. On 7514 all four
rewards were also zero. Two-rollout scouts on the other train tasks found one non-regressive valid
patch on 7646 with reward `0.03`, but an independent four-rollout check returned four zeros and no
failure reduction. The 7365, 7446, and 7509 scouts were all zero. Regression tasks 7385 and 7608
were also zero in two rollouts each, with 7608 introducing a new failure.

Navigation is therefore no longer the immediate blocker for 7607. The remaining failure is patch
semantics: the model reaches the correct implementation but does not produce a verifier-improving
change. No train or regression task produced reproducible failure reduction or reward variance, so
GRPO remains stopped. The next session should inspect the 7607 wrong patches against the gold hunk
and build semantic edit supervision before another reward scout.

The complete parent-path evidence archive is
`/root/autodl-tmp/navigation-parent-path-evidence-20260916.tar.gz` with SHA-256
`5d65bdcce1ad82d1efd417bd07f49ac477c15b37585a76f18cf88419afb0b17c`.
