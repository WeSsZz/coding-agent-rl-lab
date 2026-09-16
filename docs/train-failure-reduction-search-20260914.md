# Train failure-reduction search (2026-09-14)

The previous paired direction check found no strict success or reduction in failing tests, so this
round did not extend the old multi-step GRPO run. It first searched the six train tasks for a
rollout that could produce a verifier-measured improvement under `conservative-v2`.

## Search result

Repeated base, SFT20, SFT60, and task-isolated SFT rollouts on `moto-7365`, `moto-7509`,
`moto-7514`, and `moto-7646` produced no reduction in failing tests. `moto-7514` was the closest:
the SFT60 adapter consistently located S3 Select and produced valid source patches, but the patches
did not fix a test.

The official SFT data for `moto-7514` contains six gold hunks. Each hunk had been represented as an
independent locate/inspect/edit/verify trajectory. That layout teaches repeated searches and a test
run after every isolated hunk, even though the solution requires several accumulating edits.

`coding_agent_rl_lab.sft_cumulative` now converts the per-hunk verifier rows into one accumulating
trajectory per task. It removes duplicate exact tool calls and keeps one final verifier action. The
resulting private train-only `moto-7514` dataset has 16 completion-only examples and a maximum full
length of 3,173 tokens, below the configured 16,384-token limit. `sft_train --adapter-path` allows
these examples to continue an existing LoRA instead of discarding its learned tool protocol.

Training the cumulative data for 12 steps directly from the base model degraded tool use. Continuing
the stable 60-step SFT adapter for 12 steps retained the protocol and reached `train_loss=0.1126`
with a final-step `grad_norm=0.1678`.

## Frozen evaluation

On a curriculum verifier containing the official `moto-7514` gzip failure plus all seven passing
tests, the continued adapter achieved 1 strict success in 4 rollouts at seed 104100. The successful
rollout reduced the baseline from 1 failing test to 0, created a valid source patch, ran the verifier,
and received reward 1. The group summary was:

- strict success: 1/4
- trials reducing failures: 1/4
- valid patches: 1/4
- mean `conservative-v2` reward: 0.25

This is curriculum evidence only. The slice is derived from the train task and does not constitute a
complete SWE-Gym resolution.

The same frozen adapter was then evaluated on the complete official `moto-7514` verifier with an
independent seed, 105100. Its baseline had 3 failures and 7 passes. The result was 0 strict successes
and 0 failure reductions in 4 rollouts. Three rollouts created valid patches, but each introduced new
failures, so all received reward 0. Two patches changed decoded text into bytes and expanded the
failure count from 3 to 10. This confirms that `conservative-v2` rejects the previously exploitable
"valid patch plus verifier" behavior when tests regress.

## GRPO gate

A one-step GRPO attempt at seed 104100 did not reproduce the frozen evaluation's positive sample.
All four training rollouts received reward 0, with `reward_std=0`, `grad_norm=0`, and no effective
adapter update. The experiment therefore does not justify multi-step GRPO by itself.

A second one-step attempt at seed 104101 also produced four zero rewards. One valid patch expanded
the failure count from 1 to 8, while the other three did not create a valid verified patch. Again,
`reward_std=0`, `grad_norm=0`, and no effective update occurred. Two consecutive training groups
therefore failed the variance gate, and no multi-step run was started.

## Decision

The first real failure reduction shows that the reward can recognize a correct update and that
cumulative trajectory supervision moves the policy in a useful direction. The complete task and the
two training groups remain unsuccessful. Multi-step GRPO stays gated on a reproducible training
group with nonzero reward variance followed by a frozen post-update evaluation.

The remote evidence archive is
`/root/autodl-tmp/train-failure-reduction-evidence-20260914.tar.gz` with SHA-256
`492a0b8e3eb1f116541fe7b54c7eaf0de5b602a0f77e227a855d6993189d39d6`. A compact machine-readable
summary is stored beside this document in `train-failure-reduction-search-20260914.json`.

## Full-task follow-up: moto-7607

`moto-7607` was selected because its complete official verifier has one failing and two passing
tests. Its gold solution touches one callback service file and produced six cumulative SFT examples
(1,291–1,963 tokens). Continuing SFT60 for 5 steps gave an effective update, but frozen seed 106100
still produced 0/4 failure reductions: two patches preserved the single failure and two introduced
new failures. A further 5-step continuation also produced 0/4 reductions at seed 107100, with three
rollouts introducing new failures. No GRPO run was started.

The traces expose a narrower recovery failure: one incorrect search returned the correct callback
file as a `SUGGESTED_PATH`, but the policy opened a neighboring Lambda file. A small train-only
recovery-example builder was added to supervise that exact transition.

## Recovery follow-up (2026-09-15)

The builder extracted two distinct failed-search transitions from the 5-step and 10-step traces.
Each example ends with the gold `read_file` call to the suggested callback path. Continuing the
10-step adapter on these two examples for 2 optimizer steps produced a nonzero update
(`train_loss=0.1181`, final `loss=0.02235`, `grad_norm=0.4471`).

Frozen evaluation on the complete official `moto-7607` verifier at seed 108100 did not improve.
Across four rollouts it produced 0 strict successes, 0 failure reductions, and mean reward 0. Two
rollouts created valid verified patches and both introduced new failures. None of the 48 tool calls
contained the target callback path or the exact `SUGGESTED_PATH` trigger used by the recovery
examples.

This narrows the diagnosis: the reward is correctly rejecting regressions, while the policy's search
state distribution does not reliably reach the narrowly supervised recovery state. The two examples
teach the desired action after one observed mistake but do not improve the preceding file-selection
behavior. With all four rewards equal to zero, GRPO would have zero within-group advantage, so no
GRPO update was started.

The recovery evidence remains on AutoDL at
`/root/autodl-tmp/train-7607-recovery-evidence-20260915.tar.gz` with SHA-256
`0d4d0c4d3b069d66ddff69bc4d4e9d74ad8ae78078e9a8b40d0a1fcb6b38d3dc`.
