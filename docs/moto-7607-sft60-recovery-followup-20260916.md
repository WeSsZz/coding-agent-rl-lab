# moto-7607 SFT60 recovery follow-up (2026-09-16)

Two train-only recovery examples were rebuilt with `coding_agent_rl_lab.sft_recovery` from the
saved real rollouts at seeds 106100 and 107100. The rebuilt dataset is byte-identical to the prior
extraction and has SHA-256
`c3c6ed0a68223eef3b314e79117289dd51b2c037edcc3f97c4d6005e7930dfdb`.

The stable SFT60 adapter
(`55c947e02b404c3d150e3a2eabd4f55dd50c34641d8d68495f7fbb00c573081c`) received one optimizer
step over both examples at learning rate `5e-5` and seed `121001`. The update was effective:
`train_loss=0.2579658` and `grad_norm=2.6751328`. The output adapter SHA-256 is
`bbda71e6d0549435375f837fa7d363c8472834b2cc4bd11cc494274272f980e0`.

Frozen seed `121100` completed four rollouts against the complete official moto-7607 verifier.
There were 0/4 strict successes and 0/4 failure reductions. All four rollouts created and verified
patches, but only one patch was valid and three introduced two new failures each. Rewards were
`[0.03, 0.0, 0.0, 0.0]` (mean `0.0075`, TRL-reported group standard deviation `0.015`). The only
positive reward preserved the baseline one failure; it did not reduce failures.

GRPO was not started. Within-group reward variance was present, but the required reproducible
failure reduction was absent.

The remote evidence directory is
`/root/autodl-tmp/moto-7607-sft60-recovery-followup-20260916`. The complete archive is
`/root/autodl-tmp/moto-7607-sft60-recovery-evidence-20260916.tar.gz`, with SHA-256
`95d46453ca623cd84fc5f8e6110e985de7d21b62b20a512b7e8d46a6f9853fd2`.

The requested `bjb2:32245` route reset before the SSH banner. The target 48 GiB instance was
identified by GPU and evidence contents and used through the working Ubuntu-to-`bjb1:31719` route.
No previous evidence or adapter was overwritten.
