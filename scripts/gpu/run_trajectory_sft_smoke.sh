#!/usr/bin/env bash
set -Eeuo pipefail
: "${SESSION_STARTED_AT:?set actual provider billing start in Unix seconds}"
[[ "$SESSION_STARTED_AT" =~ ^[0-9]{10}$ ]] || { echo 'invalid billing start' >&2; exit 1; }
now=$(date +%s)
(( SESSION_STARTED_AT <= now )) || { echo 'billing start is in the future' >&2; exit 1; }
remaining=$(( SESSION_STARTED_AT + 14400 - now ))
(( remaining > 0 )) || { echo 'billing window expired' >&2; exit 1; }
# Bounds loading, training, export and reload together. This does not stop cloud billing.
exec timeout --signal=TERM --kill-after=30s "${remaining}s" \
  env TRAJECTORY_SFT_BOUNDED=1 python -m coding_agent_rl_lab.trajectory_train "$@" --train
