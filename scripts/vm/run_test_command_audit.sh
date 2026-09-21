#!/usr/bin/env bash
# Run the SWE-Gym test-command audit inside a freshly extracted copy of the repository.
# Kept as a script because the command mixes options that start with '-' with quoted values,
# which does not survive the PowerShell -> ssh -> sh quoting chain.
set -uo pipefail

STAMP=$(date +%Y%m%d-%H%M%S)
ROOT=/tmp/audit-$STAMP
mkdir -p "$ROOT"
tar -xf /tmp/repo.tar -C "$ROOT"

echo "=== extracted to $ROOT ==="
cd "$ROOT"

echo "=== image availability ==="
for image in \
  xingyaoww/sweb.eval.x86_64.conan-io_s_conan-10975:latest \
  xingyaoww/sweb.eval.x86_64.conan-io_s_conan-13757:latest \
  xingyaoww/sweb.eval.x86_64.pydantic_s_pydantic-8947:latest ; do
  printf '  %-70s ' "$image"
  docker image inspect "$image" >/dev/null 2>&1 && echo present || echo MISSING
done

echo "=== audit ==="
PYTHONPATH=src python3 scripts/audit_swe_gym_test_command.py \
  --rows /home/wesz/coding-agent-rl-lab/work/private/audit-rows.jsonl \
  --prefix "/opt/miniconda3/envs/testbed/bin/pytest -n0 -rA" \
  --output /home/wesz/coding-agent-rl-lab/work/private/test-command-audit.json \
  --test-timeout-seconds 600
STATUS=$?

echo "=== audit exit status: $STATUS ==="
echo "=== leftover containers ==="
docker ps -a --format '{{.Names}} {{.Status}}' | grep coding-agent || echo '(none)'
exit $STATUS
