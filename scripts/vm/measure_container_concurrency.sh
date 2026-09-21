#!/usr/bin/env bash
# Measure how many task containers this worker can actually run at once.
#
# The worker has 8 vCPU and 5 GB of RAM, and each task container is started with `--memory 2g`.
# That limit is a ceiling, not a reservation, so the number of containers that really fit has to
# be measured rather than derived. The result decides whether rollout parallelism can live here
# at all or must move to the GPU box.
#
# Usage: bash scripts/vm/measure_container_concurrency.sh [image] [levels...]
set -uo pipefail

IMAGE=${1:-xingyaoww/sweb.eval.x86_64.getmoto_s_moto-7365:latest}
shift || true
LEVELS=("$@")
if [ ${#LEVELS[@]} -eq 0 ]; then
  LEVELS=(1 2 3 4 6)
fi

# One unit of work: start, run a real pytest collection inside the repo, report peak RSS.
TASK='
cd /testbed
/opt/miniconda3/envs/testbed/bin/python - <<PY
import resource, subprocess, time
start = time.time()
subprocess.run(["/opt/miniconda3/envs/testbed/bin/pytest","--collect-only","-q",
                "tests/test_dynamodb"], capture_output=True)
peak_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
print(f"elapsed={time.time()-start:.1f}s peak_child_rss={peak_kb//1024}MB")
PY
'

echo "host memory (MB):"
free -m | sed -n '2p'

for level in "${LEVELS[@]}"; do
  echo
  echo "=== concurrency $level ==="
  before_free=$(free -m | awk 'NR==2{print $7}')
  start=$(date +%s.%N)
  pids=()
  for index in $(seq 1 "$level"); do
    (
      name="bench-${level}-${index}-$$"
      docker run --rm --platform linux/amd64 --network none --memory 2g --cpus 1.0 \
        --pids-limit 256 --cap-drop ALL --security-opt no-new-privileges \
        --name "$name" --entrypoint bash "$IMAGE" -lc "$TASK" >"/tmp/bench-$name.log" 2>&1
      echo "$?" > "/tmp/bench-$name.rc"
    ) &
    pids+=($!)
  done
  failures=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failures=$((failures + 1))
  done
  end=$(date +%s.%N)
  after_free=$(free -m | awk 'NR==2{print $7}')

  echo "  wall: $(echo "$end - $start" | bc)s   failures: $failures"
  echo "  available MB before/after: ${before_free} / ${after_free}"
  for log in /tmp/bench-*-$$.log; do
    [ -f "$log" ] || continue
    printf '    %-34s %s\n' "$(basename "$log")" "$(grep -o 'elapsed=.*' "$log" | head -1)"
  done
  grep -l -i "cannot allocate memory\|out of memory\|Killed" /tmp/bench-*-$$.log 2>/dev/null \
    | sed 's/^/    OOM signal in /' || true
  rm -f /tmp/bench-*-$$.log /tmp/bench-*-$$.rc
done

echo
echo "=== leftover bench containers ==="
docker ps -a --format '{{.Names}}' | grep '^bench-' || echo '(none)'
