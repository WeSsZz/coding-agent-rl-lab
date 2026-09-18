#!/bin/sh
set -eu

run_dir=/home/wesz/coding-agent-rl-lab/work/private/moto-7607-p1-v2-20260917
project_dir=/home/wesz/coding-agent-rl-lab
mkdir -p "$run_dir"
test ! -e "$run_dir/worker.pid"
test ! -e "$run_dir/tunnel.pid"
test ! -e "$run_dir/worker.token"
umask 077
openssl rand -hex 32 >"$run_dir/worker.token"

cd "$project_dir"
PYTHONPATH=src nohup python3 -m coding_agent_rl_lab.grpo_remote \
  --port 9015 \
  --task-source swe-gym \
  --task-set train \
  --task-id getmoto__moto-7607 \
  --rows-cache work/swe-gym-development-rows.jsonl \
  --test-timeout-seconds 300 \
  --reward-version conservative-v2 \
  --token-file "$run_dir/worker.token" \
  >"$run_dir/worker.log" 2>&1 &
worker_pid=$!
printf '%s\n' "$worker_pid" >"$run_dir/worker.pid"

sleep 2
kill -0 "$worker_pid"

ssh -p 31719 \
  -i /home/wesz/.ssh/id_ed25519_autodl \
  -o BatchMode=yes \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=4 \
  -N -R 127.0.0.1:9015:127.0.0.1:9015 \
  root@connect.bjb1.seetacloud.com \
  >"$run_dir/tunnel.log" 2>&1 &
tunnel_pid=$!
printf '%s\n' "$tunnel_pid" >"$run_dir/tunnel.pid"

sleep 2
kill -0 "$tunnel_pid"
printf 'worker_pid=%s tunnel_pid=%s run_dir=%s\n' "$worker_pid" "$tunnel_pid" "$run_dir"
