#!/usr/bin/env bash
# Run experiment plans on a shared Linux GPU server inside tmux, so they keep running after you disconnect.
#
#   ./server.sh start stage1 1     start plan "stage1" on GPU 1 (see nvidia-smi for a free one)
#   ./server.sh status stage1      runs done, latest log lines, GPU use
#   ./server.sh attach stage1      watch live (detach again: Ctrl-b then d)
#   ./server.sh stop stage1        stop; start again later and it resumes where it stopped
#
# Extra config overrides go after the GPU id, e.g.  ./server.sh start stage1 1 train.workers=2
#
#   ./server.sh run darkening 0 darkening_test.py --no-tta     run any script the same way; status/attach/stop by name
set -euo pipefail
cd "$(dirname "$0")"

cmd=${1:-help}
plan=${2:-stage1}
session="u2nete_${plan}"
log="logs/${plan}.log"
py=".venv/bin/python"

case "$cmd" in
  start)
    gpu=${3:?give the GPU id, e.g. ./server.sh start stage1 1}
    extra="${*:4}"
    [ -x "$py" ] || { echo "no .venv found: create it first (README, 'Running on an NVIDIA server')"; exit 1; }
    command -v tmux >/dev/null || { echo "tmux is not installed; see README for the nohup alternative"; exit 1; }
    if tmux has-session -t "$session" 2>/dev/null; then echo "already running: ./server.sh attach $plan"; exit 1; fi
    mkdir -p logs
    set_args=""
    if [ -n "$extra" ]; then set_args="--set $extra"; fi
    # nice: lower CPU priority than other users' jobs; OMP_NUM_THREADS: cap CPU threads;
    # PYTHONUNBUFFERED: write progress to the log immediately instead of in buffered chunks
    tmux new-session -d -s "$session" \
      "export CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
       nice -n 10 $py run_experiments.py --plan $plan $set_args 2>&1 | tee -a $log; \
       echo; echo '=== finished, press Enter to close ==='; read"
    echo "started '$plan' on GPU $gpu in tmux session '$session'"
    echo "  watch:  ./server.sh attach $plan      status: ./server.sh status $plan      log: $log"
    ;;
  run)
    gpu=${3:?give the GPU id, e.g. ./server.sh run darkening 0 darkening_test.py --no-tta}
    args="${*:4}"
    [ -n "$args" ] || { echo "give the script to run"; exit 1; }
    [ -x "$py" ] || { echo "no .venv found: create it first (README, 'Running on an NVIDIA server')"; exit 1; }
    if tmux has-session -t "$session" 2>/dev/null; then echo "already running: ./server.sh attach $plan"; exit 1; fi
    mkdir -p logs
    tmux new-session -d -s "$session" \
      "export CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
       nice -n 10 $py $args 2>&1 | tee -a $log; \
       echo; echo '=== finished, press Enter to close ==='; read"
    echo "started '$args' on GPU $gpu in tmux session '$session'; log: $log"
    ;;
  status)
    if tmux has-session -t "$session" 2>/dev/null; then echo "running (tmux session $session)"; else echo "not running"; fi
    $py run_experiments.py --plan "$plan" --dry-run 2>/dev/null | head -1 || true
    if [ -f "$log" ]; then echo "--- last log lines"; tail -n 5 "$log"; fi
    if command -v nvidia-smi >/dev/null; then
      nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
    fi
    ;;
  attach)
    tmux attach -t "$session"
    ;;
  stop)
    if tmux has-session -t "$session" 2>/dev/null; then
      tmux send-keys -t "$session" C-c
      sleep 3
      tmux kill-session -t "$session" 2>/dev/null || true
      echo "stopped '$plan'. Finished runs are kept; './server.sh start $plan <gpu>' resumes."
    else
      echo "not running"
    fi
    ;;
  *)
    sed -n '2,11p' "$0"
    ;;
esac
