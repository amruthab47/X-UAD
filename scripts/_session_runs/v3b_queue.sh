#!/bin/bash
# Revision-3b queue (starts after the v3 queue) (2026-09-25). Runs jobs from v3b_jobs.txt, 3 at a time,
# in file order. Each job writes <out>/<tag>.{json,log,_pred.npz,.pt}; a job whose
# JSON already exists is skipped, so re-launching resumes the queue.
while [ -f "$(dirname "$0")/../../results_v3/QUEUE_RUNNING" ]; do sleep 600; done
cd "$(dirname "$0")/../.."
export PY="$PWD/.venv/Scripts/python.exe"
export OUT=results_v3
mkdir -p "$OUT"
touch "$OUT/QUEUE_RUNNING"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/_session_runs/keep_awake.ps1 &
run_job () {
  tag=$1; shift
  if [ -f "$OUT/$tag.json" ]; then echo "skip $tag"; return; fi
  echo "$(date '+%F %T') start $tag"
  "$PY" scripts/run_experiment.py --data data --out "$OUT" --tag "$tag" \
        --threads 2 --workers 0 "$@" > "$OUT/$tag.log" 2>&1
  echo "$(date '+%F %T') done  $tag exit=$?"
}
export -f run_job
grep -v '^#' scripts/_session_runs/v3b_jobs.txt | grep -v '^\s*$' | \
  xargs -P 3 -L 1 bash -c 'run_job "$@"' _ >> "$OUT/queue.log" 2>&1
rm -f "$OUT/QUEUE_RUNNING"
echo "$(date '+%F %T') queue finished" >> "$OUT/queue.log"
