#!/usr/bin/env bash
# Pilot vs. the current-code run with the same seed, at the same training update.
# Usage: bash pilot_memory/analyze.sh            (latest pilot snapshot)
set -u
cd "$(dirname "$0")"
P=../venv/bin/python
snap=$(ls runs/memory_conn_s0/snapshots | tail -n 1)
echo "== training success at matched steps (last 40 updates up to $snap; pilot includes trap starts)"
$P - "$snap" <<'PY'
import csv, sys
u = int(sys.argv[1][1:6])
for name, path in [("pilot (memory+traps)", "runs/memory_conn_s0/log.csv"), ("current maze_conn_s0", "../runs/maze_conn_s0/log.csv")]:
    rows = [r for r in csv.DictReader(open(path)) if u - 40 < int(r["update"]) <= u]
    v = [float(r["success_rate"]) for r in rows if r["success_rate"] != "nan"]
    print(f"  {name:22s} {sum(v)/len(v):.0%} success (updates {u-39}-{u})")
PY
echo "== 100 standard practice mazes (corner start, sizes 3-6), same mazes for both"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
$P analyze.py "$PWD/src" "$PWD/runs" memory_conn_s0 "$snap" 2>&1 | grep -v "^\[INFO\]\|pybullet build"
$P analyze.py "$PWD/../src" "$PWD/../runs" maze_conn_s0 "$snap" 2>&1 | grep -v "^\[INFO\]\|pybullet build"
