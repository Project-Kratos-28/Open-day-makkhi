#!/usr/bin/env bash
# Pilot vs. the old-code run with the same seed, at the same training update.
# Usage: bash pilot_goalsight/analyze.sh            (latest pilot snapshot)
set -u
cd "$(dirname "$0")"
P=../venv/bin/python
snap=$(ls runs/goalsight_conn_s0/snapshots | tail -n 1)
echo "== training success at matched steps (last 40 updates up to $snap)"
$P - "$snap" <<'PY'
import csv, sys
u = int(sys.argv[1][1:6])
for name, path in [("pilot (goal sight)", "runs/goalsight_conn_s0/log.csv"), ("old maze_conn_s0", "../runs/maze_conn_s0/log.csv")]:
    rows = [r for r in csv.DictReader(open(path)) if u - 40 < int(r["update"]) <= u]
    v = [float(r["success_rate"]) for r in rows if r["success_rate"] != "nan"]
    print(f"  {name:20s} {sum(v)/len(v):.0%} success (updates {u-39}-{u})")
PY
echo "== behaviour near the goal, 30 practice mazes (4x4-6x6)"
$P analyze.py src runs goalsight_conn_s0 "$snap" 2>&1 | grep -v "^\[INFO\]\|pybullet build"
$P analyze.py ../src ../runs maze_conn_s0 "$snap" 2>&1 | grep -v "^\[INFO\]\|pybullet build"
