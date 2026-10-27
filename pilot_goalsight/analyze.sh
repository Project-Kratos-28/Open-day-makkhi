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
# One CPU thread each (tiny networks: more threads only add overhead); both run side by side.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
$P analyze.py src runs goalsight_conn_s0 "$snap" > /tmp/analyze_pilot_$$.txt 2>&1 &
$P analyze.py ../src ../runs maze_conn_s0 "$snap" > /tmp/analyze_old_$$.txt 2>&1 &
wait
grep -hv "^\[INFO\]\|pybullet build" /tmp/analyze_pilot_$$.txt /tmp/analyze_old_$$.txt
rm -f /tmp/analyze_pilot_$$.txt /tmp/analyze_old_$$.txt
