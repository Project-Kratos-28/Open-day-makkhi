#!/usr/bin/env bash
# Goal line-of-sight pilot: one real-fly (connectome) network, seed 0, on grid mazes,
# with the code in pilot_goalsight/src (goal-visible input + goal radius 0.40 m).
# Compared afterwards against runs/maze_conn_s0 (same seed, same mazes, old code)
# with:  bash pilot_goalsight/analyze.sh
set -u
cd "$(dirname "$0")"
STEPS=${STEPS:-5000000}
systemd-run --user --unit=pilot-goalsight --working-directory="$PWD" \
  -p MemoryMax=5G -p MemorySwapMax=0 -p OOMPolicy=continue \
  bash -c "mkdir -p runs/goalsight_conn_s0 && ../venv/bin/python src/train_ppo.py --model connectome --task grid \
    --name goalsight_conn_s0 --seed 0 --steps $STEPS > runs/goalsight_conn_s0/stdout.txt 2>&1 \
    && touch runs/goalsight_conn_s0/done"
echo "pilot started; progress: tail -f pilot_goalsight/runs/goalsight_conn_s0/stdout.txt"
