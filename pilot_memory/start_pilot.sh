#!/usr/bin/env bash
# Memory + trap-practice pilot: one real-fly (connectome) network, seed 0, on grid mazes,
# with the code in pilot_memory/src:
#   fix 1: training looks back 128 steps (~4.3 s) through the network's memory, not 32 (~1 s)
#   fix 3: 30% of training mazes start at the end of a misleading dead end
# Compared afterwards against runs/maze_conn_s0 (same seed, current code) with:
#   bash pilot_memory/analyze.sh
set -u
cd "$(dirname "$0")"
STEPS=${STEPS:-5000000}
systemd-run --user --unit=pilot-memory --working-directory="$PWD" \
  -p MemoryMax=6G -p MemorySwapMax=0 -p OOMPolicy=continue \
  bash -c "mkdir -p runs/memory_conn_s0 && ../venv/bin/python src/train_ppo.py --model connectome --task grid \
    --name memory_conn_s0 --seed 0 --steps $STEPS > runs/memory_conn_s0/stdout.txt 2>&1 \
    && touch runs/memory_conn_s0/done"
echo "pilot started; progress: tail -f pilot_memory/runs/memory_conn_s0/stdout.txt"
