#!/usr/bin/env bash
# The full experiment, in order: 20 maze runs -> pick best versions -> maze tests
# (best and final versions) -> 20 obstacle runs -> obstacle test.
# Safe to stop and start again: finished runs are skipped, unfinished ones resume
# from their last save point, and the selection/tests simply run again.
#
#   systemd-run --user --unit=everything --working-directory="$PWD" \
#     -p MemoryMax=22G -p MemorySwapMax=0 -p OOMPolicy=continue bash src/run_everything.sh
set -eu
cd "$(dirname "$0")/.."
LOG=runs/experiments.log
MODELS="connectome shuffled gru baseline"
SEEDS="0 1 2 3 4"
export OMP_NUM_THREADS=1   # evaluation: tiny networks, one thread per worker process

echo "== everything: start $(date '+%F %T')" >> $LOG
TASK=grid MODELS="$MODELS" SEEDS="$SEEDS" PARALLEL=5 bash src/run_experiments.sh >> $LOG 2>&1
echo "maze selection + tests start $(date +%T)" >> $LOG
(cd src && ../venv/bin/python select_best.py --workers 20 \
   && ../venv/bin/python evaluate.py --task grid --checkpoint best --workers 20 \
   && ../venv/bin/python evaluate.py --task grid --checkpoint last --workers 20) > runs/maze_eval_stdout.txt 2>&1
echo "maze tests done $(date +%T)" >> $LOG
MODELS="$MODELS" SEEDS="$SEEDS" PARALLEL=5 bash src/run_experiments.sh >> $LOG 2>&1
(cd src && ../venv/bin/python evaluate.py --workers 20) > runs/eval_stdout.txt 2>&1
echo "== everything: done $(date '+%F %T')" >> $LOG
