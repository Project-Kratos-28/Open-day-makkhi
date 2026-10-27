#!/usr/bin/env bash
# Main comparison: connectome vs baseline (and the shuffled-wiring control),
# one run per seed for each model, 5M steps per run.
# Runs PARALLEL at a time. Each run writes runs/<name>/log.csv, model.pt, stdout.txt.
# Safe to re-run: finished runs (marked with runs/<name>/done) are skipped.
#
#   bash src/run_experiments.sh            # start (or resume) everything
#   SEEDS="0 1 2 3 4" bash src/run_experiments.sh   # more seeds
#   MODELS="shuffled" SEEDS="0 1 2 3 4" bash src/run_experiments.sh
#   TASK=grid MODELS="connectome shuffled gru baseline" SEEDS="0 1 2 3 4" bash src/run_experiments.sh
#       grid mazes, trained from scratch; runs are named maze_<prefix>_s<seed>
#   pkill -f train_ppo.py                  # stop everything
set -u
cd "$(dirname "$0")/.."
STEPS=${STEPS:-5000000}
PARALLEL=${PARALLEL:-3}
SEEDS=${SEEDS:-0 1 2}
MODELS=${MODELS:-connectome baseline}
TASK=${TASK:-obstacles}

jobs=()
declare -A PREFIX=([connectome]=conn [baseline]=base [shuffled]=shuf [gru]=gru)
for seed in $SEEDS; do
  for model in $MODELS; do
    name=${PREFIX[$model]}_s$seed
    if [ "$TASK" = grid ]; then
      extra="--task grid"
      name=maze_$name
    else
      extra=""
    fi
    jobs+=("$model $name $seed $extra")
  done
done

run_one() {
  read -r model name seed extra <<<"$1"
  if [ -f "runs/$name/done" ]; then echo "skip $name (already done)"; return; fi
  mkdir -p "runs/$name"
  echo "start $name $(date +%T)"
  if venv/bin/python src/train_ppo.py --model "$model" --name "$name" --seed "$seed" \
       --steps "$STEPS" $extra > "runs/$name/stdout.txt" 2>&1; then
    touch "runs/$name/done"; echo "done  $name $(date +%T)"
  else
    echo "FAIL  $name $(date +%T) (see runs/$name/stdout.txt)"
    return 1
  fi
}
export -f run_one
export STEPS

printf '%s\n' "${jobs[@]}" | xargs -P "$PARALLEL" -I{} bash -c 'run_one "$@"' _ {} || { echo "some runs FAILED $(date +%T)"; exit 1; }
echo "all finished $(date +%T)"
