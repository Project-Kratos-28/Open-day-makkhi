"""
Pick each grid-maze run's best version on practice mazes.

Training saves a weights snapshot every 20 updates (runs/<run>/snapshots/).
Training can wobble (e.g. drift into a slow, cautious style late on), so
instead of always using the last one, every snapshot after the first 25% of
training flies the same practice mazes and the best is copied to
runs/<run>/model_best.pt, which evaluate.py then tests. The rule was fixed
before any final results were seen (DECISIONS.md Step 15); evaluate.py
--checkpoint last also tests the final version, and both are reported.

Practice mazes use seeds from 500,000; the final test uses seeds from
1,000,000, so the test mazes play no part in the choice (that would be cheating).

    python src/select_best.py                 # all finished maze_* runs
    python src/select_best.py maze_conn_s0 --episodes 40
"""
import argparse
import csv
import shutil
from multiprocessing import Pool

import torch

from evaluate import fly, load_agent
from grid_maze_env import GridMazeAviary
from ppo import RUNS_DIR

VALIDATION_SEED = 500_000


def score(job):
    path, episodes = job
    torch.set_num_threads(1)
    agent, _ = load_agent(path)
    env = GridMazeAviary()   # the training mix of sizes, styles and loops
    outcomes, _ = fly(agent, env, range(VALIDATION_SEED, VALIDATION_SEED + episodes))
    env.close()
    return path, outcomes.count("goal") / episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--from-fraction", type=float, default=0.25,
                    help="only consider snapshots after this fraction of training")
    args = ap.parse_args()

    runs = args.runs or sorted(d.name for d in RUNS_DIR.glob("maze_*") if (d / "done").exists())
    jobs = []
    for run in runs:
        snaps = sorted((RUNS_DIR / run / "snapshots").glob("u*.pt"))
        jobs += [(s, args.episodes) for s in snaps[int(len(snaps) * args.from_fraction):]]
    print(f"scoring {len(jobs)} snapshots from {len(runs)} runs on {args.episodes} practice mazes each")
    with Pool(args.workers) as pool:
        scores = pool.map(score, jobs)

    for run in runs:
        mine = [(p, s) for p, s in scores if p.parent.parent == RUNS_DIR / run]
        if not mine:
            raise SystemExit(f"{run}: no snapshots to choose from")
        with open(RUNS_DIR / run / "validation.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["snapshot", "practice_success_rate"])
            w.writerows([(p.name, s) for p, s in mine])
        # Best score; on a tie, the later snapshot (more training).
        best_path, best = max(mine, key=lambda ps: (ps[1], ps[0].name))
        shutil.copy(best_path, RUNS_DIR / run / "model_best.pt")
        print(f"  {run:16s} best {best_path.name} practice success {best:.0%} "
              f"(last snapshot {mine[-1][1]:.0%})")


if __name__ == "__main__":
    main()
