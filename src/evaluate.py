"""
Final scoreboard: every trained network flies the same held-out mazes,
at the training difficulty and at harder levels it never trained on.

    python src/evaluate.py                       # all finished conn_s*/shuf_s*/base_s*/gru_s* runs
    python src/evaluate.py --episodes 20 conn_s0 # quick check of one run
    python src/evaluate.py --task grid           # all finished maze_* runs, on grid mazes

Maze seeds start at 1,000,000, far from anything used in training, so these
layouts are new to every network, and identical for all of them. The drone
uses its best-guess action (no random exploration).

For grid-maze runs the network tested is model_best.pt (the snapshot that
scored best on separate practice mazes, see select_best.py) if it exists.

Writes runs/eval_results.csv (grid: runs/eval_grid_results.csv), one row per
run x difficulty, and prints a summary.
"""
import argparse
import csv
from multiprocessing import Pool

import numpy as np
import torch

from baseline_rnn import make_model
from grid_maze_env import ALGORITHMS, TRAIN_BRAID, GridMazeAviary
from maze_env import ACT_DIM, OBS_DIM, MazeAviary
from ppo import RUNS_DIR, Agent

EVAL_SEED = 1_000_000
# name: (pillars, wall segments). "standard" is what both networks trained on.
LEVELS = {"standard": (8, 2), "hard": (14, 4), "very_hard": (20, 6)}
# Grid mazes: all three maze styles mixed, loops as in training. 7x7 is never trained on.
GRID_LEVELS = {"4x4": (4,), "6x6": (6,), "7x7_new": (7,)}


def make_level_env(task, level):
    if task == "grid":
        return GridMazeAviary(sizes=GRID_LEVELS[level], algorithms=ALGORITHMS, braid_range=TRAIN_BRAID)
    return MazeAviary(*LEVELS[level])


def load_agent(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    if a.get("obs_dim", 23) != OBS_DIM:
        raise SystemExit(f"{path} was trained with {a.get('obs_dim', 23)} sensor inputs; the env now "
                         f"has {OBS_DIM} (32 lidar rays), so it can't be loaded")
    agent = Agent(make_model(a["model"], OBS_DIM, ACT_DIM, a["min_synapses"], a["seed"]),
                  squash=bool(a.get("squash", 0)))
    agent.load_state_dict(ckpt["agent"])
    return agent.eval(), a


def fly(agent, env, seeds):
    """Fly one deterministic episode per maze seed. Returns (outcomes, seconds-to-goal list)."""
    outcomes, goal_times = [], []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        h = agent.core.initial_state(1)
        done = False
        while not done:
            with torch.no_grad():
                action, h = agent.act(torch.as_tensor(obs).unsqueeze(0), h)
            obs, _, term, trunc, info = env.step(action[0].numpy())
            done = term or trunc
        outcomes.append(info["outcome"])
        if info["outcome"] == "goal":
            goal_times.append(env.env_steps * env.dt)
    return outcomes, goal_times


def evaluate_run(job):
    run, episodes, task, which = job
    torch.set_num_threads(1)
    path = RUNS_DIR / run / ("model_best.pt" if task == "grid" and which == "best" else "model.pt")
    if not path.exists():
        raise SystemExit(f"{path} missing (for grid runs, run select_best.py first)")
    agent, train_args = load_agent(path)
    checkpoint = f"{path.name}@u{torch.load(path, map_location='cpu', weights_only=False)['update']}"

    rows = []
    for level in (GRID_LEVELS if task == "grid" else LEVELS):
        env = make_level_env(task, level)
        outcomes, goal_times = fly(agent, env, range(EVAL_SEED, EVAL_SEED + episodes))
        env.close()
        rows.append({"run": run, "model": train_args["model"], "seed": train_args["seed"],
                     "checkpoint": checkpoint, "level": level, "episodes": episodes,
                     "success_rate": outcomes.count("goal") / episodes,
                     "crash_rate": outcomes.count("crash") / episodes,
                     "timeout_rate": outcomes.count("timeout") / episodes,
                     "mean_time_to_goal_s": float(np.mean(goal_times)) if goal_times else float("nan")})
        print(f"  {run:8s} {level:9s} success {rows[-1]['success_rate']:4.0%}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="run names (default: every finished conn_s*/shuf_s*/base_s*/gru_s* run)")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--task", choices=["obstacles", "grid"], default="obstacles")
    ap.add_argument("--checkpoint", choices=["best", "last"], default="best",
                    help="grid runs: test model_best.pt (picked on practice mazes) or the final model.pt")
    args = ap.parse_args()

    prefixes = ("maze_",) if args.task == "grid" else ("conn_", "base_", "shuf_", "gru_")
    levels = GRID_LEVELS if args.task == "grid" else LEVELS
    runs = args.runs or sorted(d.name for d in RUNS_DIR.glob("*_s*")
                               if d.name.startswith(prefixes) and (d / "done").exists())
    if not runs:
        raise SystemExit("no finished runs to evaluate")
    print(f"evaluating {len(runs)} runs x {len(levels)} levels x {args.episodes} mazes")
    with Pool(min(args.workers, len(runs))) as pool:
        rows = [r for run_rows in pool.map(evaluate_run, [(r, args.episodes, args.task, args.checkpoint) for r in runs])
                for r in run_rows]

    out = RUNS_DIR / (f"eval_grid_{args.checkpoint}_results.csv" if args.task == "grid" else "eval_results.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\nsaved {out}\n\nsuccess rate, mean ± sample std across runs (n = number of runs):")
    for level in levels:
        for model in ("connectome", "shuffled", "baseline", "gru"):
            s = [r["success_rate"] for r in rows if r["level"] == level and r["model"] == model]
            if s:
                print(f"  {level:9s} {model:10s} {np.mean(s):5.1%} ± {np.std(s, ddof=1) if len(s) > 1 else 0:4.1%}  (n={len(s)})")


if __name__ == "__main__":
    main()
