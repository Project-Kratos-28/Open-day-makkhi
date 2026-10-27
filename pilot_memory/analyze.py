"""
Fly one run's snapshot on the standard practice mazes (corner start, the training
mix of sizes; seeds from 500,000, as in select_best.py) and report goals, crashes,
timeouts, and how many timeouts were loops (the drone kept going back to the same
cells: some cell entered 3+ times).

    python analyze.py <src dir> <runs dir> <run> <snapshot|model.pt> [episodes]
"""
import sys
from multiprocessing import Pool

src, runs, run, snap = sys.argv[1:5]
episodes = int(sys.argv[5]) if len(sys.argv) > 5 else 100
sys.path.insert(0, src)


def fly_one(seed):
    import numpy as np
    import torch
    from evaluate import load_agent
    from grid_maze_env import ARENA_HALF, GridMazeAviary
    torch.set_num_threads(1)
    path = f"{runs}/{run}/{'snapshots/' if snap.startswith('u') else ''}{snap}"
    agent, _ = load_agent(path)
    env = GridMazeAviary()
    obs, _ = env.reset(seed=seed)
    h, done, cells, entries = agent.core.initial_state(1), False, [], {}
    while not done:
        with torch.no_grad():
            a, h = agent.act(torch.as_tensor(obs).unsqueeze(0), h)
        obs, _, term, trunc, info = env.step(a[0].numpy())
        done = term or trunc
        c = tuple(np.clip(((env.pos[0, :2] + ARENA_HALF) // env.cell).astype(int), 0, env.n - 1))
        if not cells or cells[-1] != c:
            cells.append(c)
            entries[c] = entries.get(c, 0) + 1
    env.close()
    return env.n, info["outcome"], max(entries.values()) >= 3


if __name__ == "__main__":
    with Pool(10) as pool:
        res = pool.map(fly_one, range(500_000, 500_000 + episodes))
    out = [o for _, o, _ in res]
    loops = sum(1 for _, o, lp in res if o == "timeout" and lp)
    big = [o for n, o, _ in res if n >= 5]
    print(f"  {run} @ {snap}: goal {out.count('goal')}/{episodes}, crash {out.count('crash')}, "
          f"timeout {out.count('timeout')} (of which loops {loops}); "
          f"5x5-6x6 only: goal {big.count('goal')}/{len(big)}")
