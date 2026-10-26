"""
Compare a network's behaviour near the goal: success, goal near-misses (got within
0.6 m but didn't finish) and how much steering responds to the goal direction,
split by whether the goal is in line of sight.

    python analyze.py <src dir> <runs dir> <run name> [snapshot]   (see analyze.sh)
"""
import sys
src, runs_dir, run = sys.argv[1:4]
snap = sys.argv[4] if len(sys.argv) > 4 else None
sys.path.insert(0, src)
import numpy as np
import pybullet as p
import torch
from evaluate import load_agent
from grid_maze_env import GridMazeAviary, N_RAYS

path = f"{runs_dir}/{run}/" + (f"snapshots/{snap}" if snap else "model.pt")
agent, _ = load_agent(path)
env = GridMazeAviary(sizes=(4, 5, 6))
outcomes, near_miss, resp = [], 0, {True: [], False: []}
for seed in range(500_000, 500_030):           # practice mazes (never the test mazes)
    obs, _ = env.reset(seed=seed)
    h = agent.core.initial_state(1)
    done, closest = False, 9.0
    while not done:
        o = torch.as_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            a, h2 = agent.act(o, h)
            b = np.arctan2(obs[N_RAYS], obs[N_RAYS + 1])
            turns = []
            for db in (+0.52, -0.52):          # goal direction shifted 30 deg left / right
                o2 = o.clone()
                o2[0, N_RAYS], o2[0, N_RAYS + 1] = float(np.sin(b + db)), float(np.cos(b + db))
                turns.append(agent.act(o2, h)[0][0, 0].item())
        pos = env.pos[0]
        hit = p.rayTest(list(pos), [*env.goal, pos[2]], physicsClientId=env.CLIENT)[0]
        visible = hit[0] == -1 or hit[0] == env.DRONE_IDS[0]
        d = np.linalg.norm(env.goal - pos[:2])
        closest = min(closest, d)
        if d < 2.0:
            resp[bool(visible)].append(turns[0] - turns[1])
        h = h2
        obs, _, term, trunc, info = env.step(a[0].numpy())
        done = term or trunc
    outcomes.append(info["outcome"])
    near_miss += info["outcome"] != "goal" and closest < 0.6
n = len(outcomes)
print(f"{run} {snap or 'model.pt'}: goal {outcomes.count('goal')}/{n}, crash {outcomes.count('crash')}, "
      f"timeout {outcomes.count('timeout')}, near-misses (within 0.6 m, not finished) {near_miss}")
print(f"  steering response to goal direction within 2 m (about 1.0 = steers at the goal, 0 = ignores it): "
      f"goal visible {np.mean(resp[True]):+.2f} (n={len(resp[True])}), "
      f"not visible {np.mean(resp[False]) if resp[False] else float('nan'):+.2f} (n={len(resp[False])})")
