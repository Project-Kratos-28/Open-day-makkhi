"""
Watch a trained network fly the maze in the 3D simulator window.

    python src/watch.py conn_s0                 # 5 episodes, real-time, GUI
    python src/watch.py base_s0 --episodes 10 --seed 123
    python src/watch.py conn_s0 --no-gui --episodes 50   # fast, just print the success rate
    python src/watch.py maze_conn_s0 --size 5   # a grid-maze run, on 5x5 mazes

Loads runs/<name>/model.pt (saved every few minutes during training, so it
also works on a run that's still going), or model_best.pt for a grid-maze run
once select_best.py has picked it. The drone uses its best-guess
action (no random exploration), and gets a fresh random maze each episode.
"""
import argparse
import time

import pybullet as p
import torch

from evaluate import load_agent
from grid_maze_env import ALGORITHMS, TRAIN_BRAID, TRAIN_SIZES, GridMazeAviary
from maze_env import ARENA_HALF, MazeAviary
from ppo import RUNS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run name, e.g. conn_s0 (a folder in runs/)")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1000, help="maze layouts; same seed = same mazes")
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed in the GUI (2 = twice as fast)")
    ap.add_argument("--size", type=int, help="grid-maze runs: maze size (cells per side); default the training mix")
    ap.add_argument("--width", type=int, default=1920, help="native PyBullet window width")
    ap.add_argument("--height", type=int, default=1080, help="native PyBullet window height")
    args = ap.parse_args()

    if not args.no_gui:
        original_connect = p.connect
        width = max(640, args.width)
        height = max(360, args.height)

        def native_gui_connect(mode, key=0, options="", **kwargs):
            if mode == p.GUI:
                options = f"--width={width} --height={height}"
            return original_connect(mode, key=key, options=options, **kwargs)

        p.connect = native_gui_connect

    best = RUNS_DIR / args.run / "model_best.pt"
    path = best if best.exists() else RUNS_DIR / args.run / "model.pt"
    agent, train_args = load_agent(path)
    update = torch.load(path, map_location="cpu", weights_only=False)["update"]
    print(f"{args.run}: {train_args['model']} network, {path.name} from update {update} "
          f"(~{update * train_args['n_envs'] * train_args['rollout_len']:,} training steps)")

    if train_args.get("task") == "grid":
        env = GridMazeAviary(sizes=(args.size,) if args.size else TRAIN_SIZES, algorithms=ALGORITHMS,
                             braid_range=TRAIN_BRAID, gui=not args.no_gui)
    else:
        env = MazeAviary(n_pillars=train_args["n_pillars"], n_walls=train_args["n_walls"], gui=not args.no_gui)
    outcomes = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        if not args.no_gui:
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=env.CLIENT)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=env.CLIENT)
            p.resetDebugVisualizerCamera(cameraDistance=2.6 * ARENA_HALF, cameraYaw=0, cameraPitch=-65,
                                         cameraTargetPosition=[0, 0, 0], physicsClientId=env.CLIENT)
        h = agent.core.initial_state(1)
        done = False
        while not done:
            t0 = time.time()
            with torch.no_grad():
                action, h = agent.act(torch.as_tensor(obs).unsqueeze(0), h)
            obs, _, term, trunc, info = env.step(action[0].numpy())
            done = term or trunc
            if not args.no_gui:
                time.sleep(max(0.0, env.dt / args.speed - (time.time() - t0)))
        outcomes.append(info["outcome"])
        print(f"episode {ep + 1}: {info['outcome']:7s} after {env.env_steps * env.dt:4.1f} s")
        if not args.no_gui:
            time.sleep(1.0)
    n = len(outcomes)
    print(f"reached goal {outcomes.count('goal')}/{n}, crashed {outcomes.count('crash')}/{n}, "
          f"timed out {outcomes.count('timeout')}/{n}")
    env.close()


if __name__ == "__main__":
    main()
