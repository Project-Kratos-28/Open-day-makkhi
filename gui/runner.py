"""
Process runner for executing simulation runs either embedded in the GUI with live frame streaming,
or as an external PyBullet OS window subprocess.
"""
from dataclasses import dataclass, field
import os
import re
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PIL import Image

# Ensure project root & src are on python path
from gui.config import PROJECT_ROOT, PYTHON_EXE, RUNS_DIR, SRC_DIR, WATCH_SCRIPT
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@dataclass
class EpisodeResult:
    episode: int
    outcome: str  # 'goal', 'crash', 'timeout'
    time_s: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class RunSummary:
    total_episodes: int
    goals: int
    crashes: int
    timeouts: int
    success_rate: float
    avg_time_s: float
    min_time_s: float
    max_time_s: float


EPISODE_RE = re.compile(r"episode\s+(\d+):\s+(\w+)\s+after\s+([\d\.]+)\s*s", re.IGNORECASE)
SUMMARY_RE = re.compile(
    r"reached goal\s+(\d+)/(\d+),\s*crashed\s+(\d+)/\d+,\s*timed out\s+(\d+)/\d+",
    re.IGNORECASE,
)


class SimulationRunner:
    """Supports both direct in-GUI embedded 3D rendering and external subprocess execution."""

    def __init__(
        self,
        on_frame: Optional[Callable[[Image.Image, Dict[str, Any]], None]] = None,
        on_line: Optional[Callable[[str], None]] = None,
        on_episode: Optional[Callable[[EpisodeResult], None]] = None,
        on_summary: Optional[Callable[[RunSummary], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_finished: Optional[Callable[[int], None]] = None,
    ):
        self.on_frame = on_frame
        self.on_line = on_line
        self.on_episode = on_episode
        self.on_summary = on_summary
        self.on_status = on_status
        self.on_finished = on_finished

        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._is_running = False
        self._stop_requested = False
        self._lock = threading.Lock()
        self.episode_results: List[EpisodeResult] = []
        self.camera_mode = "overview"  # "overview" or "chase"

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._is_running

    def set_camera_mode(self, mode: str) -> None:
        self.camera_mode = mode

    def start(
        self,
        run_name: str,
        size: int = 4,
        episodes: int = 5,
        speed: float = 1.0,
        layout_seed: int = 1000,
        embedded: bool = True,
    ) -> List[str]:
        """Starts simulation either embedded inside GUI or in external PyBullet window."""
        with self._lock:
            if self._is_running:
                raise RuntimeError("A simulation run is already in progress.")
            self._is_running = True
            self._stop_requested = False
            self.episode_results.clear()

        cmd = [
            PYTHON_EXE,
            str(WATCH_SCRIPT),
            run_name,
            "--size",
            str(size),
            "--episodes",
            str(episodes),
            "--speed",
            str(speed),
            "--seed",
            str(layout_seed),
        ]

        if embedded:
            # Run in worker thread with direct PyBullet camera frame capture
            if self.on_status:
                self.on_status(f"Starting embedded 3D simulation: {run_name} (Size {size}x{size})...")
            self._thread = threading.Thread(
                target=self._run_embedded,
                args=(run_name, size, episodes, speed, layout_seed),
                daemon=True,
            )
            self._thread.start()
        else:
            # Run in external OS window using subprocess
            if self.on_status:
                self.on_status(f"Launching external PyBullet window: {run_name}...")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            if "DISPLAY" not in env:
                env["DISPLAY"] = ":1"

            self._process = subprocess.Popen(
                [PYTHON_EXE, "-u"] + cmd[1:],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            self._thread = threading.Thread(target=self._read_subprocess_output, daemon=True)
            self._thread.start()

        return cmd

    def stop(self) -> None:
        """Stops the active simulation immediately."""
        with self._lock:
            self._stop_requested = True
            proc = self._process

        if proc:
            try:
                proc.terminate()
            except Exception:
                pass

        if self.on_status:
            self.on_status("Stopping simulation...")

    def _run_embedded(
        self,
        run_name: str,
        size: int,
        episodes: int,
        speed: float,
        layout_seed: int,
    ) -> None:
        """Embedded simulation loop rendering camera frames directly into GUI."""
        import pybullet as p
        import torch
        from evaluate import load_agent
        from grid_maze_env import ALGORITHMS, ARENA_HALF, FLY_HEIGHT, GOAL_RADIUS, TRAIN_BRAID, GridMazeAviary
        from maze_env import MazeAviary

        run_dir = RUNS_DIR / run_name
        best_pt = run_dir / "model_best.pt"
        ckpt_path = best_pt if best_pt.exists() else (run_dir / "model.pt")

        if not ckpt_path.exists():
            if self.on_line:
                self.on_line(f"[Error] Checkpoint file {ckpt_path} not found.")
            with self._lock:
                self._is_running = False
            if self.on_finished:
                self.on_finished(-1)
            return

        try:
            agent, train_args = load_agent(ckpt_path)
            ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            update = ckpt_data.get("update", 0)

            msg = f"{run_name}: {train_args.get('model', 'connectome')} network, {ckpt_path.name} (update {update})"
            if self.on_line:
                self.on_line(msg)

            # Initialize environment in direct headless mode (we render via camera image)
            task = train_args.get("task", "grid")
            if task == "grid":
                env = GridMazeAviary(
                    sizes=(size,),
                    algorithms=ALGORITHMS,
                    braid_range=TRAIN_BRAID,
                    gui=False,
                )
            else:
                env = MazeAviary(
                    n_pillars=train_args.get("n_pillars", 8),
                    n_walls=train_args.get("n_walls", 2),
                    gui=False,
                )

            # Camera render dimensions
            cam_w, cam_h = 480, 340
            render_every_n_steps = 2  # render at ~30 FPS

            for ep in range(episodes):
                if self._stop_requested:
                    break

                obs, _ = env.reset(seed=layout_seed + ep)

                # Add visual glowing goal sphere in DIRECT mode
                vis_goal = p.createVisualShape(
                    p.GEOM_SPHERE,
                    radius=GOAL_RADIUS,
                    rgbaColor=[0, 1, 0, 0.7],
                    physicsClientId=env.CLIENT,
                )
                p.createMultiBody(
                    baseVisualShapeIndex=vis_goal,
                    basePosition=[*env.goal, FLY_HEIGHT],
                    physicsClientId=env.CLIENT,
                )

                h = agent.core.initial_state(1)
                done = False
                step_idx = 0

                while not done and not self._stop_requested:
                    t0 = time.time()

                    with torch.no_grad():
                        action, h = agent.act(torch.as_tensor(obs).unsqueeze(0), h)

                    obs, _, term, trunc, info = env.step(action[0].numpy())
                    done = term or trunc
                    step_idx += 1

                    # Render frame
                    if self.on_frame and (step_idx % render_every_n_steps == 0 or done):
                        drone_pos = env.pos[0]
                        if self.camera_mode == "chase":
                            view_mat = p.computeViewMatrixFromYawPitchRoll(
                                cameraTargetPosition=drone_pos.tolist(),
                                distance=1.8,
                                yaw=float(np.degrees(env.heading)),
                                pitch=-28,
                                roll=0,
                                upAxisIndex=2,
                                physicsClientId=env.CLIENT,
                            )
                        else:  # Overview
                            view_mat = p.computeViewMatrixFromYawPitchRoll(
                                cameraTargetPosition=[0, 0, 0],
                                distance=2.6 * ARENA_HALF,
                                yaw=0,
                                pitch=-65,
                                roll=0,
                                upAxisIndex=2,
                                physicsClientId=env.CLIENT,
                            )

                        proj_mat = p.computeProjectionMatrixFOV(
                            fov=60.0,
                            aspect=float(cam_w) / float(cam_h),
                            nearVal=0.1,
                            farVal=30.0,
                            physicsClientId=env.CLIENT,
                        )

                        cam_img = p.getCameraImage(
                            cam_w,
                            cam_h,
                            view_mat,
                            proj_mat,
                            renderer=p.ER_TINY_RENDERER,
                            physicsClientId=env.CLIENT,
                        )
                        rgba_data = np.reshape(cam_img[2], (cam_h, cam_w, 4)).astype(np.uint8)
                        pil_img = Image.fromarray(rgba_data)

                        cur_spd = float(np.linalg.norm(env.vel[0])) if hasattr(env, "vel") else 0.0
                        outcome_status = info.get("outcome") if info else None
                        telemetry = {
                            "episode": ep + 1,
                            "time_s": env.env_steps * env.dt,
                            "pos": [float(drone_pos[0]), float(drone_pos[1]), float(drone_pos[2])],
                            "speed": cur_spd,
                            "status": outcome_status or "FLYING",
                        }
                        self.on_frame(pil_img, telemetry)

                    # Throttle simulation speed
                    elapsed = time.time() - t0
                    time.sleep(max(0.001, (env.dt / max(0.1, speed)) - elapsed))

                outcome = info.get("outcome", "timeout")
                flight_time = env.env_steps * env.dt
                res = EpisodeResult(episode=ep + 1, outcome=outcome, time_s=flight_time)
                self.episode_results.append(res)

                if self.on_episode:
                    self.on_episode(res)
                if self.on_line:
                    self.on_line(f"episode {ep + 1}: {outcome:7s} after {flight_time:4.1f} s")

                self._update_live_summary()
                time.sleep(0.4)

            env.close()

            # Final summary log line
            n = len(self.episode_results)
            g = sum(1 for r in self.episode_results if r.outcome == "goal")
            c = sum(1 for r in self.episode_results if r.outcome == "crash")
            t = sum(1 for r in self.episode_results if r.outcome == "timeout")
            sum_line = f"reached goal {g}/{n}, crashed {c}/{n}, timed out {t}/{n}"
            if self.on_line:
                self.on_line(sum_line)

        except Exception as e:
            if self.on_line:
                self.on_line(f"[Simulation Error]: {e}")
        finally:
            with self._lock:
                self._is_running = False
            if self.on_status:
                self.on_status("Simulation finished.")
            if self.on_finished:
                self.on_finished(0)

    def _read_subprocess_output(self) -> None:
        """Reads output when running in external window mode."""
        proc = self._process
        if not proc or not proc.stdout:
            return

        exit_code = 0
        try:
            for line in iter(proc.stdout.readline, ""):
                line_clean = line.rstrip("\r\n")
                if not line_clean:
                    continue

                if self.on_line:
                    self.on_line(line_clean)

                ep_match = EPISODE_RE.search(line_clean)
                if ep_match:
                    res = EpisodeResult(
                        episode=int(ep_match.group(1)),
                        outcome=ep_match.group(2).lower(),
                        time_s=float(ep_match.group(3)),
                    )
                    self.episode_results.append(res)
                    if self.on_episode:
                        self.on_episode(res)
                    self._update_live_summary()

                sum_match = SUMMARY_RE.search(line_clean)
                if sum_match:
                    self._emit_final_summary(
                        int(sum_match.group(2)),
                        int(sum_match.group(1)),
                        int(sum_match.group(3)),
                        int(sum_match.group(4)),
                    )

            proc.stdout.close()
            exit_code = proc.wait()
        except Exception as e:
            if self.on_line:
                self.on_line(f"[Process Error]: {e}")
            exit_code = -1
        finally:
            with self._lock:
                self._is_running = False
                self._process = None
            if self.on_status:
                self.on_status("Completed." if exit_code == 0 else f"Stopped ({exit_code}).")
            if self.on_finished:
                self.on_finished(exit_code)

    def _update_live_summary(self) -> None:
        if not self.episode_results or not self.on_summary:
            return
        total = len(self.episode_results)
        goals = sum(1 for r in self.episode_results if r.outcome == "goal")
        crashes = sum(1 for r in self.episode_results if r.outcome == "crash")
        timeouts = sum(1 for r in self.episode_results if r.outcome == "timeout")
        goal_times = [r.time_s for r in self.episode_results if r.outcome == "goal"]
        all_times = [r.time_s for r in self.episode_results]

        avg_t = sum(goal_times) / len(goal_times) if goal_times else (sum(all_times) / total if all_times else 0.0)
        min_t = min(goal_times) if goal_times else (min(all_times) if all_times else 0.0)
        max_t = max(goal_times) if goal_times else (max(all_times) if all_times else 0.0)

        self.on_summary(
            RunSummary(
                total_episodes=total,
                goals=goals,
                crashes=crashes,
                timeouts=timeouts,
                success_rate=(goals / total) * 100.0 if total > 0 else 0.0,
                avg_time_s=avg_t,
                min_time_s=min_t,
                max_time_s=max_t,
            )
        )

    def _emit_final_summary(self, total: int, goals: int, crashes: int, timeouts: int) -> None:
        if not self.on_summary:
            return
        goal_times = [r.time_s for r in self.episode_results if r.outcome == "goal"]
        all_times = [r.time_s for r in self.episode_results]
        avg_t = sum(goal_times) / len(goal_times) if goal_times else (sum(all_times) / total if all_times else 0.0)
        min_t = min(goal_times) if goal_times else (min(all_times) if all_times else 0.0)
        max_t = max(goal_times) if goal_times else (max(all_times) if all_times else 0.0)

        self.on_summary(
            RunSummary(
                total_episodes=total,
                goals=goals,
                crashes=crashes,
                timeouts=timeouts,
                success_rate=(goals / total) * 100.0 if total > 0 else 0.0,
                avg_time_s=avg_t,
                min_time_s=min_t,
                max_time_s=max_t,
            )
        )
