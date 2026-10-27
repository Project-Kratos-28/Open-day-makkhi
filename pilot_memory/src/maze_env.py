"""
Obstacle-maze navigation task for a simulated Crazyflie quadcopter, built on
gym-pybullet-drones.

The drone flies at a fixed height through a walled arena of random pillars
and wall segments, from a start corner to a goal in the opposite corner.
A new random layout is generated every episode.

Action (2 numbers in [-1, 1]), the same kind of command the fly's PFL
neurons produce:
    turn     -> turning rate (left/right)
    forward  -> forward speed (0 .. MAX_SPEED)
The built-in PID controller turns these into rotor speeds, so the network
learns navigation, not low-level stabilisation.

Observation (all egocentric, i.e. relative to the drone's heading):
    N_RAYS lidar readings around the drone (0 = nothing in range, 1 = touching)
    sin/cos of the goal's bearing, distance to goal, goal in line of sight (1/0)
    forward and sideways speed, last turn command, height error

Run this file directly for a scripted smoke test: python src/maze_env.py [--gui]
"""
import sys

import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

ARENA_HALF = 2.5        # arena is 5 m x 5 m
WALL_HEIGHT = 1.2
FLY_HEIGHT = 0.5
N_RAYS = 32             # 11.25 deg apart; with 16 (22.5 deg) thin wall ends slipped between rays
RAY_RANGE = 1.5
MAX_SPEED = 1.0         # m/s
MAX_TURN_RATE = 2.0     # rad/s
GOAL_RADIUS = 0.4       # was 0.3: at full speed the drone's turning circle (~0.5 m) made it orbit just outside
EPISODE_SEC = 20
CLEARANCE = 0.6         # keep obstacles this far from start and goal

# Reward per control step: PROGRESS_REWARD per metre of progress towards the goal,
# minus TIME_COST, plus SPEED_BONUS x (progress speed / MAX_SPEED), where progress
# speed only counts movement towards the goal (along the route in mazes). The
# bonus stops "creeping" from paying, earns nothing for circling, and is smaller
# than the time cost, so it can never be farmed. With gamma 0.99 the discounted future time cost is at most
# TIME_COST / 0.01 = 3, below the crash penalty, so crashing on purpose never pays.
PROGRESS_REWARD = 5.0
TIME_COST = 0.03
SPEED_BONUS = 0.01
GOAL_REWARD = 10.0
CRASH_PENALTY = 10.0

OBS_DIM = N_RAYS + 4 + 4
ACT_DIM = 2


class MazeAviary(BaseRLAviary):

    def __init__(self, n_pillars=8, n_walls=2, gui=False, ctrl_freq=30):
        self.n_pillars = n_pillars
        self.n_walls = n_walls
        self.start = np.array([-ARENA_HALF + 0.5, -ARENA_HALF + 0.5])
        self.goal = -self.start
        self._layout_rng = np.random.default_rng()
        super().__init__(initial_xyzs=np.array([[*self.start, FLY_HEIGHT]]),
                         ctrl_freq=ctrl_freq, gui=gui,
                         obs=ObservationType.KIN, act=ActionType.VEL)
        self.SPEED_LIMIT = MAX_SPEED
        self.dt = 1.0 / ctrl_freq
        self.max_steps = int(EPISODE_SEC * ctrl_freq)

    # ---- spaces --------------------------------------------------------

    def _actionSpace(self):
        return spaces.Box(-1.0, 1.0, (ACT_DIM,), dtype=np.float32)

    def _observationSpace(self):
        return spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)

    # ---- episode -------------------------------------------------------

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._layout_rng = np.random.default_rng(seed)
        self.heading = self._initial_heading()
        self.turn_cmd = 0.0
        self.env_steps = 0
        self.outcome = None
        # Clear the autopilot's memory (last attitude, integral terms), so an
        # episode doesn't depend on how the previous one ended.
        if hasattr(self, "ctrl"):
            for c in self.ctrl:
                c.reset()
        obs, info = super().reset(seed=seed, options=options)
        self.prev_dist = self._goal_dist()
        return obs, info

    def _initial_heading(self):
        """Start facing roughly towards the goal, with some randomness."""
        to_goal = self.goal - self.start
        return np.arctan2(to_goal[1], to_goal[0]) + self._layout_rng.uniform(-0.5, 0.5)

    def _addObstacles(self):
        self.obstacle_ids = []
        h = WALL_HEIGHT / 2
        a = ARENA_HALF
        for cx, cy, hx, hy in [(0, a, a, 0.05), (0, -a, a, 0.05), (a, 0, 0.05, a), (-a, 0, 0.05, a)]:
            self._box([cx, cy, h], [hx, hy, h], [0.5, 0.5, 0.5, 1])

        rng = self._layout_rng
        placed = 0
        while placed < self.n_walls:
            c = rng.uniform(-a + 1, a - 1, 2)
            if self._too_close(c, 1.0):
                continue
            half_len = rng.uniform(0.4, 0.9)
            hx, hy = (half_len, 0.05) if rng.random() < 0.5 else (0.05, half_len)
            self._box([*c, h], [hx, hy, h], [0.8, 0.4, 0.2, 1])
            placed += 1
        placed = 0
        while placed < self.n_pillars:
            c = rng.uniform(-a + 0.4, a - 0.4, 2)
            if self._too_close(c, CLEARANCE):
                continue
            r = rng.uniform(0.08, 0.2)
            self._box([*c, h], [r, r, h], [0.2, 0.5, 0.8, 1])
            placed += 1

        if self.GUI:
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=GOAL_RADIUS, rgbaColor=[0, 1, 0, 0.4],
                                      physicsClientId=self.CLIENT)
            p.createMultiBody(baseVisualShapeIndex=vis, basePosition=[*self.goal, FLY_HEIGHT],
                              physicsClientId=self.CLIENT)

    def _too_close(self, c, d):
        return np.linalg.norm(c - self.start) < d or np.linalg.norm(c - self.goal) < d

    def _box(self, pos, half_extents, rgba):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.CLIENT)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba,
                                  physicsClientId=self.CLIENT)
        self.obstacle_ids.append(p.createMultiBody(0, col, vis, pos, physicsClientId=self.CLIENT))

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(ACT_DIM), -1, 1)
        self.turn_cmd = action[0]
        self.heading += self.turn_cmd * MAX_TURN_RATE * self.dt
        speed = (action[1] + 1) / 2 * MAX_SPEED
        vz = np.clip(2.0 * (FLY_HEIGHT - self.pos[0, 2]), -0.5, 0.5)
        v = np.array([speed * np.cos(self.heading), speed * np.sin(self.heading), vz])
        norm = np.linalg.norm(v)
        vel_action = np.zeros((1, 4))
        if norm > 1e-6:
            vel_action[0, :3] = v / norm
            vel_action[0, 3] = min(norm / self.SPEED_LIMIT, 1.0)
        self.env_steps += 1
        return super().step(vel_action)

    # ---- observation / reward ------------------------------------------

    def _goal_dist(self):
        return np.linalg.norm(self.goal - self.pos[0, :2])

    def _lidar(self):
        pos = self.pos[0]
        angles = self.heading + np.arange(N_RAYS) * 2 * np.pi / N_RAYS
        dirs = np.stack([np.cos(angles), np.sin(angles), np.zeros(N_RAYS)], 1)
        # Start just outside the drone's own body (collision radius 0.06 m). Starting
        # further out would put the start inside a wall when the drone is very close
        # to one, and pybullet doesn't report a hit from inside a box.
        starts = pos + 0.065 * dirs
        hits = p.rayTestBatch(list(starts), list(pos + RAY_RANGE * dirs), physicsClientId=self.CLIENT)
        frac = np.array([h[2] if h[0] != -1 else 1.0 for h in hits])
        return 1.0 - frac

    def _computeObs(self):
        if not hasattr(self, "heading"):   # called once inside the base __init__
            return np.zeros(OBS_DIM, dtype=np.float32)
        to_goal = self.goal - self.pos[0, :2]
        bearing = np.arctan2(to_goal[1], to_goal[0]) - self.heading
        c, s = np.cos(self.heading), np.sin(self.heading)
        vx, vy = self.vel[0, :2]
        obs = np.concatenate([
            self._lidar(),
            [np.sin(bearing), np.cos(bearing), self._goal_dist() / (2 * ARENA_HALF), self._goal_visible()],
            [(c * vx + s * vy) / MAX_SPEED, (-s * vx + c * vy) / MAX_SPEED,
             self.turn_cmd, (FLY_HEIGHT - self.pos[0, 2]) / FLY_HEIGHT],
        ])
        return obs.astype(np.float32)

    def _goal_visible(self):
        """1 if there's a clear line of sight from the drone to the goal, else 0. Tells the
        network when the goal's direction can be trusted (in a maze it usually points
        through a wall, so trained networks learned to ignore it, even in the goal's own cell)."""
        pos = self.pos[0]
        target = np.array([*self.goal, pos[2]])
        d = target - pos
        n = np.linalg.norm(d)
        if n < 0.07:
            return 1.0
        hit = p.rayTest(list(pos + 0.065 * d / n), list(target), physicsClientId=self.CLIENT)[0]
        return 1.0 if hit[0] == -1 else 0.0

    def _crashed(self):
        drone = self.DRONE_IDS[0]
        if p.getContactPoints(bodyA=drone, physicsClientId=self.CLIENT):
            return True
        # Only count a real flip as a crash: at 0.8 rad, rapid stop/go commands
        # alone could trip it (autopilot overshoot) with nothing hit.
        tilted = np.abs(self.rpy[0, :2]).max() > 1.2
        outside = np.abs(self.pos[0, :2]).max() > ARENA_HALF or not 0.1 < self.pos[0, 2] < WALL_HEIGHT
        return tilted or outside

    def _computeReward(self):
        if not hasattr(self, "heading"):
            return 0.0
        dist = self._goal_dist()
        progress = self.prev_dist - dist
        speed_frac = np.clip(progress / (MAX_SPEED * self.dt), 0.0, 1.0)
        reward = PROGRESS_REWARD * progress - TIME_COST + SPEED_BONUS * speed_frac
        self.prev_dist = dist
        if dist < GOAL_RADIUS:
            self.outcome = "goal"
            reward += GOAL_REWARD
        elif self._crashed():
            self.outcome = "crash"
            reward -= CRASH_PENALTY
        return reward

    def _computeTerminated(self):
        return self.outcome is not None

    def _computeTruncated(self):
        if self.outcome is None and self.env_steps >= self.max_steps:
            self.outcome = "timeout"
            return True
        return False

    def _computeInfo(self):
        return {"outcome": self.outcome}


def make_env(seed, n_pillars, n_walls):
    """Builds one maze env. Lives here, not in train_ppo.py, so that the env worker
    processes only import this file (pybullet + numpy) and not torch."""
    env = MazeAviary(n_pillars=n_pillars, n_walls=n_walls)
    env.reset(seed=seed)
    return env


if __name__ == "__main__":
    import time
    gui = "--gui" in sys.argv
    env = MazeAviary(gui=gui)
    print("obs space", env.observation_space.shape, "action space", env.action_space.shape)
    rng = np.random.default_rng(0)
    outcomes = []
    t0 = time.time()
    steps = 0
    for ep in range(5):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            # Scripted "bug" policy: steer toward the goal, swerve away from
            # the closest obstacle in front. Just proves the env works.
            front = obs[:N_RAYS][[0, 1, 2, -1, -2]]
            turn = np.clip(2 * obs[N_RAYS], -1, 1)
            if front.max() > 0.5:
                turn = 1.0 if obs[1:4].sum() < obs[N_RAYS - 4:N_RAYS - 1].sum() else -1.0
            obs, r, term, trunc, info = env.step(np.array([turn, 0.3]))
            assert np.isfinite(obs).all()
            steps += 1
            done = term or trunc
            if gui:
                time.sleep(env.dt)
        outcomes.append(info["outcome"])
        print(f"episode {ep}: {info['outcome']} after {env.env_steps} steps, "
              f"final dist to goal {env._goal_dist():.2f} m")
    print(f"{steps / (time.time() - t0):.0f} env steps/sec")
    env.close()
