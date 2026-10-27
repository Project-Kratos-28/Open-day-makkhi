"""
Real-maze task: corridors and dead ends on a square grid of cells, built
inside the same 5 m x 5 m arena as the obstacle task, with the same drone,
sensors, observation and action (so obstacle-trained networks can be
fine-tuned on it directly).

Every episode a new maze is generated from a seed, with a random
difficulty and a random "style":
    size      cells per side (bigger = narrower corridors, longer routes)
    algorithm recursive backtracker (long winding corridors, few branches),
              Kruskal or Prim (many short dead ends)
    braid     fraction of dead ends opened up into loops (0 = exactly one route)

The drone starts in one corner cell and the goal is the centre of the
opposite corner cell. It still observes the straight-line direction to the
goal, which in a maze often points through a wall, so it has to learn when
to go *away* from the goal.

Reward: progress along the shortest route through the maze (not straight-
line distance), minus a small time cost; +10 for the goal, -10 for a crash.
The route distance comes from a smooth distance map: shortest path on a
5 cm grid of free space, computed once per maze, read with bilinear
interpolation (a cell-by-cell route distance jumped at corners).

Run directly for a smoke test: python src/grid_maze_env.py [--gui]
"""
import sys
from collections import deque

import numpy as np
import pybullet as p
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from maze_env import (ARENA_HALF, FLY_HEIGHT, GOAL_RADIUS, MAX_SPEED, N_RAYS, WALL_HEIGHT,
                      MazeAviary)

ALGORITHMS = ("backtracker", "kruskal", "prim")
WALL_HALF_THICK = 0.04
# Training mixes these at random; the test also uses a size never trained on.
TRAIN_SIZES = (3, 4, 5, 6)
TRAIN_BRAID = (0.0, 0.3)
# Trap practice (training only): this fraction of training mazes starts the drone
# at the end of a dead end, facing its end wall, as if it had just flown in.
# Dead ends that point towards the goal are preferred: that is where the goal
# direction pulls the drone back in. Practice and test mazes always start in
# the corner (trap_frac 0), so they are unchanged.
TRAIN_TRAP_FRAC = 0.3
MAX_EPISODE_SEC = 90
GRID_RES = 0.05          # m, resolution of the route-distance map
DRONE_RADIUS = 0.06      # m, Crazyflie collision radius
ROUTE_MARGIN = 0.10      # m, extra clearance the shaped route keeps from walls


# ---- maze generation (pure numpy, no simulator) -------------------------
# A maze is the set of open passages between neighbouring cells (r, c).

def _neighbours(cell, n):
    r, c = cell
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if 0 <= r + dr < n and 0 <= c + dc < n:
            yield (r + dr, c + dc)


def _backtracker(n, rng):
    open_ = set()
    stack, seen = [(0, 0)], {(0, 0)}
    while stack:
        cell = stack[-1]
        options = [nb for nb in _neighbours(cell, n) if nb not in seen]
        if not options:
            stack.pop()
            continue
        nb = options[rng.integers(len(options))]
        open_.add(frozenset((cell, nb)))
        seen.add(nb)
        stack.append(nb)
    return open_


def _kruskal(n, rng):
    parent = {(r, c): (r, c) for r in range(n) for c in range(n)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    edges = [((r, c), (r + dr, c + dc)) for r in range(n) for c in range(n)
             for dr, dc in ((1, 0), (0, 1)) if r + dr < n and c + dc < n]
    open_ = set()
    for k in rng.permutation(len(edges)):
        a, b = edges[k]
        if find(a) != find(b):
            parent[find(a)] = find(b)
            open_.add(frozenset((a, b)))
    return open_


def _prim(n, rng):
    open_, inside = set(), {(0, 0)}
    frontier = [((0, 0), nb) for nb in _neighbours((0, 0), n)]
    while frontier:
        a, b = frontier.pop(rng.integers(len(frontier)))
        if b in inside:
            continue
        open_.add(frozenset((a, b)))
        inside.add(b)
        frontier += [(b, nb) for nb in _neighbours(b, n) if nb not in inside]
    return open_


def _braid(open_, n, frac, rng):
    """Open a wall at a fraction `frac` of dead ends, creating loops."""
    cells = [(r, c) for r in range(n) for c in range(n)]
    for k in rng.permutation(len(cells)):
        cell = cells[k]
        links = [nb for nb in _neighbours(cell, n) if frozenset((cell, nb)) in open_]
        if len(links) == 1 and rng.random() < frac:
            closed = [nb for nb in _neighbours(cell, n) if frozenset((cell, nb)) not in open_]
            open_.add(frozenset((cell, closed[rng.integers(len(closed))])))
    return open_


def generate_maze(n, algorithm, braid, rng):
    open_ = {"backtracker": _backtracker, "kruskal": _kruskal, "prim": _prim}[algorithm](n, rng)
    return _braid(open_, n, braid, rng)


def route_distance_map(open_, n, goal_xy, res=GRID_RES, half=ARENA_HALF):
    """
    Shortest-path distance (m) to goal_xy through free space, on a res-sized grid
    covering the arena (8-connected, diagonal steps cost sqrt 2). Grid points
    inside a wall get the value of their nearest free neighbour, so the map can
    be read anywhere. Returns an array indexed [y_index, x_index].
    """
    m = int(round(2 * half / res)) + 1
    coords = -half + np.arange(m) * res
    cell = 2 * half / n
    # A grid point is blocked if it lies on a closed wall between two cells.
    x, y = np.meshgrid(coords, coords)          # [iy, ix]
    fx, fy = (x + half) / cell, (y + half) / cell
    # Walls are widened by the drone's radius plus a safety margin (otherwise the
    # shortest route, and so the reward, pulls the drone onto wall corners with
    # zero clearance), and are always several grid points thick (so paths can't
    # slip through).
    band = WALL_HALF_THICK + DRONE_RADIUS + ROUTE_MARGIN
    near_vx = np.abs(fx - np.round(fx)) * cell < band
    near_hy = np.abs(fy - np.round(fy)) * cell < band
    blocked = np.zeros((m, m), dtype=bool)
    # Each wall also extends `band` past its ends (the physical wall overhangs the
    # cell corner by its half-thickness, plus the drone's radius).
    for r in range(n):
        for c in range(n):
            if c + 1 < n and frozenset(((r, c), (r, c + 1))) not in open_:
                along = (y + half > r * cell - band) & (y + half < (r + 1) * cell + band)
                blocked |= near_vx & (np.round(fx) == c + 1) & along
            if r + 1 < n and frozenset(((r, c), (r + 1, c))) not in open_:
                along = (x + half > c * cell - band) & (x + half < (c + 1) * cell + band)
                blocked |= near_hy & (np.round(fy) == r + 1) & along
    blocked |= (np.abs(x) > half - band) | (np.abs(y) > half - band)

    idx = np.arange(m * m).reshape(m, m)
    free = ~blocked
    rows, cols, w = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        # Edge from grid point (iy, ix) to (iy+dy, ix+dx), for every point where both exist.
        y0, y1 = 0, m - dy
        x0, x1 = max(0, -dx), m - max(0, dx)
        a = (slice(y0, y1), slice(x0, x1))
        b = (slice(y0 + dy, y1 + dy), slice(x0 + dx, x1 + dx))
        ok = free[a] & free[b]
        if dy and dx:   # diagonal: no cutting corners
            ok &= free[(slice(y0 + dy, y1 + dy), slice(x0, x1))] & free[(slice(y0, y1), slice(x0 + dx, x1 + dx))]
        rows.append(idx[a][ok]); cols.append(idx[b][ok])
        w.append(np.full(ok.sum(), res * np.hypot(dx, dy)))
    graph = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(m * m, m * m)).tocsr()
    gi = int(round((goal_xy[1] + half) / res)) * m + int(round((goal_xy[0] + half) / res))
    dist = dijkstra(graph, directed=False, indices=gi).reshape(m, m)
    # Points inside the widened walls get the value of the nearest free point plus
    # the distance to it, so near walls the map rises away from the free route
    # (gently pushing the drone off walls) instead of pulling it onto wall ends.
    reach = np.isfinite(dist)
    near, (iy, ix) = distance_transform_edt(~reach, sampling=res, return_indices=True)
    dist = np.where(reach, dist, dist[iy, ix] + near)
    return np.where(np.isfinite(dist), dist, dist[np.isfinite(dist)].max())


def cell_distances(open_, n, goal):
    """Number of cells along the shortest route from every cell to `goal`."""
    dist = {goal: 0}
    queue = deque([goal])
    while queue:
        cell = queue.popleft()
        for nb in _neighbours(cell, n):
            if nb not in dist and frozenset((cell, nb)) in open_:
                dist[nb] = dist[cell] + 1
                queue.append(nb)
    return dist


# ---- simulator env --------------------------------------------------------

class GridMazeAviary(MazeAviary):
    """
    sizes, algorithms, braid_range: what each new maze is drawn from. Pass a
    single size / algorithm and braid_range=(b, b) for a fixed difficulty.
    """

    def __init__(self, sizes=TRAIN_SIZES, algorithms=ALGORITHMS, braid_range=TRAIN_BRAID,
                 gui=False, ctrl_freq=30, trap_frac=0.0):
        self.sizes, self.algorithms, self.braid_range = sizes, algorithms, braid_range
        self.trap_frac = trap_frac
        self._maze_rng = np.random.default_rng()
        self._new_maze()   # the base __init__ builds the world once
        super().__init__(n_pillars=0, n_walls=0, gui=gui, ctrl_freq=ctrl_freq)
        self._set_start_goal()

    def _new_maze(self):
        rng = self._maze_rng
        self.n = int(rng.choice(self.sizes))
        self.algorithm = str(rng.choice(self.algorithms))
        self.braid = float(rng.uniform(*self.braid_range))
        self.open = generate_maze(self.n, self.algorithm, self.braid, rng)
        self.cell = 2 * ARENA_HALF / self.n
        self.goal_cell = (self.n - 1, self.n - 1)
        self.dist = cell_distances(self.open, self.n, self.goal_cell)
        self.route_map = route_distance_map(self.open, self.n, self._center(self.goal_cell))
        self.start_cell, self.trap_exit = (0, 0), None
        # (no extra random draws when trap_frac is 0, so practice/test mazes stay the same)
        if self.trap_frac > 0 and rng.random() < self.trap_frac:
            self._pick_trap(rng)
        # Centre-line route length (cell to cell), used for the time budget.
        self.route_m = self.dist[self.start_cell] * self.cell

    def _pick_trap(self, rng):
        """Start at a dead end, one of the 3 most misleading ones: those whose way out is
        longest compared with their straight-line distance to the goal."""
        n, gr, gc = self.n, *self.goal_cell
        ends = []
        for cell in ((r, c) for r in range(n) for c in range(n)):
            links = [nb for nb in _neighbours(cell, n) if frozenset((cell, nb)) in self.open]
            if len(links) == 1 and cell not in ((0, 0), self.goal_cell):
                ends.append((self.dist[cell] - np.hypot(cell[0] - gr, cell[1] - gc), cell, links[0]))
        if ends:
            ends.sort(reverse=True)
            _, self.start_cell, self.trap_exit = ends[rng.integers(min(3, len(ends)))]

    def _center(self, cell):
        r, c = cell
        return np.array([-ARENA_HALF + (c + 0.5) * self.cell, -ARENA_HALF + (r + 0.5) * self.cell])

    def _set_start_goal(self):
        self.start = self._center(self.start_cell)
        self.goal = self._center(self.goal_cell)
        self.INIT_XYZS = np.array([[*self.start, FLY_HEIGHT]])
        # Budget: the shortest route at 25% of top speed, plus 10 s, leaving
        # time to explore a few dead ends (a scripted route-follower needs ~40%).
        self.max_steps = int(min(10 + self.route_m / (0.25 * MAX_SPEED), MAX_EPISODE_SEC) / self.dt)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._maze_rng = np.random.default_rng(seed)
        self._new_maze()
        self._set_start_goal()
        return super().reset(seed=seed, options=options)

    def _initial_heading(self):
        """Face along a random open passage out of the start cell (facing the goal
        diagonal pointed at the start cell's inner corner: early crashes in small cells).
        In a trap start, face the dead end's end wall instead (away from its only exit)."""
        if self.trap_exit is not None:
            d = self.start - self._center(self.trap_exit)
            return np.arctan2(d[1], d[0]) + self._maze_rng.uniform(-0.3, 0.3)
        exits = [nb for nb in _neighbours((0, 0), self.n) if frozenset(((0, 0), nb)) in self.open]
        d = self._center(exits[self._maze_rng.integers(len(exits))]) - self.start
        return np.arctan2(d[1], d[0]) + self._maze_rng.uniform(-0.2, 0.2)

    def _addObstacles(self):
        self.obstacle_ids = []
        h, a, t = WALL_HEIGHT / 2, ARENA_HALF, WALL_HALF_THICK
        grey, brown = [0.5, 0.5, 0.5, 1], [0.8, 0.4, 0.2, 1]
        for cx, cy, hx, hy in [(0, a, a, t), (0, -a, a, t), (a, 0, t, a), (-a, 0, t, a)]:
            self._box([cx, cy, h], [hx, hy, h], grey)
        # Interior walls: one box per closed passage between neighbouring cells.
        half = self.cell / 2 + t
        for r in range(self.n):
            for c in range(self.n):
                x, y = self._center((r, c))
                if c + 1 < self.n and frozenset(((r, c), (r, c + 1))) not in self.open:
                    self._box([x + self.cell / 2, y, h], [t, half, h], brown)
                if r + 1 < self.n and frozenset(((r, c), (r + 1, c))) not in self.open:
                    self._box([x, y + self.cell / 2, h], [half, t, h], brown)
        if self.GUI:
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=GOAL_RADIUS, rgbaColor=[0, 1, 0, 0.4],
                                      physicsClientId=self.CLIENT)
            p.createMultiBody(baseVisualShapeIndex=vis, basePosition=[*self.goal, FLY_HEIGHT],
                              physicsClientId=self.CLIENT)

    def _route_dist(self, xy):
        """Smooth shortest-route distance (m) from point xy to the goal (bilinear read of route_map)."""
        ij = (np.asarray(xy, dtype=float)[::-1] + ARENA_HALF) / GRID_RES
        return float(map_coordinates(self.route_map, ij.reshape(2, 1), order=1, mode="nearest")[0])

    def _goal_dist(self):
        """Distance to the goal along the maze's shortest route (used for the progress reward).
        Close to the goal this is ~ the straight-line distance, so the goal test still works."""
        if not hasattr(self, "route_map"):
            return 0.0
        xy = self.pos[0, :2]
        if np.linalg.norm(self.goal - xy) < GOAL_RADIUS:
            return float(np.linalg.norm(self.goal - xy))
        return self._route_dist(xy)

    def _computeObs(self):
        # Same observation as the obstacle task, with the straight-line goal
        # distance (the base uses _goal_dist, which here is the route distance).
        obs = super()._computeObs()
        if hasattr(self, "heading"):
            obs[N_RAYS + 2] = np.linalg.norm(self.goal - self.pos[0, :2]) / (2 * ARENA_HALF)
        return obs

    def _computeInfo(self):
        return {"outcome": self.outcome, "maze_size": self.n}


def make_grid_env(seed, sizes=TRAIN_SIZES, algorithms=ALGORITHMS, braid_range=TRAIN_BRAID):
    """Builds one training grid-maze env (see maze_env.make_env for why this lives here)."""
    env = GridMazeAviary(sizes=sizes, algorithms=algorithms, braid_range=braid_range,
                         trap_frac=TRAIN_TRAP_FRAC)
    env.reset(seed=seed)
    return env


if __name__ == "__main__":
    import time
    gui = "--gui" in sys.argv
    # Check every generator gives a connected maze.
    rng = np.random.default_rng(0)
    for alg in ALGORITHMS:
        for n in (3, 6, 8):
            m = generate_maze(n, alg, 0.0, rng)
            assert len(cell_distances(m, n, (n - 1, n - 1))) == n * n
            assert len(m) == n * n - 1, "a perfect maze has exactly one route"
    print("generators OK: all mazes connected, perfect mazes are trees")

    def route_follower(env):
        """Scripted pilot that knows the maze: fly cell centre to cell centre along the route."""
        xy = env.pos[0, :2]
        r, c = np.clip(((xy + ARENA_HALF) // env.cell).astype(int)[::-1], 0, env.n - 1)
        cell = (int(r), int(c))
        if cell == env.goal_cell:
            target = env.goal
        else:
            nxt = min((nb for nb in _neighbours(cell, env.n) if frozenset((cell, nb)) in env.open),
                      key=lambda nb: env.dist[nb])
            target, centre = env._center(nxt), env._center(cell)
            axis = 0 if nxt[0] != cell[0] else 1     # line up with the corridor first
            if abs((xy - centre)[axis]) > 0.12 * env.cell:
                target = centre
        b = np.arctan2(*(target - xy)[::-1]) - env.heading
        b = (b + np.pi) % (2 * np.pi) - np.pi
        return np.array([np.clip(3 * b, -1, 1), 0.2 if abs(b) < 0.3 else -1.0])

    t0, steps = time.time(), 0
    for sizes in ((3,), (6,), (7,), TRAIN_SIZES):
        env = GridMazeAviary(sizes=sizes, gui=gui)
        outcomes, returns, used = [], [], []
        for ep in range(8):
            obs, _ = env.reset(seed=100 * sizes[0] + ep)
            done, ret = False, 0.0
            while not done:
                obs, r, term, trunc, info = env.step(route_follower(env))
                assert np.isfinite(obs).all() and np.isfinite(r)
                ret += r
                steps += 1
                done = term or trunc
                if gui:
                    time.sleep(env.dt)
            outcomes.append(info["outcome"])
            returns.append(ret)
            used.append(env.env_steps / env.max_steps)
        print(f"sizes {sizes}: route-follower {outcomes.count('goal')}/8 goals, mean return "
              f"{np.mean(returns):.1f}, time used {np.mean(used):.0%} of budget")
        env.close()
    print(f"{steps / (time.time() - t0):.0f} env steps/sec")
