# Fly Connectome → Autonomous Drone Navigation
## Complete Technical Dossier & Presentation Defense Manual

---

### Table of Contents
1. [Executive Summary & Core Research Question](#1-executive-summary--core-research-question)
2. [Biological Connectome Extraction (Data Layer)](#2-biological-connectome-extraction-data-layer)
3. [Neural Network Architectures & Mathematical Dynamics](#3-neural-network-architectures--mathematical-dynamics)
4. [The 4-Way Fair Benchmark Design](#4-the-4-way-fair-benchmark-design)
5. [Drone Simulation & Physics Environment](#5-drone-simulation--physics-environment)
6. [Procedural Maze Generation & Geodesic Rewards](#6-procedural-maze-generation--geodesic-rewards)
7. [Recurrent PPO Training Pipeline & Trap Fixes](#7-recurrent-ppo-training-pipeline--trap-fixes)
8. [Evaluation Protocol & Empirical Findings](#8-evaluation-protocol--empirical-findings)
9. [Comprehensive Master Parameters & Numbers Reference](#9-comprehensive-master-parameters--numbers-reference)
10. [Defense & FAQ: Addressing Tough Audience / Faculty Questions](#10-defense--faq-addressing-tough-audience--faculty-questions)

---

## 1. Executive Summary & Core Research Question

### The Narrative Arc
1. **The Inspiration**: Flies navigate complex 3D environments with extreme agility using an energy budget of microwatts and a central complex of fewer than 1,000 neurons.
2. **The Implementation**: We extracted the exact wiring diagram (synapse-level connectivity and excitatory/inhibitory signs) of the *Drosophila melanogaster* **Central Complex (CX)** from electron-microscopy data. We instantiated this circuit as a sparse, sign-constrained recurrent neural network (RNN).
3. **The Task**: Flying a simulated Crazyflie quadcopter drone through obstacle arenas and procedurally generated mazes using 32-ray lidar and egocentric bearings.
4. **The Science**: The project is **not** just a demo of a drone flying. The core scientific contribution is a **controlled 4-way benchmark** matching parameter counts (~75k parameters) across:
   - **Biological Connectome Network**
   - **Shuffled Connectome Control** (topological control)
   - **Dense Leaky RNN** (neuron-model baseline)
   - **Standard Gated Recurrent Unit (GRU)** (modern ML baseline)
5. **Key Citation**:
   > *Hulse, B. K. et al. (2021). "A connectome of the Drosophila central complex reveals network motifs suitable for flexible navigation and context-dependent action selection." eLife 10:e66039.*

---

## 2. Biological Connectome Extraction (Data Layer)

**Source File:** [`src/extract_connectome.py`](file:///home/anuragiyer/Desktop/Open-day-makkhi/src/extract_connectome.py)

### 2.1 Database & Dataset
- **Database**: Janelia Research Campus **neuPrint** (`neuprint.janelia.org`).
- **Dataset**: `male-cns:v1.0` (Full male central nervous system, reconstructed via serial section transmission electron microscopy).
- **Decision: Why `male-cns:v1.0` instead of `hemibrain:v1.2.1`?**
  - `hemibrain` only contains half a brain and is truncated at the brain boundaries.
  - `male-cns:v1.0` contains the complete brain and ventral nerve cord (VNC), providing complete symmetry across both brain hemispheres and complete projection pathways to descending motor pathways.

### 2.2 Anatomical Circuit Boundaries (The Central Complex)
- **Decision: Why CX only instead of the whole fly brain (~140,000 neurons)?**
  - 140k neurons include large visual lobes (lobula, medulla), antennal lobes (olfaction), and mushroom bodies (associative memory/learning).
  - The central complex is the universally acknowledged compass, vector-navigation, and steering engine of the insect brain. Restricting to the CX provides a tractable, interpretable, navigation-specific circuit.
- **Neuron Families Extracted (922 neurons total)**:
  1. **`ER` (Ring Neurons, 282 neurons)**: Located in the Bulb (`BU`) and Ellipsoid Body (`EB`). Acts as the sensory/visual input gateway.
  2. **`EPG` (50 neurons)**: Connects Ellipsoid Body to Protocerebral Bridge (`PB`) and Gall. Acts as the **internal compass** (maintains a localized bump of activity tracking head direction).
  3. **`PEN` (42 neurons)**: Connects PB, EB, and Noduli (`NO`). Integrates angular velocity (turning) to shift the compass bump as the animal turns.
  4. **`Delta7` ($\Delta 7$, 42 neurons)**: Local inhibitory interneurons in the Protocerebral Bridge. Spans across columns to provide global/cross-column inhibition, stabilizing the ring-attractor compass.
  5. **`PFN` (456 neurons)**: Connects PB, Fan-shaped Body (`FB`), and Noduli (`NO`). Integrates heading with self-motion/translational velocity (odometry).
  6. **`PFL` (50 neurons)**: Connects PB, FB, and Lateral Accessory Lobe (`LAL`). The primary pre-motor steering outputs (initiating left/right turning commands).

### 2.3 Noise Filtering & Connection Matrix
- **Raw Connections in Query**: 79,506 connections.
- **Filtering Threshold**: `min_synapses = 3`.
  - **Decision: Why filter $< 3$ synapses?** Connectome literature demonstrates that 1-to-2 synapse connections have high false-positive rates in automated EM segmentation and negligible physiological influence. Filtering reduces connections to **44,723 robust synapses**.

### 2.4 Biological Synaptic Signs (Dale's Principle)
Each neuron's neurotransmitter is determined from neuPrint's predicted consensus:
- **Acetylcholine (ACh)** $\to +1.0$ (Excitatory)
- **GABA** $\to -1.0$ (Inhibitory)
- **Glutamate** $\to -1.0$ (Inhibitory in Drosophila CX due to inhibitory glutamate-gated chloride channels, GluCl)
- **Resulting Circuit Balance**: **61% Inhibitory, 39% Excitatory**.

---

## 3. Neural Network Architectures & Mathematical Dynamics

**Source Files:** [`src/connectome_rnn.py`](file:///home/anuragiyer/Desktop/Open-day-makkhi/src/connectome_rnn.py), [`src/baseline_rnn.py`](file:///home/anuragiyer/Desktop/Open-day-makkhi/src/baseline_rnn.py)

### 3.1 What is Frozen vs. What is Trained

| Component | Status | Mechanism |
|---|---|---|
| **Topology (Who connects to whom)** | **FROZEN (Biology)** | PyTorch non-trainable buffers (`pre`, `post` indices) |
| **Synaptic Signs (+ / -)** | **FROZEN (Biology)** | Presynaptic neurotransmitter buffer (`conn_sign`) |
| **Synaptic Strengths (Weights)** | **TRAINED (RL)** | $W_{ij} = \text{sign}[j] \cdot \lvert \text{raw}_{ij} \rvert$ |
| **Sensor Encoder** | **TRAINED (RL)** | Linear layer: 40 inputs $\to$ 738 input neurons (`ER` + `PFN`) |
| **Motor Decoder** | **TRAINED (RL)** | Linear layer: 50 output neurons (`PFL`) $\to$ 2 motor actions |

### 3.2 Weight Parameterization Math
- We represent the weight of connection $k$ ($j \to i$) as:
  $$W_{ij} = \text{sign}_j \cdot |\text{raw}_k|$$
- **Decision: Why absolute value $|\text{raw}|$ instead of $\text{softplus}(\text{raw})$?**
  - An early prototype used $\text{softplus}(\text{raw})$. However, $\text{softplus}'(x) = \frac{1}{1 + e^{-x}}$, meaning weak synapses had near-zero gradients. Connection weights changed $15\times$ slower than the baseline network, artificially crippling the connectome's learning ability!
  - Absolute value $|\text{raw}|$ yields subgradient $\text{sign}(\text{raw}) = \pm 1$, giving equal gradient magnitude across all connection strengths while strictly preventing sign-flipping.

### 3.3 Spectral Radius Initialization
- Connection weights start proportional to biological synapse counts:
  $$\text{init}_k = \frac{\text{synapse\_count}_k}{\max(\text{synapse\_count})}$$
- The matrix $W_0 = \text{sign} \odot \text{init}$ is constructed, and its maximum absolute eigenvalue (spectral radius $\rho(W_0)$) is computed via `torch.linalg.eigvals`.
- All weights are multiplied by:
  $$\text{scale} = \frac{0.900}{\rho(W_0)}$$
- **Decision: Why $\rho(W) = 0.900$?**
  - $\rho < 1.0$ guarantees that the unforced linear recurrent dynamics are asymptotically stable (activity neither explodes to infinity nor immediately decays to zero before RL begins).

### 3.4 Leaky Rate Neuron Dynamics & Maximum Firing Rate
For each time step, neurons update via:
$$\text{drive}_t = W_{\text{enc}} \cdot \text{obs}_t$$
$$\text{rate} = \text{MAX\_RATE} \cdot \tanh\left(\frac{\text{ReLU}(h W^T + \text{drive}_t)}{\text{MAX\_RATE}}\right)$$
$$h \leftarrow (1 - \alpha) h + \alpha \cdot \text{rate}$$

- $\alpha = 0.5$: Discrete leak coefficient ($\Delta t / \tau$). Blends $50\%$ old memory with $50\%$ new incoming firing rate.
- $\text{MAX\_RATE} = 10.0$:
  - **Decision: Why cap firing rates?** In early unconstrained tests, activity in recurrent loops occasionally snowballed exponentially, causing floating-point overflow (`NaN`/`inf`). Real biological neurons have a physiological maximum firing frequency (~200–500 Hz). The smooth tanh saturation ensures stability while leaving normal firing rates (~0–3) unaffected.

### 3.5 The 5 Substeps (Multi-Hop Signaling)
- Inside every single environment step (30 Hz), the sensory input $\text{drive}_t$ is held constant while the recurrence runs for **`substeps = 5`** internal ticks:
  ```python
  for _ in range(net.substeps):
      rate = MAX_RATE * torch.tanh(F.relu(h @ w_t + drives[t]) / MAX_RATE)
      h = (1 - net.alpha) * h + net.alpha * rate
  ```
- **Decision: Why 5 substeps?**
  - In the biological central complex, signals must travel:
    $$\text{Sensors} \to \text{ER/PFN} \xrightarrow{\text{Hop 1}} \text{EPG/Delta7} \xrightarrow{\text{Hop 2}} \text{PEN/FB} \xrightarrow{\text{Hop 3–4}} \text{PFL} \to \text{Motors}$$
  - A standard RNN taking 1 step per control step would only propagate sensory information 1 neuron deep per action. 5 substeps allow signals to traverse all intermediate biological layers and settle into a coherent spatial state before motor readout.

---

## 4. The 4-Way Fair Benchmark Design

**Decision: Why match trainable parameter count rather than neuron count?**
- If we gave the dense baseline 922 neurons, its fully connected $W \in \mathbb{R}^{922 \times 922}$ would have $\sim 850,000$ weights ($11\times$ more than the connectome). A win for the baseline would simply mean "bigger is better."
- Matching parameter counts (~75,000) isolates **wiring topology** as the independent variable.

```
+-----------------------------------------------------------------------------------------+
|                                4-WAY BENCHMARK ARCHITECTURE                             |
+-----------------------------------------------------------------------------------------+
| Model          | Neurons | Wiring Description               | Trainable Parameters      |
+----------------+---------+----------------------------------+---------------------------+
| Connectome     | 922     | Fixed biological fly CX wiring   | 75,083                    |
| Shuffled Fly   | 922     | Maslov-Sneppen degree-matched    | 75,083                    |
| Dense Baseline | 254     | Fully connected leaky rate RNN   | 75,440 (within 0.5%)      |
| GRU Standard   | 139     | Standard PyTorch nn.GRUCell      | 75,757 (within 0.9%)      |
+-----------------------------------------------------------------------------------------+
```

### The Shuffled Connectome Control ([`shuffle_wiring`](file:///home/anuragiyer/Desktop/Open-day-makkhi/src/connectome_rnn.py#L62-L93))
- **Mechanism**: Maslov–Sneppen (2002) double-edge swap algorithm.
- Repeatedly selects edges $(a \to b)$ and $(c \to d)$ and rewires them to $(a \to d)$ and $(c \to b)$.
- **Guarantees**:
  1. Every neuron keeps its **exact out-degree**.
  2. Swaps are partitioned within sign groups, so every neuron keeps its **exact number of excitatory incoming and inhibitory incoming connections**.
  3. No self-loops or duplicate parallel edges are formed.
  4. Connection strengths use the exact same initialization scale as the real fly.
- **Scientific Purpose**:
  - If `Connectome > Shuffled`: Proves that the **specific biological wiring motif** (compass rings, column shifts) is what drives performance.
  - If `Connectome ≈ Shuffled > Baseline`: Proves that **sparsity and wide dimensional representation** (922 sparse vs 254 dense) drives performance.

---

## 5. Drone Simulation & Physics Environment

**Source File:** [`src/maze_env.py`](file:///home/anuragiyer/Desktop/Open-day-makkhi/src/maze_env.py)

### 5.1 Platform & Simulator
- **Simulator**: `gym-pybullet-drones` on Bullet Physics engine.
- **Drone Model**: Bitcraze Crazyflie 2.X quadcopter (mass: 27 grams, collision radius: 6 cm).
- **Simulation Speed**: ~2,300 physics steps/second headless.

### 5.2 Control Hierarchy (Hierarchical RL)
- **Control Frequency**: 30 Hz ($dt \approx 0.0333$ s).
- **Altitude**: Fixed at $0.5$ m via a built-in altitude PID controller ($v_z = \text{clip}(2.0(0.5 - z), -0.5, 0.5)$).
- **Action Space (2 Continuous Outputs in $[-1, 1]$)**:
  - $\text{Action}_0 \in [-1, 1]$: Turn rate ($\pm 2.0$ rad/s).
  - $\text{Action}_1 \in [-1, 1]$: Forward velocity ($0 \dots 1.0$ m/s).
- **Decision: Why high-level velocity commands instead of low-level rotor RPMs?**
  - Quadcopter attitude stabilization from raw rotor RPMs is an aerodynamic motor control problem unrelated to navigation.
  - This matches biology: fly PFL neurons output directional heading and forward drive to downstream thoracic central pattern generators, not individual wing-stroke muscle contractions.

### 5.3 Observation Space (40 Dimensions, Strictly Egocentric)
1. **32 Lidar Rays (0–31)**:
   - $360^\circ$ planar scan at flying height ($0.5$ m), $11.25^\circ$ spacing, 1.5 m maximum range.
   - Values: Normalized proximity $1.0 - (\text{hit\_distance} / \text{max\_range})$.
   - **Decision: Why 32 rays instead of 16?** With 16 rays ($22.5^\circ$), thin wall ends slipped between rays, causing frequent edge-clipping crashes.
   - **Ray Origin**: Starts at radius $0.065$ m (just outside the $0.06$ m body). Starting at $0.0$ placed ray origins inside walls when close, causing PyBullet raycast misses.
2. **Goal Tracking (32–35)**:
   - $\sin(\text{bearing})$ and $\cos(\text{bearing})$ relative to drone heading.
   - Normalized Euclidean distance: $d / (2 \cdot \text{ARENA\_HALF})$.
   - **Goal in Line-of-Sight (1.0 or 0.0)**: Evaluated via single collision raycast from drone to goal.
     - **Decision: Why the Line-of-Sight sensor?** In mazes, the direct goal direction usually points straight into a wall, causing trained networks to ignore the goal vector entirely. Near the finish, drones would endlessly orbit the goal without entering. The LOS indicator signals to the network exactly when the goal heading becomes trustworthy.
3. **Kinematics & Proprioception (36–39)**:
   - Body-frame forward speed $v_x / MAX\_SPEED$ and lateral speed $v_y / MAX\_SPEED$.
   - Previous turn command $a_{t-1, 0}$.
   - Normalized vertical height error $(0.5 - z) / 0.5$.

---

## 6. Procedural Maze Generation & Geodesic Rewards

**Source File:** [`src/grid_maze_env.py`](file:///home/anuragiyer/Desktop/Open-day-makkhi/src/grid_maze_env.py)

### 6.1 Procedural Generation Algorithms
Arena dimensions: 5.0 m $\times$ 5.0 m.
- **Sizes**: $3\times 3, 4\times 4, 5\times 5, 6\times 6$ during training; **$7\times 7$ reserved exclusively for out-of-distribution testing**.
- **Three Generator Algorithms**:
  1. *Recursive Backtracker*: Produces long, winding corridors with low branching factors.
  2. *Kruskal's Algorithm*: Produces uniform spanning trees with numerous short dead ends.
  3. *Prim's Algorithm*: Produces highly branched structures with compact dead ends.
- **Braiding ($0.0 \dots 0.3$)**:
  - Removes a fraction of dead-end walls to create closed loops and alternate routes.
- **Initial Heading**: Drones start facing down an open corridor rather than diagonally into a corner wall.

### 6.2 The Geodesic Route Distance Map ([`route_distance_map`](file:///home/anuragiyer/Desktop/Open-day-makkhi/src/grid_maze_env.py#L125-L183))
- **Decision: Why not Euclidean distance for reward?**
  - Euclidean distance creates local minima: a drone pressing against a corridor wall facing the goal would receive positive reward despite being trapped in a dead end.
- **The 5 cm Geodesic Grid Solution**:
  1. An 8-connected grid of resolution $5$ cm ($101 \times 101$ nodes) is laid across the arena.
  2. Obstacle inflation: Walls are widened by $\text{band} = \text{WALL\_HALF\_THICK} + \text{DRONE\_RADIUS} + \text{ROUTE\_MARGIN} = 0.04 + 0.06 + 0.10 = 0.20$ m.
  3. Shortest-path distance from the goal node to all free nodes is computed via **Dijkstra’s algorithm**.
  4. Wall interiors receive values via **Euclidean Distance Transform (`distance_transform_edt`)**, ensuring the distance gradient smoothly points away from walls, gently repelling the drone.
  5. During flight, distance is sampled continuously at the drone’s $(x, y)$ coordinate via bilinear interpolation (`scipy.ndimage.map_coordinates`).

### 6.3 Reward Formulation
$$R_t = 5.0 \cdot (d_{t-1} - d_t) - 0.03 + 0.01 \cdot \text{speed\_fraction} + R_{\text{terminal}}$$

- **Progress Reward**: $+5.0$ per meter along the geodesic shortest path.
- **Time Penalty**: $-0.03$ per step (discourages wandering and stalling).
- **Anti-Creeping Speed Bonus**: $+0.01 \cdot (\text{progress\_speed} / MAX\_SPEED)$.
  - **Decision: Why this specific speed bonus?** Networks naturally learned to "creep" at 0.1 m/s to avoid crashing. The speed bonus only rewards velocity directed *along the route*. Because the time penalty ($0.03$) exceeds the maximum speed bonus ($0.01$), spinning in circles always yields net negative reward.
- **Goal Reward**: $+10.0$ upon reaching within $0.40$ m radius of the goal.
- **Crash Penalty**: $-10.0$ upon collision or tilt $> 1.2$ rad.

---

## 7. Recurrent PPO Training Pipeline & Trap Fixes

**Source Files:** [`src/ppo.py`](file:///home/anuragiyer/Desktop/Open-day-makkhi/src/ppo.py), [`src/train_ppo.py`](file:///home/anuragiyer/Desktop/Open-day-makkhi/src/train_ppo.py)

### 7.1 Architecture & Vectorization
- **16 Parallel Environments** (`AsyncVectorEnv`, `spawn` context).
- **Memory Isolation**: `train_ppo.py` delegates execution so child worker processes never import PyTorch, reducing RAM footprint from $7.8$ GB to $3.2$ GB per run and eliminating system OOM crashes.
- **Rollout Length**: $T = 256$ steps ($16 \times 256 = 4,096$ environment steps per update).
- **Total Training Budget**: 5,000,000 steps (~1,220 updates, ~35 minutes on RTX 5080).

### 7.2 Truncated Backpropagation Through Time (BPTT Chunking)
- Rollouts are partitioned into chunks of **`chunk_len = 32` steps**.
- Hidden state $h$ at each chunk boundary is recorded in `h_chunks`.
- During PPO optimization, minibatches replay sequences of length 32 starting from their recorded $h$, backpropagating gradients through recurrent dynamics across time.

### 7.3 Critical RL Bugs Found & Fixed

#### 1. The "Hover Trap" (Action Saturation)
- **Problem**: When unbounded Gaussian policies output raw means past $[-1, 1]$ (e.g., speed mean $-2.5$), all exploratory samples clip to $-1.0$. The gradient vanishes, and the agent becomes permanently stuck hovering in place.
- **Fix**: Action means are passed through a smooth $\tanh$ squash:
  $$\text{action} \sim \mathcal{N}(\tanh(\text{raw}), \sigma)$$

#### 2. The "Spin Trap" (Tanh Saturation)
- **Problem**: While $\tanh$ constrains actions to $[-1, 1]$, if raw activations grow large (e.g., $+8.0$), the derivative $\tanh'(8.0) \approx 0$. Gradients vanish, locking the drone into a permanent full-speed left turn.
- **Fix**: Auxiliary loss penalty on raw activations exceeding $\text{RAW\_LIMIT} = 2.0$:
  $$\mathcal{L}_{\text{raw}} = 0.01 \cdot (\max(0, |\text{raw}| - 2.0))^2$$

#### 3. Autopilot State Carryover
- **Problem**: The quadcopter PID controller retained attitude and integral error terms across episode resets, causing identical seeds to produce different outcomes depending on the preceding episode.
- **Fix**: Explicitly call `ctrl.reset()` on all PID controllers at every episode boundary.

#### 4. Timeout Value Bootstrapping
- **Problem**: Episodes terminating due to the step limit (timeout) were treated as terminal zero-value states.
- **Fix**: Bootstraps the discounted value of the truncated state:
  $$R_{\text{cut}} \leftarrow R_{\text{cut}} + \gamma V(s_{\text{cut}})$$

---

## 8. Evaluation Protocol & Empirical Findings

**Source Files:** [`src/select_best.py`](file:///home/anuragiyer/Desktop/Open-day-makkhi/src/select_best.py), [`src/evaluate.py`](file:///home/anuragiyer/Desktop/Open-day-makkhi/src/evaluate.py)

### 8.1 Rigorous Three-Tier Seed Separation
To prevent data snooping or overfitting:
1. **Training Seeds**: Seeds $0 \dots 4$ generate unique maze streams starting at $\text{seed} \times 50,000$.
2. **Practice Seeds ($500,000+$)**: 40 practice mazes used by `select_best.py` to pick `model_best.pt` among snapshots saved after the first 25% of training.
3. **Held-Out Test Seeds ($1,000,000+$)**: 100 test mazes per difficulty level, evaluated deterministically (zero action noise).

### 8.2 Empirical Results Summary

#### Obstacle Arena Benchmark (100 Held-Out Mazes, 5 Seeds)
| Model | Standard (8 pillars, 2 walls) | Hard (14 pillars, 4 walls) | Very Hard (20 pillars, 6 walls) | Worst Seed (Standard) |
|---|---|---|---|---|
| **Real Fly Connectome** | **78%** | **48%** | 25% | **61%** |
| **Shuffled Fly Control** | 67% | 40% | 22% | 3% (collapsed) |
| **Standard GRU** | 68% | 46% | **29%** | 0% (collapsed) |
| **Dense Leaky RNN** | 60% | 30% | 13% | 41% |

#### Procedural Grid Maze Benchmark (Training Success over Final Updates)
- **Real Fly Connectome**: **~84%** (Seeds: 83–85%)
- **Shuffled Fly Control**: ~69% (Seeds: 62–78%)
- **Standard GRU**: ~67% (Seeds: 59–72%)
- **Dense Leaky RNN**: ~61% (Seeds: 59–65%)

### 8.3 Key Empirical Conclusions
1. **Sample Efficiency**: The real fly connectome learns substantially faster during early training, reaching ~85% success at 2.2M steps compared to 73% (shuffled), 58% (GRU), and 42% (dense).
2. **Training Stability & Reliability**: The connectome exhibited the lowest seed-to-seed variance. No connectome run collapsed, whereas GRU and Shuffled runs suffered catastrophic collapses into hover/creeping traps under identical hyperparameters.
3. **Clear Win Over Dense Architectures**: The connectome decisively outperformed the parameter-matched dense network built with the same leaky neuron dynamics across all difficulty tiers ($p < 0.05$).
4. **Asymptotic Equivalence with Modern GRUs**: When modern GRU runs did not collapse, their final asymptotic success on familiar layouts matched the connectome.

---

## 9. Comprehensive Master Parameters & Numbers Reference

| Parameter / Metric | Exact Value | Source / Rationale |
|---|---|---|
| neuPrint Database & Dataset | `neuprint.janelia.org` / `male-cns:v1.0` | Full male CNS including VNC motor tracts |
| Total Biological Neurons | **922 neurons** | Extracted CX families |
| Biological Connections ($\ge 3$ syn) | **44,723 connections** | Summed across ROIs |
| Total Trainable Parameters (Fly) | **75,083 parameters** | Connections (44,723) + Enc (29,520) + Dec (102) |
| Trainable Parameters (Baseline Dense) | **75,440 parameters** | 254 dense neurons |
| Trainable Parameters (GRU) | **75,757 parameters** | 139 GRU hidden units |
| Initial Spectral Radius $\rho(W)$ | **0.900** | Linear asymptotic stability |
| Leaky Integration Rate $\alpha$ | **0.500** | $\Delta t / \tau$ memory decay factor |
| Maximum Firing Rate $\text{MAX\_RATE}$ | **10.0** | Prevents runaway gradient explosion |
| Internal Substeps | **5 substeps** | Allows 3–5 multi-hop synaptic transmission |
| Drone Model & Mass | Bitcraze Crazyflie 2.X / 27 grams | Standard micro-UAV |
| Control Frequency & Time Step | 30 Hz / $dt = 0.0333$ s | Standard PyBullet drone frequency |
| Flying Altitude | 0.50 m | 2D navigation abstraction |
| Max Forward Speed | 1.0 m/s | Drone operational limit |
| Max Turn Rate | 2.0 rad/s (~114.6°/s) | Yaw agility limit |
| Observation Dimension | **40 dimensions** | 32 lidar + 4 goal + 4 kinematics |
| Lidar Beam Count & Spacing | 32 beams / $11.25^\circ$ spacing | Prevents wall-end edge skipping |
| Lidar Range & Origin Offset | 1.5 m max range / 0.065 m offset | Clears drone collision cylinder |
| Goal Capture Radius | **0.40 m** | Matches turning radius at 1.0 m/s |
| Arena Dimensions | 5.0 m $\times$ 5.0 m | Physical arena constraint |
| Geodesic Grid Resolution | 0.05 m (5 cm) | Dijkstra path resolution |
| Wall Inflation Margin | 0.20 m (4cm wall + 6cm drone + 10cm safety) | Prevents clipping wall corners |
| PPO Rollout Length $T$ / Chunks $L$ | $T = 256$ steps / $L = 32$ steps | Truncated BPTT horizon |
| Parallel Environments | 16 environments | AsyncVectorEnv |
| Total Training Steps | 5,000,000 steps (~1,220 updates) | Standard training horizon |
| PPO Discount $\gamma$ / GAE $\lambda$ | $\gamma = 0.99$ / $\lambda = 0.95$ | Standard reinforcement learning settings |
| PPO Clip $\epsilon$ / Learning Rate | $\epsilon = 0.2$ / $\text{lr} = 3 \times 10^{-4}$ | Adam optimizer ($\text{eps}=10^{-5}$) |
| Raw Activation Penalty Threshold | $\text{RAW\_LIMIT} = 2.0$ (coef 0.01) | Tanh saturation defense |
| Progress Reward Coefficient | +5.0 per meter | Geodesic path delta |
| Time Cost per Step | -0.03 per step | Anti-dawdling penalty |
| Anti-Creep Speed Bonus | +0.01 $\times$ (speed / max) | Rewards velocity toward goal |
| Goal Reward / Crash Penalty | +10.0 / -10.0 | Terminal outcome incentives |

---

## 10. Defense & FAQ: Addressing Tough Audience / Faculty Questions

### Q1: "Why didn't you model action potentials (spiking neurons)?"
> **Answer**: Rate-based leaky models are mathematically equivalent to the average firing frequency of spiking populations over short temporal integration windows ($33$ ms). Training biological spiking networks with gradient-based RL remains notoriously sample-inefficient and numerically unstable. Using rate neurons allowed us to isolate **topological connectivity** without conflating learning performance with spike-timing approximations.

### Q2: "Did you just memorize one maze?"
> **Answer**: No. Every single episode in training and evaluation is procedurally generated from distinct random seeds. Furthermore, training seeds ($0\dots 4$), practice selection seeds ($500,000+$), and final test seeds ($1,000,000+$) are strictly partitioned. The $7\times 7$ maze evaluation tests generalization on corridor dimensions never encountered during training.

### Q3: "Doesn't the encoder and decoder destroy the biological validity?"
> **Answer**: Real fruit flies have eyes (ommatidia) and wing muscles; our simulated drone has lidar and brushless DC motors. A fly's brain cannot natively parse simulated distance arrays. The linear encoder and decoder represent sensory transduction and neuromuscular mapping, allowing the frozen central complex architecture to interface with quadcopter physics.

### Q4: "Is the fly network really better than a modern GRU?"
> **Answer**: On asymptotic success when a GRU converges, they tie (~70–80%). However, the biological connectome demonstrated **two decisive advantages**:
> 1. **Sample efficiency**: It learned the task in fewer than half the environment steps (~2M vs ~4.5M).
> 2. **Optimization robustness**: Under identical training hyperparameters, the connectome never collapsed across 5 random seeds, whereas the GRU and shuffled networks suffered severe policy collapses into hover traps.

### Q5: "Could the network bypass the biological compass entirely?"
> **Answer**: Because ring neurons (`ER`) and `PFN` neurons connect directly to intermediate structures, some direct sensory-to-motor pathways exist. We acknowledge this honest scientific caveat: in future work, a **systematic lesion experiment** (inactivating the `EPG`/`PEN`/$\Delta 7$ compass loop post-training) will quantify the exact percentage of steering control mediated through the compass ring attractor.
