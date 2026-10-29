# Fly connectome → drone navigation

Can the wiring of a fruit fly's navigation centre fly a drone? This project takes the
**central complex** of the fruit fly (*Drosophila*) from the male CNS connectome, turns it into a
recurrent neural network with the fly's real wiring (who connects to whom, and whether each
connection excites or inhibits), and trains it with reinforcement learning (PPO) to fly a
simulated quadcopter through obstacle arenas and procedurally generated mazes.

It is compared against networks with the **same number of trainable parameters**:

| Network | What it is |
|---|---|
| **Real fly** (`connectome`) | 922 neurons (EPG, PEN, PFL, PFN, ER, Δ7), 44,723 connections with fixed wiring and signs; only connection strengths are trained |
| **Shuffled fly** (`shuffled`) | Same neurons and connection strengths, wiring randomly rewired (every neuron keeps its numbers of outgoing, excitatory-incoming and inhibitory-incoming connections) |
| **GRU** (`gru`) | A standard recurrent network |
| **Dense RNN** (`baseline`) | Fully connected, same neuron model as the fly network |

**Work in progress.** [DECISIONS.md](DECISIONS.md) is a plain-language log of every decision,
bug, fix and result so far, including the caveats.

## Setup

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The connectome data is already in `data/`. To re-download it you need a neuPrint token
(from your account page at neuprint.janelia.org):

```bash
export NEUPRINT_TOKEN=...
python src/extract_connectome.py
```

## Running

```bash
# Train one network (obstacle arena, or --task grid for mazes)
python src/train_ppo.py --model connectome --task grid --name maze_conn_s0 --seed 0

# All four networks x 5 seeds, 5 at a time
TASK=grid MODELS="connectome shuffled gru baseline" SEEDS="0 1 2 3 4" PARALLEL=5 bash src/run_experiments.sh

# Pick each run's best snapshot on practice mazes, then test on held-out mazes
python src/select_best.py
python src/evaluate.py --task grid --checkpoint best
python src/evaluate.py --task grid --checkpoint last

# Launch the interactive Open Day GUI
python launch_gui.py

# Or watch a trained drone directly from the command line in the 3D simulator
python src/watch.py maze_conn_s0 --size 4
```

## Code

| File | Purpose |
|---|---|
| `src/extract_connectome.py` | Download the central-complex neurons and connections from neuPrint |
| `src/connectome_rnn.py` | Connectome RNN (fixed wiring and signs, trainable strengths) and the shuffled control |
| `src/baseline_rnn.py` | Parameter-matched dense RNN and GRU |
| `src/maze_env.py` | Obstacle-arena drone task (gym-pybullet-drones) |
| `src/grid_maze_env.py` | Procedurally generated grid mazes with a route-distance reward |
| `src/ppo.py`, `src/train_ppo.py` | Recurrent PPO training |
| `src/select_best.py`, `src/evaluate.py` | Snapshot selection on practice mazes; final test on held-out mazes |
| `src/make_plots.py`, `src/watch.py` | Figures; 3D playback |
| `pilot_goalsight/` | Pilot of a "goal in line of sight" sensor (see DECISIONS.md Step 16) |

## Credits

- Connectome: male CNS v1.0 (`male-cns:v1.0`), via [neuPrint](https://neuprint.janelia.org), Janelia Research Campus.
- Central complex circuitry: Hulse, B. K. et al. (2021). *A connectome of the Drosophila central complex reveals network motifs suitable for flexible navigation and context-dependent action selection.* eLife 10:e66039.
- Simulator: [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) (Panerati et al., 2021).
