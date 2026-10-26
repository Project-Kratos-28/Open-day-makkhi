# Project brief — fly connectome → drone maze navigation

Read this whole file before doing anything. It's written so a fresh Claude
session (no prior conversation context) can pick up exactly where this one
left off. If anything here looks stale compared to what's actually in this
directory, trust the directory over this file and flag the mismatch to the
user before proceeding.

## The goal

College open day in ~10 days (target: **2026-11-03** — confirm this date
with the user, it was derived from "17 days out" stated around 2026-10-17).
The user needs to demo robotics/AI research to CS department professors who
control their funding. This is a build-it-and-demo-it task under a hard
deadline, not open-ended research — bias toward "working and simple" over
"complete and ambitious."

## The idea, in one paragraph

Take the real wiring diagram of a fruit fly's **central complex** — the
insect's actual navigation/heading circuit, not its whole brain — from a
public connectome database, and use that wiring as the fixed architecture
of a recurrent neural network. Train the free parameters (connection
strengths, plus new sensor/motor interface layers) with reinforcement
learning so it can pilot a *simulated* quadcopter through a maze full of
obstacles. The deliverable is a **comparison**: this biologically-wired
network vs. a standard neural network of the same size, on the same task.
That comparison is the actual research contribution for the pitch — not
just "look, it flies through a maze."

## Why this specific scope (don't let it balloon)

- NOT the whole fly brain (~140k neurons, what the viral "fly plays Beat
  Saber / fly in Minecraft" demos used). That's too big and not
  specifically about navigation.
- Instead: just the **central complex**, ~hundreds of neurons, cell types
  `EPG`, `PEN1`, `PEN2`, `PFL`, `PFN`, in brain regions `EB` (ellipsoid
  body — the "compass"), `PB` (protocerebral bridge), `FB` (fan-shaped
  body — where steering decisions get made), `NO` (noduli — odometry),
  `BU` (bulb — sensory input stage).
- Backed by a real citation: **Hulse et al. 2021, eLife, "A connectome of
  the Drosophila central complex reveals network motifs suitable for
  flexible navigation and context-dependent action selection."** Cite this
  in the pitch doc — it's not just the user's own claim that this circuit
  is navigation-relevant.
- Simulation only. No physical drone, no physical flight. This was an
  explicit scope decision to keep the project achievable in the time left.

## Technical architecture

1. **Data source**: neuPrint, dataset `male-cns:v1.0` (NOT the older
   hemibrain dataset — deliberately chosen because it's the full CNS
   including the ventral nerve cord, i.e. it also contains the real motor
   neurons). Accessed via the `neuprint-python` package and a personal API
   token from the user's neuPrint account page.
2. **Extraction**: pull every neuron of the five cell types above, and
   every synapse between them, as two tables (neuron list, connection
   list with synapse-count weights).
3. **Network**: a custom PyTorch recurrent network where:
   - **Frozen** (comes straight from biology, never trained): which
     neurons connect to which (topology/mask), and whether each
     connection is excitatory or inhibitory (sign).
   - **Trained** (via RL): the numeric strength of each existing
     connection (synapse count is just the starting guess), plus a new
     **sensor encoder** (maze/obstacle sensor data → activity on the
     real sensory input neurons) and a new **motor decoder** (activity on
     the real output neurons → drone thrust/yaw/pitch/roll commands).
     These interface layers are not biological — they have to be learned
     from scratch because a fly's real sensory neurons don't know what a
     simulated lidar reading is.
4. **Drone simulator**: `gym-pybullet-drones` (GitHub:
   utiasDSL/gym-pybullet-drones — not on PyPI, must be `git clone`d and
   `pip install -e .`'d, see Current status below). Chosen over Isaac Sim
   for faster setup; the user's existing Isaac Sim experience is from an
   unrelated rover project (`/home/aarsh/Downloads/athena_description`,
   a 4WIS rover digital twin — do not mix these two projects' files).
5. **Training**: PPO via `stable-baselines3`, run twice — once with the
   connectome-derived network, once with a plain baseline NN of matching
   parameter count — and compare results (training speed, sample
   efficiency, robustness, whatever turns out cleanest to measure).

## Environment (confirmed working as of 2026-10-24)

- Ubuntu 24.04.5 LTS, kernel 6.8.0-147-generic, Python 3.12.3
- GPU: NVIDIA RTX 5080, 16GB VRAM, driver 580.178.04, CUDA 13.0 — plenty
  for training this
- 42GB free on `/home` at last check
- `python3.12-venv` apt package was missing and has since been installed
  (it provides `ensurepip`, required for `python3 -m venv` to work)

## Current status of this directory (check this against reality — it may
## be stale by the time you read it)

- `requirements.txt` — present, lists `neuprint-python numpy pandas torch
  stable-baselines3 gymnasium` (gym-pybullet-drones deliberately excluded,
  see below)
- `venv/` — created, should have the above packages installed via `pip
  install -r requirements.txt` (confirm with `pip list` inside it)
- `gym-pybullet-drones/` — NOT yet cloned as of this writing. User was
  told to run:
  ```bash
  git clone https://github.com/utiasDSL/gym-pybullet-drones.git
  cd gym-pybullet-drones && pip install -e . && cd ..
  ```
- `src/extract_connectome.py` — written, **not yet run against live
  data**. It expects a `NEUPRINT_TOKEN` env var (from the user's neuPrint
  account page) and will save `data/neurons.csv` and `data/connections.csv`.
  It was written from documented `neuprint-python` API calls
  (`fetch_neurons`, `fetch_adjacencies`) but never actually executed, so
  **a column name may not match on first run** (e.g. whatever field holds
  the neurotransmitter/sign prediction) — the script prints all available
  columns on every run specifically so this can be diagnosed and fixed
  fast rather than guessed at.
- `data/` — empty. Extraction has not been run yet.
- No network/training code has been written yet.

## Remaining plan (day budget, originally ~10 days from 2026-10-24 —
## recompute from today's actual date)

| Day | Task |
|---|---|
| 1 | Finish env setup, get neuPrint token, run `extract_connectome.py`, fix any column-name mismatches, confirm `data/*.csv` look sane |
| 2 | Build the masked-RNN module (`src/connectome_rnn.py` or similar): load the two CSVs, build mask + sign matrix, trainable magnitude matrix, sanity-check shapes and that frozen parts stay frozen |
| 3–4 | Build the Gymnasium-compatible maze/obstacle env on top of `gym-pybullet-drones`, wire up the sensor encoder and motor decoder, get one full dummy forward pass running end to end (random weights is fine, just prove the loop works) |
| 5–8 | Real PPO training runs: connectome model and baseline NN of equal size, on the same maze task. Expect this to be the bottleneck — have a reward-shaping or smaller-maze fallback ready by day 7–8 if not converging |
| 9 | Lock in whichever result is working, produce comparison plots (training curves, success rate, whatever's cleanest) |
| 10 | Poster/demo prep and rehearsal — buffer day, don't schedule real work into it if avoidable |

## How to work with this user

- They're a student, time-pressured, wants direct actionable steps over
  long explanations at this stage of the project.
- They do their own `sudo`/interactive terminal commands themselves (paste
  screenshots back); don't expect to run those yourself, just hand them
  the exact command.
- They're also writing a pitch in a Google Doc in parallel — if asked
  about framing/wording for that, the key angles already validated with
  them: (a) be precise that the connectome gives architecture+sign, not a
  "ready-made skill" — training still does real work; (b) lead with the
  Hulse et al. citation for credibility; (c) frame the project as a
  controlled comparison (connectome net vs. baseline net), not just a demo,
  since that's what makes it a fundable research question rather than a
  cute trick.
