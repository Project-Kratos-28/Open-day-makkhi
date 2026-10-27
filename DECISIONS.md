# Project log: fly connectome → drone maze

A plain-language record of what we've done, what we decided, and why.
Newest steps are at the bottom. Open day: **2026-11-03**.

---

## The big picture

We take the real wiring of the part of the fruit fly brain that handles
navigation (the **central complex**), turn it into an artificial neural
network, and teach it to fly a simulated drone through a maze. Then we
compare it against an ordinary neural network of the same size on the
same task. **The comparison is the research result**, not the flying.

---

## Step 1: Getting the fly wiring data (2026-10-24) ✅

**What we did:** downloaded the neurons and their connections from
neuPrint (an online database of fly brain wiring), dataset `male-cns:v1.0`.
Saved to `data/neurons.csv` and `data/connections.csv`.
Script: `src/extract_connectome.py`.

**Result:** 922 neurons and 79,506 neuron-to-neuron connections.

| Neuron family | Count | Role in the fly | Role in our drone |
|---|---|---|---|
| ER (ring neurons) | 282 | carry visual/landmark info into the compass | **inputs**: receive sensor data |
| EPG | 50 | the compass itself (tracks heading) | internal |
| PEN | 42 | rotate the compass when the fly turns | internal |
| Δ7 (Delta7) | 42 | inhibition that keeps the compass stable | internal |
| PFN | 456 | combine heading with self-motion | **inputs**: receive sensor data |
| PFL | 50 | decide which way to steer | **outputs**: drive the motors |

**Decisions:**

- **Only the navigation circuit, not the whole brain.** The whole brain
  is about 140,000 neurons. That's too big, and most of it has nothing to
  do with navigation. This circuit is the one Hulse et al. 2021 (eLife)
  showed is built for navigation, which gives us a citation to stand on.
- **Matched neuron types by name pattern** (e.g. "anything starting with
  PFN"). The database splits families into subtypes (PFNa, PFNd, …), so
  exact names found nothing.
- **Added the ER ring neurons.** Without them the circuit had no natural
  "way in" for sensor information.
- **Added the Δ7 neurons.** Without them, everything inside the compass
  was excitatory (only "go" signals, no "stop" signals). Real compasses
  in the fly depend on Δ7 inhibition to stay stable, so leaving them out
  would make the "it's the real wiring" claim weaker.
- **Bug fixed along the way:** the database connection was being thrown
  away immediately (a quirk of the `neuprint` library), so the script now
  keeps hold of it.

---

## Step 2: Turning the wiring into a neural network (2026-10-24) ✅

**What we did:** built the network in `src/connectome_rnn.py`.
Run `python src/connectome_rnn.py` to check it still works.

**How it works, simply:** every neuron has an activity level. Each time
step, each neuron adds up signals from the neurons wired into it
(positive for excitatory, negative for inhibitory) and updates its
activity. Because activity carries over between steps, the network has
**memory**.

**What's fixed (from biology) vs. learned (from training):**

| Fixed: never changes | Learned: training adjusts it |
|---|---|
| Which neuron connects to which | How strong each connection is |
| Whether each connection excites or inhibits | Sensor encoder: drone sensors → input neurons |
| | Motor decoder: output neurons → drone commands |

**Decisions:**

- **Ignored connections with fewer than 3 synapses.** Half of all
  connections are just 1–3 synapses, which is often noise. This left
  44,723 connections. It's a single setting (`min_synapses`) if we want to
  revisit it.
- **Each connection's sign comes from the sending neuron's chemical.**
  Acetylcholine = excitatory; GABA and glutamate = inhibitory. These are
  standard assumptions for this brain region. 61% of connections turn out
  to be inhibitory.
- **Starting strengths are proportional to the real synapse counts**, then
  scaled down so the network starts out stable (activity doesn't explode
  or die out).
- **The network runs 5 internal "thinking" steps per drone step**, because
  signals have to pass through several neurons to get from the inputs to
  the steering outputs.
- **Size: 57,473 learnable numbers.** The ordinary network we compare
  against must have about the same number, or the comparison isn't fair.

**Checks that passed:** activity stays stable over 200 steps with random
input (803 of 922 neurons active); training changes the connection
strengths but never the wiring or the signs.

---

## Step 3: Choosing how to train (2026-10-24) ✅ decided

**The problem:** the training method (PPO, learning by trial and error)
comes from an off-the-shelf library that assumes the network has no
memory. Ours does.

**Options we considered:**

- **A: Quick hack.** Feed the network its own previous activity as if it
  were another sensor. Fast, but the network can never learn *how to
  remember*, which is exactly the fly circuit's strength.
- **B: Our own training loop that understands memory.** More code (about
  200 lines), but learning can trace back through time ("I crashed because
  I forgot that turn 20 steps ago").

**Decision: B.** A professor could fairly object to A that we never let
the network use the thing that makes it special. B gives a fair,
defensible comparison. **Fallback:** if B isn't working by day 5, switch to A.

---

## Step 4: The maze (2026-10-24) ✅

**What we did:** built the drone task in `src/maze_env.py`, on top of the
`gym-pybullet-drones` simulator. Run `python src/maze_env.py --gui` to
watch a crude hand-written pilot try it.

**The task:** a 5 m × 5 m walled arena with random pillars and wall
segments. The drone starts in one corner and must reach a goal in the
opposite corner without touching anything. **Every episode gets a new
random layout**, so the network can't just memorise one route.

**What the network sees** (23 numbers, all measured relative to where the
drone is facing):
- 16 distance sensors in a circle around the drone, like a simple lidar
- which direction the goal is, and how far away
- its own speed (forward and sideways), its last turn command, and
  whether it's too high or too low

**What the network controls** (2 numbers): **turn left/right** and
**forward speed**.

**Scoring (the "reward" that training tries to maximise):** points for
getting closer to the goal, +10 for reaching it, −10 for crashing, and a
tiny penalty per second so it doesn't dawdle. An episode ends on goal,
crash, or after 20 seconds.

**Decisions:**

- **The network steers; the drone's built-in autopilot handles balance.**
  Learning to keep a quadcopter in the air (from raw rotor speeds) is a
  hard problem of its own, and has nothing to do with navigation. It
  would eat our training time. Conveniently, this also matches the
  biology: the fly's central complex outputs "turn left/right, speed
  up/slow down" (via the PFL neurons), not individual wing-muscle commands.
- **Fixed flying height (0.5 m), so the problem is 2D.** It's simpler and
  enough to show navigation. The autopilot holds the height.
- **Top speed raised to 1 m/s.** The simulator's default for this drone
  is 0.25 m/s, which would make each episode painfully slow to simulate.
- **Sensors are relative to the drone's heading** ("obstacle on my left",
  not "obstacle at map position x, y"). That's how a fly experiences the
  world, and it's the kind of input the compass circuit is built for.

**Checks that passed:** the drone holds its height and flies at the
commanded speed; crashes, goals and timeouts are detected correctly; the
simulator runs at ~2,300 steps per second.

---

## Step 5: The comparison network and the training loop (2026-10-24) ✅

**What we did:**
- `src/baseline_rnn.py`: the **ordinary network** we compare against.
- `src/train_ppo.py`: the **memory-aware training loop** (option B from
  step 3).

**The baseline network** has everything the same as the connectome
network except the wiring:

| | Connectome network | Baseline network |
|---|---|---|
| Neurons | 922 (real fly neurons) | 238 |
| Wiring | real fly connections only | every neuron to every neuron |
| Excitatory/inhibitory | fixed by biology | free, learned |
| Learnable numbers | 62,537 | 62,834 (within 0.5%) |
| Neuron behaviour, training method, settings | identical | identical |

**Decision: match the number of learnable numbers, not the number of
neurons.** The fair question is "given the same learning budget, does
biological wiring help?" With 922 fully connected neurons the baseline
would have about 850,000 learnable numbers, 14× more, and any win for
it would just mean "bigger is better." So the baseline gets fewer neurons
(238), each connected to all the others.

**How training works, simply:** 16 copies of the maze run in parallel.
The drone flies for 256 steps in each, we look at which actions turned
out well, and we nudge the network towards them. Then we repeat. To
learn *memory*, the flights are cut into 32-step pieces and each piece is
replayed starting from what the network "remembered" at that point, so
learning can trace a crash back to something it forgot several seconds
earlier.

**Speed:** about 2,600 steps per second, so a 5-million-step training run
takes around 30 minutes per network.

**Usage:**
```bash
python src/train_ppo.py --model connectome --name conn_s0 --steps 5000000
python src/train_ppo.py --model baseline   --name base_s0 --steps 5000000
```
Results go to `runs/<name>/log.csv` (progress) and `runs/<name>/model.pt`
(the trained network).

---

## Step 6: First learning test, and a fairness bug we found (2026-10-24) ✅

**What we did:** a short practice run (1 million steps, ~8 minutes) of
each network, to check that learning actually happens before committing
to long runs.

**First result:**

| | Connectome network | Baseline network |
|---|---|---|
| Reaches the goal (end of run) | 6% | 78% |
| Crashes | 94% | 9% |

**This was not a real result. It was a bug in our setup.** We checked
how much each network's internal connections had changed during
training: the baseline's by 27%, the connectome's by only 2%. The way we
had protected the connections' signs (excitatory stays excitatory)
had a side effect: weak connections could only change very slowly. The
connectome network was effectively learning in slow motion, so the
comparison was rigged against it.

**Decision: fixed it so both networks' connections learn at comparable
speed.** Signs are still protected (they still can't flip), just in a way
that doesn't slow learning down. The old run is kept in
`runs/test1m_conn_softplus/` for reference.

**Result after the fix** (average over the last part of the 1M-step run):

| | Connectome (fixed) | Baseline | Connectome (before fix) |
|---|---|---|---|
| Reaches the goal | 82% | 76% | 8% |
| Crashes | 18% | 11% | 92% |
| Runs out of time | 0% | 13% | 0% |
| Average flight length | 170 steps | 304 steps | 222 steps |

**What we can and can't say yet:**
- Both networks learn the task. That's what this test was for.
- Hints of different "styles": the connectome network starts learning
  later (still 0% at 170k steps, when the baseline was already at 43%),
  then flies faster and more directly, but crashes a bit more. The
  baseline is more cautious and sometimes runs out of time.
- **This is one short run of each, so none of this is a conclusion yet.**
  Results vary a lot from run to run in this kind of training. The real
  runs (next step) repeat each network several times with different
  random seeds.

**Why this matters for the pitch:** if we had compared the networks
without catching this, any "the fly wiring is worse" conclusion would
have been an artefact of our code, not a fact about the biology. This
kind of check is worth mentioning: it shows the comparison is controlled.

---

## Step 7: The real training runs (started 2026-10-24, 18:53) 🔄

**What we're doing:** training each network **3 times**, each from a
different random starting point ("seed"), for 5 million steps each:
`conn_s0`, `conn_s1`, `conn_s2` and `base_s0`, `base_s1`, `base_s2`.
Three run at a time, about 50 minutes per batch, so roughly
**1 hour 40 minutes in total**.

**Decision: 3 runs per network, not 1.** Step 6 showed results swing a
lot within a single run. With 3 runs each we can report an average and
how much it varies (error bars), so the comparison holds up to questions.

**Decision: going up to 5 runs per network, then deciding on 10.**
With only 3 vs. 3 runs, even a perfectly clean win could happen by luck
about 1 time in 20, which is right at the edge of convincing. Reinforcement
learning results with few runs are known to be unreliable (Henderson et
al. 2018, "Deep Reinforcement Learning that Matters"). So seeds 3 and 4
(`conn_s3`, `conn_s4`, `base_s3`, `base_s4`) are queued to start
automatically once the first six finish. After seeing 5 vs. 5, we'll
decide whether to go to 10 each (about 4–5 more hours, could run overnight).
Expected finish for all 10 runs: around 21:50.

**Decision: 5 million steps per run** (5× the test). The test runs were
still improving at 1 million steps, so the longer runs let both networks
reach their best.

**How to check on it or control it:**
- Progress so far: `cat runs/experiments.log`
- One run's numbers: `tail runs/conn_s0/log.csv`
- Stop everything: `pkill -f train_ppo.py`
- Resume (skips finished runs): `SEEDS="0 1 2 3 4" bash src/run_experiments.sh`

---

## Step 8: Two problems, fixed, and a clean restart (2026-10-24, 20:44) ✅

### Problem 1: runaway neurons
**What happened:** one baseline training run (`base_s2`) crashed 12% of
the way in. Activity in its simulated neurons snowballed until the numbers
overflowed. A connectome run (`conn_s0`) had the same blow-up later but
recovered by luck.

**Cause:** nothing limited how active a simulated neuron could get. This
was a flaw in the neuron model that **both** networks share, not something
about either wiring.

**Fix:** a maximum firing rate. Activity now levels off smoothly towards
a ceiling instead of growing forever. Real neurons have a maximum firing
rate too, so this is biologically sensible. Normal activity levels are
barely affected.

**Decision: redo all 10 runs on the fixed code.** For a fair comparison,
every run has to use exactly the same code. The 5 runs that had finished
are kept in `runs/v1_uncapped/` for reference only. (For the record: in
those runs the connectome network reached the goal ~85% of the time vs.
39–54% for the baseline, but those numbers don't count, because they ran
on the buggy code.)

### Problem 2: the computer's desktop kept crashing
**What happened:** twice, the whole desktop session died (all windows
closed, training killed) a few seconds to a minute after starting 4 or 5
training runs at once.

**Cause (confirmed from the system logs):** the computer ran out of
memory. Each run starts 16 copies of the simulator, and each copy was
needlessly loading the whole AI library (~350 MB each). That's ~7.8 GB per
run, so 4 runs needed more than the 30 GB the computer has. When memory
ran out, Linux killed desktop apps (browser, Claude app), which took the
whole desktop down with them.

**Fixes:**
- **The simulator copies no longer load the AI library.** Memory per run
  dropped from ~7.8 GB to ~3.2 GB, with no change in speed. (Code moved
  from `train_ppo.py` into `ppo.py`; you still run `train_ppo.py` exactly
  as before.)
- **Training now runs in its own walled-off "container" with a 22 GB
  memory limit.** If it ever runs out of memory again, Linux kills a
  training run instead of your desktop.
- **Save points.** Every training run now saves its full progress about
  every 5 minutes. If anything crashes, re-running the same command
  continues from the last save point instead of starting over. We tested
  this by killing a run on purpose: it resumed exactly where it left off.

**Now running:** all 10 runs (5 per network), 4 at a time, finishing
around **23:35**.

**How to check on it or control it now:**
- Progress: `cat runs/experiments.log`
- Is it running: `systemctl --user status train`
- Stop everything: `systemctl --user stop train`
- Resume after a crash or stop (continues from save points):
  ```bash
  systemd-run --user --unit=train -p MemoryMax=22G -p MemorySwapMax=0 -p OOMPolicy=continue \
    --setenv=SEEDS="0 1 2 3 4" --setenv=PARALLEL=4 --working-directory=$PWD \
    bash -c 'bash src/run_experiments.sh > runs/experiments.log 2>&1'
  ```

---

## Step 8b: A third network, the "shuffled fly" control (2026-10-24, 21:40)

**Why:** early results showed the fly network well ahead of the baseline.
But the two differ in more than "fly wiring vs. not":
- the fly network has 922 neurons vs. 238;
- it is sparsely connected;
- its weights start from real synapse counts;
- it splits its learnable numbers differently.

Any of those, rather than the fly's specific wiring, could explain the
gap. That's the first thing an expert would ask about.

**What it is:** the fly network with its connections **randomly
rewired**. Everything else is identical: the same 922 neurons, the same
number of connections, and every neuron keeps exactly as many incoming and
outgoing connections as it has in the real fly. The same
excitatory/inhibitory balance, the same input and output neurons, and the
same synapse-count values, just attached to different pairs. Each of its 5
runs gets its own random rewiring. (About 20% of the shuffled connections
land on real fly connections by chance, because a few "hub" neurons are
connected to almost everything in their group.)

**How to read the result:**

| If… | it means… |
|---|---|
| fly > shuffled > baseline | the specific fly wiring matters **and** sparse/wide structure helps |
| fly ≈ shuffled > baseline | it's the sparse, wide structure, not the fly's specific wiring |
| fly > shuffled ≈ baseline | it's specifically the fly's wiring |

Any of these is a legitimate finding. This control just tells us *which*
claim we can honestly make.

**Runs:** `shuf_s0` … `shuf_s4`, queued to start automatically after the
current 10 finish (expected finish around 01:00).
Check with `systemctl --user status train-shuffled`; stop with
`systemctl --user stop train-shuffled`.

---

## Step 8c: Main training finished (2026-10-24, 23:46) ✅

All 10 main runs (5 fly, 5 ordinary) trained for 5 million steps each with no
crashes or errors. Training-time scores (average of the last 10 updates; these
include random exploration, so they are noisy and **not** the final result):

| | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | average |
|---|---|---|---|---|---|---|
| Fly connectome: reaches goal | 73% | 89% | 86% | 71% | 75% | ~79% |
| Ordinary net: reaches goal | 60% | 57% | 76% | 40% | 39% | ~54% |

What it looks like so far: the fly network reaches the goal more often. When the
ordinary network fails, it is mostly by **running out of time** (wandering)
rather than crashing. The real comparison is the Step 9 test on unseen mazes.

## Step 9: The final test and the robustness test (done 2026-10-25, 02:40) ✅

**Script:** `src/evaluate.py`, run after training finishes. It writes
`runs/eval_results.csv`.

**What it does:** every trained network (all 20) flies the **same 100
mazes** at three difficulty levels:

| Level | Pillars | Wall segments | Seen in training? |
|---|---|---|---|
| standard | 8 | 2 | same *kind* of maze, but these exact layouts are new |
| hard | 14 | 4 | **no**: harder than anything in training |
| very hard | 20 | 6 | **no** |

**Decisions:**
- **Test mazes come from seeds starting at 1,000,000,** far away from
  anything used in training, so no network has seen these layouts.
- **Identical mazes for every network**, so differences come from the
  networks, not from luck of the draw.
- **No random exploration during the test:** the drone always takes its
  best-guess action, which is how it would be used for real.
- **The robustness test (hard and very hard) is the stronger claim.** It
  asks which wiring copes better with difficulty it never trained for,
  not just which scores higher on familiar mazes.
- **Known limitation:** at the very hard level, some random layouts may
  be impossible (the way blocked completely). That affects both networks
  equally, since they get the same mazes, but it means 100% isn't
  reachable there.

### Results (`runs/eval_results.csv`, figures in `figures/`)

Success rate (reaches the goal), mean of 5 seeds, on 100 new mazes per level:

| Network | Standard | Hard (new) | Very hard (new) | Worst seed (standard) |
|---|---|---|---|---|
| Real fly wiring | **78%** | **48%** | 25% | **61%** |
| Shuffled fly wiring | 67% | 40% | 22% | 3% |
| GRU (standard network) | 68% | 46% | **29%** | 0% |
| Dense, same neurons | 60% | 30% | 13% | 41% |

Per seed, standard level: real fly 77/91/95/61/66, shuffled 87/81/67/95/3,
GRU 0/92/78/91/78, dense 51/63/81/41/66.

**What we found, in plain words:**
1. **The real fly wiring is the most *reliable*.** Every one of its 5 seeds learned a
   good policy (worst 61%). The shuffled fly and the GRU each have one seed that
   failed completely (3% and 0%). Leave those out and they are as good as or slightly
   better than the real fly.
2. **The failures were a "hover trap".** The drone learns to sit still until time runs
   out. That's "safe" because it never crashes, but it never scores either. Counted over
   training, the networks fell into this trap: real fly 2 times, GRU 4, shuffled 6,
   dense 12. `gru_s0` collapsed into it near the end (at 4.7M steps) and never got out.
   `shuf_s4` kept falling in and out, and its final save landed in a bad spell.
   Switching random exploration back on during the test didn't rescue either one, so
   the test setup is fair.
3. **The fly-style wiring clearly beats a dense network built from the same simple
   neurons**, at every level. The gap grows as the mazes get harder (60 vs 78, 30 vs 48,
   13 vs 25).
4. **Against the standard GRU, it's a tie on average.** The fly is ahead on standard
   mazes, about equal on hard, and slightly behind on very hard. With 5 seeds and this
   much spread, these differences are **not** statistically solid.
5. **The real fly learns fastest early on** (see `figures/learning_curves.png`): about 85%
   at 2.2M steps vs. 73% shuffled, 58% GRU and 42% dense.

**Honest one-liner for the poster:** *"Wiring copied from the fly's navigation centre
learns faster and more reliably than a randomly rewired copy or a standard AI
network of the same size, and clearly beats a dense network built from the same
neurons. On average, though, a standard GRU ends up about as good."*

**Caveats to say out loud:** only 5 seeds per network; the test is the final save
point (no picking of the best checkpoint, which would be cheating); the hover trap is
a known failure mode of this training method (PPO), not something unique to our
networks.

**Figures** (made by `src/make_plots.py`):
- `figures/learning_curves.png`: success during training, mean ± spread over seeds.
- `figures/final_test.png`: final-test success per difficulty, one dot per seed.

---

## Watching a trained drone: `src/watch.py` (2026-10-24)

Training runs **headless**: the simulator computes all the physics but
draws nothing, which is hundreds of times faster than drawing it. To
actually *see* a trained network fly, use:

```bash
python src/watch.py conn_s0                  # opens the 3D window, 5 random mazes
python src/watch.py base_s0 --speed 2        # baseline network, twice as fast
python src/watch.py conn_s0 --no-gui --episodes 50   # no window, just the success rate
```

It works on runs that are still training (they save progress every few
minutes). Using the same `--seed` gives the same mazes, so you can show
both networks flying the *same* course side by side at the demo.

---

## Decision: real mazes come next, as a separate phase (2026-10-24, 23:55)

**Question raised:** the current arena is an open room with obstacles, not a proper
maze with corridors and dead ends. Can the drone solve a real maze?

**Answer:** probably not yet. The current task can be solved by "point at the goal and
dodge things". A real maze needs memory and exploration (going *away* from the goal
to get around a wall), which this training never asked for.

**Decision:** keep the two tasks separate, and finish within the next two days.
1. **Tonight:** finish the obstacle-avoidance experiment (fly vs. shuffled vs.
   ordinary), run the final test and make the plots. That's the first, complete result.
2. **Next:** build a real-maze environment (grid mazes with corridors and dead ends,
   still made from a seed) and train a second set of networks on it. These will be the
   same three kinds of network, either starting fresh or starting from the
   obstacle-trained ones.

**Why:** the obstacle experiment is almost done and gives a clean result to show. The
maze task is where the fly's navigation and memory circuit *might* matter most, so it's
worth doing as its own experiment rather than changing the current one halfway through.

## Step 8c (update): shuffled-fly runs finished (2026-10-25, 01:10) ✅

Final training-time scores (last 10 updates; noisy, not the final test):

| | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | average |
|---|---|---|---|---|---|---|
| Real fly | 73% | 89% | 86% | 71% | 69% | ~78% |
| Shuffled fly | 89% | 80% | 84% | 93% | 47% | ~78% |
| Ordinary | 60% | 57% | 76% | 40% | 61% | ~59% |

What it looks like: by the end of training, the shuffled fly is about **as good as the
real fly**. Both clearly beat the ordinary network. Earlier in training (at 2.2M steps)
the real fly was ahead (84% vs. 76%), so the real wiring may help it **learn faster**
without making it better in the end. The Step 9 test will check this properly.

(Note: a harmless "unexpected EOF" line in experiments.log came from editing
run_experiments.sh while it was running. All 5 runs finished and saved normally.)

## Step 8d: A fourth network, a standard GRU (2026-10-25, 00:40)

**Why:** our "ordinary" network was homemade. It uses the same simple neurons and
training settings as the fly network, and those settings were tuned while testing
the fly network. So beating it only shows the fly wiring beats *our* dense network.
An expert might ask: "would it beat a real, standard AI network?"

**What:** a **GRU** (Gated Recurrent Unit), the off-the-shelf memory network people
normally use for this kind of task. Its neurons have "gates", little dials that
decide how much old memory to keep and how much new information to let in. That
makes it good at memory and forgiving about training settings.

- Size: 133 neurons, 63,310 trainable settings (fly: 62,537, about 1% apart).
- One update per drone step (it doesn't need the fly net's 5 internal ticks).
- Same training code, same mazes, same 5M steps, seeds 0–4 (runs `gru_s0`–`gru_s4`).
- Queued to start automatically when the shuffled runs finish (`train-gru` unit).

**How to read the result:** if the fly network beats the GRU, it beats a standard AI
network of the same size (a strong claim). If the GRU wins, the honest claim is "fly
wiring beats an equivalent dense network built the same way, but a standard network
is still better."

## Step 10: Real mazes (started 2026-10-25, 03:00) 🔄

**Decision:** use **procedurally generated** mazes (made by code from a seed), with all
maze styles mixed together and a random difficulty every episode. We chose this over
an external maze library because:
- it gives unlimited variety (a library has hundreds of mazes at most, so the drone
  could memorise them);
- difficulty is a dial;
- the train/test split stays clean (seeds);
- it fits the existing simulator.

**The maze environment** (`src/grid_maze_env.py`):
- **Same arena, drone, sensors and controls** as the obstacle task. Because the inputs
  and outputs are identical, obstacle-trained networks can carry on learning here.
- **Grid of cells, random size each episode:** 3×3 up to 6×6 in training. Bigger means
  narrower corridors (1.7 m down to 0.8 m) and longer routes.
- **Three maze-building algorithms, picked at random each episode:**
  - *recursive backtracker:* long winding corridors;
  - *Kruskal* and *Prim:* lots of short dead ends.
- **Random "braiding":** 0–30% of dead ends are opened up, which creates loops and
  extra routes.
- **Start and goal:** start in one corner cell, goal in the opposite corner.
- **What the drone still senses:** the straight-line direction to the goal. In a maze
  that often points through a wall, so it has to learn to go *away* from the goal
  sometimes.
- **Reward:** progress along the **shortest route through the maze**, not straight-line
  distance. Otherwise the drone would be rewarded for pressing against a wall that
  happens to face the goal.
- **Time limit:** 10 s + enough time to fly the shortest route at 25% of top speed,
  capped at 90 s.
- **Check:** a scripted pilot that knows the route solved 40 out of 40 test mazes
  (3×3 to 7×7), so every maze is solvable within the time limit.

**Picking the best version (fixes the hover trap fairly):**
- Training now also saves a small snapshot every 20 updates.
- Afterwards, `src/select_best.py` flies each snapshot from the second half of training
  on 40 **practice mazes** (seeds from 500,000) and keeps the best one.
- The final test still uses separate mazes (seeds from 1,000,000), so the test plays
  no part in the choice.

**Final test levels:** 4×4, 6×6 and **7×7**. The 7×7 size is never seen in training,
so it's the robustness test. All three styles are mixed.

**Pilot (in progress):** two fly networks, about 1.5M steps each. One starts from its
obstacle training (`conn_s0`), one starts from scratch. This decides which way the
full experiment (4 networks × 5 seeds) is run.

**Pilot result (stopped at about 1M steps):**

| Pilot | Reaches goal during training |
|---|---|
| From scratch | 50–63% |
| From obstacle training | 13–27% |

- **Surprise: from scratch learns mazes much better.** Likely reason: the obstacle-trained
  network has a strong "beeline to the goal" habit, which sends it into walls in a maze.
  It also explores less because it's already confident.
- **By maze size** (from-scratch pilot, 15 practice mazes each): 4×4 7/15, 6×6 2/15,
  7×7 0/15. Most failures in the big mazes are crashes, because the corridors are narrower.

**Decision:** train all maze networks **from scratch**. It learns better and is cleaner
to explain.

**Full maze experiment launched 2026-10-25, 03:26:**
- Fly, shuffled fly, GRU and dense, seeds 0–4 = 20 runs, 5M steps each, 5 at a time.
- Runs `runs/maze_*`, under the `train-maze` systemd unit.
- Expected to finish around 09:30.
- Then: `python src/select_best.py`, then `python src/evaluate.py --task grid`, then plots.

## Step 11: The "hover trap" was partly a bug, fixed (2026-10-26, 15:50)

**Question raised:** "Why is the GRU all over the place? Are we training it right?"

**Checked:** the right models are being trained (GRU 133 neurons / 63,310 settings,
etc.), with identical, textbook training settings for all four.

**Found:** a flaw in how commands are handled.
- **How commands work:** the network outputs a turn and a speed command, and the
  simulator cuts anything outside −1…+1 down to the limit.
- **How it learns:** during training it tries random variations around its chosen
  command.
- **What went wrong:** if the chosen command drifts far past the limit (e.g. speed
  −2.5), every variation gets cut to the same "stop". The network gets no feedback,
  can't learn its way back, and is stuck hovering or spinning.
- **Measured on the maze runs:** the GRU's speed command was past the limit 86% of the
  time (81% far past it), vs. 16–26% for the real fly runs. This is what the
  "hover trap" really was.

**Fix:** pass the command through a smooth squash (`tanh`) so it can never go past ±1.
Random variations then always give feedback. This is standard practice. It's in
`Agent` in `src/ppo.py` (`--squash 1`, now the default), and the same for all four
networks.

**Decisions:**
- The 5 maze runs in progress were trained with the flaw. They're stopped and archived
  in `runs/v2_maze_unsquashed/`. The maze experiment restarts from scratch with the fix.
- The **obstacle results (Step 9) were trained without the fix**, and we keep them, but
  with this caveat: part of the real fly's "reliability" advantage may be that its
  wiring happened to avoid this trap better (a bug, not biology). Rerun them with the
  fix if time allows.
- Before restarting, five independent reviewers are checking the whole codebase for
  further bugs.

## Step 12: Full bug review by five independent reviewers (2026-10-26)

**What we did:** five independent reviewers each checked one part of the project:
the training algorithm, the simulators, the networks/fairness, the testing pipeline,
and an outside-expert audit of the whole project. Every finding below was re-checked,
fixed and tested before restarting.

**Verdict:** no bug that makes the obstacle numbers wrong (one reviewer re-flew 4 runs
and reproduced `eval_results.csv` exactly). But several things made the comparison
less fair or less clean than we claimed.

### Bugs and unfairness found, and fixed

| # | Problem (plain words) | Effect | Fix |
|---|---|---|---|
| 1 | **Hover trap** (Step 11): commands could drift past ±1 and stop giving feedback | GRU and shuffled runs got stuck hovering | tanh squash on the command |
| 2 | **Shuffled net started with 6–8× stronger connections** than the real fly. Each was scaled separately, and the real wiring has one very strong inhibitory loop among the ring neurons that pulls its whole scale down | The fly-vs-shuffled comparison was not "same weights, only positions differ" | The shuffled net now uses the real fly's scale, so its connection strengths are identical (verified) |
| 3 | **Shuffle didn't keep each neuron's mix of + and − inputs.** E.g. the output (PFL) neurons got ~700 instead of 1,282 excitatory inputs | The shuffled net's outputs started heavily inhibited | Shuffle + and − connections separately with "edge swaps" (Maslov–Sneppen), which keeps every neuron's out-, excitatory-in and inhibitory-in counts exactly. Overlap with the real wiring is now 27% |
| 4 | **The autopilot (PID) remembered the previous episode** (its last tilt and built-up correction) | The same maze could end differently depending on the episode before it (one test went goal → crash) | Reset the autopilot at every episode start |
| 5 | **Training "seeds" shared most of their mazes**: seed 0 and seed 1 saw 15 of the same 16 maze sequences | The 5 seeds were less independent than claimed, so seed-to-seed spread was understated | Each seed now gets its own maze streams (seed × 50,000 + env), never touching practice or test mazes |
| 6 | **Timeouts were treated like a final ending**, but the drone can't see the clock | Noisy learning signal, worst for slow or hovering drones | The value of the cut-off state is now added (standard "bootstrapping") |
| 7 | **Maze reward jumped at corners,** and by up to ~9 points at some cells in mazes with loops | An unfair, noisy reward | A smooth "distance through the maze" map (shortest path on a 5 cm grid, walls widened by the drone's size) |
| 8 | **Lidar missed a wall in the last few cm** (rays started inside the wall) | A blind spot just before a crash | Rays start just outside the drone's body (6.5 cm) |
| 9 | Pipeline: failed runs didn't make the script fail; fine-tuned and from-scratch maze runs could get the same names; maze tests could silently use the wrong saved version | Could mix up results without warning | Fixed; results now record which saved version was tested; resuming checks the settings match |
| 10 | Spreads used the "population" standard deviation | Spreads looked ~11% smaller | Sample standard deviation everywhere |

Also new: a column `action_saturation` in every training log shows how often the
command sits at its limit, so we can show the hover trap is gone.

### What this means for the obstacle results (Step 9)
They were made with bugs 1–8 present. Specifically (from the outside-expert audit):
- **"More reliable" mostly came from the hover-trap bug**, which hit one GRU seed and one
  shuffled seed. Without those two seeds, shuffled (83%) and GRU (85%) are as good as the
  fly (78%).
- **None of the final-test differences are statistically significant** with 5 seeds
  (permutation tests: fly vs dense p≈0.13–0.20; vs shuffled and GRU p>0.6).
- **The one robust result:** early in training, the real fly beat its shuffled copy in all
  5 seed pairs (p≈0.008). That comparison was made with the old, unequal weight scaling
  (bug 2), though.
- **Fair-claims rule from now on:** report means with every seed shown, say "not significant
  at n=5" where true, and pick the comparison metric *before* looking at the results.

### Honest design caveats (not bugs; say these if asked)
- **Unequal inputs and outputs:** the fly net gets sensor input into 738 of its 922 neurons
  and reads 50 output neurons. The dense net feeds and reads all 238.
- **Unequal critic:** the "critic" (the part that predicts reward) reads the whole network, so
  it is bigger for bigger networks.
- **Unequal tuning:** the training settings and the neuron model were tuned while working on
  the fly net.
- **The fly circuit may be partly bypassed:** many input neurons connect directly to the output
  neurons, so the compass circuit (EPG/PEN/Δ7) may barely be used. A "lesion test" (switch those
  neurons off and re-test) would show it.
- **Mazes don't strictly need memory:** start and goal are corners on the outer wall, so "follow one wall"
  solves every maze without loops. Don't claim the maze tests memory.
- **7×7 is mainly a narrower-corridor test**, not a much longer maze. Report it as that.

## Step 13: Mid-training check found a "spin trap", fixed (2026-10-26, 17:20)

**What we did:** at 75% of the first maze batch, a fresh reviewer checked the runs and
flew each network through practice mazes to see *how* they succeed and fail.

**Bug found: the squash fix (Step 11) had a milder version of the same trap.**
- The squash curve (`tanh`) keeps commands inside ±1, but if the network's raw output keeps
  growing (e.g. +8), the command stays pinned at the limit and the learning signal shrinks
  to almost nothing (about a millionth).
- **Real fly, seed 0:** 74% success at 3.2M steps, then got stuck with its turn command at
  full left (flying in circles) and fell to 0–13% for over a million steps.
- **Dense, seed 0:** was sliding into it. **GRU, seed 0:** fell in briefly and recovered.
- Whether a run falls in is luck, and with 5 seeds that could decide the comparison.

**Fix:** a small penalty on the raw output beyond ±2 (`--raw-coef 0.01`, `RAW_LIMIT = 2`
in `src/ppo.py`). It's not a hard cap; it gently pulls the output back so feedback keeps
flowing. **Tested on the collapsed run:** after about 5 minutes of extra training with the
penalty, "stuck at full left" fell from 76% to 7% and success rose from 21% to 61%.

The training log now shows saturation by direction (`sat_turn_left`, `sat_turn_right`,
`sat_full_speed`, `sat_stop`, `raw_over_limit`), because "at full speed" is fine but
"stuck turning" is not.

**Decision:** restart all 20 maze runs with the fix. The interrupted runs are archived in
`runs/v3_maze_spintrap/`.

**What the reviewer learned about behaviour** (useful for explaining results):
- **Small mazes (3×3–5×5):** every network solves these almost perfectly. Big mazes separate
  them.
- **The real fly flies fast:** about 0.9 m/s, often at full throttle, finishing faster. When
  lost, it circles.
- **The shuffled fly, GRU and dense network creep:** about 0.2 m/s with frequent stops; when lost,
  they inch along until time runs out.
- **Why creeping makes sense to them:** a crash costs −10 at once, while progress pays the
  same whether it's fast or slow.
- **Crashes are almost always at wall corners and wall ends.** The 16 lidar rays are 22.5°
  apart, so a thin wall end can slip between them (a design caveat).
- **Mazes with loops are easier** (about 64% vs 33%), because a wrong turn isn't a dead end.
  **Prim-style mazes** (many short dead ends) are the hardest.
- **Checked and fine:** no blow-ups, the reward matches the outcomes, crashes are real
  collisions, the distance map is correct, and no maze leaks between training, practice and
  test.

## Step 14: Speed reward and remaining fixes, before the final restart (2026-10-26, 18:00)

**Why:** you asked to reward speed (the networks had learned to creep) and to fix every
remaining known issue *now*, not after another long run. The maze runs started at 17:30
were stopped and archived in `runs/v4_maze_prespeed/`.

**1. A small reward for speed, done safely.**
- A plain speed reward can be farmed by flying in fast circles forever (the spin trap again).
- So the bonus (0.01 per step at full speed) is **smaller than the time cost** (raised from
  0.01 to 0.03 per step). Flying fast without getting closer to the goal still loses reward
  overall; it just loses less than hovering.
- **Checked:** the total discounted time cost is at most 0.03 / (1 − 0.99) = 3, well below the
  −10 crash penalty, so crashing on purpose never pays.
- **Tested with scripted behaviours** (5×5 mazes, total reward):

| Behaviour | Total reward |
|---|---|
| Fly the route fast | **+33.8** |
| Medium speed | +30.5 |
| Creep | +3.8 |
| Hover or spin in place | −39.6 |
| Circle at full speed | crashes (worst once future rewards are discounted) |

  Faster is better, and there's no way to farm reward.

**2. 32 lidar rays instead of 16** (11.25° apart). Crashes were concentrated at thin wall
ends that slipped between rays 22.5° apart. This changes the input size from 23 to 39, so
every network's size changes a little but stays matched:

| Network | Trainable settings |
|---|---|
| Fly | 74,345 |
| GRU (138 neurons) | 74,384 |
| Dense (253 neurons) | 74,637 |

The old obstacle runs can no longer be loaded with the new sensors. They're rerun anyway.

**3. Start heading in mazes:** the drone now starts facing along an open corridor, instead of
diagonally at its start cell's inner corner (half of all crashes used to happen in the
first few seconds).

**4. Route map:** walls in the distance map now also extend past their ends, by the wall's
half-thickness plus the drone's radius (the shaped route used to cut about 10 cm past wall tips).

**Checks:** the scripted pilot still solves 32 of 32 mazes (3×3 to 7×7), and parameter counts
are still matched. Two more fresh reviewers (a reward "red team" and a review of all of
today's code changes) checked everything before the restart.

**Their findings, all fixed before launch:**

| # | Found | Fix |
|---|---|---|
| 1 | **The distance map still pulled the drone onto wall corners.** The shaped route had zero clearance, and the values filled in near walls were too steep. | The route now keeps 10 cm extra clearance. Points near walls get "nearest free point + distance", so walls gently push the drone away. Check: on 15,190 reachable moves toward wall ends, moving closer usually *costs* reward, and only 0.03% are over-rewarded (slightly). |
| 2 | **Rapid stop/go commands could count as a "crash" with nothing hit** (autopilot overshoot past the 0.8 rad tilt limit). | Only a real flip (>1.2 rad) counts as a crash. |
| 3 | **The speed bonus made circling slightly better than hovering.** | The bonus now pays only for speed *towards the goal* (along the route), so circling earns nothing from it. |
| 4 | **Old saved runs (16-ray sensors) crashed the tools** with an unclear error. | Runs now save their sensor size; old runs are refused with a clear message. The unused "fine-tune from obstacle runs" option was removed. |
| 5 | The experiment script still reported success when a run failed. | It now exits with an error. |
| 6 | Small things: the raw-output penalty also applied to unsquashed agents; the resume check now compares more settings; wrong indices in a smoke test. | Fixed. |

**Caveat noted, not changed:** training flies with random exploration noise, while the test
uses the network's best-guess command. The size of that difference depends on each network's
learned noise level. It's the same rule for every network; mention it if asked.

**Final checks:** scripted pilot 32 of 32 mazes; reward ranking still fast > medium > creep >
hover/circle; all four networks train, save and resume.

## Step 15: Midway health check, and fixing the "best version" rule in advance (2026-10-26, 18:40)

**Health check at 50% of the first batch:** no errors, and the traps are fixed. Commands
stuck at a limit only 0–10% of the time, raw outputs past the threshold only 1.5–9%
(compare 60–86% in the old bad runs). The real fly, dense and shuffled runs are learning
steadily.

**GRU wobble (not a code bug):** the GRU reached 67% at about 1.6M steps, then drifted to
about 20%. Flying its best and its current version on 12 practice mazes:

| GRU version | Reaches goal | Speed | "Stop" commands |
|---|---|---|---|
| Best | 8/12 | 0.54 m/s | 18% |
| Now | 4/12 | 0.24 m/s | 38% |

It has slipped back into slow creeping. Its commands are *not* pinned, so it can recover.
The usual cause is a constant learning rate; the standard remedy (lowering it over training)
would mean another restart, so we don't change it now.

**Rule fixed now, before any final results are seen** (so it can't be bent to favour anyone):
- The best-version picker considers **every snapshot after the first 25% of training** (it
  was 50%), for all networks.
- It chooses on practice mazes only.
- We report **both** the best-version and the final-version test scores
  (`evaluate.py --task grid --checkpoint best` and `--checkpoint last`). A big gap
  between the two means the network is unstable, so instability stays visible.

## Step 16: The drone orbits the goal; a "can I see the goal?" fix, ready to pilot (2026-10-26, 20:30)

**Spotted by you while watching:** near the goal, especially with walls close by, the drone
goes back and forth and never finishes.

**Measured:**
- **How often:** the real fly network (seed 1) got within 0.30–0.46 m of the goal (it counts at
  0.30 m) and then ran out of time in 2 of 15 4×4 mazes and 4 of 15 6×6 mazes.
- **What it does:** a traced episode shows it arriving at full speed, just missing the circle,
  turning the *wrong way* (the goal is on its left but it turns right), looping out about 1 m,
  coming back, and repeating.

**Why (tested, not guessed):** we shifted the goal direction the network sees by 30° and
measured how much its steering changed. A goal-steering drone would change by about 1.0;
all the networks change by only about 0.0–0.26. **They have learned to mostly ignore the goal
direction.** In a maze it usually points through a wall, so they navigate by the walls instead.
At the very end the direction becomes exactly what's needed, but the network can't tell when it
has become trustworthy.

**Fix candidate (your question "are you sure?" led to a better one):** the first idea, a sharper
"how close am I" input, probably wouldn't help, since the network ignores the direction even
when close. Instead:
1. **Add a "goal in line of sight" input (1/0)**, checked with one ray from the drone to the goal.
   It tells the network exactly when the goal direction can be trusted. Checked: it reads
   visible 100% of the time in the goal cell, and 18% of the time 3+ cells away (a straight
   corridor to the goal).
2. **Goal circle 0.30 → 0.40 m** as a safety net (the closest misses were 0.30–0.46 m).

**Plan:** test first with **one** real-fly network (seed 0) before changing anything.
- The code is in a separate copy, `pilot_goalsight/`, so the running 20-run experiment is not
  affected. (Runs that start later reload the code, so editing the main code now would have
  mixed old and new.)
- Start: `bash pilot_goalsight/start_pilot.sh`
- Compare with the old seed-0 run at the same point: `bash pilot_goalsight/analyze.sh`. It reports
  success, goal near-misses, and steering response with and without the goal in sight.
- **Waiting for your go-ahead to start.**

## Step 17: Maze runs finished; pilot started; plan for tomorrow (2026-10-27, 02:45)

- **All 20 maze runs finished training** (02:09). Best-version picking was started, then
  stopped once the pilot finished. Those runs will be replaced by the goal-sight version,
  and their picking and testing can be rerun at any time (`select_best.py`,
  `evaluate.py --task grid`) if needed.
- **Goal-sight pilot (one real-fly network, seed 0), training-time numbers at matched points:**

| Point | Pilot | Old seed-0 run |
|---|---|---|
| Update 296 | 64% success, 4% timeouts | 59% success, 14% timeouts |
| Update 752 | 73% success, 18% crashes, 9% timeouts | 65% success, 22% crashes, 13% timeouts |

- **Pilot result (finished 03:16; compared with the old seed-0 run, both at update 1,220):**

| Measure | Pilot (goal sight) | Old run |
|---|---|---|
| Training success (last 40 updates) | **83%** | 72% |
| 30 practice mazes (4×4–6×6): goals | **22** | 18 |
| crashes | 5 | 4 |
| timeouts | **3** | 8 |
| near-misses at the goal | **0** | 1 |
| Steering toward the goal within 2 m, goal **visible** | **+0.32** | +0.05 |
| Steering toward the goal within 2 m, goal **hidden** | +0.04 | +0.02 |

  **The network learned to use the sensor.** It steers about 6× more strongly toward a goal it can
  see, and still ignores the direction when a wall is in the way. So the gain isn't only the
  bigger goal circle. Caveat: one run vs one run, and +0.32 is still far from full "beeline" steering.
  (The analysis script now uses one CPU thread per analysis; with all 24 threads it crawled.)
- **Decision (yours):** after the pilot, stop for the night. Tomorrow, retrain everything with
  the goal-sight changes (the line-of-sight sensor and the 0.40 m goal circle). That means
  moving them from `pilot_goalsight/src` into `src/`, archiving the current maze runs, and
  rerunning the 20 maze runs and the 20 obstacle runs. Then the tuned GRU (and dense, if time).

## Step 18: Full retrain with the goal-sight fix (started 2026-10-27, 13:11)

- **The fix moved into the main code** (`src/maze_env.py`, the only file that differed from the
  pilot copy): the "goal in line of sight" input and the 0.40 m goal circle. There are now 40 inputs.
- **Networks still size-matched:**

| Network | Trainable settings |
|---|---|
| Fly | 75,083 |
| Dense (254 neurons) | 75,440 |
| GRU (139 neurons) | 75,757 |

- **Archived:** the last maze runs (without goal sight) in `runs/v5_maze_nosight/`, and the old
  obstacle runs (16 rays, pre-fix) in `runs/v6_obstacles_16rays/`, together with their
  `eval_results.csv`.
- **One restartable script runs everything in order** (`src/run_everything.sh`, systemd unit
  `everything`):
  1. 20 maze runs;
  2. best-version picking;
  3. maze tests (best and final versions);
  4. 20 obstacle runs;
  5. obstacle test.

  If stopped, run the same command again: finished runs are skipped, and the rest resume
  from their save points.
- **Expected:**

| Stage | Expected around |
|---|---|
| Maze training done | 19:00 |
| Maze tests done | 20:00 |
| Obstacle training and test done | 01:30 |

- **What actually happened (2026-10-27):**
  - All 20 maze runs finished at 18:58, with no errors.
  - Average training success over the last 60 updates:

    | Network | Average success | Seeds |
    |---|---|---|
    | Real fly | ~84% | 83–85% |
    | Shuffled | ~69% | 62–78% |
    | GRU | ~67% | 59–72% |
    | Dense | ~61% | 59–65% |

    These are training numbers, not test results.
  - Best-version picking turned out to be slow: each snapshot takes about 4–5 minutes, so the 920 snapshots need about 4 hours in total.
  - Paused at 20:15 (no overnight power), partway through picking. That step has no save points, so it starts again from scratch when resumed (same command). Maze tests and obstacle runs come after it.

## Step 19 — Pilot: longer memory training + trap practice (prepared 2026-10-27, not started)

- **Problem (user-observed, `watch.py maze_conn_s2 --size 5`, 5th maze, seed 1004):**
  - The drone went up from the start into a big dead-end area that leads towards the goal.
  - It then flew back and forth between two dead ends for 40 s and timed out.
  - It never went back past the start to take the real route along the bottom.
- **How common it is:** on 20 practice mazes, `maze_conn_s0` (final version) timed out 4 times, and **all 4 were loops** (it entered some cell 3 or more times).
- **Causes:**
  1. Training only looks back 32 steps (about 1 s) through the network's memory, so remembering "I already checked that corridor" for longer is never trained directly.
  2. The goal direction keeps pulling the drone back into dead ends that point towards the goal.
  3. Watching and testing use the network's exact choice (no randomness), so a loop never breaks by chance.
- **Fix 1 (longer memory training):** `--chunk-len` goes from 32 to **128** steps (about 4.3 s).
  - This is the same for every network, since it is purely a training setting.
  - It costs time. In a 4-update test run on its own: 1280 steps/s at 32, 840 at 128, 650 at 256.
  - 256 (the whole 8.5 s rollout) would make training about 2 times slower; 128 makes it about 1.5 times slower. We picked 128 as the middle ground: it covers a trip into a dead end and back.
- **Fix 3 (trap practice), and why it isn't simply "more dead ends":**
  - I measured the current mazes first. They already have plenty of dead ends: 5×5 mazes have 2–6 dead ends, and the deepest detour is about 5–7 cells. Fewer loops (braid 0) barely changes this.
  - What's missing is practice at *getting out of* a dead end against the pull of the goal.
  - So **30% of training mazes start the drone at the end of a dead end, facing the end wall**, as if it had just flown in.
  - The dead end is one of the 3 most misleading ones in that maze: those where the way out is longest compared with the straight-line distance to the goal.
  - The time budget is based on the route from that start.
- **Checks:**
  - Practice and test mazes are **unchanged**: the first 40 seeds of each match the current code exactly (layout, start, heading, budget). That's because trap starts are only on in training and draw no extra random numbers otherwise.
  - The scripted route-follower reaches the goal from 40 out of 40 trap starts.
  - Hovering in place after a trap start never crashes (0 of 20).
  - A 2-update training dry run works.
- **The pilot:**
  - One real fly network, seed 0, 5M steps. Start it with `bash pilot_memory/start_pilot.sh`.
  - Compare it with `bash pilot_memory/analyze.sh`, which matches `runs/maze_conn_s0` at the same training update and flies both on the same 100 practice mazes. It reports goals, crashes, timeouts, how many timeouts were loops, and the 5×5–6×6 mazes on their own.
- **Decision rule:** if the pilot has fewer loops and at least as many goals, both fixes go into the main code and all 20 maze runs are retrained, like the goal-sight pilot (Step 17).

## Status and what's next

| Day | Task | Status |
|---|---|---|
| 1 | Download fly wiring data | ✅ done |
| 2 | Build the connectome network | ✅ done |
| 3–4 | Maze environment + memory-aware training loop | ✅ done; both networks learn the task |
| 5–8 | Real training runs: connectome net vs. ordinary net | ✅ all 10 finished 2026-10-24 23:46 |
| 5–8 | Shuffled-fly control (5 seeds) | ✅ finished 01:10 |
| 5–8 | GRU standard-network baseline (5 seeds) | ✅ finished 02:12 |
| 9 | Final test (20 networks × 3 levels × 100 mazes) + plots | ✅ done 02:40 |
| next | Real mazes: full training (20 runs), all fixes up to Step 14 | ✅ finished 2026-10-27 02:09 (superseded by the goal-sight retrain) |
| now | **Full retrain with goal sight:** 20 maze + 20 obstacle runs, plus tests (`src/run_everything.sh`) | 🔄 started 2026-10-27 13:11 |
| later | **Tuned GRU:** short tuning runs (input layer, standard init, decaying learning rate, 1–2 learning rates), picked on practice mazes only; then 5 GRU seeds with the winner; report "GRU (untuned)" and "GRU (tuned)" side by side. If time: the same for the dense network. | ⏳ planned (2026-10-26). Why: the GRU was unstable (65% → 24%) using settings tuned on the fly network; an untuned baseline is an easy target for criticism. |

**To resume the maze training** (picks up from each run's last save point, skips finished runs):

```
cd ~/Downloads/fly_connectome_drone
systemd-run --user --unit=train-maze --working-directory="$PWD" -p MemoryMax=22G -p MemorySwapMax=0 -p OOMPolicy=continue bash -c 'TASK=grid MODELS="connectome shuffled gru baseline" SEEDS="0 1 2 3 4" PARALLEL=5 bash src/run_experiments.sh >> runs/experiments.log 2>&1'
```

Paused because of voltage fluctuations overnight (not safe to leave the laptop
plugged in). Remaining: about 5 hours of training.
| 10 | Poster/demo prep, rehearsal (buffer) | ⏳ |
