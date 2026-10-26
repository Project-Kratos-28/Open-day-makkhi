"""
Recurrent PPO for ConnectomeRNN / BaselineRNN on the maze task.

Unlike stable-baselines3's PPO, this keeps the network's memory (its neuron
activity) across time steps and trains through it: rollouts are cut into
chunks of CHUNK_LEN steps, and during the update each chunk is replayed from
the neuron activity it started with, so gradients flow back through time.

Usage:
    python src/train_ppo.py --model connectome --steps 5000000 --name conn_s0
    python src/train_ppo.py --model baseline   --steps 5000000 --name base_s0

Writes runs/<name>/log.csv (one row per update) and runs/<name>/model.pt.

Crash-safe: model.pt is a full save point (network, optimiser, random-number
state, progress), written every SAVE_EVERY updates (~5 min). Re-running the
same command picks up from the last save point instead of starting over.
On resume the mazes and the network's short-term memory start fresh, which
is the same as the start of a new batch of episodes.
"""

# The training code lives in ppo.py. This file stays tiny on purpose: the 16 env
# worker processes are started with "spawn", and each one re-imports this script.
# Importing torch here would cost every worker ~350 MB of RAM (~5.5 GB per run).
if __name__ == "__main__":
    from ppo import main
    main()
