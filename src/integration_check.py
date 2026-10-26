"""
Closes the full loop once: MazeAviary <-> ConnectomeRNN, with the real
observation/action dimensions (not the placeholder 16/4 used in
connectome_rnn.py's own self-test). Confirms the connectome network can
actually drive the drone sim end to end before any RL training starts.

Run: python src/integration_check.py
"""
import numpy as np
import torch

from maze_env import MazeAviary
from connectome_rnn import ConnectomeRNN, count_params


def main():
    env = MazeAviary(gui=False)
    obs_dim = env.observation_space.shape[-1]
    action_dim = env.action_space.shape[-1]
    print(f"obs_dim={obs_dim}  action_dim={action_dim}")

    net = ConnectomeRNN(obs_dim, action_dim)
    print(f"connectome net: {net.n} neurons, {count_params(net)} trainable params")

    obs, info = env.reset(seed=0)
    h = net.initial_state(batch_size=1)

    total_reward = 0.0
    for t in range(300):
        obs_t = torch.as_tensor(obs.reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            action, h = net(obs_t, h)
        action_np = action.numpy().reshape(env.action_space.shape)
        action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
        obs, reward, terminated, truncated, info = env.step(action_np)
        total_reward += reward
        if terminated or truncated:
            print(f"  episode ended at step {t} (terminated={terminated}, truncated={truncated}), "
                  f"reward so far {total_reward:.2f}")
            obs, info = env.reset(seed=0)
            h = net.initial_state(batch_size=1)

    assert np.isfinite(total_reward)
    print(f"OK: connectome net drove the drone for 300 steps, total reward {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    main()
