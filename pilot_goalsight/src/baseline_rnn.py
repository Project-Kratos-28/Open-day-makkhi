"""
Baseline for the comparison: an ordinary, fully connected recurrent network
with the same dynamics as ConnectomeRNN (leaky rate neurons, ReLU, same
substeps) and about the same number of trainable parameters, but with no
biological wiring: every neuron may connect to every other, and each
connection is free to be excitatory or inhibitory.

Run this file directly to print the parameter match: python src/baseline_rnn.py
"""
import math

import torch
import torch.nn as nn

from connectome_rnn import ConnectomeRNN, count_params, run_leaky_rnn


def matched_hidden_size(target_params, obs_dim, action_dim):
    """Smallest n with n*n + n*(obs_dim+1) + action_dim*(n+1) >= target_params."""
    b = obs_dim + 1 + action_dim
    n = (-b + math.sqrt(b * b + 4 * (target_params - action_dim))) / 2
    return math.ceil(n)


class BaselineRNN(nn.Module):

    def __init__(self, obs_dim, action_dim, n=None, target_params=None,
                 substeps=5, alpha=0.5, spectral_radius=0.9):
        super().__init__()
        if n is None:
            n = matched_hidden_size(target_params, obs_dim, action_dim)
        self.n = n
        self.substeps = substeps
        self.alpha = alpha
        w0 = torch.randn(n, n) / math.sqrt(n)
        w0 *= spectral_radius / torch.linalg.eigvals(w0).abs().max().item()
        self.w = nn.Parameter(w0)
        self.encoder = nn.Linear(obs_dim, n)
        self.decoder = nn.Linear(n, action_dim)

    def weight(self):
        return self.w

    def drive(self, obs):
        return self.encoder(obs)

    def readout(self, h):
        return self.decoder(h)

    def initial_state(self, batch_size, device=None):
        return torch.zeros(batch_size, self.n, device=device or self.w.device)

    def forward(self, obs, h):
        actions, hs = self.forward_sequence(obs[None], h)
        return actions[0], hs[0]

    def forward_sequence(self, obs_seq, h, reset_seq=None):
        return run_leaky_rnn(self, obs_seq, h, reset_seq)


def gru_hidden_size(target_params, obs_dim, action_dim):
    """Smallest n with 3*(n*obs_dim + n*n + 2n) + action_dim*(n+1) >= target_params."""
    b = 3 * (obs_dim + 2) + action_dim
    n = (-b + math.sqrt(b * b + 12 * (target_params - action_dim))) / 6
    return math.ceil(n)


class GRUNet(nn.Module):
    """
    Second, stronger baseline: a standard GRU, the off-the-shelf memory network
    used in RL, sized to the connectome net's parameter count. Unlike the other
    nets it uses its own (gated) neurons and takes one step per env step, so it
    is not handicapped by dynamics that were tuned on the connectome.
    """

    def __init__(self, obs_dim, action_dim, target_params):
        super().__init__()
        self.n = gru_hidden_size(target_params, obs_dim, action_dim)
        self.cell = nn.GRUCell(obs_dim, self.n)
        self.decoder = nn.Linear(self.n, action_dim)

    def initial_state(self, batch_size, device=None):
        return torch.zeros(batch_size, self.n, device=device or self.decoder.weight.device)

    def forward(self, obs, h):
        actions, hs = self.forward_sequence(obs[None], h)
        return actions[0], hs[0]

    def forward_sequence(self, obs_seq, h, reset_seq=None):
        hs = []
        for t in range(obs_seq.shape[0]):
            if reset_seq is not None:
                h = h * (~reset_seq[t]).unsqueeze(-1)
            h = self.cell(obs_seq[t], h)
            hs.append(h)
        hs = torch.stack(hs)
        return self.decoder(hs), hs


def make_model(kind, obs_dim, action_dim, min_synapses=3, seed=0):
    """
    kind: 'connectome', 'baseline' (sized to match the connectome net),
    'shuffled' (connectome net with randomly rewired connections; `seed`
    picks the rewiring, so each training seed gets its own random wiring)
    or 'gru' (standard GRU, also size-matched).
    """
    if kind == "shuffled":
        return ConnectomeRNN(obs_dim, action_dim, min_synapses=min_synapses, shuffle_seed=seed)
    connectome = ConnectomeRNN(obs_dim, action_dim, min_synapses=min_synapses)
    if kind == "connectome":
        return connectome
    if kind == "baseline":
        return BaselineRNN(obs_dim, action_dim, target_params=count_params(connectome))
    if kind == "gru":
        return GRUNet(obs_dim, action_dim, target_params=count_params(connectome))
    raise ValueError(kind)


if __name__ == "__main__":
    from maze_env import OBS_DIM, ACT_DIM
    c = make_model("connectome", OBS_DIM, ACT_DIM)
    b = make_model("baseline", OBS_DIM, ACT_DIM)
    print(f"connectome: {c.n} neurons, {count_params(c)} params")
    print(f"baseline:   {b.n} neurons, {count_params(b)} params")
    h = b.initial_state(4)
    for _ in range(200):
        a, h = b(torch.randn(4, OBS_DIM), h)
    assert torch.isfinite(h).all()
    print(f"baseline after 200 random steps: mean rate {h.mean():.3f}, max {h.max():.3f}")
