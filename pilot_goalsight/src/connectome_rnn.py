"""
Recurrent network whose wiring is the fly central complex (from
data/neurons.csv and data/connections.csv, see extract_connectome.py).

Frozen (biology):  which neuron connects to which, and the sign of each
                   connection (from the presynaptic neuron's transmitter).
Trained (RL):      the strength of each existing connection, a sensor
                   encoder (obs -> input neurons) and a motor decoder
                   (output neurons -> action).

Weight of the connection pre -> post is  sign[pre] * |raw[k]|,
where raw is one trainable number per connection, so training can change
a connection's strength but never its sign or whether it exists.

Run this file directly for a sanity check: python src/connectome_rnn.py
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Transmitter -> sign. Glutamate is inhibitory in the CX (GluCl receptors).
NT_SIGN = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0}

# ER ring neurons carry sensory (visual/landmark) input into the EB; PFN
# neurons get self-motion input via the noduli. PFL neurons are the CX's
# steering outputs (to the LAL, downstream of which are descending neurons).
INPUT_FAMILIES = ("ER", "PFN")
OUTPUT_FAMILIES = ("PFL",)
FAMILY_REGEX = r"^(EPG|PEN|PFL|PFN|ER|Delta7)"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Maximum firing rate. Without a ceiling, activity occasionally ran away to
# infinity during training (killed one baseline run, hit one connectome run).
# Rates saturate smoothly towards MAX_RATE; typical rates (~0-3) are barely affected.
MAX_RATE = 10.0


def load_connectome(neurons_csv=DATA_DIR / "neurons.csv",
                    connections_csv=DATA_DIR / "connections.csv",
                    min_synapses=3):
    """Returns (neurons DataFrame indexed 0..N-1, pre_idx, post_idx, synapse_counts)."""
    neurons = pd.read_csv(neurons_csv)
    neurons["family"] = neurons["type"].str.extract(FAMILY_REGEX)[0]
    unknown_nt = set(neurons["consensusNt"]) - set(NT_SIGN)
    if unknown_nt:
        raise ValueError(f"No sign defined for transmitter(s) {unknown_nt}; add to NT_SIGN.")
    neurons = neurons.sort_values(["family", "type", "bodyId"]).reset_index(drop=True)

    conns = pd.read_csv(connections_csv)
    conns = conns[conns["weight"] >= min_synapses]
    index_of = pd.Series(neurons.index, index=neurons["bodyId"])
    pre = index_of[conns["bodyId_pre"]].to_numpy()
    post = index_of[conns["bodyId_post"]].to_numpy()
    return neurons, pre, post, conns["weight"].to_numpy(dtype=np.float32)


def shuffle_wiring(pre, post, n, seed, groups=None, swaps_per_edge=10):
    """
    Randomly rewire the connections while keeping, for every neuron, how many
    connections it sends and receives and the set of synapse counts. Only
    *which* neuron talks to which is randomised. No self-loops or duplicate pairs.

    Method: repeated double-edge swaps (Maslov & Sneppen 2002). Pick two
    connections a->b and c->d and rewire them to a->d and c->b, unless that would
    create a self-loop or a connection that already exists. Every swap keeps all
    in- and out-degrees exactly.

    groups: optional label per connection (e.g. its sign); swaps only happen
    within a group, so every neuron also keeps how many excitatory and how many
    inhibitory inputs it receives.
    """
    rng = np.random.default_rng(seed)
    pre, post = np.asarray(pre).copy(), np.asarray(post).copy()
    groups = np.zeros(len(pre), dtype=int) if groups is None else np.asarray(groups)
    existing = set(zip(pre.tolist(), post.tolist()))
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        n_tries = swaps_per_edge * len(idx)
        picks = rng.choice(idx, size=(n_tries, 2))
        for i, j in picks:
            a, b, c, d = pre[i], post[i], pre[j], post[j]
            if a == d or c == b or (a, d) in existing or (c, b) in existing:
                continue
            existing -= {(a, b), (c, d)}
            existing |= {(a, d), (c, b)}
            post[i], post[j] = d, b
    return pre, post


class ConnectomeRNN(nn.Module):
    """
    Leaky rate network: h <- (1 - alpha) * h + alpha * relu(W h + encoder(obs)),
    run for `substeps` internal steps per env step so signals can cross the
    several synaptic hops from ER/PFN inputs to PFL outputs.

    shuffle_seed: if given, the wiring is randomly rewired (see shuffle_wiring)
    as a control: same neurons, connection strengths, and per-neuron numbers of
    outgoing, excitatory incoming and inhibitory incoming connections, but not
    the fly's actual circuit.
    """

    def __init__(self, obs_dim, action_dim, neurons_csv=DATA_DIR / "neurons.csv",
                 connections_csv=DATA_DIR / "connections.csv", min_synapses=3,
                 substeps=5, alpha=0.5, spectral_radius=0.9, shuffle_seed=None):
        super().__init__()
        neurons, pre, post, syn = load_connectome(neurons_csv, connections_csv, min_synapses)
        self.neurons = neurons
        self.n = len(neurons)
        self.substeps = substeps
        self.alpha = alpha
        sign = torch.tensor(neurons["consensusNt"].map(NT_SIGN).to_numpy(), dtype=torch.float32)

        # Start each strength proportional to its synapse count, then scale
        # all of them so the real fly's initial W has the requested spectral
        # radius (keeps activity from exploding or dying out before training).
        init = torch.as_tensor(syn) / syn.max()
        self.register_buffer("pre", torch.tensor(pre, dtype=torch.long))
        self.register_buffer("post", torch.tensor(post, dtype=torch.long))
        w0 = self._dense(sign[self.pre] * init)
        scale = spectral_radius / torch.linalg.eigvals(w0).abs().max().item()
        # The shuffled control reuses the real fly's scale, so both start with
        # identical connection strengths and differ only in who connects to
        # whom. (Rescaling the shuffled net separately made its weights 6-8x
        # larger, because the real wiring's largest mode is one strong
        # inhibitory loop among the ring neurons.)
        if shuffle_seed is not None:
            pre, post = shuffle_wiring(pre, post, self.n, shuffle_seed, groups=sign.numpy()[pre])
            self.pre = torch.tensor(pre, dtype=torch.long)
            self.post = torch.tensor(post, dtype=torch.long)
        self.register_buffer("conn_sign", sign[self.pre])
        # Strengths are learned directly (abs keeps the sign fixed). An earlier
        # softplus version made small weights learn ~15x slower than the
        # baseline's, which biased the comparison against the connectome.
        self.raw = nn.Parameter(init * scale)

        is_in = neurons["family"].isin(INPUT_FAMILIES).to_numpy()
        is_out = neurons["family"].isin(OUTPUT_FAMILIES).to_numpy()
        self.register_buffer("input_idx", torch.as_tensor(np.flatnonzero(is_in)))
        self.register_buffer("output_idx", torch.as_tensor(np.flatnonzero(is_out)))
        self.encoder = nn.Linear(obs_dim, len(self.input_idx))
        self.decoder = nn.Linear(len(self.output_idx), action_dim)

    def _dense(self, values):
        w = torch.zeros(self.n, self.n, device=values.device, dtype=values.dtype)
        return w.index_put((self.post, self.pre), values)  # w[post, pre]

    def weight(self):
        return self._dense(self.conn_sign * self.raw.abs())

    def initial_state(self, batch_size, device=None):
        return torch.zeros(batch_size, self.n, device=device or self.raw.device)

    def drive(self, obs):
        h_shape = (*obs.shape[:-1], self.n)
        return obs.new_zeros(h_shape).index_copy(-1, self.input_idx, self.encoder(obs))

    def readout(self, h):
        return self.decoder(h[..., self.output_idx])

    def forward(self, obs, h):
        """obs: [B, obs_dim], h: [B, N] -> (action [B, action_dim], new h)."""
        actions, hs = self.forward_sequence(obs[None], h)
        return actions[0], hs[0]

    def forward_sequence(self, obs_seq, h, reset_seq=None):
        """
        obs_seq: [T, B, obs_dim], h: [B, N] state before the first step,
        reset_seq: optional [T, B] bool, True where h must be zeroed before
        that step (a new episode started). Returns (actions [T, B, A], hs [T, B, N]).
        Builds the weight matrix once for the whole sequence.
        """
        return run_leaky_rnn(self, obs_seq, h, reset_seq)


def run_leaky_rnn(net, obs_seq, h, reset_seq=None):
    """Shared dynamics for ConnectomeRNN and the baseline, so they differ only in wiring."""
    w_t = net.weight().T
    drives = net.drive(obs_seq)
    hs = []
    for t in range(obs_seq.shape[0]):
        if reset_seq is not None:
            h = h * (~reset_seq[t]).unsqueeze(-1)
        for _ in range(net.substeps):
            rate = MAX_RATE * torch.tanh(F.relu(h @ w_t + drives[t]) / MAX_RATE)
            h = (1 - net.alpha) * h + net.alpha * rate
        hs.append(h)
    hs = torch.stack(hs)
    return net.readout(hs), hs


def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    obs_dim, action_dim, batch = 16, 4, 8
    net = ConnectomeRNN(obs_dim, action_dim)

    print(f"neurons: {net.n}  connections: {len(net.raw)}  "
          f"inhibitory: {(net.conn_sign < 0).float().mean():.0%}")
    print(f"input neurons: {len(net.input_idx)}  output neurons: {len(net.output_idx)}")
    print(net.neurons["family"].value_counts().to_string())
    print(f"trainable params: {count_params(net)} "
          f"(connections {net.raw.numel()}, encoder {count_params(net.encoder)}, "
          f"decoder {count_params(net.decoder)})")
    print(f"initial spectral radius: {torch.linalg.eigvals(net.weight()).abs().max():.3f}")

    # Frozen parts are buffers, not parameters.
    assert {n for n, _ in net.named_parameters()} == {
        "raw", "encoder.weight", "encoder.bias", "decoder.weight", "decoder.bias"}
    mask_before = net.weight() != 0
    sign_before = torch.sign(net.weight())

    # Random 200-step rollout: activity should stay finite and not die out.
    h = net.initial_state(batch)
    for t in range(200):
        action, h = net(torch.randn(batch, obs_dim), h)
    assert torch.isfinite(h).all() and torch.isfinite(action).all()
    print(f"after 200 steps: mean rate {h.mean():.3f}, max {h.max():.3f}, "
          f"active neurons {(h > 1e-3).any(0).sum().item()}/{net.n}")

    # One gradient step through a short rollout; topology and signs must survive.
    opt = torch.optim.Adam(net.parameters(), lr=1e-1)
    h = net.initial_state(batch)
    loss = 0
    for t in range(10):
        action, h = net(torch.randn(batch, obs_dim), h)
        loss = loss + action.pow(2).mean()
    loss.backward()
    assert net.raw.grad is not None and net.raw.grad.abs().sum() > 0
    opt.step()
    w = net.weight()
    assert torch.equal(w != 0, mask_before), "topology changed"
    assert torch.equal(torch.sign(w)[mask_before], sign_before[mask_before]), "sign flipped"
    print("OK: gradients reach connection strengths; topology and signs unchanged after update")
