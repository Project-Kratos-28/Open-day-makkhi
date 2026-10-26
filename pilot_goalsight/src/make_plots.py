"""
Comparison figures for the poster, saved to figures/:

  learning_curves.png  success rate during training, mean across seeds (band = ±1 std)
  final_test.png       held-out test success at each difficulty, one dot per seed

    python src/make_plots.py
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS, FIGS = ROOT / "runs", ROOT / "figures"
SEEDS = range(5)

# (run prefix, model name in eval_results.csv, label, colour). Fixed order and colours.
MODELS = [("conn", "connectome", "Real fly wiring", "#2a78d6"),
          ("shuf", "shuffled", "Shuffled fly wiring", "#eb6834"),
          ("gru", "gru", "GRU (standard network)", "#1baf7a"),
          ("base", "baseline", "Dense RNN (same neuron model)", "#eda100")]
SHORT = ["Real\nfly", "Shuffled\nfly", "GRU", "Dense\nRNN"]
LEVELS = [("standard", "Standard\n(trained on)"), ("hard", "Hard\n(new)"), ("very_hard", "Very hard\n(new)")]

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def learning_curve(prefix, grid, window=20):
    """Success rate per seed, smoothed over `window` updates and resampled onto `grid` (env steps)."""
    curves = []
    for s in SEEDS:
        rows = list(csv.DictReader(open(RUNS / f"{prefix}_s{s}" / "log.csv")))
        steps = np.array([int(r["env_steps"]) for r in rows])
        succ = np.array([float(r["success_rate"]) for r in rows])
        smooth = np.convolve(succ, np.ones(window) / window, mode="valid")
        curves.append(np.interp(grid, steps[window - 1:], smooth))
    return np.array(curves)


def plot_learning_curves():
    grid = np.linspace(0.1e6, 5e6, 200)
    fig, ax = plt.subplots(figsize=(8, 5))
    ends = []
    for prefix, _, label, colour in MODELS:
        c = learning_curve(prefix, grid) * 100
        mean, std = c.mean(0), c.std(0, ddof=1)
        ax.fill_between(grid / 1e6, mean - std, mean + std, color=colour, alpha=0.12, linewidth=0)
        ax.plot(grid / 1e6, mean, color=colour, linewidth=2)
        ends.append([mean[-1], label, colour])
    # Direct labels at the line ends, nudged apart so they don't overlap.
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 5)
    for y, label, colour in ends:
        ax.text(5.08, y, label, color=INK, va="center", fontsize=10)
        ax.plot([5.0, 5.06], [y, y], color=colour, linewidth=2)
    ax.set(xlim=(0, 5), ylim=(0, 100), xlabel="Training experience (million drone steps)",
           ylabel="Reaches the goal (%)")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Learning to fly through obstacles: mean of 5 seeds, band = ±1 std",
                 loc="left", fontsize=12, color=INK)
    fig.subplots_adjust(right=0.74)
    fig.savefig(FIGS / "learning_curves.png", dpi=200)
    plt.close(fig)


def plot_final_test():
    rows = list(csv.DictReader(open(RUNS / "eval_results.csv")))
    fig, axes = plt.subplots(1, len(LEVELS), figsize=(12, 5), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (level, title) in zip(axes, LEVELS):
        for i, (_, model, label, colour) in enumerate(MODELS):
            vals = np.array([float(r["success_rate"]) * 100 for r in rows
                             if r["level"] == level and r["model"] == model])
            x = i + rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(x, vals, s=46, color=colour, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.plot([i - 0.28, i + 0.28], [vals.mean()] * 2, color=INK, linewidth=2, zorder=4)
            ax.text(i + 0.31, vals.mean(), f"{vals.mean():.0f}%", ha="left", va="center",
                    fontsize=9.5, color=INK, fontweight="bold", zorder=5)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(SHORT, fontsize=9.5)
        ax.set_xlim(-0.5, len(MODELS) - 0.1)
        ax.set_title(title, fontsize=11, color=INK)
        ax.set_ylim(-3, 103)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Reaches the goal (%)")
    fig.suptitle("Final test on 100 unseen mazes per level: dots = individual seeds, line = mean",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(FIGS / "final_test.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    FIGS.mkdir(exist_ok=True)
    plot_learning_curves()
    plot_final_test()
    print(f"saved {FIGS / 'learning_curves.png'} and {FIGS / 'final_test.png'}")
