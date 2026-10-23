"""Shared helpers: paths, DQN hyperparameters, and learning-curve plotting."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save plots to files; no GUI needed
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.monitor import load_results

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SOLVED = 200.0

# Tuned DQN settings for LunarLander (from the RL Baselines3 Zoo).
# SB3's plain defaults also learn, but much more slowly.
DQN_KWARGS = dict(
    learning_rate=6.3e-4,
    batch_size=128,
    buffer_size=50_000,          # replay buffer: past transitions sampled at random
    learning_starts=0,
    gamma=0.99,                  # discount: how much future reward counts
    target_update_interval=250,  # how often the frozen "target network" is refreshed
    train_freq=4,
    gradient_steps=-1,
    exploration_fraction=0.12,   # epsilon decays 1.0 -> 0.1 over first 12% of training
    exploration_final_eps=0.1,
    policy_kwargs=dict(net_arch=[256, 256]),
)


def moving_average(x, window=100):
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        window = max(1, len(x))
    return np.convolve(x, np.ones(window) / window, mode="valid")


def episodes_to_solve(rewards, window=100):
    """First episode index where the 100-episode moving average reaches 200, else None."""
    ma = moving_average(rewards, window)
    hits = np.where(ma >= SOLVED)[0]
    return int(hits[0] + window) if len(hits) else None


def plot_curves(runs, out_path, title="LunarLander-v3 learning curve"):
    """runs: dict label -> monitor log folder. Rewards in the logs are always the ORIGINAL env reward."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, folder in runs.items():
        df = load_results(str(folder))
        r = df["r"].to_numpy()
        ax.plot(np.arange(1, len(r) + 1), r, alpha=0.15)
        ma = moving_average(r)
        ax.plot(np.arange(len(r) - len(ma) + 1, len(r) + 1), ma, linewidth=2,
                label=f"{label} (100-ep moving avg)", color=ax.lines[-1].get_color())
    ax.axhline(SOLVED, linestyle="--", color="green", label="Solved (200)")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward (original env reward)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {out_path}")
