"""Phase 4 - Rigorous reward-shaping study: several seeds, equal budget, mean +/- std.

Fixes the weak spots of the single-seed Phase 3 comparison:
  * every run (including stock) trains the SAME number of steps, no early stopping
  * N seeds per variant, run in parallel
  * every number comes from greedy evaluation on the ORIGINAL reward (same fairness rule as
    reward_shaping.py: the shaping wrapper only changes what the agent learns from)
  * no "best checkpoint" cherry-picking: we report the mean over the last checkpoints

    python src/shaping_study.py --run --seeds 5 --workers 4
    python src/shaping_study.py --report
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from common import DQN_KWARGS, RESULTS, SOLVED

STUDY = RESULTS / "study"
VARIANTS = ["stock", "tilt", "time", "fuel"]
EVAL_FREQ = 10_000
EVAL_EPISODES = 20


def run_one(job):
    name, seed, timesteps = job
    out = STUDY / f"{name}_s{seed}"
    if (out / "evaluations.npz").exists() and (out / "done").exists():
        return f"skip {name} s{seed}"
    out.mkdir(parents=True, exist_ok=True)

    import gymnasium as gym
    import torch
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor

    from reward_shaping import SHAPERS

    torch.set_num_threads(1)
    env = Monitor(gym.make("LunarLander-v3"))            # Monitor INSIDE the wrapper
    if name != "stock":
        env = SHAPERS[name](env)
    eval_env = Monitor(gym.make("LunarLander-v3"))       # always the stock reward
    cb = EvalCallback(eval_env, eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES,
                      log_path=str(out), deterministic=True, verbose=0)
    model = DQN("MlpPolicy", env, seed=seed, verbose=0, **DQN_KWARGS)
    model.learn(total_timesteps=timesteps, callback=cb)
    (out / "done").write_text("ok")
    return f"done {name} s{seed}"


def run(seeds, timesteps, workers):
    jobs = [(n, s, timesteps) for s in range(seeds) for n in VARIANTS]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for msg in ex.map(run_one, jobs):
            print(msg, flush=True)


def load_curves(name):
    curves, steps = [], None
    for d in sorted(STUDY.glob(f"{name}_s*")):
        f = d / "evaluations.npz"
        if not f.exists():
            continue
        z = np.load(f)
        steps = z["timesteps"]
        curves.append(z["results"].mean(axis=1))
    n = min(len(c) for c in curves)
    return steps[:n], np.array([c[:n] for c in curves])


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, fig_data = {}, {}
    for name in VARIANTS:
        steps, c = load_curves(name)
        last = c[:, -5:].mean(axis=1)                     # mean of last 5 checkpoints, per seed
        first = [int(steps[np.argmax(x >= SOLVED)]) if (x >= SOLVED).any() else None for x in c]
        reached = [f for f in first if f is not None]
        rows[name] = dict(
            seeds=len(c),
            final_mean=float(last.mean()), final_std=float(last.std(ddof=1)) if len(c) > 1 else 0.0,
            seeds_reaching_200=len(reached),
            median_first_200=float(np.median(reached)) if reached else None,
            auc=float(c.mean()),                          # mean eval return over training
        )
        fig_data[name] = (steps, c)

    print(f"{'variant':<8}{'seeds':>6}{'final (last 5 evals)':>26}{'reach 200':>11}{'median step':>13}{'AUC':>8}")
    for n, r in rows.items():
        med = f"{r['median_first_200']:.0f}" if r["median_first_200"] else "-"
        print(f"{n:<8}{r['seeds']:>6}{r['final_mean']:>14.1f} +/- {r['final_std']:<7.1f}"
              f"{r['seeds_reaching_200']:>6}/{r['seeds']:<4}{med:>13}{r['auc']:>8.1f}")
    (STUDY / "summary.json").write_text(json.dumps(rows, indent=2))

    fig, ax = plt.subplots(figsize=(9, 5))
    for n, (steps, c) in fig_data.items():
        m, s = c.mean(axis=0), c.std(axis=0)
        ax.plot(steps, m, label=f"{n} (n={len(c)})", linewidth=2)
        ax.fill_between(steps, m - s, m + s, alpha=0.15)
    ax.axhline(SOLVED, linestyle="--", color="green", label="Solved (200)")
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Eval return on stock reward (20 eps)")
    ax.set_title("Reward shaping, multi-seed (mean +/- std)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "shaping_study.png", dpi=150)
    print(f"Saved -> {RESULTS/'shaping_study.png'}  and {STUDY/'summary.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    STUDY.mkdir(parents=True, exist_ok=True)
    if a.run:
        run(a.seeds, a.timesteps, a.workers)
    if a.report:
        report()
