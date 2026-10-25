"""Phase 6 - Continuous LunarLander with PPO (policy-gradient, a different family from DQN).

In `continuous=True` the agent outputs 2 real numbers instead of choosing 1 of 4 buttons:
  action[0] main engine  (< 0 off, 0..1 throttle)      action[1] lateral engine (-1..1)
DQN cannot do this directly (it takes an argmax over a finite action set). PPO learns a
Gaussian policy pi(a|s) plus a value function, and updates with a clipped objective so a
single bad batch can't wreck the policy.

Hyperparameters follow the RL Baselines3 Zoo for LunarLanderContinuous (PPO).
Scoring is the same stock reward; `evaluate` below is the only place that says "solved".

    python src/ppo_continuous.py --timesteps 1000000
"""
import argparse
import json

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from common import RESULTS, SOLVED

PPO_KWARGS = dict(n_steps=1024, batch_size=64, gae_lambda=0.98, gamma=0.999,
                  n_epochs=4, ent_coef=0.01)
N_ENVS = 16


def make_env():
    return gym.make("LunarLander-v3", continuous=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(RESULTS / "ppo_continuous"))
    a = p.parse_args()
    torch.set_num_threads(2)
    out = __import__("pathlib").Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(make_env, n_envs=N_ENVS, seed=a.seed)
    eval_env = Monitor(make_env())
    cb = EvalCallback(eval_env, eval_freq=max(25_000 // N_ENVS, 1), n_eval_episodes=20,
                      best_model_save_path=str(out), log_path=str(out),
                      deterministic=True, verbose=1)
    model = PPO("MlpPolicy", env, seed=a.seed, verbose=0, device="cpu", **PPO_KWARGS)
    model.learn(total_timesteps=a.timesteps, callback=cb, progress_bar=False)
    model.save(out / "final_model")

    best = PPO.load(out / "best_model", device="cpu")
    scores, _ = evaluate_policy(best, Monitor(make_env()), n_eval_episodes=100,
                                deterministic=True, return_episode_rewards=True)
    res = dict(mean=float(np.mean(scores)), std=float(np.std(scores)),
               episodes_ge_200=int(sum(x >= SOLVED for x in scores)),
               crashes=int(sum(x < 0 for x in scores)),
               solved=bool(np.mean(scores) >= SOLVED), timesteps=a.timesteps, seed=a.seed)
    (out / "evaluation.json").write_text(json.dumps(res, indent=2))
    print(f"\n100-episode mean: {res['mean']:.1f} +/- {res['std']:.1f} | "
          f">=200: {res['episodes_ge_200']}/100 | crashes: {res['crashes']} | "
          f"{'SOLVED' if res['solved'] else 'not solved'}")


if __name__ == "__main__":
    main()
