"""Official-style check: 100 episodes on the STOCK LunarLander-v3 reward. Solved = mean >= 200.

    python src/evaluate.py                                    # best model from train_dqn.py
    python src/evaluate.py --model results/shaped_tilt/best_model.zip
    python src/evaluate.py --watch 3                          # also watch 3 landings
"""
import argparse
import json

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from common import RESULTS, SOLVED


def load(path):
    for algo in (DQN, PPO):
        try:
            return algo.load(path)
        except Exception:
            continue
    raise SystemExit(f"Could not load model at {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(RESULTS / "dqn_stock" / "best_model.zip"))
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--continuous", action="store_true")
    p.add_argument("--watch", type=int, default=0)
    args = p.parse_args()

    model = load(args.model)
    env = Monitor(gym.make("LunarLander-v3", continuous=args.continuous))  # stock reward, no shaping
    rewards, lengths = evaluate_policy(model, env, n_eval_episodes=args.episodes,
                                       deterministic=True, return_episode_rewards=True)
    rewards = np.array(rewards)
    mean, std = rewards.mean(), rewards.std()
    print(f"Model: {args.model}")
    print(f"{args.episodes}-episode mean reward: {mean:.1f} +/- {std:.1f}")
    print(f"Episodes >= 200: {(rewards >= SOLVED).sum()}/{args.episodes} | crashes (< 0): {(rewards < 0).sum()}")
    print("SOLVED" if mean >= SOLVED else "NOT SOLVED yet - train longer (--timesteps 500000)")

    (RESULTS / "evaluation.json").write_text(json.dumps(
        {"model": args.model, "episodes": args.episodes, "mean": float(mean), "std": float(std),
         "solved": bool(mean >= SOLVED)}, indent=2))

    if args.watch:
        env = gym.make("LunarLander-v3", continuous=args.continuous, render_mode="human")
        for i in range(args.watch):
            obs, _ = env.reset()
            done, total = False, 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, _ = env.step(action)
                total += r
                done = term or trunc
            print(f"Watched episode {i + 1}: {total:.1f}")
        env.close()


if __name__ == "__main__":
    main()
