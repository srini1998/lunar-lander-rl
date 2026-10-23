"""Phase 0 - Smoke test: a random agent, i.e. "episode #1 crashes".

Proves the install works and records the baseline score a trained agent must beat.

    python src/smoke_test.py              # watch it in a window
    python src/smoke_test.py --no-render  # headless, faster
"""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--no-render", action="store_true")
    args = p.parse_args()

    env = gym.make("LunarLander-v3", render_mode=None if args.no_render else "human")

    # Peek at what the agent "sees" and what it can "do"
    print("Observation space:", env.observation_space)  # 8 numbers
    print("Action space:     ", env.action_space)       # 4 discrete actions

    scores = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        total, steps, done = 0.0, 0, False
        while not done:
            action = env.action_space.sample()          # pure random - no learning
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            steps += 1
            done = terminated or truncated
        scores.append(total)
        print(f"Episode {ep + 1:2d}: score {total:8.1f}  ({steps} steps)")

    env.close()
    mean = float(np.mean(scores))
    print(f"\nRandom-agent baseline: mean {mean:.1f} +/- {np.std(scores):.1f}  (solved = 200)")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "baseline_random.json").write_text(
        json.dumps({"episodes": args.episodes, "mean": mean, "scores": scores}, indent=2)
    )
    print(f"Saved -> {RESULTS / 'baseline_random.json'}")


if __name__ == "__main__":
    main()
