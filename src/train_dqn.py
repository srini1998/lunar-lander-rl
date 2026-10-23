"""Phase 1 - Train a DQN agent on stock LunarLander-v3.

    python src/train_dqn.py                       # 300k steps, stops early once solved
    python src/train_dqn.py --timesteps 500000    # if it hasn't solved yet
    tensorboard --logdir results/tensorboard      # live charts (optional, 2nd terminal)
"""
import argparse

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor

from common import DQN_KWARGS, RESULTS, SOLVED, episodes_to_solve, plot_curves
from stable_baselines3.common.monitor import load_results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=20,
                   help="episodes per periodic check (the final 100-episode test is evaluate.py)")
    args = p.parse_args()

    run_dir = RESULTS / "dqn_stock"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Monitor logs every training episode's reward/length to CSV -> learning curve
    env = Monitor(gym.make("LunarLander-v3"), str(run_dir / "train"))
    eval_env = Monitor(gym.make("LunarLander-v3"))

    # Every 25k steps: play eval episodes greedily (no random exploration).
    # If their mean reaches 200, stop training early and keep that best model.
    stop = StopTrainingOnRewardThreshold(reward_threshold=SOLVED, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        callback_on_new_best=stop,
        eval_freq=25_000,
        n_eval_episodes=args.eval_episodes,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir),
        deterministic=True,
        verbose=1,
    )

    model = DQN("MlpPolicy", env, seed=args.seed, verbose=1,
                tensorboard_log=str(RESULTS / "tensorboard"), **DQN_KWARGS)
    model.learn(total_timesteps=args.timesteps, callback=eval_cb, tb_log_name="dqn_stock",
                progress_bar=True)
    model.save(run_dir / "final_model")

    rewards = load_results(str(run_dir))["r"].to_numpy()
    n = episodes_to_solve(rewards)
    print(f"\nTraining episodes played: {len(rewards)}")
    print("Training-curve moving average hit 200 at episode:", n if n else "not yet (exploration noise keeps it lower; check evaluate.py)")
    print(f"Models: {run_dir/'best_model.zip'} (best eval) and {run_dir/'final_model.zip'}")

    plot_curves({"DQN, stock reward": run_dir}, RESULTS / "learning_curve.png")
    print("\nNext: python src/evaluate.py")


if __name__ == "__main__":
    main()
