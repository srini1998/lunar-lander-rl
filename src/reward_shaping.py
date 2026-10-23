"""Phase 3 - Reward-shaping experiment.

Train DQN with an EXTRA shaping term on top of the stock reward, then compare against
the stock run. Fairness rule: Monitor sits INSIDE the shaping wrapper, so every logged
score is the original environment reward - the only number allowed to claim "solved".

    python src/reward_shaping.py --shaping tilt     # stronger penalty for leaning
    python src/reward_shaping.py --shaping time     # small cost per step -> land faster
    python src/reward_shaping.py --shaping fuel     # harsher main-engine cost
    python src/reward_shaping.py --compare          # plot all finished runs vs stock
"""
import argparse

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor, load_results

from common import DQN_KWARGS, RESULTS, episodes_to_solve, plot_curves


class TiltPenalty(gym.Wrapper):
    """Extra penalty proportional to |angle| (obs[4], radians). Stock already penalises tilt;
    this makes the agent care much more about staying upright."""
    def __init__(self, env, coef=1.0):
        super().__init__(env)
        self.coef = coef

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        return obs, r - self.coef * abs(float(obs[4])), term, trunc, info


class TimePenalty(gym.Wrapper):
    """Constant cost per step: hovering forever becomes expensive, so it should land sooner."""
    def __init__(self, env, cost=0.3):
        super().__init__(env)
        self.cost = cost

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        return obs, r - self.cost, term, trunc, info


class FuelPenalty(gym.Wrapper):
    """Extra cost each frame the main engine (action 2) fires, on top of the stock -0.3."""
    def __init__(self, env, extra=0.3):
        super().__init__(env)
        self.extra = extra

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        return obs, r - (self.extra if int(action) == 2 else 0.0), term, trunc, info


SHAPERS = {"tilt": TiltPenalty, "time": TimePenalty, "fuel": FuelPenalty}


def train(name, timesteps, seed):
    run_dir = RESULTS / f"shaped_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # gym env -> Monitor (logs ORIGINAL reward) -> shaping wrapper (what the agent learns from)
    env = SHAPERS[name](Monitor(gym.make("LunarLander-v3"), str(run_dir / "train")))
    eval_env = Monitor(gym.make("LunarLander-v3"))  # evaluate on stock reward only

    eval_cb = EvalCallback(eval_env, eval_freq=25_000, n_eval_episodes=20,
                           best_model_save_path=str(run_dir), log_path=str(run_dir),
                           deterministic=True, verbose=1)
    # Same budget, seed and hyperparameters as the stock run -> only the reward differs.
    # No early stopping here, so both runs can be compared over the full horizon.
    model = DQN("MlpPolicy", env, seed=seed, verbose=1,
                tensorboard_log=str(RESULTS / "tensorboard"), **DQN_KWARGS)
    model.learn(total_timesteps=timesteps, callback=eval_cb, tb_log_name=f"shaped_{name}",
                progress_bar=True)
    model.save(run_dir / "final_model")
    print(f"Done. Evaluate fairly with: python src/evaluate.py --model {run_dir/'best_model.zip'}")


def compare():
    runs = {"stock": RESULTS / "dqn_stock"}
    runs.update({f"shaped: {n}": RESULTS / f"shaped_{n}" for n in SHAPERS
                 if (RESULTS / f"shaped_{n}").exists()})
    print(f"{'run':<16}{'episodes':>10}{'eps-to-solve*':>16}{'last-100 mean':>16}{'mean ep length':>16}")
    for label, d in runs.items():
        df = load_results(str(d))
        r, l = df["r"].to_numpy(), df["l"].to_numpy()
        n = episodes_to_solve(r)
        print(f"{label:<16}{len(r):>10}{str(n) if n else 'not reached':>16}"
              f"{r[-100:].mean():>16.1f}{l[-100:].mean():>16.0f}")
    print("* first episode where the 100-episode TRAINING moving average (original reward) hits 200."
          "\n  Note the stock run may have stopped early once its eval hit 200.")
    plot_curves(runs, RESULTS / "shaping_comparison.png", "Stock vs shaped reward (scored on original reward)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shaping", choices=SHAPERS)
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compare", action="store_true")
    args = p.parse_args()
    if args.compare:
        compare()
    elif args.shaping:
        train(args.shaping, args.timesteps, args.seed)
    else:
        p.error("pass --shaping {tilt,time,fuel} or --compare")


if __name__ == "__main__":
    main()
