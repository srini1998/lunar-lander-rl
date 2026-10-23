"""Phase 2 - Before/after GIF: random agent crashing, then the trained agent landing.

    python src/record_video.py
    python src/record_video.py --model results/dqn_stock/final_model.zip --seed 7
"""
import argparse

import gymnasium as gym
import imageio
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import DQN

from common import RESULTS


def run_episode(env, policy, seed):
    frames, total = [], 0.0
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        frames.append(env.render())
        obs, r, term, trunc, _ = env.step(policy(obs))
        total += r
        done = term or trunc
    frames.append(env.render())
    return frames, total


def label(frames, text, every=2, scale=0.6):
    out = []
    for f in frames[::every]:  # skip frames + shrink to keep the GIF small
        img = Image.fromarray(f)
        img = img.resize((int(img.width * scale), int(img.height * scale)))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
        d.text((8, 5), text, fill=(255, 255, 255))
        out.append(np.array(img))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(RESULTS / "dqn_stock" / "best_model.zip"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    model = DQN.load(args.model)

    before, r0 = run_episode(env, lambda o: env.action_space.sample(), args.seed)
    after, r1 = run_episode(env, lambda o: int(model.predict(o, deterministic=True)[0]), args.seed)
    env.close()

    frames = label(before, f"BEFORE: random agent   score {r0:.0f}")
    frames += [frames[-1]] * 15   # brief pause
    after_l = label(after, f"AFTER: trained DQN   score {r1:.0f}")
    frames += after_l + [after_l[-1]] * 25

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "lander.gif"
    imageio.mimsave(out, frames, duration=1000 / 25, loop=0)  # duration in ms per frame
    print(f"Random agent: {r0:.1f} | Trained agent: {r1:.1f}")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
