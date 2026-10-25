"""Phase 5 - DQN from scratch in raw PyTorch (no Stable-Baselines3).

Every piece of DQN is spelled out so you can see what the library hides:
  * ReplayBuffer      - ring buffer of (s, a, r, s', done); sampled uniformly at random
  * QNetwork          - MLP 8 -> 256 -> 256 -> 4, outputs Q(s, a) for every action
  * epsilon-greedy    - linear decay 1.0 -> 0.1, then greedy (argmax Q) mostly
  * target network    - frozen copy, hard-updated every `target_every` steps
  * TD target         - y = r + gamma * max_a' Q_target(s', a') * (1 - done)
  * loss              - Huber(Q(s, a), y), one Adam step per `train_every` env steps

Hyperparameters mirror common.DQN_KWARGS so the comparison with SB3 is like-for-like.

    python src/dqn_scratch.py --timesteps 300000 --seed 0
"""
import argparse
import json
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import RESULTS, SOLVED

CFG = dict(lr=6.3e-4, batch=128, buffer=50_000, gamma=0.99, target_every=250,
           train_every=4, eps_start=1.0, eps_end=0.1, eps_frac=0.12, hidden=256)


class ReplayBuffer:
    def __init__(self, capacity, obs_dim):
        self.cap, self.i, self.n = capacity, 0, 0
        self.s = np.zeros((capacity, obs_dim), np.float32)
        self.s2 = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.d = np.zeros(capacity, np.float32)

    def add(self, s, a, r, s2, done):
        j = self.i
        self.s[j], self.a[j], self.r[j], self.s2[j], self.d[j] = s, a, r, s2, done
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch, rng):
        idx = rng.integers(0, self.n, size=batch)
        return (torch.from_numpy(self.s[idx]), torch.from_numpy(self.a[idx]),
                torch.from_numpy(self.r[idx]), torch.from_numpy(self.s2[idx]),
                torch.from_numpy(self.d[idx]))


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)


def evaluate(q, episodes, seed):
    """Greedy play on the stock reward."""
    env = gym.make("LunarLander-v3")
    scores = []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            with torch.no_grad():
                a = int(q(torch.as_tensor(s, dtype=torch.float32)).argmax())
            s, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
        scores.append(total)
    env.close()
    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--out", default=str(RESULTS / "dqn_scratch"))
    a = p.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    out = __import__("pathlib").Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    env = gym.make("LunarLander-v3")
    obs_dim, n_act = env.observation_space.shape[0], env.action_space.n
    q = QNetwork(obs_dim, n_act, CFG["hidden"])
    q_target = QNetwork(obs_dim, n_act, CFG["hidden"])
    q_target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=CFG["lr"])
    buf = ReplayBuffer(CFG["buffer"], obs_dim)

    s, _ = env.reset(seed=a.seed)
    ep_ret, train_returns, evals, best = 0.0, [], [], -1e9
    t0 = time.time()
    for step in range(1, a.timesteps + 1):
        frac = min(1.0, step / (CFG["eps_frac"] * a.timesteps))
        eps = CFG["eps_start"] + frac * (CFG["eps_end"] - CFG["eps_start"])
        if rng.random() < eps:
            act = int(rng.integers(n_act))
        else:
            with torch.no_grad():
                act = int(q(torch.as_tensor(s, dtype=torch.float32)).argmax())

        s2, r, term, trunc, _ = env.step(act)
        buf.add(s, act, r, s2, float(term))   # truncation is NOT a terminal state for bootstrapping
        ep_ret += r
        s = s2
        if term or trunc:
            train_returns.append(ep_ret)
            ep_ret = 0.0
            s, _ = env.reset()

        if step % CFG["train_every"] == 0 and buf.n >= CFG["batch"]:
            bs, ba, br, bs2, bd = buf.sample(CFG["batch"], rng)
            with torch.no_grad():
                y = br + CFG["gamma"] * q_target(bs2).max(dim=1).values * (1.0 - bd)
            pred = q(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()

        if step % CFG["target_every"] == 0:
            q_target.load_state_dict(q.state_dict())

        if step % a.eval_freq == 0:
            scores = evaluate(q, a.eval_episodes, seed=10_000)
            m = float(np.mean(scores))
            evals.append((step, m))
            if m > best:
                best = m
                torch.save(q.state_dict(), out / "best_q.pt")
            recent = np.mean(train_returns[-20:]) if train_returns else float("nan")
            print(f"step {step:>7} | eps {eps:.2f} | train(20) {recent:7.1f} | "
                  f"eval {m:7.1f} | {time.time()-t0:5.0f}s", flush=True)

    torch.save(q.state_dict(), out / "final_q.pt")
    # Final 100-episode test of the best checkpoint on the stock reward: the only "solved" claim.
    q.load_state_dict(torch.load(out / "best_q.pt"))
    final = evaluate(q, 100, seed=50_000)
    res = dict(mean=float(np.mean(final)), std=float(np.std(final)),
               episodes_ge_200=int(sum(x >= SOLVED for x in final)),
               crashes=int(sum(x < 0 for x in final)),
               solved=bool(np.mean(final) >= SOLVED), timesteps=a.timesteps, seed=a.seed,
               eval_curve=evals)
    (out / "evaluation.json").write_text(json.dumps(res, indent=2))
    print(f"\n100-episode mean: {res['mean']:.1f} +/- {res['std']:.1f} | "
          f">=200: {res['episodes_ge_200']}/100 | crashes: {res['crashes']} | "
          f"{'SOLVED' if res['solved'] else 'not solved'}")


if __name__ == "__main__":
    main()
