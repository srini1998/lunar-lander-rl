# CLAUDE.md — LunarLander RL project

You are Claude Code, working on Krishna's MacBook (Apple Silicon, macOS). This project trains a
DQN agent to land a rocket in Gymnasium's `LunarLander-v3` and runs a reward-shaping experiment.
All code is already written (see README.md for the concepts and layout). Your job is to **run it
end to end, fix whatever breaks, and fill in the results** — explaining the RL concepts to Krishna
as you go (he wants to understand reward shaping, not just run scripts).

## Steps

1. **Setup:** `bash setup_mac.sh` (installs swig via Homebrew, makes `.venv` on Python 3.10–3.12,
   installs requirements, self-checks). If Box2D fails to build, confirm `swig` is on PATH and the
   venv Python is ≤ 3.12, then retry. Homebrew may be at `~/.homebrew` — check before installing.
   Always `source .venv/bin/activate` before running scripts.
2. **Phase 0:** `python src/smoke_test.py --no-render` → baseline in `results/baseline_random.json`.
3. **Phase 1:** `python src/train_dqn.py` — run it in the background, it takes 15–45 min. It stops
   early when the 20-episode eval mean reaches 200. Show progress periodically.
4. **Evaluate:** `python src/evaluate.py` — must report 100-episode mean ≥ 200 on the stock reward.
   If not solved, retrain with `--timesteps 500000`.
5. **Phase 2:** `python src/record_video.py` → `results/lander.gif`. Open it for Krishna to see.
6. **Phase 3:** `python src/reward_shaping.py --shaping tilt` (then `time`, `fuel` if time allows),
   then `python src/reward_shaping.py --compare`. Evaluate each shaped model with `evaluate.py
   --model results/shaped_<name>/best_model.zip`. Explain what each shaping term did to learning
   speed and behaviour.
7. **README:** replace the two `_fill from ..._` placeholders in the Results table with the real
   numbers, and add a short Phase 3 findings table.
8. **Git:** `git init`, commit (the `.gitignore` excludes the venv, models, and TensorBoard logs).
   Ask before creating or pushing to any GitHub repo.

## Rules
- Only claim "solved" from `evaluate.py` on the stock reward. Never on a shaped reward.
- Keep the honesty notes in the README.
- Don't change the fairness setup in `reward_shaping.py` (Monitor inside the shaping wrapper).
- The scripts were syntax-checked but never executed (written in a sandbox without the packages),
  so expect and fix small runtime issues (e.g. API changes in Gymnasium/SB3 versions).

## Stretch (only if Krishna asks)
- Continuous LunarLander (`continuous=True`) with PPO, compared to DQN.
- DQN from scratch in raw PyTorch (replay buffer, target network, ε-greedy).
