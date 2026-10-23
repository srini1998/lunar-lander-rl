#!/usr/bin/env bash
# One-time setup on macOS. Run from the project folder:  bash setup_mac.sh
set -e

echo "==> Checking Homebrew"
if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Install it from https://brew.sh, then re-run this script."
  exit 1
fi

echo "==> Installing swig (needed to build Box2D)"
brew list swig >/dev/null 2>&1 || brew install swig

# Box2D builds most reliably on Python 3.10-3.12
PY=""
for v in python3.12 python3.11 python3.10; do
  if command -v "$v" >/dev/null 2>&1; then PY="$v"; break; fi
done
if [ -z "$PY" ]; then
  echo "==> Installing Python 3.12 via Homebrew"
  brew install python@3.12
  PY="$(brew --prefix)/bin/python3.12"
fi
echo "==> Using $($PY --version)"

echo "==> Creating virtual environment .venv"
$PY -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

echo "==> Verifying install"
python -c "import gymnasium as gym, stable_baselines3 as sb3; gym.make('LunarLander-v3'); print('OK  gymnasium', gym.__version__, '| SB3', sb3.__version__)"

echo
echo "Setup done. Next:"
echo "  source .venv/bin/activate"
echo "  python src/smoke_test.py          # Phase 0: random agent baseline (opens a window)"
echo "  python src/train_dqn.py           # Phase 1: train (~15-45 min on CPU)"
