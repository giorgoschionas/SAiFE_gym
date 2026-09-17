#!/bin/bash -l
# Prepare a Python environment for the domain-randomized LP PPO experiment.

set -euo pipefail

PROJECT_DIR=${SAIFE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
if [ ! -f "$PROJECT_DIR/experiments/train_robust_lp_agent.py" ] || [ ! -f "$PROJECT_DIR/requirements.txt" ]; then
  echo "ERROR: no SAiFE_gym checkout at $PROJECT_DIR. Set SAIFE_PROJECT_DIR to the project root." >&2
  exit 1
fi
PROJECT_DIR=$(cd "$PROJECT_DIR" && pwd)
VENV_DIR=${SAIFE_VENV_DIR:-$PROJECT_DIR/venv}
if [[ "$VENV_DIR" != /* ]]; then
  VENV_DIR="$PROJECT_DIR/$VENV_DIR"
fi

if [ -n "${SAIFE_PYTHON_MODULE:-}" ]; then
  if ! command -v module >/dev/null 2>&1; then
    echo "ERROR: SAIFE_PYTHON_MODULE is set, but 'module' is unavailable. Initialize your cluster's module system or unset SAIFE_PYTHON_MODULE." >&2
    exit 1
  fi
  module purge
  module load "$SAIFE_PYTHON_MODULE"
fi

mkdir -p "$(dirname "$VENV_DIR")"
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/experiments/results/domain_randomized_ppo"

cd "$PROJECT_DIR"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "ERROR: $VENV_DIR is not a virtualenv. Set SAIFE_VENV_DIR to a valid environment or a new directory." >&2
  exit 1
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

python - <<'PY'
import sys
import numpy
import stable_baselines3
import torch

print("Python:", sys.version)
print("NumPy:", numpy.__version__)
print("Stable-Baselines3:", stable_baselines3.__version__)
print("Torch:", torch.__version__)
PY

echo "Environment ready: $VENV_DIR"
