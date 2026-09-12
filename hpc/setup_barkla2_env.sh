#!/bin/bash -l
# Example Barkla2 environment setup for the domain-randomized LP PPO experiment.

set -euo pipefail

PROJECT_DIR=${SAIFE_PROJECT_DIR:-/mnt/scratch/users/$USER/rl_experiments/SAiFE_gym}
VENV_DIR=${SAIFE_VENV_DIR:-/mnt/fastscratch/users/$USER/venvs/rl_venv}

module purge
module load miniforge3/25.3.0-python3.12.10

mkdir -p "$(dirname "$VENV_DIR")"
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/experiments/results/domain_randomized_ppo"

cd "$PROJECT_DIR"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
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
