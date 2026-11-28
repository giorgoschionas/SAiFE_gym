#!/bin/bash -l
#SBATCH -J lp_fine_volatility
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --export=ALL

set -euo pipefail
cd "${SAIFE_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source hpc/run_periodic_market_sweep.sh volatility
