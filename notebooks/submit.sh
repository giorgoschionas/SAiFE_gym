#!/bin/bash -l
# Submit every combination from experiment_config.py as one SLURM array task.
#
# Usage:
#   ./submit.sh              # submit all combinations
#   ./submit.sh --dry-run    # just print the count

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PY="$SCRIPT_DIR/SAiFE_gym/notebooks/experiment_config.py"
RUNNER="$SCRIPT_DIR/run_1-1_cpu.sh"

if [ ! -f "$CONFIG_PY" ]; then
    echo "Error: $CONFIG_PY not found" >&2
    exit 1
fi
if [ ! -f "$RUNNER" ]; then
    echo "Error: $RUNNER not found" >&2
    exit 1
fi

# Load the conda env so numpy is available to experiment_config.py.
module load anaconda3/2022.10-gcc-13.2.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate SAiFE

N=$(python "$CONFIG_PY" count)
if ! [[ "$N" =~ ^[0-9]+$ ]] || [ "$N" -lt 1 ]; then
    echo "Error: experiment_config.py count returned '$N'" >&2
    exit 1
fi

LAST=$((N - 1))
echo "Found $N combination(s). Array range: 0-$LAST"

if [ "${1:-}" = "--dry-run" ]; then
    echo "Dry run — not submitting."
    exit 0
fi

sbatch --array=0-"$LAST" "$RUNNER"
