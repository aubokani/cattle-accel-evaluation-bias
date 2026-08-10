#!/bin/bash
# =============================================================================
# submit_leakage.sh — Protocol-leakage quantification (single GPU job)
# =============================================================================
# Addresses SAT Reviewer weakness #6: shuffled overlapping windows leak between
# train/test. Runs all 18 animals through 4 protocols (shuffled-overlap,
# blocked-overlap, temporal-holdout+guard-gap, non-overlapping-temporal) on the
# real accel-only STA-LSTM-H backbone and reports the inflation gap.
#
# Output: Results/leakage_protocol_per_animal.csv
#         Results/leakage_protocol_summary.csv
# =============================================================================

#SBATCH -J STALSTMH_leakage
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -p gpucomputeq
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -o logs/leakage-%j.out
#SBATCH -e logs/leakage-%j.err
#SBATCH --mail-type=END,FAIL

cd "$SLURM_SUBMIT_DIR"

module purge
module load Python/3.12.3-GCCcore-13.3.0-deep-learning-cpu

VENV_DIR="$SLURM_SUBMIT_DIR/.venv"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    exit 1
fi

mkdir -p logs
echo "==========================================="
echo " LEAKAGE  node=${SLURMD_NODENAME}  start=$(date)"
echo "==========================================="

python -u scripts/leakage_experiment.py --all --epochs 5

EXIT=$?
echo "Finished LEAKAGE at $(date) — exit ${EXIT}"
exit $EXIT
