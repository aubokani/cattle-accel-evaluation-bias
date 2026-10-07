#!/bin/bash
# =============================================================================
# submit_revision2_ladder.sh — Leakage ladder at the stated recipe + mechanism rungs (R1.M2/M3, R3.M1/M2)
# =============================================================================
# Generic task-file array (revision of 2026-10-07). Each array index runs one
# line of $TASKFILE through scripts/revision2_leakage_ladder.py. CPU partition by default (all GPUs on Ada
# were fully allocated when these runs were planned); override with -p/--gres.
#
#   python scripts/revision2_leakage_ladder.py --list-tasks <set> > Results/revision2_hpc/tasks/<name>.txt
#   N=$(wc -l < Results/revision2_hpc/tasks/<name>.txt)
#   sbatch --array=0-$((N-1)) --export=ALL,TASKFILE=Results/revision2_hpc/tasks/<name>.txt \
#          scripts/submit_revision2_ladder.sh
#   (optional EXTRA="--epochs 1 ..." for pilots)
# =============================================================================
#SBATCH -J rev2_ladder
#SBATCH -p workq
#SBATCH -c 8
#SBATCH --mem=12G
#SBATCH -t 02:30:00
#SBATCH -o logs/rev2-ladder-%A_%a.out
#SBATCH -e logs/rev2-ladder-%A_%a.err

cd "$SLURM_SUBMIT_DIR"
module purge
module load Python/3.12.3-GCCcore-13.3.0-deep-learning-cpu
source "$SLURM_SUBMIT_DIR/.venv/bin/activate" || { echo "no venv"; exit 1; }
# one core left free: a WekaFS thread may be pinned inside the allocation (see revision2_common.n_threads)
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK-1)) MKL_NUM_THREADS=$((SLURM_CPUS_PER_TASK-1))

echo "[$(date)] node=$SLURMD_NODENAME job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} task='$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$TASKFILE")'"
python -u scripts/revision2_leakage_ladder.py --task-file "$TASKFILE" --task-index "$SLURM_ARRAY_TASK_ID" ${EXTRA}
EXIT=$?
echo "[$(date)] exit $EXIT"
exit $EXIT
