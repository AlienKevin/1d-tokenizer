#!/bin/bash
#SBATCH --account=viscam
#SBATCH --partition=viscam
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=1000
#SBATCH --cpus-per-task=32
#SBATCH --job-name=eval
#SBATCH --output=%j_output.txt
#SBATCH --error=%j_error.txt

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

source .venv/bin/activate

MOUNT_DIR="/tmp/$USER/gcs/tok1d"

if ! mountpoint -q "$MOUNT_DIR"; then
    mkdir -p "$MOUNT_DIR"
    gcsfuse tok1d "$MOUNT_DIR"
fi

python eval.py

echo "Done"
exit 0
