#!/bin/bash
#SBATCH --account=krishna
#SBATCH --partition=gpu-l40s
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH --output=slurm-%x_%j.out
#SBATCH --error=slurm-%x_%j.err

set -euo pipefail

# Start from the directory where sbatch was invoked.
cd "${SLURM_SUBMIT_DIR}"

mkdir -p results/nlu/figures

. ~/.bashrc
conda activate ethics

# Per-node threading controls.
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=false

# Optional: set a persistent HF cache location on cluster storage.
# export HF_HOME=/gscratch/krishna/${USER}/.cache/huggingface

echo "[SLURM] Host: $(hostname)"
echo "[SLURM] Working dir: $(pwd)"
echo "[SLURM] Python: $(which python)"
echo "[SLURM] Job ID: ${SLURM_JOB_ID}"

# Run full pipeline: topic/emotion/ABSA merge, then figures.
srun --cpu-bind=cores python -u src/analyze_topics_emotions.py
srun --cpu-bind=cores python -u src/plot_topics_emotions.py

echo "[SLURM] Pipeline completed successfully."
