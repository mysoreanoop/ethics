#!/bin/bash
#SBATCH --account=krishna
#SBATCH --partition=gpu-l40s
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --export=ALL
#SBATCH --output=slurm-%x_%j.out
#SBATCH --error=slurm-%x_%j.err

conda activate baselines

# Optional: set a persistent HF cache location on cluster storage.
# export HF_HOME=/gscratch/kevin/.cache/huggingface

echo "[SLURM] Host: $(hostname)"
echo "[SLURM] Working dir: $(pwd)"
echo "[SLURM] Python: $(which python)"
echo "[SLURM] Job ID: ${SLURM_JOB_ID}"

# Run full pipeline: topic/emotion/ABSA merge, then figures.
srun --cpu-bind=cores python -u src/analyze_topics_emotions.py
srun --cpu-bind=cores python -u src/plot_topics_emotions.py

echo "[SLURM] Pipeline completed successfully."
