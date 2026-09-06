#!/bin/bash
#SBATCH --job-name=merge_encoder
#SBATCH --output=logs/merge_encoder_%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00

MEDCPT_PATH=/path/to/MedCPT-Query-Encoder

cd /path/to/project

srun python -u merge_lora_encoder.py \
    --base_path  ${MEDCPT_PATH} \
    --lora_path  outputs/medcpt_debias/best \
    --output_dir outputs/medcpt_debias_merged
