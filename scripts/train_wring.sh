#!/bin/bash
#SBATCH --job-name=train_wring
#SBATCH --output=logs/train_wring_%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

MEDCPT_PATH=/path/to/MedCPT-Query-Encoder

cd /path/to/WRING

srun python -u train_wring_encoder.py \
    --encoder_path              ${MEDCPT_PATH} \
    --debiaser_path             wring_debiaser.npz \
    --classifier_labeled_path   data/classifier_train_labeled.json \
    --perturb_train_files \
        perturb_dataset/phrases_ethnicity_train_perturbed.jsonl \
        perturb_dataset/phrases_gender_train_perturbed.jsonl \
    --output_dir                outputs/medcpt_wring \
    --epochs                    5 \
    --warmup_epochs             1 \
    --batch_size                32 \
    --cls_batch_size            32 \
    --lr                        2e-5 \
    --lambda_supervised         1.0 \
    --lambda_inv                0.1 \
    --lambda_wring              0.01 \
    --temperature               0.07 \
    --use_lora \
    --lora_r                    16 \
    --lora_alpha                32 \
    --gradient_checkpointing
