#!/bin/bash
#SBATCH --job-name=rag2_debias_inference
#SBATCH --output=logs/inference_%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00

LLAMA_PATH=/path/to/Meta-Llama-3-8B-Instruct
CLASSIFIER_PATH=/path/to/flan_t5_large_trained
DEBIAS_ENCODER=/path/to/medcpt_debias_merged

cd /path/to/project
mkdir -p logs results

srun python -u inference.py \
    --input_path        data/medqa_test.jsonl \
    --output_path       results/rag2_debias_results.json \
    --llm_path          ${LLAMA_PATH} \
    --classifier_path   ${CLASSIFIER_PATH} \
    --encoder_path      ${DEBIAS_ENCODER} \
    --embeddings_dir    retriever/embeddings \
    --articles_dir      retriever/articles \
    --num_pubmed_chunks 38 \
    --pubmed_group_num  10 \
    --top_k             100 \
    --top_k_final       10 \
    --load_in_8bit \
    --checkpoint_path   results/rag2_debias_ckpt.json \
    --llm_batch_size    4
