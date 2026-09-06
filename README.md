This repository contains the implementation of a retrieval-level debiasing method built on top of the RAG² pipeline ([Jeong et al., 2024](https://arxiv.org/abs/2408.02076)).

## Quick Access: Reproduce Our Results

Download the perturbed evaluation dataset and pre-built debiasing encoder, then run inference directly.

**1. Download perturbed dataset** ([link](https://drive.google.com/drive/folders/1wgDFKKpCqTUrGsvVrQepGoBK0Qz0rHJV))

Place the downloaded files under `perturb_dataset/`.

**2. Download pre-trained debiasing encoder** ([link](https://drive.google.com/file/d/1mkkQzMiwKGM0XK2Hxz7UTGIrOlEbLS7d/view?usp=sharing))

Place the downloaded model under `outputs/medcpt_debias_merged/` and set `--encoder_path outputs/medcpt_debias_merged`.

**3. Run inference with our debiasing encoder**

```bash
python inference.py \
    --input_path        perturb_dataset/phrases_rethnicity_test_prefix_perturbed_p1.jsonl \
    --output_path       results/rag2_debias_eth_p1.json \
    --llm_path          /path/to/Meta-Llama-3-8B-Instruct \
    --classifier_path   /path/to/flan_t5_large_trained \
    --encoder_path      /path/to/medcpt_debias_merged \
    --embeddings_dir    retriever/embeddings \
    --articles_dir      retriever/articles \
    --num_pubmed_chunks 38 \
    --pubmed_group_num  10 \
    --top_k             100 \
    --top_k_final       10 \
    --cot_cache_path    cot_cache/cot_cache_eth_p1.json \
    --load_in_8bit \
    --llm_batch_size    4
```

Repeat for `_p2` and `_p3` partitions, then evaluate:

```bash
python analyze/analyze_bias_propagation.py
python analyze/analyze_flip_statistics.py
```

---

## Requirements

```bash
conda env create -f environment.yml
conda activate retriever
```

Main dependencies: `torch`, `transformers`, `peft`, `faiss-gpu`, `numpy`

## Project Structure

```
├── inference.py              # RAG² inference pipeline (CoT → retrieve → answer)
├── train_encoder.py    # debiasing encoder training (L_sup + L_inv + L_rot)
├── merge_lora_encoder.py     # Merge LoRA weights into base encoder
├── generate_cot.py           # Pre-generate CoT rationales for caching
├── rotation_debias.py           # debiasing module
├── retriever/
│   ├── query_encode.py       # MedCPT query encoding
│   └── retrieve.py           # FAISS index creation and retrieval
├── classifier/
│   └── train_classifier.py   # Flan-T5 evidence helpfulness classifier (LoRA)
├── analyze/
│   ├── analyze_bias_propagation.py    # Biased Evidence %, Attraction Rate %
│   ├── analyze_flip_statistics.py     # Flip rate, Answer Stability
│   └── analyze_retrieval_consistency.py  # Jaccard retrieval consistency
└── scripts/
    ├── train_encoder.sh         # SLURM: train debiasing encoder
    ├── merge_encoder.sh       # SLURM: merge LoRA weights
    └── run_inference.sh       # SLURM: run RAG² inference
```

## Step-by-Step Usage

### 1. Build Debiaser

Encode demographic group directions from the base MedCPT encoder:

```bash
python rotation_debias.py \
    --encoder_path /path/to/MedCPT-Query-Encoder \
    --output_path  rotation_debiaser.npz \
    --concepts race gender
```

### 2. Train Classifier (Flan-T5 evidence helpfulness)

```bash
python classifier/train_classifier.py \
    --model_path  /path/to/flan-t5-large \
    --train_file  data/classifier_train_labeled.json \
    --output_dir  outputs/flan_t5_classifier \
    --epochs 40 --lora_r 16
```

### 3. Train Debiasing Encoder

Fine-tunes MedCPT with three losses: `L_sup` (InfoNCE in rotation space), `L_inv` (demographic invariance), `L_rot` (geometric alignment).

```bash
python train_encoder.py \
    --encoder_path              /path/to/MedCPT-Query-Encoder \
    --debiaser_path             rotation_debiaser.npz \
    --classifier_labeled_path   data/classifier_train_labeled.json \
    --perturb_train_files \
        perturb_dataset/phrases_ethnicity_train_perturbed.jsonl \
        perturb_dataset/phrases_gender_train_perturbed.jsonl \
    --output_dir  outputs/medcpt_debias \
    --epochs 5 --warmup_epochs 1 \
    --lambda_supervised 1.0 --lambda_inv 0.1 --lambda_rot 0.01 \
    --use_lora --lora_r 16 --lora_alpha 32 --gradient_checkpointing
```

### 4. Merge LoRA Weights

```bash
python merge_lora_encoder.py \
    --base_path  /path/to/MedCPT-Query-Encoder \
    --lora_path  outputs/medcpt_debias/best \
    --output_dir outputs/medcpt_debias_merged
```

### 5. Run Inference

```bash
python inference.py \
    --input_path        data/medqa_test.jsonl \
    --output_path       results/rag2_debias_results.json \
    --llm_path          /path/to/Meta-Llama-3-8B-Instruct \
    --classifier_path   outputs/flan_t5_classifier \
    --encoder_path      outputs/medcpt_debias_merged \
    --embeddings_dir    retriever/embeddings \
    --articles_dir      retriever/articles \
    --top_k 100 --top_k_final 10 \
    --load_in_8bit --llm_batch_size 4
```

### 6. Evaluate Bias Metrics

```bash
# Bias propagation (Biased Evidence %, Attraction Rate %)
python analyze/analyze_bias_propagation.py

# Flip rate & Answer Stability
python analyze/analyze_flip_statistics.py

# Retrieval Consistency
python analyze/analyze_retrieval_consistency.py
```


Key design choices:
- **L_sup uses the post-rotation query**: `R(f(q))` is matched against `f(d+)`, ensuring training aligns with inference (where queries are always debiased before retrieval).
- **Document embeddings are not rotated**: The document index is built from the base MedCPT Article Encoder and remains unchanged.
- **L_inv and L_rot are disabled during warmup**: Only `L_sup` is active in the first epoch for stable initialization.

## Perturbed Evaluation Dataset

The evaluation dataset is available at: (link)[https://drive.google.com/drive/folders/1wgDFKKpCqTUrGsvVrQepGoBK0Qz0rHJV]

Demographic perturbation prepends a single sentence (e.g., "The patient is of Caucasian descent.") to clinical questions. The dataset covers:
- **Ethnicity**: 5 groups × 1,115 questions = 5,575 perturbed pairs

Fairness metrics:
- **DP Gap**: max(acc per group) − min(acc per group) across perturbed groups
- **Flip Rate**: fraction of originally-correct questions answered incorrectly after perturbation
- **Retrieval Consistency**: Jaccard similarity between retrieved docs for original vs. perturbed queries

