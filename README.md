# BAND: Bias-Aware Retrieval Debiasing for Demographic Robustness in Retrieval-Augmented Medical Question Answering

**Official implementation of BAND**, accepted at the **GroundLM Workshop @ EMNLP 2026**.

[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/hrimlee/BAND-MedQA) [![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/hrimlee/BAND-MedCPT)

BAND (**Bias-Aware Neutral Debiasing**) is a rotation-based retrieval debiasing framework for improving demographic robustness in retrieval-augmented medical question answering. It mitigates **retrieval shift**—the tendency of demographic perturbations to steer retrieval toward demographically correlated but clinically irrelevant evidence—while preserving clinically relevant retrieval behavior.

## Main Results

Results on the ethnicity-perturbed MedQA test set (five ethnicity groups; 5,575 perturbed test instances):

| Method | Overall Accuracy ↑ | DP Gap ↓ | Flip Rate ↓ | Retrieval Consistency ↑ |
|:--|--:|--:|--:|--:|
| RAG² | **61.47** | 1.35 | 11.89 | 0.26 |
| **BAND** | 61.24 | **1.26** | **10.69** | **0.30** |

BAND reduces demographic sensitivity while maintaining accuracy comparable to RAG². Evidence-level analysis also shows lower **BiasedEvidence** (6.33 vs. 10.01), **AttrRate** (1.38 vs. 2.06), and **DiffDocs** (86.21 vs. 88.07).

## Quick Start

Use the released perturbed evaluation dataset and pretrained debiasing encoder to reproduce inference results.

### 1. Download resources

| Resource | Download | Place under |
|:--|:--|:--|
| Ethnicity-perturbed MedQA dataset | [🤗 Hugging Face Dataset](https://huggingface.co/datasets/hrimlee/BAND-MedQA) | `perturb_dataset/` |
| Pretrained debiasing encoder | [🤗 Hugging Face Model](https://huggingface.co/hrimlee/BAND-MedCPT) | `outputs/medcpt_debias_merged/` |

Download the released resources with the Hugging Face Hub CLI (or download them from the linked pages):

```bash
pip install -U huggingface_hub
hf download hrimlee/BAND-MedCPT --local-dir outputs/medcpt_debias_merged
hf download hrimlee/BAND-MedQA --repo-type dataset --local-dir perturb_dataset
```

**Dataset file paths:** The Hugging Face dataset provides the train, validation, and test splits. If the downloaded files are stored under `perturb_dataset/data/`, update `--input_path` to point to the relevant JSONL file. The partitioned `_p1`, `_p2`, and `_p3` files in the example below are the original evaluation layout; they are **not** automatically created by downloading the Hugging Face dataset. To run on the released test split directly, use its actual downloaded path (e.g., `perturb_dataset/data/test.jsonl`) and adjust the output and cache paths as needed.

Inference also requires a Llama-3-8B-Instruct model, a trained evidence-helpfulness classifier, PubMed article embeddings and articles, and a CoT cache. Set the paths below to your local resources. The dataset and encoder downloads alone are not sufficient to run inference.

### 2. Install dependencies

```bash
conda env create -f environment.yml
conda activate retriever
```

Main dependencies include `torch`, `transformers`, `peft`, `faiss-gpu`, and `numpy`.

### 3. Run inference

```bash
python inference.py \
    --input_path perturb_dataset/phrases_rethnicity_test_prefix_perturbed_p1.jsonl \
    --output_path results/rag2_debias_eth_p1.json \
    --llm_path /path/to/Meta-Llama-3-8B-Instruct \
    --classifier_path /path/to/flan_t5_large_trained \
    --encoder_path outputs/medcpt_debias_merged \
    --embeddings_dir retriever/embeddings \
    --articles_dir retriever/articles \
    --num_pubmed_chunks 38 \
    --pubmed_group_num 10 \
    --top_k 100 \
    --top_k_final 10 \
    --cot_cache_path cot_cache/cot_cache_eth_p1.json \
    --load_in_8bit \
    --llm_batch_size 4
```

Repeat inference for the `_p2` and `_p3` partitions, updating the input, output, and cache paths accordingly.

### 4. Evaluate

```bash
python analyze/analyze_bias_propagation.py
python analyze/analyze_flip_statistics.py
python analyze/analyze_retrieval_consistency.py
```

## Training from Scratch

### Step 1. Construct the demographic debiaser

Compute demographic directions from the pretrained MedCPT query encoder.

```bash
python rotation_debias.py \
    --encoder_path /path/to/MedCPT-Query-Encoder \
    --output_path rotation_debiaser.npz \
    --concepts race gender
```

### Step 2. Train the evidence-helpfulness classifier

```bash
python classifier/train_classifier.py \
    --model_path /path/to/flan-t5-large \
    --train_file data/classifier_train_labeled.json \
    --output_dir outputs/flan_t5_classifier \
    --epochs 40 \
    --lora_r 16
```

### Step 3. Train the debiasing encoder

BAND combines a supervised retrieval objective (`L_sup`), a demographic-invariance objective (`L_inv`), and a rotation-alignment objective (`L_rotate`).

```bash
python train_encoder.py \
    --encoder_path /path/to/MedCPT-Query-Encoder \
    --debiaser_path rotation_debiaser.npz \
    --classifier_labeled_path data/classifier_train_labeled.json \
    --perturb_train_files \
        perturb_dataset/phrases_ethnicity_train_perturbed.jsonl \
        perturb_dataset/phrases_gender_train_perturbed.jsonl \
    --output_dir outputs/medcpt_debias \
    --epochs 5 \
    --warmup_epochs 1 \
    --lambda_supervised 1.0 \
    --lambda_inv 0.1 \
    --lambda_rot 0.01 \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --gradient_checkpointing
```

**Implementation notes**

- The supervised retrieval loss matches the rotated query representation `R(f(q))` with positive evidence representations.
- The document index uses the base MedCPT Article Encoder and is not rotated.
- Only `L_sup` is active during the first warmup epoch; `L_inv` and `L_rotate` are introduced afterward.

### Step 4. Merge LoRA weights

```bash
python merge_lora_encoder.py \
    --base_path /path/to/MedCPT-Query-Encoder \
    --lora_path outputs/medcpt_debias/best \
    --output_dir outputs/medcpt_debias_merged
```

### Step 5. Run inference with the trained encoder

```bash
python inference.py \
    --input_path data/medqa_test.jsonl \
    --output_path results/rag2_debias_results.json \
    --llm_path /path/to/Meta-Llama-3-8B-Instruct \
    --classifier_path outputs/flan_t5_classifier \
    --encoder_path outputs/medcpt_debias_merged \
    --embeddings_dir retriever/embeddings \
    --articles_dir retriever/articles \
    --top_k 100 \
    --top_k_final 10 \
    --load_in_8bit \
    --llm_batch_size 4
```

## Evaluation Dataset and Metrics

We construct an ethnicity-perturbed MedQA benchmark by filtering out questions where ethnicity could change the correct answer, then prepending standardized ethnicity statements to the retained questions. The benchmark includes five groups: **Caucasian, African American, Hispanic, Asian, and Native American**.

The resulting benchmark contains **44,720 training**, **5,600 development**, and **5,575 test** perturbed instances. The test split is reserved for evaluation.

| Metric | Definition |
|:--|:--|
| **Accuracy** ↑ | Answer accuracy for each ethnicity group and overall |
| **DP Gap** ↓ | Highest minus lowest ethnicity-group accuracy |
| **Flip Rate** ↓ | Fraction of originally correct answers that become incorrect after perturbation |
| **Retrieval Consistency** ↑ | Mean Jaccard similarity between retrieved document sets for original and perturbed queries |
| **BiasedEvidence** ↓ | Percentage of perturbed queries whose retrieved evidence contains any demographic terms from the predefined lexicon |
| **AttrRate** ↓ | Percentage of perturbed queries whose retrieved evidence contains terms associated with the injected ethnicity |
| **DiffDocs** ↓ | Percentage of perturbed queries whose retrieved document set differs from the original |

`BiasedEvidence` and `AttrRate` are computed using deterministic keyword matching, not classifier or human judgments.

## Project Structure

```text
.
├── inference.py                         # RAG inference pipeline
├── train_encoder.py                     # Debiasing encoder training
├── merge_lora_encoder.py                # Merge LoRA weights
├── generate_cot.py                      # CoT rationale caching
├── rotation_debias.py                   # Rotation-based debiasing module
├── retriever/
│   ├── query_encode.py                  # MedCPT query encoding
│   └── retrieve.py                      # FAISS index and retrieval
├── classifier/
│   └── train_classifier.py              # Evidence-helpfulness classifier
├── analyze/
│   ├── analyze_bias_propagation.py      # Evidence bias and attraction
│   ├── analyze_flip_statistics.py       # Flip rate and answer stability
│   └── analyze_retrieval_consistency.py # Jaccard retrieval consistency
└── scripts/
    ├── train_encoder.sh                 # SLURM encoder training
    ├── merge_encoder.sh                 # SLURM LoRA merge
    └── run_inference.sh                 # SLURM inference
```

## Citation

If you find this repository useful, please cite our paper:

> Harim Lee, Dasom Lee, and Chaehyun Ryu. **BAND: Bias-Aware Retrieval Debiasing for Demographic Robustness in Retrieval-Augmented Medical Question Answering.** GroundLM Workshop @ EMNLP 2026.

*BibTeX will be added when the official proceedings entry is available.*

## Acknowledgements

This implementation builds on the **RAG²** pipeline introduced in *Rationale-Guided Retrieval Augmented Generation for Medical Question Answering* (Sohn et al., 2025), particularly its evidence-labeling and relevance-filtering components. We thank the authors for making their work available.
