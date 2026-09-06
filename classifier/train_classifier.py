"""
LoRA fine-tuning of Flan-T5-large for evidence helpfulness classification.

Replaces run_classifier.py (full fine-tuning) with PEFT/LoRA:
  - Trainable params: 770M → ~3.5M  (~0.5% of original)
  - GPU memory:       ~32GB → ~10GB  (1 GPU instead of 4)
  - Training time:    ~2 days → ~6-8 hours (40 epochs)

Output is compatible with inference.py — just point --classifier_path
to the merged output directory.

Usage:
    python train_lora.py \
        --model_path  /path/to/flan-t5-large \
        --train_file  /path/to/classifier_train_labeled.json \
        --output_dir  /path/to/output \
        --epochs      40 \
        --lora_r      16
"""

import json
import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm


# ── Dataset ───────────────────────────────────────────────────────────────────

class EvidenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data      = data
        self.tokenizer = tokenizer
        self.max_len   = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer(
            item["question"],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        label_ids = self.tokenizer(
            item["answer"],
            max_length=8,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).input_ids
        # Replace padding token id with -100 so it's ignored in loss
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "labels":         label_ids.squeeze(0),
        }


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load tokenizer + base model ──────────────────────────────────────────
    print(f"Loading base model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model     = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, torch_dtype=torch.float16)

    # ── Apply LoRA ───────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type    = TaskType.SEQ_2_SEQ_LM,
        r            = args.lora_r,
        lora_alpha   = args.lora_alpha,
        lora_dropout = args.lora_dropout,
        target_modules = ["q", "v"],   # encoder + decoder attention
        bias = "none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model = model.to(device)

    # ── Data ─────────────────────────────────────────────────────────────────
    print(f"Loading training data from {args.train_file} ...")
    with open(args.train_file) as f:
        all_data = json.load(f)
    random.shuffle(all_data)

    dataset    = EvidenceDataset(all_data, tokenizer, max_length=args.max_seq_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)

    # ── Optimizer & scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    total_steps = len(dataloader) * args.epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps  = int(0.05 * total_steps),
        num_training_steps = total_steps,
    )

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\nTraining for {args.epochs} epochs ({len(all_data)} samples) ...")
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch {epoch+1:3d}  loss={avg_loss:.4f}")

        # Save adapter checkpoint each epoch
        ckpt_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}")
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_pretrained(os.path.join(args.output_dir, "best"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, "best"))

    # ── Merge LoRA weights into base model for inference compatibility ────────
    print("\nMerging LoRA weights into base model ...")
    merged = model.merge_and_unload()
    merged.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Merged model saved to {args.output_dir}")
    print(f"Best adapter (unmerged) saved to {args.output_dir}/best")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",   required=True)
    parser.add_argument("--train_file",   required=True)
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--epochs",       type=int,   default=40)
    parser.add_argument("--batch_size",   type=int,   default=32,
                        help="Larger batch possible due to reduced memory")
    parser.add_argument("--lr",           type=float, default=3e-4,
                        help="LoRA typically uses higher lr than full fine-tuning")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--lora_r",       type=int,   default=16)
    parser.add_argument("--lora_alpha",   type=int,   default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
