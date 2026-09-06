"""
WRING: Classifier-guided debiased retrieval training with rotation-aware InfoNCE.

Losses:
    L = λ_supervised · InfoNCE(WRING(f(q)), f(d+))  ← InfoNCE in post-rotation space
      + λ_inv        · (1 − cos(f(q_pert), sg(f(q_orig))))
      + λ_wring      · MSE(f(q), sg(WRING(f(q))))

    L_supervised : InfoNCE over (question, helpful_evidence) pairs from
                   classifier_train_labeled.json.  Positive pairs on the diagonal;
                   in-batch negatives are other questions' helpful documents.

                   Key difference from v3: WRING rotation is applied to query
                   embeddings before InfoNCE, matching the inference-time pipeline
                   where queries are always WRING-debiased before retrieval.
                   This forces the encoder to learn representations that remain
                   medically discriminative after the demographic rotation is applied.

                   Document embeddings are NOT rotated, matching the fixed document
                   index built from the original MedCPT Article Encoder.

    L_inv, L_wring : same as v2/v3 — keep demographic invariance and WRING rotation.

Training data:
    1. classifier_train_labeled.json   → helpful (q, doc) pairs for L_supervised
    2. perturb_dataset/*_train_perturbed.jsonl → (q_orig, q_pert) pairs for L_inv / L_wring

Usage:
    python train_wring_encoder.py \\
        --encoder_path  /path/to/MedCPT-Query-Encoder \\
        --debiaser_path wring_debiaser.npz \\
        --classifier_labeled_path data/classifier_train_labeled.json \\
        --perturb_train_files \\
            perturb_dataset/phrases_ethnicity_train_perturbed.jsonl \\
            perturb_dataset/phrases_gender_train_perturbed.jsonl \\
        --output_dir classifier/model/outputs/medcpt_wring \\
        --use_lora
"""

import os
import sys
import json
import re
import argparse
import random
from itertools import cycle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "retriever"))

_EVIDENCE_RE = re.compile(r"Evidence:\s*(.+?)\n\nQuestion:", re.DOTALL)
_QUESTION_RE = re.compile(r"Question:\s*(.+)$", re.DOTALL)


# ── Datasets ─────────────────────────────────────────────────────────────────

class ClassifierDataset(Dataset):
    """
    Helpful (question, evidence) pairs from classifier_train_labeled.json.

    Only [HELPFUL] examples are loaded — each forms a positive (q, d+) pair.
    In-batch negatives are provided by other questions' documents during
    InfoNCE computation.
    """

    def __init__(self, labeled_path, n_samples=None, seed=42):
        with open(labeled_path) as f:
            raw = json.load(f)

        self.pairs = []
        for d in raw:
            if d.get("answer") != "[HELPFUL]":
                continue
            text = d["question"]
            ev_m = _EVIDENCE_RE.search(text)
            q_m  = _QUESTION_RE.search(text)
            if not ev_m or not q_m:
                continue
            self.pairs.append({
                "question": q_m.group(1).strip(),
                "evidence": ev_m.group(1).strip(),
            })

        if n_samples is not None:
            rng = random.Random(seed)
            self.pairs = rng.sample(self.pairs, min(n_samples, len(self.pairs)))

        print(f"Classifier dataset: {len(self.pairs)} helpful (question, evidence) pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


class PerturbPairDataset(Dataset):
    """
    (original_question, perturbed_question) pairs from perturb JSONL files.
    Same as v2.
    """

    def __init__(self, jsonl_files, n_samples=None, seed=42):
        groups = {}
        for fpath in jsonl_files:
            with open(fpath) as f:
                for line in f:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    if "swap_skipped" in d:
                        if d.get("swap_skipped", True):
                            continue
                    else:
                        if d.get("is_perturbed") and \
                           d.get("swapped_ethnicity") is None and \
                           d.get("original_ethnicity") is None:
                            continue
                    gid = (os.path.basename(fpath), d["_group_id"])
                    if gid not in groups:
                        groups[gid] = {"orig": None, "perts": []}
                    if not d["is_perturbed"]:
                        groups[gid]["orig"] = d["question"]
                    else:
                        groups[gid]["perts"].append(d["question"])

        self.pairs = []
        for group in groups.values():
            if group["orig"] is None or not group["perts"]:
                continue
            for pert in group["perts"]:
                self.pairs.append((group["orig"], pert))

        if n_samples is not None:
            rng = random.Random(seed)
            self.pairs = rng.sample(self.pairs, min(n_samples, len(self.pairs)))

        print(f"Perturb dataset: {len(self.pairs)} (original, perturbed) pairs "
              f"from {[os.path.basename(f) for f in jsonl_files]}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        orig, pert = self.pairs[idx]
        return {"original": orig, "perturbed": pert}


def classifier_collate(batch):
    return {
        "question": [b["question"] for b in batch],
        "evidence": [b["evidence"]  for b in batch],
    }


def perturb_collate(batch):
    return {
        "original":  [b["original"]  for b in batch],
        "perturbed": [b["perturbed"] for b in batch],
    }


# ── Encoder ───────────────────────────────────────────────────────────────────

class MedCPTEncoder(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path)

    def forward(self, **kwargs):
        kwargs.pop("token_type_ids", None)
        out = self.encoder(**kwargs)
        return out.last_hidden_state[:, 0, :]   # CLS → (B, 768)


# ── Losses ────────────────────────────────────────────────────────────────────

def infonce_loss(q_emb, d_emb, temperature=0.07):
    """
    In-batch InfoNCE (NT-Xent style).

    Positive pairs are on the diagonal: (q_i, d_i) is the positive for query i.
    All off-diagonal (q_i, d_j), j≠i are in-batch negatives.

    Note: both q_emb and d_emb come from the same student encoder.
    This is intentional — the signal teaches the query encoder to produce
    embeddings that are more similar to clinically helpful evidence text and
    less similar to random medical text, which is orthogonal to the demographic
    invariance objective.
    """
    q = F.normalize(q_emb, dim=-1)
    d = F.normalize(d_emb, dim=-1)
    sim = (q @ d.T) / temperature          # (B, B)
    labels = torch.arange(len(q), device=q.device)
    loss_q = F.cross_entropy(sim,   labels)   # q→d direction
    loss_d = F.cross_entropy(sim.T, labels)   # d→q direction (symmetric)
    return (loss_q + loss_d) / 2.0


def invariance_loss(orig_emb, pert_emb):
    return (1.0 - F.cosine_similarity(pert_emb, orig_emb.detach())).mean()


def wring_loss(q_emb, P_joint, w_joint):
    v     = q_emb.double()
    P_Cv  = v @ P_joint.T
    norms = P_Cv.norm(dim=1, keepdim=True)
    v_target = (v - P_Cv + norms * w_joint.unsqueeze(0)).detach()
    return F.mse_loss(v, v_target).float()


def wring_apply(emb, P_joint, w_joint):
    """Apply WRING debiasing in-graph (gradient flows through emb).

    Removes the demographic component and replaces it with a fixed neutral
    direction w_joint — identical to the inference-time WRING operation.
    P_joint and w_joint are fixed (no grad); gradient flows only through emb.
    """
    v    = emb.double()
    P_Cv = v @ P_joint.T                               # (B, k) demographic component
    norms = P_Cv.norm(dim=1, keepdim=True)             # (B, 1)
    v_debiased = v - P_Cv + norms * w_joint.unsqueeze(0)  # replace with neutral dir
    return v_debiased.float()


# ── Training ──────────────────────────────────────────────────────────────────

def train_one(args, lambda_supervised, lambda_inv, lambda_wring, output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}  |  "
          f"λ_supervised={lambda_supervised}  λ_inv={lambda_inv}  "
          f"λ_wring={lambda_wring}  |  output: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_path)

    # Student encoder
    student = MedCPTEncoder(args.encoder_path)
    if args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["query", "key", "value"],
            lora_dropout=0.1,
            bias="none",
        )
        student.encoder = get_peft_model(student.encoder, lora_cfg)
        student.encoder.print_trainable_parameters()
    if args.gradient_checkpointing:
        # Reduce activation memory at cost of recomputation (~30% slower, ~4x less VRAM)
        # enable_input_require_grads() required for LoRA + gradient checkpointing compatibility
        base_enc = student.encoder.base_model.model if args.use_lora else student.encoder
        base_enc.enable_input_require_grads()
        base_enc.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    student = student.to(device)

    # Classifier dataset (L_supervised)
    cls_dataset = ClassifierDataset(args.classifier_labeled_path)
    cls_loader  = DataLoader(
        cls_dataset, batch_size=args.cls_batch_size,
        shuffle=True, collate_fn=classifier_collate,
        num_workers=2, pin_memory=True,
    )
    cls_iter = cycle(cls_loader)   # cycle so it matches perturb epoch length

    # Perturb dataset (L_inv, L_wring)
    perturb_dataset = PerturbPairDataset(args.perturb_train_files)
    perturb_loader  = DataLoader(
        perturb_dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=perturb_collate,
        num_workers=2, pin_memory=True,
    )

    # Optimizer & scheduler (driven by perturb loader length)
    optimizer   = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=args.lr, weight_decay=0.01,
    )
    total_steps  = len(perturb_loader) * args.epochs
    warmup_steps = total_steps // 10
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # WRING debiaser matrices
    P_joint = w_joint = None
    if args.debiaser_path and lambda_wring > 0:
        from wring_debias import WRINGDebiaser
        debiaser = WRINGDebiaser.load(args.debiaser_path)
        debiaser.build_joint()
        P_joint = torch.tensor(debiaser._P_joint, dtype=torch.float64).to(device)
        w_joint = torch.tensor(debiaser._w_joint, dtype=torch.float64).to(device)
        print(f"Loaded WRING debiaser: P_joint {P_joint.shape}, w_joint {w_joint.shape}")

    os.makedirs(output_dir, exist_ok=True)
    best_loss, best_dir = float("inf"), os.path.join(output_dir, "best")

    for epoch in range(args.epochs):
        student.train()
        total_s, total_i, total_w, total_all, n_batches = 0., 0., 0., 0., 0

        warming_up = epoch < args.warmup_epochs
        lam_i = 0.0 if warming_up else lambda_inv
        lam_w = 0.0 if warming_up else lambda_wring
        if warming_up:
            print(f"  [warmup epoch {epoch+1}] λ_inv=0  λ_wring=0  "
                  f"(only L_supervised active)")

        for perturb_batch in tqdm(perturb_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            # ── L_supervised : InfoNCE on helpful (question, evidence) pairs ──
            cls_batch  = next(cls_iter)
            q_enc  = tokenizer(cls_batch["question"], padding=True, truncation=True,
                                max_length=128, return_tensors="pt").to(device)
            d_enc  = tokenizer(cls_batch["evidence"], padding=True, truncation=True,
                                max_length=args.evidence_max_length, return_tensors="pt").to(device)
            q_emb  = student(**q_enc)
            d_emb  = student(**d_enc)
            # Apply WRING rotation to query side only — mirrors inference where
            # queries are WRING-debiased before searching the (unrotated) doc index.
            # Gradient flows through q_emb; P_joint/w_joint are fixed tensors.
            if P_joint is not None:
                q_emb_rot = wring_apply(q_emb, P_joint, w_joint)
            else:
                q_emb_rot = q_emb
            L_s = infonce_loss(q_emb_rot, d_emb, temperature=args.temperature)

            torch.cuda.empty_cache()

            # ── L_inv : demographic invariance (perturb pairs) ────────────────
            orig_enc = tokenizer(perturb_batch["original"],  padding=True, truncation=True,
                                  max_length=128, return_tensors="pt").to(device)
            pert_enc = tokenizer(perturb_batch["perturbed"], padding=True, truncation=True,
                                  max_length=128, return_tensors="pt").to(device)
            orig_emb = student(**orig_enc)
            pert_emb = student(**pert_enc)
            L_i = invariance_loss(orig_emb, pert_emb) if lam_i > 0 else torch.tensor(0.0)

            # ── L_wring : geometric rotation toward debiased subspace ─────────
            L_w = torch.tensor(0.0, device=device)
            if P_joint is not None and lam_w > 0:
                L_w = wring_loss(orig_emb, P_joint, w_joint)

            L = lambda_supervised * L_s + lam_i * L_i + lam_w * L_w

            optimizer.zero_grad()
            L.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_s   += L_s.item()
            total_i   += L_i.item()
            total_w   += L_w.item()
            total_all += L.item()
            n_batches += 1

        avg_s = total_s / n_batches
        avg_i = total_i / n_batches
        avg_w = total_w / n_batches
        avg   = total_all / n_batches
        print(f"Epoch {epoch+1}: loss={avg:.4f}  "
              f"(supervised={avg_s:.4f}  inv={avg_i:.4f}  wring={avg_w:.4f})")

        ckpt_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
        os.makedirs(ckpt_dir, exist_ok=True)
        student.encoder.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

        if avg < best_loss:
            best_loss = avg
            os.makedirs(best_dir, exist_ok=True)
            student.encoder.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"  ↑ Best model saved → {best_dir}")

    print(f"Training complete. Best loss: {best_loss:.4f}")
    return best_loss


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--encoder_path",  required=True)
    parser.add_argument("--output_dir",    required=True)
    parser.add_argument("--debiaser_path", default=None)
    parser.add_argument("--classifier_labeled_path", required=True,
                        help="Path to classifier_train_labeled.json")
    parser.add_argument("--perturb_train_files", nargs="+", required=True)

    parser.add_argument("--epochs",           type=int,   default=5)
    parser.add_argument("--warmup_epochs",    type=int,   default=1)
    parser.add_argument("--batch_size",       type=int,   default=32,
                        help="Batch size for perturb pairs (L_inv, L_wring)")
    parser.add_argument("--cls_batch_size",   type=int,   default=8,
                        help="Batch size for classifier pairs (L_supervised)")
    parser.add_argument("--lr",               type=float, default=2e-5)
    parser.add_argument("--lambda_supervised",type=float, default=1.0,
                        help="Weight of InfoNCE supervised contrastive loss")
    parser.add_argument("--lambda_inv",       type=float, default=0.1,
                        help="Weight of demographic invariance loss")
    parser.add_argument("--lambda_wring",     type=float, default=0.01,
                        help="Weight of WRING rotation loss")
    parser.add_argument("--temperature",      type=float, default=0.07,
                        help="InfoNCE temperature (lower = sharper contrast)")
    parser.add_argument("--evidence_max_length", type=int, default=256,
                        help="Max token length for evidence encoding (abstracts fit in 256)")

    parser.add_argument("--use_lora",              action="store_true")
    parser.add_argument("--lora_r",                type=int, default=16)
    parser.add_argument("--lora_alpha",            type=int, default=32)
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save GPU memory (~4x less VRAM, ~30% slower)")

    args = parser.parse_args()
    train_one(args, args.lambda_supervised, args.lambda_inv, args.lambda_wring, args.output_dir)


if __name__ == "__main__":
    main()
