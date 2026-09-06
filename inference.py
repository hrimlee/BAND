"""
Full RAG² inference pipeline with accuracy evaluation.

Flow per question:
  1. Generate CoT rationale with LLaMA3
  2. Encode rationale with MedCPT query encoder
  3. Retrieve top-k evidence from PubMed (+ PMC/CPG/Textbook if available)
  4. Filter evidence with trained Flan-T5 classifier
  5. Generate final answer with LLaMA3 + filtered evidence
  6. Compare to gold answer → accuracy

Usage:
    python inference.py \
        --input_path      data/medqa_test.jsonl \
        --output_path     results/medqa_test_results.json \
        --llm_path        /path/to/llama3 \
        --classifier_path /path/to/flan-t5-finetuned \
        --embeddings_dir  retriever/embeddings \
        --articles_dir    retriever/articles \
        --num_pubmed_chunks 38 \
        --pubmed_group_num  10 \
        --top_k 100 \
        --top_k_final 10
"""

import json
import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# Retriever utilities
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "retriever"))
import query_encode as qe
import retrieve as rt


# ── prompt templates ──────────────────────────────────────────────────────────

COT_SYSTEM = (
    "You are a medical expert. Given a multiple choice medical question, "
    "provide a concise clinical rationale (2-3 sentences) explaining the "
    "key medical concepts. Focus on pathophysiology, not on identifying the answer."
)

ANSWER_TEMPLATE = """{evidence_block}

Based on the evidence above, answer the following question.
Reply with ONLY the option letter (A, B, C, or D).

Question: {question}

Options:
{options}

Answer:"""


# ── helpers ───────────────────────────────────────────────────────────────────

def load_data(path):
    if path.endswith(".jsonl"):
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]
    with open(path) as f:
        return json.load(f)


def format_options(item):
    opts = item.get("options", {})
    if isinstance(opts, dict):
        return "\n".join(f"{k}) {v}" for k, v in sorted(opts.items()))
    return (
        f"A) {item.get('opa', '')}\n"
        f"B) {item.get('opb', '')}\n"
        f"C) {item.get('opc', '')}\n"
        f"D) {item.get('opd', '')}"
    )


def get_gold(item):
    """Return gold answer letter (A/B/C/D)."""
    return item.get("answer_idx") or item.get("cop", "")


def generate_rationale(llm, tokenizer, item):
    user = (
        f"Question: {item['question']}\n\n"
        f"Options:\n{format_options(item)}\n\nRationale:"
    )
    messages = [
        {"role": "system", "content": COT_SYSTEM},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to(llm.device)
    with torch.no_grad():
        out = llm.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def score_evidence(classifier, clf_tok, question, evidences, device=None, batch_size=32):
    """Return helpfulness probability for each evidence string (batched)."""
    helpful_id     = clf_tok.convert_tokens_to_ids("[HELPFUL]")
    not_helpful_id = clf_tok.convert_tokens_to_ids("[NOT_HELPFUL]")

    prompts = [
        (
            "Given the following evidence, determine whether it helps answer "
            f"the provided question.\n\nEvidence: {ev}\n\nQuestion: {question}"
        )
        for ev in evidences
    ]

    all_scores = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        clf_device = next(classifier.parameters()).device
        inputs = clf_tok(
            batch, return_tensors="pt", truncation=True,
            max_length=512, padding=True,
        ).to(clf_device)
        with torch.no_grad():
            out = classifier.generate(
                **inputs,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
            )
        logits = out.scores[0]
        pair = torch.stack([logits[:, helpful_id], logits[:, not_helpful_id]], dim=1)
        probs = torch.softmax(pair, dim=1)[:, 0].tolist()
        all_scores.extend(probs)

    return all_scores


def generate_answer(llm, tokenizer, item, evidences):
    """Generate final A/B/C/D answer conditioned on filtered evidence."""
    if evidences:
        ev_block = "Relevant evidence:\n" + "\n".join(
            f"[{i+1}] {e[:300]}" for i, e in enumerate(evidences[:5])
        )
    else:
        ev_block = "No relevant evidence available."

    prompt_text = ANSWER_TEMPLATE.format(
        evidence_block=ev_block,
        question=item["question"],
        options=format_options(item),
    )
    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(llm.device)
    with torch.no_grad():
        out = llm.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()
    for ch in text:
        if ch in "ABCD":
            return ch
    return "A"


# ── public API (used by sociodemographic_experiment.py) ───────────────────────

def run_inference(data, args, llm=None, llm_tok=None, classifier=None, clf_tok=None):
    """
    Run the full RAG² pipeline on `data`.
    Models can be passed in to avoid reloading between calls.
    Returns list of result dicts.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models if not supplied
    if llm is None:
        print("Loading LLM...")
        llm_tok = AutoTokenizer.from_pretrained(args.llm_path, use_fast=False)
        if llm_tok.pad_token is None:
            llm_tok.pad_token = llm_tok.eos_token
        if getattr(args, "load_in_8bit", False):
            load_kwargs = {"load_in_8bit": True}
            print("  [load_in_8bit] LLaMA loaded in 8-bit quantization (~8GB GPU)")
            # transformers auto-sets device_map for 8-bit, causing dispatch_model to
            # call model.to() which raises ValueError for quantized models.
            # Fix: patch PreTrainedModel.to at the source where ValueError is raised,
            # replacing it with a no-op during dispatch, then restore it.
            import accelerate.big_modeling as _bm
            import transformers.modeling_utils as _mu
            _orig_dispatch = _bm.dispatch_model
            _orig_pretrained_to = _mu.PreTrainedModel.to
            def _patched_dispatch(model, **kwargs):
                _mu.PreTrainedModel.to = lambda self, *a, **kw: self
                try:
                    return _orig_dispatch(model, **kwargs)
                finally:
                    _mu.PreTrainedModel.to = _orig_pretrained_to
            _bm.dispatch_model = _patched_dispatch
            _mu.dispatch_model = _patched_dispatch
        else:
            load_kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
        llm = AutoModelForCausalLM.from_pretrained(args.llm_path, **load_kwargs)
        if getattr(args, "load_in_8bit", False):
            # dispatch_model patch skips model.to(), leaving non-quantized buffers
            # (e.g. rotary inv_freq) on CPU. Move all CPU buffers to GPU manually.
            target_device = next(p.device for p in llm.parameters())
            for buf in llm.buffers():
                if buf.device != target_device:
                    buf.data = buf.data.to(target_device)
            print(f"  [8bit] Moved CPU buffers to {target_device}")
        llm.eval()

    if classifier is None and args.classifier_path:
        print("Loading classifier...")
        clf_device = "cpu" if getattr(args, "clf_on_cpu", False) else device
        if clf_device == "cpu":
            print("  [clf_on_cpu] Classifier will run on CPU to save GPU memory")
        if args.classifier_base_path:
            from peft import PeftModel
            clf_tok    = AutoTokenizer.from_pretrained(args.classifier_base_path)
            base_model = AutoModelForSeq2SeqLM.from_pretrained(args.classifier_base_path, torch_dtype=torch.bfloat16)
            classifier = PeftModel.from_pretrained(base_model, args.classifier_path, torch_dtype=torch.bfloat16).to(clf_device)
            print(f"  LoRA adapter loaded: {args.classifier_path}")
        else:
            clf_tok    = AutoTokenizer.from_pretrained(args.classifier_path)
            classifier = AutoModelForSeq2SeqLM.from_pretrained(args.classifier_path).to(clf_device)
        classifier.eval()

    # Generate CoT rationales and encode queries
    no_cot     = getattr(args, "no_cot", False)
    cache_path = getattr(args, "cot_cache_path", None)

    if no_cot:
        print("Skipping CoT (--no_cot): encoding raw questions directly...")
        rationales = [item["question"] for item in data]
    elif cache_path and os.path.exists(cache_path):
        import json as _json
        cached = _json.load(open(cache_path, encoding="utf-8"))
        if len(cached) >= len(data):
            rationales = cached[:len(data)]
            print(f"Loaded CoT cache: {len(rationales)} rationales from {cache_path}")
        else:
            print(f"WARNING: cache {len(cached)} < data {len(data)}, re-generating...")
            rationales = []
            for item in tqdm(data, desc="CoT"):
                rationales.append(generate_rationale(llm, llm_tok, item))
    else:
        print("Generating CoT rationales...")
        rationales = []
        for item in tqdm(data, desc="CoT"):
            rat = generate_rationale(llm, llm_tok, item)
            rationales.append(rat)

    print("Encoding queries with MedCPT...")
    encoder_path = getattr(args, "encoder_path", None)
    xq = qe.query_encode(rationales, model_path=encoder_path)

    # Optional: rotation debiasing on query embeddings
    if getattr(args, "debiaser_path", None):
        from rotation_debias import RotationDebiaser
        debiaser = RotationDebiaser.load(args.debiaser_path)
        xq = debiaser.debias_batch(xq)
        print(f"Applied rotation debiasing  (concepts: {list(debiaser.P.keys())})")

    # Retrieve from PubMed
    print("Retrieving from PubMed...")
    pubmed_I_array = []
    for start in range(0, args.num_pubmed_chunks, args.pubmed_group_num):
        idx = rt.pubmed_index_create(
            os.path.join(args.embeddings_dir, "pubmed"),
            start, args.pubmed_group_num, args.num_pubmed_chunks,
        )
        I_temp = []
        for s in range(0, len(xq), 1024):
            _, I = idx.search(xq[s : s + 1024], args.top_k)
            I_temp.extend(I)
        pubmed_I_array.append(I_temp)
        del idx

    pubmed_evidences = rt.pubmed_decode(
        pubmed_I_array,
        os.path.join(args.articles_dir, "pubmed"),
        args.pubmed_group_num,
        args.num_pubmed_chunks,
    )

    # Stage 1: Filter evidences with classifier (batched per question)
    all_helpful_ev = []
    for i, item in enumerate(tqdm(data, desc="Classifier")):
        evidences = pubmed_evidences[i]
        if classifier is not None:
            scores = score_evidence(classifier, clf_tok, item["question"], evidences, device)
            paired = sorted(zip(scores, evidences), reverse=True)
            helpful_ev = [e for sc, e in paired if sc > 0.5][: args.top_k_final]
            if not helpful_ev:
                helpful_ev = [e for _, e in paired][: args.top_k_final]
        else:
            helpful_ev = evidences[: args.top_k_final]
        all_helpful_ev.append(helpful_ev)

    # Stage 2: Batch LLaMA answer generation
    llm_tok.padding_side = "left"
    if llm_tok.pad_token is None:
        llm_tok.pad_token = llm_tok.eos_token

    llm_batch_size = getattr(args, "llm_batch_size", 4)

    def _build_answer_prompt(item, evidences):
        if evidences:
            ev_block = "Relevant evidence:\n" + "\n".join(
                f"[{i+1}] {e[:300]}" for i, e in enumerate(evidences[:5])
            )
        else:
            ev_block = "No relevant evidence available."
        prompt_text = ANSWER_TEMPLATE.format(
            evidence_block=ev_block,
            question=item["question"],
            options=format_options(item),
        )
        return llm_tok.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False, add_generation_prompt=True,
        )

    all_preds = []
    for start in tqdm(range(0, len(data), llm_batch_size), desc="Answer gen"):
        batch_items = data[start : start + llm_batch_size]
        batch_ev    = all_helpful_ev[start : start + llm_batch_size]
        prompts = [_build_answer_prompt(it, ev) for it, ev in zip(batch_items, batch_ev)]
        inputs = llm_tok(
            prompts, return_tensors="pt", truncation=True,
            max_length=2048, padding=True,
        ).to(llm.device)
        with torch.no_grad():
            out = llm.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=llm_tok.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        for seq in out:
            text = llm_tok.decode(seq[input_len:], skip_special_tokens=True).strip()
            pred = next((ch for ch in text if ch in "ABCD"), "A")
            all_preds.append(pred)

    # Collect results
    results = []
    correct = 0
    for i, item in enumerate(data):
        pred = all_preds[i]
        gold = get_gold(item)
        is_correct = pred.upper() == gold.upper()
        if is_correct:
            correct += 1

        result = {
            "id": item.get("id", str(i)),
            "question": item["question"],
            "gold_answer": gold,
            "predicted_answer": pred,
            "correct": is_correct,
            "rationale": rationales[i],
            "top_evidence": all_helpful_ev[i][:3],
        }
        # Preserve perturb dataset metadata if present
        for key in ("is_perturbed", "_group_id",
                    "original_ethnicity", "swapped_ethnicity",
                    "original_gender",   "swapped_gender",
                    "swap_skipped"):
            if key in item:
                result[key] = item[key]
        results.append(result)

        # Checkpoint every 100 items so progress survives job timeout
        if args.checkpoint_path and (i + 1) % 100 == 0:
            _acc = correct / (i + 1) * 100
            _out = {"accuracy": _acc, "correct": correct, "total": i + 1,
                    "results": results, "_checkpoint": True}
            with open(args.checkpoint_path, "w") as _f:
                json.dump(_out, _f, ensure_ascii=False, indent=2)

    accuracy = correct / len(data) * 100 if data else 0.0
    return results, accuracy, llm, llm_tok, classifier, clf_tok


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--llm_path", required=True)
    parser.add_argument("--classifier_path", default=None,
                        help="Trained Flan-T5 path; omit to skip filtering")
    parser.add_argument("--classifier_base_path", default=None,
                        help="Base Flan-T5 path for LoRA classifier (required when --classifier_path is a LoRA adapter dir)")
    parser.add_argument("--embeddings_dir", default="retriever/embeddings")
    parser.add_argument("--articles_dir", default="retriever/articles")
    parser.add_argument("--num_pubmed_chunks", type=int, default=38)
    parser.add_argument("--pubmed_group_num", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--top_k_final", type=int, default=10)
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Use only the first N examples (for quick tests)")
    parser.add_argument("--debiaser_path", default=None,
                        help="Path to pre-built rotation debiaser (.npz); omit to skip")
    parser.add_argument("--encoder_path", default=None,
                        help="Custom MedCPT encoder path (e.g. debiased fine-tuned); omit for default")
    parser.add_argument("--cot_cache_path", default=None,
                        help="Path to pre-generated CoT rationales JSON (list of strings)")
    parser.add_argument("--no_cot", action="store_true",
                        help="Skip CoT generation; encode raw questions as MedCPT queries")
    parser.add_argument("--checkpoint_path", default=None,
                        help="Save partial results here every 100 items (survives job timeout)")
    parser.add_argument("--llm_batch_size", type=int, default=4,
                        help="Batch size for LLaMA answer generation (default 4)")
    parser.add_argument("--clf_on_cpu", action="store_true",
                        help="Load Flan-T5 classifier on CPU to save GPU memory (~1.5GB freed)")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Load LLaMA in 8-bit via bitsandbytes (~8GB vs 16GB); requires: pip install bitsandbytes")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    data = load_data(args.input_path)
    if args.n_samples:
        data = data[: args.n_samples]
    print(f"Loaded {len(data)} examples")

    results, accuracy, *_ = run_inference(data, args)

    print(f"\nAccuracy: {accuracy:.2f}%  ({sum(r['correct'] for r in results)}/{len(results)})")

    output = {
        "accuracy": accuracy,
        "correct": sum(r["correct"] for r in results),
        "total": len(results),
        "results": results,
    }
    with open(args.output_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
