"""
Generate LLaMA3 Chain-of-Thought rationales for MedQA/MedMCQA questions.
Rationales are used as retrieval queries in the RAG² pipeline.

Usage:
    python generate_cot.py \
        --input_path data/medqa_train.jsonl \
        --output_path data/medqa_train_cot.json \
        --model_name_or_path /path/to/llama3 \
        --max_new_tokens 256
"""

import json
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are a medical expert. Given a multiple choice medical question, "
    "provide a concise clinical rationale (2-3 sentences) explaining the "
    "key medical concepts needed. Focus on pathophysiology and clinical "
    "reasoning, not on identifying which answer is correct."
)


def make_user_prompt(item):
    opts = item.get("options", {})
    if isinstance(opts, dict):
        opt_str = "\n".join(f"{k}) {v}" for k, v in sorted(opts.items()))
    else:
        # MedMCQA format
        opt_str = (
            f"A) {item.get('opa', '')}\n"
            f"B) {item.get('opb', '')}\n"
            f"C) {item.get('opc', '')}\n"
            f"D) {item.get('opd', '')}"
        )
    return f"Question: {item['question']}\n\nOptions:\n{opt_str}\n\nRationale:"


def load_data(path):
    if path.endswith(".jsonl"):
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--model_name_or_path",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    print(f"Loading model from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    data = load_data(args.input_path)
    print(f"Loaded {len(data)} examples from {args.input_path}")

    # Resume from existing output
    results = []
    if os.path.exists(args.output_path):
        with open(args.output_path) as f:
            results = json.load(f)
        print(f"Resuming from index {len(results)}")
    start = len(results)

    for item in tqdm(data[start:], desc="Generating CoT", initial=start, total=len(data)):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_prompt(item)},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        rationale = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        result = dict(item)
        result["rationale"] = rationale
        result["query"] = rationale  # used as retrieval query
        results.append(result)

        # Checkpoint every 100 items
        if len(results) % 100 == 0:
            with open(args.output_path, "w") as f:
                json.dump(results, f, ensure_ascii=False)

    with open(args.output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} results to {args.output_path}")


if __name__ == "__main__":
    main()
