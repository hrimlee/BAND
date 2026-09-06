"""
Merge LoRA weights into the base MedCPT encoder and save as a full model.

Usage:
    python merge_lora_encoder.py \
        --base_path  /path/to/MedCPT-Query-Encoder \
        --lora_path  outputs/medcpt_debias/best \
        --output_dir outputs/medcpt_debias_merged
"""

import argparse
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path",  required=True)
    parser.add_argument("--lora_path",  required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    print(f"Loading base model from {args.base_path}")
    base = AutoModel.from_pretrained(args.base_path)
    tok  = AutoTokenizer.from_pretrained(args.base_path)

    print(f"Loading LoRA weights from {args.lora_path}")
    model = PeftModel.from_pretrained(base, args.lora_path)

    print("Merging LoRA weights into base model...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to {args.output_dir}")
    merged.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
