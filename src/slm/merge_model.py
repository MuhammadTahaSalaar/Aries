import argparse
import shutil
import torch
from pathlib import Path
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="triage")
    parser.add_argument("--base_model", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    args = parser.parse_args()

    lora_dir = BASE_DIR / "models" / "slm_lora" / args.dataset
    merged_dir = BASE_DIR / "models" / "slm_merged" / args.dataset

    if not lora_dir.exists():
        print(f"Error: {lora_dir} does not exist. Run finetune.py first.")
        return

    print(f"Loading PEFT model from {lora_dir}...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(lora_dir),
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )

    print("Merging LoRA adapters into base weights...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to {merged_dir}...")
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(merged_dir))

    # Export tokenizer from the base model to keep full vocab assets for GGUF conversion.
    # Some GGUF conversion paths require a physical tokenizer.model file.
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    tokenizer.save_pretrained(str(merged_dir), legacy_format=True)

    tokenizer_model = merged_dir / "tokenizer.model"
    if not tokenizer_model.exists():
        vocab_file = getattr(tokenizer, "vocab_file", None)
        if vocab_file:
            vocab_path = Path(vocab_file)
            if vocab_path.exists():
                shutil.copy2(vocab_path, tokenizer_model)

    if not tokenizer_model.exists():
        raise FileNotFoundError(
            f"Missing {tokenizer_model}. Cannot run GGUF conversion without tokenizer.model."
        )

    # Also save fast tokenizer JSON for runtime compatibility with local inference paths.
    tokenizer_fast = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer_fast.save_pretrained(str(merged_dir))
    print("Merge successful! Ready for llama.cpp GGUF conversion.")

if __name__ == "__main__":
    main()