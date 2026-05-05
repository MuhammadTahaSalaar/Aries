#!/usr/bin/env python3
###############################################################################
# ARIES — SLM Fine-tuning using QLoRA
#
# Fine-tunes a base model (Phi-3-mini) using the JSONL files produced by
# `preprocess_slm_datasets.py`. Uses 4-bit quantization and LoRA for
# memory-efficient training on a single GPU.
#
# Usage:
#   python src/slm/finetune.py --dataset triage
###############################################################################

import os
import argparse
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Base dirs
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "slm_finetuning"
CHECKPOINT_DIR = BASE_DIR / "checkpoints" / "slm"
OUTPUT_MODEL_DIR = BASE_DIR / "models" / "slm_lora"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--dataset", type=str, choices=["triage", "ner", "summarizer"], default="triage")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Dataset — our JSONL has {"messages": [...]} conversational format.
    # SFTTrainer detects the "messages" column and applies the chat template
    # automatically; no manual formatting step is needed.
    dataset_file = PROCESSED_DIR / f"{args.dataset}_slm_train.jsonl"
    if not dataset_file.exists():
        print(f"Error: {dataset_file} not found. Run scripts/preprocess_slm_datasets.py first.")
        return

    print(f"Loading dataset from {dataset_file}...")
    dataset = load_dataset("json", data_files=str(dataset_file), split="train")
    dataset = dataset.train_test_split(test_size=0.02)

    # 2. Tokenizer
    # Do NOT use trust_remote_code=True — Phi-3 is natively supported in transformers>=4.40.
    # Using it downloads a custom modeling_phi3.py that mismatches the installed config
    # format (rope_scaling "type" vs "rope_type"), causing a KeyError at load time.
    print(f"Loading tokenizer {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. 4-bit Quantization Config (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 4. Load Base Model
    # attn_implementation="eager" avoids flash-attention window_size errors on Hydra
    # where flash_attn is not installed.
    print(f"Loading base model {args.model_id} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # 5. LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # 6. Training config (SFTConfig = TrainingArguments + SFT-specific args)
    # max_length replaces the old max_seq_length (renamed in TRL 1.x).
    # dataset_text_field is NOT set — SFTTrainer auto-detects the "messages"
    # column and applies the tokenizer's chat template internally.
    task_out_dir = CHECKPOINT_DIR / args.dataset
    os.makedirs(task_out_dir, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(task_out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=10,
        max_steps=-1,
        num_train_epochs=args.epochs,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        report_to="none",
        max_length=2048,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # 7. Trainer — peft_config passed here so SFTTrainer manages the PEFT wrapping
    # cleanly alongside its own data collation logic.
    # EarlyStoppingCallback: stops when eval_loss hasn't improved for 3 consecutive
    # eval checkpoints (3 × 200 steps = 600 steps of patience).
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.001)],
    )

    print("Starting training...")
    trainer.train()

    # 8. Save LoRA Adapters
    final_out = OUTPUT_MODEL_DIR / args.dataset
    print(f"Saving LoRA adapters to {final_out}...")
    trainer.model.save_pretrained(str(final_out))
    tokenizer.save_pretrained(str(final_out))
    print("Done!")

if __name__ == "__main__":
    main()

