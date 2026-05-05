#!/usr/bin/env python3
###############################################################################
# ARIES — SLM Fine-tuning using QLoRA
#
# Fine-tunes a base model (e.g., Llama-3 or Phi-3) using the JSONL files 
# produced by `preprocess_slm_datasets.py`. Uses 4-bit quantization and LoRA
# for memory-efficient training on a single GPU.
#
# Usage:
#   python slurm/finetune_slm.py --dataset triage
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
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Base dirs
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "slm_finetuning"
CHECKPOINT_DIR = BASE_DIR / "checkpoints" / "slm"
OUTPUT_MODEL_DIR = BASE_DIR / "models" / "slm_lora"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="microsoft/Phi-3-mini-4k-instruct", help="Base model ID from HuggingFace")
    parser.add_argument("--dataset", type=str, choices=["triage", "ner", "summarizer", "all"], default="triage", help="Which dataset to fine-tune on")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Dataset Path
    dataset_file = PROCESSED_DIR / f"{args.dataset}_slm_train.jsonl"
    if not dataset_file.exists():
        print(f"Error: Dataset {dataset_file} not found. Run scripts/preprocess_slm_datasets.py first.")
        return

    print(f"Loading dataset from {dataset_file}...")
    dataset = load_dataset("json", data_files=str(dataset_file), split="train")

    # Split for basic eval
    dataset = dataset.train_test_split(test_size=0.05)

    # 2. Tokenizer
    print(f"Loading tokenizer {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix for fp16

    # 3. 4-bit Quantization Config (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 4. Load Base Model
    print(f"Loading base model {args.model_id} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False
    
    # Prepare model for kbit training
    model = prepare_model_for_kbit_training(model)

    # 5. LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, peft_config)

    # 6. Training Arguments
    task_out_dir = CHECKPOINT_DIR / args.dataset
    os.makedirs(task_out_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(task_out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=10,
        max_steps=-1,
        num_train_epochs=args.epochs,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        fp16=False,
        bf16=True, # Use bf16 on Ampere/Hopper
        optim="paged_adamw_8bit",
        report_to="none" # Switch to 'mlflow' if mlflow running locally
    )

    # 7. Formatting function for ChatML (if dataset is just "messages")
    def format_chat_template(example):
        # HuggingFace tokenizer can apply chat templates directly
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)}
    
    train_dataset = dataset["train"].map(format_chat_template)
    eval_dataset = dataset["test"].map(format_chat_template)

    # 8. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_args,
    )

    # 9. Train
    print("Starting training...")
    trainer.train()

    # 10. Save LoRA Adapters
    final_out = OUTPUT_MODEL_DIR / args.dataset
    print(f"Saving final LoRA adapters to {final_out}...")
    trainer.model.save_pretrained(str(final_out))
    tokenizer.save_pretrained(str(final_out))
    print("Done!")

if __name__ == "__main__":
    main()
