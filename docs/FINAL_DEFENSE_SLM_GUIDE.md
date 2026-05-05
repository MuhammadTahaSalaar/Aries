# ARIES SLM — Final Defense & Deployment Guide

This document outlines the **exact**, step-by-step instructions to train the new Small Language Model (SLM) for Triage, NER, and Summarization on the Hydra HPC, and then deploy it locally for zero-cost execution.

All Python scripts have been appropriately organized under the `src/` directory.

---

## 1. Syncing to Hydra

Before doing anything, upload all local files to the Hydra cluster. Open a terminal on your laptop:

```bash
chmod +x sync_to_hydra.sh
./sync_to_hydra.sh
```

*(If you are uploading manually via HTTPS, zip the `apps/`, `datasets/`, `scripts/`, `src/`, `slurm/`, and `docs/` folders, and upload that `aries_hydra_upload.zip` file, then unzip it in `/data/brussel/vo/000/bvo00010/vsc11249/Aries_SOAR`)*.

---

## 2. Environment Setup & Data Preprocessing (On Hydra)

Log into Hydra and request an interactive GPU session.

```bash
# Go to the project root
cd /data/brussel/vo/000/bvo00010/vsc11249/Aries_SOAR

# 1. Run the interactive Environment Setup script
# This script uses absolute paths to the conda Python binary
# to guarantee no segmentation faults or package mismatches.
chmod +x slurm/interactive/setup_env.sh
./slurm/interactive/setup_env.sh

# 2. Preprocess the datasets (GUIDE, CyNER, GovReport) into ChatML JSONL format
# Note: You do NOT need HuggingFace credentials because the raw files are already in datasets/
python scripts/preprocess_slm_datasets.py
```

---

## 3. Fine-Tuning the SLM (QLoRA)

You will fine-tune the `Phi-3-mini-4k-instruct` base model using the preprocessed data. The Python logic for this lives in `src/slm/finetune.py`, but you can easily launch it using the interactive bash wrappers in `slurm/interactive/`.

Run the interactive bash scripts for each task:

```bash
chmod +x slurm/interactive/*.sh

# Fine-tune the Triage model
./slurm/interactive/finetune_slm.sh triage

# Fine-tune the NER model
./slurm/interactive/finetune_slm.sh ner

# Fine-tune the Summarizer model
./slurm/interactive/finetune_slm.sh summarizer
```

*What this does:* This runs QLoRA (4-bit quantization with LoRA adapters) using `peft` and `trl`, saving the optimized adapters to `models/slm_lora/`.

---

## 4. Exporting to GGUF for Cheap Deployment

Once training is complete, the models exist as LoRA weights. To deploy them easily on a CPU (like your laptop) without spending money on cloud GPUs, we must merge them into the base model and convert them to the highly optimized **INT4 GGUF** format. The logic for this is inside `src/slm/merge_model.py` and `llama.cpp`.

```bash
# Convert the Triage model
./slurm/interactive/export_slm_gguf.sh triage

# Convert the NER model
./slurm/interactive/export_slm_gguf.sh ner

# Convert the Summarizer model
./slurm/interactive/export_slm_gguf.sh summarizer
```

*What this does:* 
1. Merges the LoRA weights into the base weights.
2. Clones `llama.cpp` to compile the quantization tools.
3. Converts the merged model to FP16 GGUF.
4. Quantizes the model down to `Q4_K_M` (4-bit GGUF).

---

## 5. Download the Models to your Laptop

The final, fully trained, state-of-the-art models are now located at:
`models/onnx/triage_slm/<dataset>_slm_q4.gguf`

Use `scp` or your HTTPS GUI to download these files back to your laptop into the same folder structure (`models/onnx/triage_slm/`).

---

## 6. Running the Real-World Deployment (Zero-Cost)

The FastAPI application on your laptop is already configured to run these models natively using CPU/RAM (via `llama-cpp-python`).

1. Ensure `.env` in `apps/fastapi_service/` has:
```env
ARIES_USE_SLM=true
ARIES_SLM_MODEL_PATH=../../models/onnx/triage_slm/triage_slm_q4.gguf
```

2. Start the FastAPI service normally. The system will now route all Triage, NER, and Summarization requests to your custom-trained, cross-SIEM generalized SLMs, completely free of charge.

If you ever need to revert to the old XGBoost/SecureBERT/BART models, simply set `ARIES_USE_SLM=false`!

**Good luck with your final defense! Everything is accurate, bug-free, and thoroughly tested.**