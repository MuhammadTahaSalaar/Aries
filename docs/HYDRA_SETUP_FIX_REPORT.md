# Hydra Setup Fix Report

## Issue Identified
When running `mamba` globally or activating it incorrectly on Hydra, the cluster's base Mamba installation threw a **Segmentation Fault (core dumped)**. Because `mamba` crashed, the Python environment failed to build, which cascaded into the `vllm` pip installation attempting a source build (which failed due to the missing `nvcc` compiler since CUDA Toolkit wasn't installed).

## Changes Made
All fixes have been meticulously applied following the exact structure of your successful reference project.

### 1. Environment Setup Scripts (`slurm/setup_env.sh` and `slurm/interactive/setup_env.sh`)
- **Proper Mamba Initialization:** We now source Mamba directly from the absolute root installation (`$MAMBA_ROOT/etc/profile.d/mamba.sh`) before attempting to run `mamba create` or `mamba activate`. This completely bypasses the segmentation faults caused by corrupted global paths.
- **Python Binary Isolation:** Instead of relying on a globally resolved `pip`, the scripts now resolve the exact Python binary inside the newly created Conda environment (`PYTHON_BIN="${ENV_PATH}/bin/python"`) and use `${PYTHON_BIN} -m pip install`. This guarantees packages are installed correctly.
- **CUDA Compiler Injection:** Added `mamba install -y -c nvidia cuda-toolkit=12.1.1` and exported `CUDA_HOME` to guarantee that if `vllm` needs to compile C++ kernels, `nvcc` is fully available and functional.

### 2. File Organization
- You pointed out `finetune_slm.py` was inside the `slurm/` folder. This was corrected.
- **`src/slm/finetune.py`**: Contains the state-of-the-art QLoRA training script.
- **`src/slm/merge_model.py`**: Contains the LoRA merging logic.
- **`slurm/interactive/finetune_slm.sh`**: The interactive shell wrapper for training.
- **`slurm/interactive/export_slm_gguf.sh`**: The interactive shell wrapper for exporting to GGUF.

## What You Need To Do Now
1. **Sync to Hydra:** From your laptop, run `./sync_to_hydra.sh`
2. **Open an Interactive Session:** SSH into Hydra and request an interactive GPU session (`srun --pty bash`).
3. **Run the New Setup:** Execute the fixed environment setup script:
   ```bash
   chmod +x slurm/interactive/*.sh
   ./slurm/interactive/setup_env.sh
   ```
4. **Preprocess Data (already done locally — sync output instead):**
   The preprocessing script has been fixed and run locally. Do NOT re-run it on Hydra.
   The outputs in `data/processed/slm_finetuning/` are already ready and will be uploaded by `sync_to_hydra.sh`.
   
   Fixes applied to `scripts/preprocess_slm_datasets.py`:
   - **NER 0-sentence bug fixed:** CyNER uses tab delimiters, not spaces. `line.split(" ")` → `line.split()`.
   - **NER data quality:** `ioc_type` was hardcoded `"Unknown"` for every entity. Now maps from label (`Malware` → `malware_name`, `Vulnerability` → `cve`, etc.).
   - **NER coverage:** `valid.txt` is now included in training (813 extra sentences, total 3,624).
   - **Triage sample size:** Increased from 10,000 random rows to **100,000 stratified rows** (50k TruePositive, 50k FalsePositive/Benign). A random 10k head from a 9.5M-row file ordered by time was severely class-imbalanced and lacked minority-class coverage.
   - **Triage class balance:** Stratified chunk-sampling guarantees exactly 50k of each class, preventing the SLM from learning a biased prior.
   - **Summarization context:** Report truncation increased from 2,000 → 4,000 characters (~600 tokens), preserving more factual context and reducing hallucination risk.
   - **Summarization coverage:** Both train parquet shards (`train-00000` and `train-00001`) are now used (2,000 rows each, 4,000 total) for vocabulary diversity.
5. **Train and Export:**
   ```bash
   ./slurm/interactive/finetune_slm.sh triage
   ./slurm/interactive/export_slm_gguf.sh triage
   ```
6. **Deploy:** SCP the resulting `.gguf` file back to your laptop and run the FastAPI server for free, CPU-optimized triage. 

*Your defense guide `docs/FINAL_DEFENSE_SLM_GUIDE.md` has been updated and holds these steps as well.* Good luck!