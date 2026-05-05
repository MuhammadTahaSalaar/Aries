# ARIES SLM Architecture & Hydra Deployment Guide

## 1. The New Architecture: Semantic Triage via SLM

ARIES has evolved from using isolated, task-specific models (XGBoost for Triage, SecureBERT for NER, and BART for Summarization) to a unified **Small Language Model (SLM)** architecture (e.g., Phi-3-mini or Llama-3-8B). 

### Why the Shift?
The legacy XGBoost model used Target Encoding trained on Microsoft GUIDE data. When ingesting alerts from new SIEMs (like Wazuh or Splunk), unseen categorical values collapsed to benign priors, severely degrading the detection of critical cross-SIEM alerts. 

The new **Semantic Architecture** parses the raw JSON and narrative text of an alert directly using prompt engineering, decoupling the AI from vendor-specific data schemas. 

### Component Breakdown
1. **Semantic Triage (`src/triage/slm_inference.py`)**: The SLM reads the alert as a SOC analyst would, outputting structured JSON (`grade`: TruePositive/FalsePositive, and `confidence`).
2. **Generative NER (`src/nlp/ner/slm_inference.py`)**: The SLM extracts STIX 2.1-aligned entities (Malware, Indicators, Systems) and events. Legacy regex logic remains as a safety net to validate extracted IPs, hashes, and domains.
3. **Fact-Grounded Summarization (`src/nlp/summarizer/slm_inference.py`)**: The SLM generates executive and analyst summaries without the hallucination issues present in the legacy BART-base model.

---

## 2. Hydra HPC Workflow: Setup & Fine-Tuning

Since local environments lack GPU resources, all heavy lifting (fine-tuning, dataset processing, and quantization) must be executed on the VUB Hydra HPC cluster.

### Step 2.1: Environment Setup
We have updated the `slurm/setup_env.sh` script to include GenAI requirements (`vllm`, `llama-cpp-python`, `onnxruntime-genai`, `bitsandbytes`, `transformers`).

On the Hydra login node, run:
```bash
cd /data/FYP/Aries
sbatch slurm/setup_env.sh
```
Monitor the setup:
```bash
tail -f aries_setup_*.out
```

### Step 2.1b: Syncing Local Repo to Hydra
If you have made local changes, sync them to Hydra using the provided script. This script automatically creates the remote directory if it does not exist and respects `.rsync_exclude`.
```bash
# On your local machine
chmod +x sync_to_hydra.sh
./sync_to_hydra.sh
```

### Step 2.2: Dataset Acquisition
Once the environment is ready, download the datasets to your Hydra storage (`/data/brussel/vo/000/bvo00010/vsc11249/Aries_SOAR/datasets/`). Most datasets are hosted on Hugging Face.

**Credentials Needed:**
You will need a free Hugging Face account and an Access Token (`HF_TOKEN`) to download them. No special gated access is usually required, but logging in is best practice.
```bash
export HF_TOKEN="your_huggingface_token"
```

**Datasets to Download:**
*   **Triage**: Augmented GUIDE dataset.
    *   *Source*: `microsoft/GUIDE` on Hugging Face.
*   **NER**: **CyberNER (2024/2025)** - STIX 2.1 harmonized dataset (CyNER + APTNER).
    *   *Source*: `yasirech-chammakhy/CyberNER` on Hugging Face / GitHub.
*   **Summarization**: S-RM 2025 Cyber Incident Insights / GovReport.
    *   *Source*: `ccdv/govreport-summarization` on Hugging Face.

You can use the preprocessing scripts in `scripts/preprocess_slm_datasets.py` to format these raw datasets into SLM-ready JSONL conversational prompts.

### Step 2.3: Fine-Tuning (QLoRA)
Write a Python script (e.g., `slurm/finetune_slm.py`) utilizing HuggingFace `SFTTrainer` and `bitsandbytes` to fine-tune your chosen base model (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`) on the aforementioned datasets. Submit this job to the GPU partition:
```bash
sbatch --partition=gpu --gres=gpu:1 --mem=32G --wrap="python slurm/finetune_slm.py"
```

### Step 2.4: Quantization to GGUF
To run the model cost-effectively in production, convert your fine-tuned model to **INT4 GGUF** format using `llama.cpp` scripts on Hydra. This compresses an 8B model from ~16GB to ~4.5GB, allowing it to run entirely on CPU/RAM if necessary.

---

## 3. MLflow Integration

Once your SLM is fine-tuned and quantized into a `.gguf` or `.onnx` file, it must be registered with MLflow so the ARIES FastAPI service can dynamically pull it at startup.

### Step 3.1: Log the Artifact to MLflow
In your Hydra training script, log the final GGUF file to MLflow:

```python
import mlflow

mlflow.set_tracking_uri("http://<YOUR_MLFLOW_SERVER>:5000")
mlflow.set_experiment("aries/slm-unified")

with mlflow.start_run() as run:
    # Log training metrics...
    mlflow.log_metric("eval_loss", 0.12)
    
    # Log the GGUF file as an artifact to MinIO
    mlflow.log_artifact("path/to/fine_tuned_model.gguf", artifact_path="slm_model")
    
    # Optionally log the system prompts used
    mlflow.log_dict({"triage_prompt": "...", "ner_prompt": "..."}, "prompts.json")
```

### Step 3.2: Configure FastAPI to use the SLM
Update your `.env` file in the FastAPI service to point to the new model and enable the SLM routing:

```env
ARIES_USE_SLM=true
ARIES_SLM_MODEL_PATH=/tmp/aries_models/slm/fine_tuned_model.gguf
```
*Note: In production, the FastAPI lifespan event should be updated to download the `slm_model` artifact from MLflow/MinIO to the `ARIES_SLM_MODEL_PATH` just like it currently does for XGBoost.*

---

## 4. Cost-Effective Real-World Deployment

For a real-world SOAR platform, deployment costs for LLMs can spiral out of control. Here are the most cost-effective deployment tiers depending on your data privacy requirements:

### Option 1: The "Practically Free" Self-Hosted CPU Route (Highest Privacy, High Latency)
Because SOAR platforms operate asynchronously (Kafka queues), a triage latency of 2–5 seconds is often acceptable. 
*   **Hardware**: A cheap VPS (e.g., Hetzner CPX31: 4 vCPU, 8GB RAM for ~$9/month) or a repurposed local office server.
*   **Execution**: Use `llama-cpp-python` (already implemented in `src/.../slm_inference.py`) to run the INT4 quantized GGUF model strictly on CPU RAM.
*   **Cost**: ~$0 to $10/month. 

### Option 2: Cheap GPU Cloud Instances (High Privacy, Low Latency)
If you require sub-second triage without buying a physical GPU:
*   **Hardware**: Rent a serverless GPU or cheap instance from **RunPod**, **Vast.ai**, or **Lambda Labs**. An RTX 3090 or RTX 4000 Ada costs ~$0.20 to $0.30 per hour.
*   **Execution**: Deploy the SLM using **vLLM** as an OpenAI-compatible API container alongside your FastAPI service. Update `slm_inference.py` to point to `http://vllm-container:8000/v1/completions`.
*   **Cost**: ~$150 - $200/month (if running 24/7). 

### Option 3: Serverless API APIs (Zero Infrastructure, Low Privacy)
If the SIEM data does not contain strictly confidential PII, or if you scrub it before sending:
*   **Providers**: **Groq** (extremely fast Llama-3 inference, generous free tier), **Cloudflare Workers AI**, or **Together AI**.
*   **Execution**: Replace the local `Llama` instantiation in `slm_inference.py` with standard HTTP requests to the Groq/Together API.
*   **Cost**: Essentially $0/month for standard SOC volumes, paying only fractions of a cent per 1M tokens if you exceed free tiers.

### Recommended Stack for ARIES MVP
For the final project demonstration and early real-world use:
1. **Train on Hydra (GPU)**.
2. **Export to INT4 GGUF**.
3. **Deploy using Option 1 (CPU / `llama.cpp`)**. The provided code in `slm_inference.py` is already perfectly optimized for this. As long as your local machine or cheap VPS has 8GB of free RAM, the entire AI-driven SOAR pipeline will run locally and privately at zero recurring cost.
