# VUB Hydra HPC - GPU Resources for ARIES

## Available GPU Partitions

| Partition | Nodes | GPUs/Node | GPU Memory | Total GPUs | Best For |
|-----------|-------|-----------|------------|------------|----------|
| `pascal_gpu` | 4 | 2× P100 | 16 GB | 8 | Light testing, small models |
| `ampere_gpu` | 6 | 2× A100 | 40 GB | 12 | **Recommended** - BART-base |
| `ampere_gpu` (big_local_ssd) | 4 | 2× A100 | 40 GB | 8 | Large datasets |
| `hopper_gpu` | 5 | 2× H200 | **140 GB** | 10 | Large models (BART-large) |

## Resource Limits Per Node

| Partition | CPUs | RAM | Local Disk | Network |
|-----------|------|-----|------------|---------|
| `pascal_gpu` | 24 cores (2×12) | 256 GB | 2 TB HDD | 10 Gbps |
| `ampere_gpu` | 32 cores (2×16) | 256 GB | 2 TB SSD | 100 Gbps EDR-IB |
| `hopper_gpu` | 48 cores (2×24) | 754 GB | 447 GB SSD | 400 Gbps NDR-IB |

## ARIES Training Recommendations

### XGBoost Triage (9.4M samples)
- **Partition**: Any GPU partition (XGBoost uses GPU efficiently)
- **GPUs**: 1 (single GPU is sufficient)
- **RAM**: 64-128 GB
- **Time**: ~2-4 hours

```bash
sbatch slurm/train_xgboost.sh  # Uses 1 GPU on ampere_gpu
```

### BART Summarization
| Partition | Model | Batch Size | Effective Batch | Est. Time |
|-----------|-------|------------|-----------------|-----------|
| `pascal_gpu` | bart-base | 2/GPU | 32 | ~24h |
| `ampere_gpu` | bart-base | 8/GPU | 128 | ~8h |
| `hopper_gpu` | **bart-large** | 16/GPU | 256 | ~4h |

```bash
# Default (A100s)
sbatch slurm/train_bart.sh

# For faster training on H200s
sbatch --partition=hopper_gpu slurm/train_bart.sh
```

### Full Pipeline

```bash
# Standard (ampere_gpu A100s)
./slurm/submit_all.sh

# High-performance (hopper_gpu H200s)  
./slurm/submit_all.sh --partition=hopper_gpu

# Budget option (pascal_gpu P100s)
./slurm/submit_all.sh --partition=pascal_gpu
```

## Multi-GPU Training

**Maximum GPUs per node: 2**

ARIES automatically uses both GPUs on the node:
- BART uses Accelerate DDP with 2 GPUs
- Effective batch size = `batch_per_gpu × 2 × gradient_accumulation`

For 4+ GPUs, you would need multi-node (not currently configured).

## Batch Size Auto-Configuration

The BART training script automatically detects GPU memory and configures:

| GPU | Memory | Batch/GPU | Grad Accum | Model |
|-----|--------|-----------|------------|-------|
| P100 | 16 GB | 2 | 8 | bart-base |
| A100 | 40 GB | 8 | 4 | bart-base |
| H200 | 140 GB | 16 | 2 | **bart-large** |

## Storage Paths

```
$VSC_DATA_VO_USER/Aries_SOAR/
├── envs/aries/          # Conda environment
├── mlruns/              # MLflow tracking (file://)
├── checkpoints/         # Training checkpoints
│   ├── xgboost/
│   └── bart/
├── models/              # Final models
│   ├── xgboost/
│   ├── bart/
│   └── onnx/
├── data/processed/      # Preprocessed data
├── src/                 # Source code
└── slurm/               # SLURM scripts
```

## Example Job Submission

```bash
# 1. Setup (once)
sbatch slurm/setup_env.sh

# 2. Monitor
squeue -u $USER

# 3. After setup completes, train
./slurm/submit_all.sh --partition=hopper_gpu

# Check job status
sacct -j <job_id> --format=JobID,JobName,State,Elapsed,MaxRSS
```

## Troubleshooting

### Queue Times
- `ampere_gpu`: Popular, may have 1-2h wait
- `hopper_gpu`: Newer, often shorter queue
- `pascal_gpu`: Usually shortest queue

### Out of Memory
The scripts automatically handle OOM by:
1. Saving emergency checkpoint
2. Clearing CUDA cache
3. Logging OOM event to MLflow

### Check GPU Availability
```bash
sinfo -p ampere_gpu -N -l
sinfo -p hopper_gpu -N -l
```
