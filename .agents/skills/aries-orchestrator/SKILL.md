---
name: aries-orchestrator
description: Core workflow and state manager for the ARIES SOAR platform, ML Triage Engine, and Dashboard UI.
---

# ARIES Project State & Rules

## 1. Assessment & Execution Protocol
When asked to evaluate the current approach, you must first read the local codebase, specifically targeting the ML Triage Engine and the data pipelines. Attempt to run the existing evaluation scripts locally to capture stdout/stderr and determine if the failure is due to logic bugs, data formatting, or a fundamentally flawed architectural approach.

## 2. Research Protocol
If the current approach is deemed inadequate, you must use your research tools to investigate the current meta for Security Orchestration, Automation, and Response (SOAR) platforms, specifically looking at recent advancements in local LLM triage and Named Entity Recognition (NER). 

## 3. Hydra Cluster Training Rules
If a new model training run is required, you cannot execute it directly. Instead, you must generate the complete Slurm batch scripts and data sync commands for the user. 
**CRITICAL CLUSTER RULE:** When generating any Slurm script for Hydra, you MUST include the following lines before any execution to ensure Hugging Face and Torch caches are stored in scratch space, avoiding home directory quota limits:
`export HF_HOME=$VSC_SCRATCH`
`export TORCH_HOME=$VSC_SCRATCH`

## 4. State Tracking
Maintain a highly structured `PLAN.md` in the root directory. Update this file automatically after the assessment phase, outlining the agreed-upon architecture changes before writing any new code.