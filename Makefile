DOCKER_DIR=MLOps
COMPOSE_FILE=$(DOCKER_DIR)/docker-compose.yml
VENV = .venv
VENV_BIN = $(VENV)/bin
PYTHON = python

# ── MLOps Docker stack ────────────────────────────────────────────────

install:
	docker compose \
		--env-file $(DOCKER_DIR)/.env \
		--file $(COMPOSE_FILE) \
		up --detach

test:
	test -d $(VENV) || python3 -m venv $(VENV)
	$(VENV_BIN)/pip install -r $(DOCKER_DIR)/tests/requirements.txt
	$(VENV_BIN)/python $(DOCKER_DIR)/tests/test_mlflow.py

stack-down:
	docker compose -f $(COMPOSE_FILE) down

# ── Local preprocessing ──────────────────────────────────────────────

preprocess-triage:
	$(PYTHON) -m src.triage.run_preprocessing

preprocess-ner:
	$(PYTHON) -m src.nlp.ner.run_preprocessing

preprocess-bart:
	$(PYTHON) -m src.nlp.summarizer.run_preprocessing

preprocess-all: preprocess-triage preprocess-ner preprocess-bart

# ── Local training (CPU / single GPU) ────────────────────────────────

train-triage:
	$(PYTHON) -m src.triage.run_training

train-ner:
	$(PYTHON) -m src.nlp.ner.run_training

train-bart:
	$(PYTHON) -m src.nlp.summarizer.run_training

train-all: train-triage train-ner train-bart

# ── ONNX export ──────────────────────────────────────────────────────

export-onnx:
	$(PYTHON) -m src.export_onnx --all

export-triage-onnx:
	$(PYTHON) -m src.export_onnx --triage

export-ner-onnx:
	$(PYTHON) -m src.export_onnx --ner

export-bart-onnx:
	$(PYTHON) -m src.export_onnx --summarizer

# ── HPC (submit to Hydra SLURM) ─────────────────────────────────────

hpc-setup:
	sbatch slurm/setup_env.sh

hpc-xgboost:
	sbatch slurm/train_xgboost.sh

hpc-ner:
	sbatch slurm/train_ner.sh

hpc-bart:
	sbatch slurm/train_bart.sh

hpc-export:
	sbatch slurm/export_onnx.sh

hpc-all:
	./slurm/submit_all.sh

# ── Cleanup ──────────────────────────────────────────────────────────

clean:
	docker compose -f $(COMPOSE_FILE) down
	rm -rf $(VENV)

clean-processed:
	rm -rf data/processed/*

clean-checkpoints:
	rm -rf checkpoints/triage/* checkpoints/ner/* checkpoints/bart/*

clean-models:
	rm -rf models/triage/* models/ner/* models/summarizer/* models/onnx/*

clean-all: clean clean-processed clean-checkpoints clean-models

# ── Convenience ──────────────────────────────────────────────────────

all: install test

.PHONY: install test stack-down \
        preprocess-triage preprocess-ner preprocess-bart preprocess-all \
        train-triage train-ner train-bart train-all \
        export-onnx export-triage-onnx export-ner-onnx export-bart-onnx \
        hpc-setup hpc-xgboost hpc-ner hpc-bart hpc-export hpc-all \
        clean clean-processed clean-checkpoints clean-models clean-all all