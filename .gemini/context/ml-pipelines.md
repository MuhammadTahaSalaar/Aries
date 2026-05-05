# ML Pipelines

## Pipeline Registry

| Pipeline              | Model                          | Dataset(s)                   | Entry Point                 | REQ           |
| --------------------- | ------------------------------ | ---------------------------- | --------------------------- | ------------- |
| Alert Triage          | XGBoost → ONNX                 | GUIDE (13.6 M rows)          | `src/triage/`               | REQ-02,03,04  |
| NER / IOC Extraction  | SecureBERT-NER fine-tuned → ONNX | CyNER + CASIE               | `src/nlp/ner/`              | REQ-09,11,12  |
| Incident Summarization | BART fine-tuned → ONNX        | GovReport; eval on APTnotes  | `src/nlp/summarizer/`       | REQ-10        |

All training artifacts are persisted in MLflow backed by MinIO/S3. Production
inference uses **ONNX Runtime exclusively**; PyTorch is training-only.

## Dataset Registry

### GUIDE — Alert Triage

- Location: `datasets/GUIDE/GUIDE_Train.csv` · `GUIDE_Test.csv`
- Size: ~9.5 M train rows · ~4.1 M test rows
- Label column: `IncidentGrade` → `TruePositive` / `FalsePositive` / `BenignPositive`
- Key features: `MitreTechniques`, `AlertTitle`, `Category`, `ThreatFamily`,
  `SuspicionLevel`, `EntityType`, `DeviceName`, `IpAddress`, `Sha256`

### CyNER (MITRE) — NER Fine-Tuning

- Location: `datasets/CyNER/` (BIO-tagged CoNLL splits)
- Entities: `Malware`, `Tool`, `Indicator`, `Vulnerability`, `Organization`, `Person`, `Location`

### CASIE — Cybersecurity Event Extraction

- Location: `datasets/CASIE/`
- 1,000 JSON-annotated articles; event types: Databreach, Phishing, Ransom,
  Discover-Vulnerability, Patch-Vulnerability

### GovReport — Summarization Fine-Tuning

- HuggingFace: `ccdv/govreport-summarization`
- Location: `datasets/gov_reports/`
- 17,517 train / 973 val / 973 test; up to 9,000 token input

### APTnotes — Evaluation Corpus

- Location: `datasets/APTnotes/` (400+ PDFs, 2008–2023)
- Used for held-out summarization evaluation only

### SecureBERT Backbone

- HF identifier: `ehsanaghaei/SecureBERT` (RoBERTa retrained on security corpus)
- Successor: `cisco-ai/SecureBERT2.0-base`

## Model Artifact Locations

```
models/
├── ner/          — SecureBERT NER fine-tuned weights (safetensors + tokenizer)
├── triage/       — XGBoost triage model
├── summarizer/   — BART fine-tuned weights
└── onnx/         — Production ONNX exports
    ├── ner.onnx / ner.opt.onnx
    ├── triage.onnx
    └── summarizer/  (encoder + decoder)
```

## Triage Pipeline Details

- 3-class output: `BenignPositive`, `FalsePositive`, `TruePositive`
- Binary scoring: `TP=1, FP/BP=0`
- Risk score overrides: `risk >= 50 → TruePositive`; `risk >= 35 and BP → FalsePositive`
- `auto_close` gated on `risk_score < 35`
- `asset_criticality` and `behavioral_score` included in `TriageResult`
- Enrichment uses severity, entity keywords, and MITRE tactics (not stub 0.5/0.5)

## NER Pipeline Details

- 11 entity types. IOC post-processing via `_recover_missed_iocs()` (regex for
  CVEs, SHA256/SHA1, IPs, URLs, emails).
- `_fix_partial_entities()` expands truncated CVE tokens.
- Deduplication runs after fix step.

## Summarization Pipeline Details

- **Do not** add instruction prefixes (BART-base is not instruction-tuned).
- `_clean_summary()` uses sentence-level source grounding (keeps faithful
  sentences, drops hallucinated tail).
- `_lead_sentences()` fallback when all output is hallucinated.
- Dynamic `min_new_tokens` scaling prevents forced hallucination past EOS.
- Two modes: `executive` and `analyst`.
