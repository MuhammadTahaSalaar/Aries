"""
ARIES — SLM Dataset Preprocessing Scripts

This script processes the local datasets (GUIDE, CyNER, gov_reports) already stored 
in the `datasets/` directory into standard JSONL conversational format (ChatML-style) 
for Instruction Fine-Tuning (SFTTrainer).

No Hugging Face credentials or internet access is required, as this script uses 
the offline data shipped with the repository.

Expected output format per row:
{"messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
]}
"""

import json
import os
import argparse
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("Error: 'pandas' is required to read the CSV and Parquet datasets. Run 'pip install pandas pyarrow'.")
    exit(1)

# Base dirs
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "slm_finetuning"

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def preprocess_triage_guide(max_rows=100000):
    """
    Reads datasets/GUIDE/GUIDE_Train.csv with stratified sampling to ensure balanced
    TruePositive / FalsePositive classes. Samples up to max_rows total (50k each class).
    The GUIDE dataset has 9.5M rows — a random head() would be heavily class-imbalanced
    since the file is sorted by time; stratified chunking guarantees coverage of both classes.
    """
    print(f"Preprocessing Triage Dataset (local GUIDE_Train.csv, target={max_rows} stratified rows)...")
    in_file = DATA_DIR / "GUIDE" / "GUIDE_Train.csv"
    out_file = PROCESSED_DIR / "triage_slm_train.jsonl"

    if not in_file.exists():
        print(f"Error: Could not find {in_file}. Make sure datasets are synced.")
        return

    target_per_class = max_rows // 2
    tp_rows, fp_rows = [], []
    tp_count, fp_count = 0, 0

    for chunk in pd.read_csv(in_file, chunksize=50000, low_memory=False):
        grade_col = chunk.get("IncidentGrade", pd.Series(dtype=str))
        tp = chunk[grade_col == "TruePositive"]
        fp = chunk[grade_col.isin(["FalsePositive", "BenignPositive"])]

        tp_need = target_per_class - tp_count
        fp_need = target_per_class - fp_count

        if tp_need > 0 and len(tp) > 0:
            tp_rows.append(tp.iloc[:tp_need])
            tp_count += min(tp_need, len(tp))
        if fp_need > 0 and len(fp) > 0:
            fp_rows.append(fp.iloc[:fp_need])
            fp_count += min(fp_need, len(fp))

        if tp_count >= target_per_class and fp_count >= target_per_class:
            break

    df = pd.concat(tp_rows + fp_rows, ignore_index=True).sample(frac=1, random_state=42)
    print(f"  Stratified sample: {tp_count} TruePositive, {fp_count} FalsePositive/Benign")

    with open(out_file, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            grade = str(row.get("IncidentGrade", "FalsePositive"))
            if grade == "BenignPositive":
                grade = "FalsePositive"

            alert_json = {
                "normalized_title": str(row.get("AlertTitle", "Unknown")),
                "severity": str(row.get("Severity", "Medium")),
                "category": str(row.get("Category", "Unknown")),
                "suspicion_level": str(row.get("SuspicionLevel", "Unknown"))
            }

            system_msg = ("You are a senior SOC analyst. Your task is to evaluate the following "
                          "SIEM alert and classify it as TruePositive or FalsePositive.\n"
                          "Respond ONLY with a valid JSON object containing exactly two keys:\n"
                          "1. \"grade\": A string, either \"TruePositive\" or \"FalsePositive\".\n"
                          "2. \"confidence\": A float between 0.0 and 1.0 representing your confidence.\n"
                          "Do not output any markdown formatting, explanation, or other text.")
            user_msg = f"Alert Context:\n{json.dumps(alert_json, indent=2)}"
            assistant_msg = json.dumps({"grade": grade, "confidence": 0.95 if grade == "TruePositive" else 0.85})

            chat = {"messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ]}
            f.write(json.dumps(chat) + "\n")

    print(f"Saved Triage tuning data to {out_file} ({len(df)} rows)")


# Maps CyNER IOB label to a meaningful ioc_type string for the assistant response.
_LABEL_TO_IOC_TYPE = {
    "Malware": "malware_name",
    "Indicator": "indicator",
    "System": "hostname",
    "Vulnerability": "cve",
    "Organization": "organization",
}


def _parse_iob_file(path: Path):
    """Parse a CyNER-style IOB file (tab-delimited, UTF-8, may have CRLF endings).
    Returns a list of (tokens, tags) tuples.

    The file format is:  TOKEN<TAB>TAG\n  with blank lines between sentences.
    Note: line.split() (not split(' ')) handles both tab and space delimiters.
    """
    sentences = []
    current_tokens: list = []
    current_tags: list = []

    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()  # removes \r\n, \n, leading whitespace
            if not line:
                if current_tokens:
                    sentences.append((current_tokens, current_tags))
                    current_tokens = []
                    current_tags = []
                continue
            parts = line.split()  # splits on ANY whitespace (tab OR space)
            if len(parts) >= 2:
                current_tokens.append(parts[0])
                current_tags.append(parts[-1])

    if current_tokens:  # flush last sentence if file doesn't end with blank line
        sentences.append((current_tokens, current_tags))

    return sentences


def preprocess_ner_cyner():
    """
    Reads CyNER train.txt + valid.txt (IOB format, tab-delimited) and converts to
    conversational JSONL for SFT. Both splits are combined since valid.txt provides
    ~1,200 additional labelled sentences and there is a separate test.txt held out.
    ioc_type is inferred from the entity label rather than hardcoded to 'Unknown'.
    """
    print("Preprocessing NER Dataset (CyNER train.txt + valid.txt)...")
    out_file = PROCESSED_DIR / "ner_slm_train.jsonl"

    all_sentences = []
    for split in ("train.txt", "valid.txt"):
        path = DATA_DIR / "CyNER" / split
        if not path.exists():
            print(f"  Warning: {path} not found, skipping.")
            continue
        parsed = _parse_iob_file(path)
        print(f"  Parsed {len(parsed)} sentences from {split}")
        all_sentences.extend(parsed)

    if not all_sentences:
        print("Error: no NER sentences parsed. Check dataset path.")
        return

    system_msg = ("You are a cybersecurity Named Entity Recognition (NER) extractor.\n"
                  "Extract all STIX 2.1 aligned entities from the text. "
                  "Valid labels are: Malware, Indicator, System, Vulnerability, Organization.\n"
                  "Also extract any broad Security Events (e.g., Ransomware, Phishing, Backdoor).\n"
                  "Respond ONLY with a valid JSON object matching this schema:\n"
                  "{\n"
                  "  \"entities\": [\n"
                  "    {\"text\": \"string\", \"label\": \"string\", \"ioc_type\": \"string\"}\n"
                  "  ],\n"
                  "  \"events\": [\n"
                  "    {\"type\": \"string\", \"keyword\": \"string\"}\n"
                  "  ]\n"
                  "}\n"
                  "Do not output any markdown formatting, explanation, or other text.")

    with open(out_file, "w", encoding="utf-8") as f:
        for tokens, tags in all_sentences:
            text = " ".join(tokens)
            entities = []
            current_ent = None

            for token, tag in zip(tokens, tags):
                if tag.startswith("B-"):
                    if current_ent:
                        entities.append(current_ent)
                    label = tag[2:]
                    ioc_type = _LABEL_TO_IOC_TYPE.get(label, label.lower())
                    current_ent = {"text": token, "label": label, "ioc_type": ioc_type}
                elif tag.startswith("I-") and current_ent and current_ent["label"] == tag[2:]:
                    current_ent["text"] += " " + token
                else:
                    if current_ent:
                        entities.append(current_ent)
                        current_ent = None
            if current_ent:
                entities.append(current_ent)

            assistant_msg = json.dumps({"entities": entities, "events": []})
            chat = {"messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Text:\n{text}"},
                {"role": "assistant", "content": assistant_msg}
            ]}
            f.write(json.dumps(chat) + "\n")

    print(f"Saved NER tuning data to {out_file} ({len(all_sentences)} sentences total)")


def preprocess_summarization_govreport(max_rows=4000):
    """
    Reads both gov_reports train parquet shards and maps to incident summarization prompts.
    Uses 4000 char truncation (up from 2000) to preserve more document context, which
    directly improves factual grounding and reduces hallucination in generated summaries.
    Both train shards are sampled evenly to maximise vocabulary diversity.
    """
    print(f"Preprocessing Summarization Dataset (local gov_reports, target={max_rows} rows)...")
    out_file = PROCESSED_DIR / "summarizer_slm_train.jsonl"

    parquet_dir = DATA_DIR / "gov_reports" / "document"
    shards = [
        parquet_dir / "train-00000-of-00002.parquet",
        parquet_dir / "train-00001-of-00002.parquet",
    ]
    per_shard = max_rows // len(shards)

    frames = []
    for shard in shards:
        if not shard.exists():
            print(f"  Warning: {shard} not found, skipping.")
            continue
        try:
            df_shard = pd.read_parquet(shard).head(per_shard)
            frames.append(df_shard)
            print(f"  Loaded {len(df_shard)} rows from {shard.name}")
        except Exception as e:
            print(f"  Could not read {shard.name}: {e}")

    if not frames:
        print("Error: no summarization data loaded.")
        return

    df = pd.concat(frames, ignore_index=True)

    system_msg = ("You are a cybersecurity analyst. Your task is to summarize the following "
                  "security incident report.\n"
                  "Provide a detailed analyst summary in 8 to 15 sentences, covering all technical details.\n"
                  "Be completely factual. Do not hallucinate any details, IPs, or actors not present in the text.\n"
                  "Respond ONLY with the summary text. Do not include any JSON formatting or preamble.")

    with open(out_file, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            report_text = str(row.get("report", ""))[:4000]  # 4k chars preserves ~600 tokens of context
            summary_text = str(row.get("summary", ""))
            if not summary_text.strip():
                continue  # skip rows with empty ground-truth summaries

            chat = {"messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Incident Report:\n{report_text}"},
                {"role": "assistant", "content": summary_text}
            ]}
            f.write(json.dumps(chat) + "\n")

    print(f"Saved Summarization tuning data to {out_file} ({len(df)} rows from {len(frames)} shards)")


def main():
    parser = argparse.ArgumentParser(description="Preprocess local ARIES SLM datasets.")
    parser.add_argument("--all", action="store_true", help="Process all datasets")
    args = parser.parse_args()

    ensure_dir(PROCESSED_DIR)

    preprocess_triage_guide()
    preprocess_ner_cyner()
    preprocess_summarization_govreport()
    
    print("\nAll datasets processed successfully!")
    print(f"Check output directory: {PROCESSED_DIR}")

if __name__ == "__main__":
    main()
