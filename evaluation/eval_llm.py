#!/usr/bin/env python3
import csv
import sys
from typing import Dict
from pathlib import Path

WANTED_COLS = ["Argument ID", "Context ID", "judge_score"]

ALIASES = {
    "argument id": "Argument ID",
    "argument_id": "Argument ID",
    "context id": "Context ID",
    "context_id": "Context ID",
    "judge score": "judge_score",
    "judge_score": "judge_score",
}

# model name -> annotator_id
MODELS = {
    "deepseek-r1": 11,
    "gpt-4o": 12,
    "llama-3.1-70b": 13
}

# Output schema (final combined file)
NEW_COLS = ["argument_id", "context_version", "annotator_id", "convincingness_rating"]

def normalize(name: str) -> str:
    return name.strip().lower()

def build_field_map(fieldnames) -> Dict[str, str]:
    """Map canonical names to actual TSV field names."""
    norm_map = {normalize(f): f for f in fieldnames if f is not None}
    out = {}
    for alias_norm, canon in ALIASES.items():
        if alias_norm in norm_map and canon not in out:
            out[canon] = norm_map[alias_norm]
    missing = [c for c in WANTED_COLS if c not in out]
    if missing:
        raise SystemExit(
            f"Error: Missing required column(s) in TSV header: {', '.join(missing)}\n"
            f"Found columns: {', '.join(fieldnames)}"
        )
    return out

def extract_one_file(tsv_path: Path, annotator_id: int, writer: csv.writer):
    with tsv_path.open("r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile, delimiter="\t")
        if reader.fieldnames is None:
            raise SystemExit(f"Error: {tsv_path} appears to have no header row.")

        field_map = build_field_map(reader.fieldnames)

        for row in reader:
            out_row = [
                row.get(field_map["Argument ID"], "").strip(),   # argument_id
                row.get(field_map["Context ID"], "").strip(),     # context_version
                str(annotator_id),                                 # annotator_id
                row.get(field_map["judge_score"], "").strip(),    # convincingness_rating
            ]
            writer.writerow(out_row)

def main():
    base_dir = Path("data")
    combined_output = base_dir / "annotations_llm.csv"
    MODEL_FILES = {
        "deepseek-r1": base_dir / "Annotations_LLM_deepseek-r1.tsv",
        "gpt-4o": base_dir / "Annotations_LLM_gpt-4o.tsv",
        "llama-3.1-70b": base_dir / "Annotations_LLM_llama-3.1-70b.tsv",
    }

    # Open the output once and stream-append all rows
    combined_output.parent.mkdir(parents=True, exist_ok=True)
    with combined_output.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile, delimiter="\t")
        writer.writerow(NEW_COLS)  # header

        for model_name, annot_id in MODELS.items():
            # Resolve input path
            tsv_path = MODEL_FILES.get(
                model_name,
                base_dir / f"Annotations_LLM_{model_name}.tsv"
            )

            if not Path(tsv_path).exists():
                print(f"Warning: missing file for model '{model_name}': {tsv_path}", file=sys.stderr)
                continue

            extract_one_file(Path(tsv_path), annot_id, writer)
            print(f"Processed {model_name} -> annotator_id {annot_id} from {tsv_path}", file=sys.stderr)

    print(f"Combined annotations saved to: {combined_output}")

if __name__ == "__main__":
    main()
