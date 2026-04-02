#!/usr/bin/env python3
"""
Compute perplexity scores for OCR output using BoKenlm-syl language model.

Perplexity measures how well the language model predicts text:
  - Lower perplexity: text closely resembles clean Tibetan (better OCR)
  - Higher perplexity: text contains noise/errors (worse OCR)

Two perplexity scores are computed for each image:
  1. perplexity_raw: minimal processing (strip newlines) + tokenize + compute
  2. perplexity_normalized: full CER normalization pipeline + tokenize + compute

The normalization pipeline includes:
  - Botok Unicode normalization
  - Remove all whitespace
  - Fold consecutive tshegs into one
  - Remove placeholder characters (K, O, B, I, S)

Usage
-----
  # Single run:
  python scripts/compute_perplexity.py --run-id 2026-03-15T15-41-30

  # All runs in runs/catalog.csv that don't have results yet:
  python scripts/compute_perplexity.py

  # Re-compute everything:
  python scripts/compute_perplexity.py --force

Output
------
  analysis/perplexity/{benchmark_id}/{run_id}.csv — columns: file_name, perplexity_raw, perplexity_normalized
"""

import argparse
import csv
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

import kenlm
from botok import WordTokenizer
from huggingface_hub import hf_hub_download

# Import the shared normalization function from calculate_cer
from calculate_cer import prepare, load_ground_truth

# ── Configuration ──────────────────────────────────────────────────────────────

BASE          = Path(__file__).parent.parent.parent / "data"
CATALOG       = BASE / "runs" / "catalog.csv"
INFERENCE     = BASE / "inference"
ANALYSIS      = BASE / "analysis" / "perplexity"

# HuggingFace model details
HF_REPO_ID    = "openpecha/BoKenlm-syl-v0.1"
HF_MODEL_FILE = "BoKenlm-syl-v0.1.arpa"


# ── Model loading ──────────────────────────────────────────────────────────────

def load_kenlm_model() -> kenlm.Model:
    """
    Download BoKenlm-syl.arpa from HuggingFace (cached) and load with kenlm.
    """
    print("Loading KenLM model from HuggingFace …")
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILE)
    model = kenlm.Model(model_path)
    print(f"  Model loaded: {model_path}")
    return model


# ── Tokenization ───────────────────────────────────────────────────────────────

def load_tokenizer() -> WordTokenizer:
    """
    Load botok syllable tokenizer for Tibetan.
    """
    tok = WordTokenizer()
    return tok


def tokenize_syllables(text: str, tokenizer: WordTokenizer) -> str:
    """
    Tokenize text into syllables using botok's word tokenizer.
    Returns space-joined syllable tokens suitable for kenlm.
    """
    if not text or not text.strip():
        return ""

    words = tokenizer.tokenize(text)
    # Each word is already syllable-level; join with spaces for kenlm
    return " ".join(w.text for w in words)


# ── Perplexity computation ─────────────────────────────────────────────────────

def compute_perplexity(tokenized_text: str, model: kenlm.Model) -> float | None:
    """
    Compute perplexity for tokenized text.
    Returns None if text is empty or invalid.
    """
    if not tokenized_text or not tokenized_text.strip():
        return None

    try:
        ppl = model.perplexity(tokenized_text)
        return round(ppl, 4) if ppl is not None else None
    except Exception:
        # Handle any tokenization errors from kenlm gracefully
        return None


# ── Text processing ───────────────────────────────────────────────────────────

def prepare_raw(text: str) -> str:
    """
    Minimal preprocessing for raw perplexity:
    Just strip newlines and leading/trailing whitespace.
    """
    return text.strip("\n\r").strip()


# ── Per-run computation ────────────────────────────────────────────────────────

def compute_run(
    benchmark_id: str,
    run_id: str,
    model: kenlm.Model,
    tokenizer: WordTokenizer,
) -> Path:
    """
    Load inference CSV, compute both perplexity scores for each file_name,
    write output CSV.
    """
    # Construct input and output paths
    input_dir = INFERENCE / benchmark_id
    input_path = input_dir / f"{run_id}.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Inference CSV not found: {input_path}")

    print(f"  Reading inference results from {input_path.name} …")
    with input_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    # Create output directory and file
    out_dir = ANALYSIS / benchmark_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.csv"

    # Compute perplexity for each row
    records = []
    for row in rows:
        fn = row.get("file_name", "")
        transcription = row.get("transcription", "")

        # Skip rows with missing filename
        if not fn:
            continue

        try:
            # Raw perplexity (minimal preprocessing)
            raw_text = prepare_raw(transcription)
            raw_tokens = tokenize_syllables(raw_text, tokenizer)
            ppl_raw = compute_perplexity(raw_tokens, model)

            # Normalized perplexity (full CER normalization pipeline)
            norm_text = prepare(transcription, is_hypothesis=True)
            norm_tokens = tokenize_syllables(norm_text, tokenizer)
            ppl_norm = compute_perplexity(norm_tokens, model)
        except Exception:
            ppl_raw = None
            ppl_norm = None

        rec = {
            "file_name": fn,
            "perplexity_raw": ppl_raw if ppl_raw is not None else "",
            "perplexity_normalized": ppl_norm if ppl_norm is not None else "",
        }
        records.append(rec)

    # Sort by file_name and write output
    records.sort(key=lambda r: r["file_name"])
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file_name", "perplexity_raw", "perplexity_normalized"])
        writer.writeheader()
        writer.writerows(records)

    # Compute summary statistics (skip empty cells)
    raw_vals = [float(r["perplexity_raw"]) for r in records if r["perplexity_raw"]]
    norm_vals = [float(r["perplexity_normalized"]) for r in records if r["perplexity_normalized"]]

    mean_raw = sum(raw_vals) / len(raw_vals) if raw_vals else float("nan")
    mean_norm = sum(norm_vals) / len(norm_vals) if norm_vals else float("nan")

    print(f"  {len(records)} images — mean PPL raw: {mean_raw:.4f}, normalized: {mean_norm:.4f} — written to {out_path}")
    return out_path


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-id", metavar="RUN_ID",
                        help="Compute perplexity for a single run")
    parser.add_argument("--benchmark-id", metavar="BENCHMARK_ID",
                        default="20260315",
                        help="Benchmark ID (default: %(default)s)")
    parser.add_argument("--force", action="store_true",
                        help="Re-compute even if output already exists")
    args = parser.parse_args()

    print("Loading KenLM model …")
    model = load_kenlm_model()

    print("Loading tokenizer …")
    tokenizer = load_tokenizer()
    print()

    if args.run_id:
        runs = [{"benchmark_id": args.benchmark_id, "run_id": args.run_id}]
    else:
        if not CATALOG.exists():
            sys.exit(f"ERROR: {CATALOG} not found. Pass --run-id or create runs/catalog.csv.")
        with CATALOG.open(encoding="utf-8") as fh:
            runs = list(csv.DictReader(fh))
        print(f"Found {len(runs)} run(s) in catalog.\n")

    for run in runs:
        bid = run["benchmark_id"]
        run_id = run["run_id"]
        out = ANALYSIS / bid / f"{run_id}.csv"

        if not args.force and out.exists():
            print(f"SKIP {run_id} (already computed — use --force to redo)")
            continue

        print(f"RUN  {run_id}")
        try:
            compute_run(bid, run_id, model, tokenizer)
        except FileNotFoundError as exc:
            print(f"  ERROR: {exc}")
        except Exception as exc:
            print(f"  ERROR: unexpected error — {exc}")
        print()


if __name__ == "__main__":
    main()
