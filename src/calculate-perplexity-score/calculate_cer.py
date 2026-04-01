#!/usr/bin/env python3
"""
Compute Character Error Rate (CER) for each image in a benchmark run.

Pre-processing pipeline (applied to both reference and hypothesis):
  1. Apply botok normalization to the hypothesis (ground truth is assumed to
     be already normalized).
  2. Remove all whitespace (spaces, newlines, …).
  3. Fold runs of multiple tshegs (U+0F0B) into a single tsheg.
  4. Remove placeholder characters (K, O, B, I, S) from both strings so that
     OCR systems incur no penalty for substituting or deleting them:
       K – Dakini script character (not encodable in Unicode)
       O – ornamental symbol
       B – broken / damaged paper
       I – illegible character
       S – struck-through character

Usage
-----
  # Single run:
  python scripts/compute_cer.py --run-id 2026-03-15T15-41-30

  # All runs in runs/catalog.csv that don't have results yet:
  python scripts/compute_cer.py

  # Re-compute everything:
  python scripts/compute_cer.py --force

Output
------
  analysis/cer/{benchmark_id}/{run_id}.csv  — columns: file_name, cer
"""

import argparse
import csv
import io
import re
import sys
from pathlib import Path

import boto3
import zstandard as zstd
from rapidfuzz.distance import Levenshtein

# botok normalization (same pipeline as normalize_transcriptions.py)
from botok.utils.corpus_normalization import normalize_spaces
from botok.utils.lenient_normalization import normalize_graphical
from botok.utils.unicode_normalization import normalize_unicode

# ── Configuration ──────────────────────────────────────────────────────────────

BASE          = Path(__file__).parent.parent
BENCHMARK     = BASE / "transcriptions" / "benchmark.csv"
CATALOG       = BASE / "runs" / "catalog.csv"
FULL_CATALOG  = BASE / "catalog" / "full_catalog.csv"
ANALYSIS      = BASE / "analysis" / "cer"

S3_BUCKET   = "bec.bdrc.io"
S3_PREFIX   = "evaluation_benchmark"

# Placeholder characters that represent unencodable or ornamental content.
# Neither substitution nor deletion of these is penalised: they are removed
# from both the reference and the hypothesis before CER is computed.
#   K – Dakini script character (not encodable in Unicode)
#   O – ornamental symbol
#   B – broken / damaged paper
#   I – illegible character
#   S – struck-through character
#
# Note: we remove placeholders from *both* sides rather than subtracting
# insertion penalties per instance. The alignment computed by Levenshtein has
# no memory of where placeholders were, so a fixed per-instance correction
# would be inaccurate and could produce negative CER on damaged images.
PLACEHOLDERS = frozenset("KOBIS")

_MULTI_TSHEG_RE = re.compile("\u0F0B{2,}")


# ── Text normalization ─────────────────────────────────────────────────────────

def normalize_hypothesis(text: str) -> str:
    """Full botok normalization for OCR output (identical to normalize_transcriptions.py)."""
    text = text.replace("\u00A0", " ")
    text = normalize_spaces(text, tibetan_specific=False)
    text = normalize_unicode(text)
    text = normalize_graphical(text)
    text = text.strip(" \n\r")
    return text


def prepare(text: str, is_hypothesis: bool = False) -> str:
    """
    Shared post-processing applied to both reference and hypothesis:
      1. Botok normalization (hypothesis only — ground truth is pre-normalized).
      2. Strip all whitespace.
      3. Fold consecutive tshegs into one.
      4. Remove placeholder characters (K, O, B, I).
    """
    if is_hypothesis:
        text = normalize_hypothesis(text)
    # Remove all whitespace
    text = "".join(text.split())
    # Fold multiple tshegs
    text = _MULTI_TSHEG_RE.sub("\u0F0B", text)
    # Remove placeholder characters
    text = "".join(c for c in text if c not in PLACEHOLDERS)
    return text


# ── CER ────────────────────────────────────────────────────────────────────────

def cer(hypothesis: str, reference: str) -> float:
    """
    Character Error Rate = edit_distance(hyp, ref) / len(ref).
    Returns 0.0 when both strings are empty, 1.0 when only reference is empty.
    """
    hyp = prepare(hypothesis, is_hypothesis=True)
    ref = reference   # already prepared at load time
    if not ref:
        return 0.0 if not hyp else 1.0
    dist = Levenshtein.distance(hyp, ref)
    return dist / len(ref)


# ── Ground truth loader ────────────────────────────────────────────────────────

def load_ground_truth() -> dict[str, str]:
    """
    Return {file_name: prepared_reference}.
    Ground truth is assumed already normalized by botok; we only apply the
    shared whitespace / tsheg / placeholder post-processing.
    """
    gt: dict[str, str] = {}
    with BENCHMARK.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            gt[row["file_name"]] = prepare(row["transcription"], is_hypothesis=False)
    return gt


# ── S3 helpers ─────────────────────────────────────────────────────────────────

def s3_key(benchmark_id: str, run_id: str) -> str:
    return f"{S3_PREFIX}/results/{benchmark_id}/{run_id}/results.csv.zst"


def download_results(benchmark_id: str, run_id: str) -> list[dict]:
    """Download and decompress results.csv.zst from S3, return list of rows."""
    key = s3_key(benchmark_id, run_id)
    s3  = boto3.client("s3")
    try:
        resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
    except s3.exceptions.NoSuchKey:
        raise FileNotFoundError(f"s3://{S3_BUCKET}/{key} not found")
    compressed = io.BytesIO(resp["Body"].read())
    dctx       = zstd.ZstdDecompressor()
    # Use stream_reader so content-size header is not required
    with dctx.stream_reader(compressed) as reader:
        text = io.TextIOWrapper(reader, encoding="utf-8").read()
    return list(csv.DictReader(io.StringIO(text)))


# ── Catalog loader ─────────────────────────────────────────────────────────────

def load_full_catalog() -> tuple[dict[str, dict], list[str]]:
    """
    Return ({file_name: catalog_row}, [ordered fieldnames]) from full_catalog.csv.
    The 'new_filename' key is used to index rows.
    """
    index: dict[str, dict] = {}
    fields: list[str] = []
    with FULL_CATALOG.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        for row in reader:
            fn = row.get("new_filename", "").strip()
            if fn:
                index[fn] = row
    return index, fields


# ── Per-run computation ────────────────────────────────────────────────────────

def compute_run(benchmark_id: str, run_id: str, gt: dict[str, str],
                catalog: dict[str, dict] | None = None,
                catalog_fields: list[str] | None = None) -> Path:
    """Download results, compute CER for each image, write output CSV."""
    print(f"  Downloading results for {run_id} …")
    rows = download_results(benchmark_id, run_id)

    out_dir = ANALYSIS / benchmark_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.csv"

    base_fields = ["file_name", "cer"]
    extra_fields = [f for f in (catalog_fields or []) if f != "new_filename"]
    fieldnames = base_fields + extra_fields

    missing_gt = 0
    records = []
    for row in rows:
        fn  = row.get("file_name", "")
        hyp = row.get("transcription", "")
        if fn not in gt:
            missing_gt += 1
            continue
        rec = {"file_name": fn, "cer": round(cer(hyp, gt[fn]), 6)}
        if catalog is not None:
            cat_row = catalog.get(fn, {})
            for f in extra_fields:
                rec[f] = cat_row.get(f, "")
        records.append(rec)

    if missing_gt:
        print(f"  WARNING: {missing_gt} image(s) have no ground-truth entry — skipped.")

    records.sort(key=lambda r: r["file_name"])
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    avg = sum(r["cer"] for r in records) / len(records) if records else float("nan")
    print(f"  {len(records)} images — mean CER: {avg:.4f} — written to {out_path}")
    return out_path


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-id", metavar="RUN_ID",
                        help="Compute CER for a single run")
    parser.add_argument("--benchmark-id", metavar="BENCHMARK_ID",
                        default="20260315",
                        help="Benchmark ID (default: %(default)s)")
    parser.add_argument("--force", action="store_true",
                        help="Re-compute even if output already exists")
    parser.add_argument("--catalog", action="store_true",
                        help="Merge catalog/full_catalog.csv columns into the "
                             "output CER CSV")
    args = parser.parse_args()

    print("Loading ground truth …")
    gt = load_ground_truth()
    print(f"  {len(gt)} ground-truth entries loaded.")

    catalog: dict[str, dict] | None = None
    catalog_fields: list[str] | None = None
    if args.catalog:
        catalog, catalog_fields = load_full_catalog()
        print(f"  {len(catalog)} catalog entries loaded ({len(catalog_fields)} columns).")
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
        bid    = run["benchmark_id"]
        run_id = run["run_id"]
        out    = ANALYSIS / bid / f"{run_id}.csv"

        if not args.force and out.exists():
            print(f"SKIP {run_id} (already computed — use --force to redo)")
            continue

        print(f"RUN  {run_id}")
        try:
            compute_run(bid, run_id, gt, catalog, catalog_fields)
        except FileNotFoundError as exc:
            print(f"  ERROR: {exc}")
        except Exception as exc:
            print(f"  ERROR: unexpected error — {exc}")
        print()


if __name__ == "__main__":
    main()
