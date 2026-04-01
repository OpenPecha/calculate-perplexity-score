#!/usr/bin/env python3
"""
Download and decompress results.csv.zst files from S3 for each benchmark run.

Usage
-----
  # All runs in runs/catalog.csv:
  python download_results.py

  # Single run:
  python download_results.py --run-id 2026-03-15T15-41-30

Output
------
  analysis/downloads/{benchmark_id}/{run_id}.csv  — uncompressed results.csv
"""

import argparse
import csv
import io
import sys
from pathlib import Path

import boto3
import zstandard as zstd

# ── Configuration ──────────────────────────────────────────────────────────────

BASE    = Path(__file__).parent.parent
CATALOG = BASE / "runs" / "catalog.csv"
DOWNLOADS = BASE / "analysis" / "downloads"

S3_BUCKET = "bec.bdrc.io"
S3_PREFIX = "evaluation_benchmark"


# ── S3 helpers ─────────────────────────────────────────────────────────────────

def s3_key(benchmark_id: str, run_id: str) -> str:
    """Construct S3 key for results.csv.zst"""
    return f"{S3_PREFIX}/results/{benchmark_id}/{run_id}/results.csv.zst"


def download_and_save_results(benchmark_id: str, run_id: str) -> Path:
    """Download results.csv.zst from S3, decompress, and save as CSV."""
    print(f"  Downloading and decompressing {run_id} …")

    key = s3_key(benchmark_id, run_id)
    s3 = boto3.client("s3")

    try:
        resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
    except s3.exceptions.NoSuchKey:
        raise FileNotFoundError(f"s3://{S3_BUCKET}/{key} not found")

    # Download and decompress
    compressed = io.BytesIO(resp["Body"].read())
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(compressed) as reader:
        text = io.TextIOWrapper(reader, encoding="utf-8").read()

    # Save to file
    out_dir = DOWNLOADS / benchmark_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.csv"

    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(text)

    # Count rows for feedback
    rows = list(csv.DictReader(io.StringIO(text)))
    print(f"  Saved {len(rows)} records to {out_path}")

    return out_path


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-id", metavar="RUN_ID",
                        help="Download results for a single run")
    parser.add_argument("--benchmark-id", metavar="BENCHMARK_ID",
                        default="20260315",
                        help="Benchmark ID (default: %(default)s)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if output already exists")
    args = parser.parse_args()

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
        out = DOWNLOADS / bid / f"{run_id}.csv"

        if not args.force and out.exists():
            print(f"SKIP {run_id} (already downloaded — use --force to redo)")
            continue

        print(f"RUN  {run_id}")
        try:
            download_and_save_results(bid, run_id)
        except FileNotFoundError as exc:
            print(f"  ERROR: {exc}")
        except Exception as exc:
            print(f"  ERROR: unexpected error — {exc}")
        print()


if __name__ == "__main__":
    main()
