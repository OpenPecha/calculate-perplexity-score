#!/usr/bin/env python3
"""
Generate the Tibetan OCR Perplexity Dashboard as a single HTML file.

Reads perplexity results from analysis/perplexity/ and CER results from
analysis/cer/, joins with catalog metadata, computes summary statistics
and per-dimension breakdowns, then injects the data into an HTML template
to produce a self-contained dashboard.

Usage:
    python scripts/generate_perplexity_dashboard.py
"""

import csv
import json
import math
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent / "data"

RUNS_CATALOG  = BASE / "runs" / "catalog.csv"
FULL_CATALOG  = BASE / "full_catalog.csv"
PPL_DIR       = BASE / "analysis" / "perplexity"
CER_DIR       = BASE / "analysis" / "cer"
TEMPLATE      = Path(__file__).resolve().parent / "template_perplexity_dashboard.html"
OUTPUT        = BASE / "analysis" / "perplexity_dashboard" / "index.html"

# 2026-03-15T15-41-30: gemini-2.5-flash run with thinking_budget=null (incomplete/broken)
# 2026-03-22T11-38-23: reserved exclusion slot for a future known-bad run
EXCLUDED_RUNS = frozenset({"2026-03-22T11-38-23", "2026-03-15T15-41-30"})

# Dimension labels must match the data-dim values used in the HTML tabs.
DIMENSIONS = [
    ("Script Type",    "script_3 types"),
    ("Script Category","script_8 categories"),
    ("Technology",     "technology"),
    ("Legibility",     "legibility"),
    ("Format",         "format"),
    ("Period",         "script_period"),
    ("Popularity",     "script_popularity on BDRC"),
]

ENGINE_COLORS = {
    "gemini":        "#D4793A",
    "google_vision": "#3B6BCA",
    "transkribus":   "#5AA867",
    "qwen":          "#7B5EA7",
    "dots":          "#D4A843",
    "dots_mocr":     "#C94A4A",
    "paddleocr":     "#E8734A",
    "ground_truth":  "#4ade80",
}

ENGINE_LABELS = {
    "gemini":        "Gemini",
    "google_vision": "Google Vision",
    "transkribus":   "Transkribus",
    "qwen":          "Qwen",
    "dots":          "Dots OCR",
    "dots_mocr":     "Dots MOCR",
    "paddleocr":     "PaddleOCR",
    "ground_truth":  "Ground Truth",
}

ENGINE_DISPLAY = {
    "gemini":        "Gemini",
    "google_vision": "Google Vision",
    "transkribus":   "Transkribus",
    "qwen":          "Qwen",
    "dots":          "Dots",
    "dots_mocr":     "Dots MOCR",
    "paddleocr":     "PaddleOCR",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_options(raw: str) -> dict:
    s = (raw or "").strip()
    if not s:
        return {}
    start, end = s.find("{"), s.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {}


def make_model_display(engine: str, model: str) -> str:
    m = model.split("/")[-1] if "/" in model else model
    if any(part in m.lower() for part in engine.lower().split("_")):
        return m
    return f"{ENGINE_DISPLAY.get(engine, engine)} / {m}"


def make_options_str(engine: str, opts: dict) -> str:
    parts: list[str] = []
    if engine == "gemini":
        parts.append(f"t={opts.get('temperature', '?')}")
        tb = opts.get("thinking_budget")
        if tb is not None:
            parts.append(f"think={tb}")
        p = opts.get("prompt")
        if p:
            parts.append(p)
    elif engine == "google_vision":
        hints = opts.get("language_hints", [])
        parts.append(f"hints={hints}")
    elif engine == "transkribus":
        lid = opts.get("layout_model_id")
        parts.append(f"layout={lid}" if lid else "no-layout")
    elif engine == "qwen":
        parts.append(f"t={opts.get('temperature', '?')}")
        rp = opts.get("repetition_penalty")
        if rp:
            parts.append(f"rep={rp}")
        if opts.get("sync"):
            parts.append("sync")
    elif engine in ("dots", "dots_mocr"):
        pass  # no configurable options exposed in catalog for these engines
    elif engine == "paddleocr":
        parts.append(opts.get("prompt_type", "?"))
        parts.append(f"t={opts.get('temperature', '?')}")
        parts.append("layout=on" if opts.get("use_layout_detection") else "layout=off")
    return " · ".join(parts)


def pearson_r(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation coefficient between two equal-length lists."""
    n = len(xs)
    if n < 3:
        return None
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx  = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy  = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return round(num / (dx * dy), 4)


def safe_round(v, n=4):
    return round(v, n) if v is not None else None


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_runs() -> list[dict]:
    rows = []
    with RUNS_CATALOG.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["run_id"] not in EXCLUDED_RUNS:
                rows.append(r)
    return rows


def load_catalog() -> dict[str, dict]:
    catalog: dict[str, dict] = {}
    with FULL_CATALOG.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            fn = r.get("new_filename", "").strip()
            if fn:
                catalog[fn] = r
    return catalog


def load_ppl(benchmark_id: str, run_id: str) -> dict[str, dict] | None:
    """Load perplexity CSV → {filename: {"raw": float|None, "norm": float|None}}."""
    path = PPL_DIR / benchmark_id / f"{run_id}.csv"
    if not path.exists():
        return None
    result: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            fn = row.get("file_name", "").strip()
            if not fn:
                continue
            raw_s  = row.get("perplexity_raw", "").strip()
            norm_s = row.get("perplexity_normalized", "").strip()
            result[fn] = {
                "raw":  float(raw_s)  if raw_s  else None,
                "norm": float(norm_s) if norm_s else None,
            }
    return result


def load_cer(benchmark_id: str, run_id: str) -> dict[str, float] | None:
    """Load CER CSV → {filename: cer_value (capped at 1.0)}."""
    path = CER_DIR / benchmark_id / f"{run_id}.csv"
    if not path.exists():
        return None
    cer: dict[str, float] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            fn = row.get("file_name", "").strip()
            try:
                cer[fn] = min(float(row["cer"]), 1.0)
            except (ValueError, KeyError):
                continue
    return cer


# ── Per-run statistics ─────────────────────────────────────────────────────────

def compute_run_stats(
    cer:    dict[str, float],
    ppl:    dict[str, dict] | None,
    catalog: dict[str, dict],
) -> dict:
    """Compute per-run summary stats and per-dimension breakdowns."""

    # ── Summary scalars ──────────────────────────────────────────────────────
    cer_vals = list(cer.values())
    n_cer = len(cer_vals)

    if n_cer == 0:
        return {}

    ppl_raw_vals, ppl_norm_vals = [], []
    if ppl:
        for fn in cer:
            pt = ppl.get(fn)
            if pt and pt["raw"] is not None:
                ppl_raw_vals.append(pt["raw"])
            if pt and pt["norm"] is not None:
                ppl_norm_vals.append(pt["norm"])

    def _median(vals):
        return safe_round(statistics.median(vals), 2) if vals else None

    def _mean(vals):
        return safe_round(statistics.mean(vals), 2) if vals else None

    def _pct(vals, q):
        if len(vals) >= 4:
            return safe_round(statistics.quantiles(sorted(vals), n=4)[q], 4)
        if vals:
            return safe_round(sorted(vals)[0 if q == 0 else -1], 4)
        return None

    cer_s = sorted(cer_vals)

    # Pearson r between CER and log(PPL) for images that have both
    corr_raw = corr_norm = None
    if ppl:
        paired_raw  = [(cer[fn], math.log(ppl[fn]["raw"]))
                       for fn in cer if fn in ppl and ppl[fn]["raw"] and ppl[fn]["raw"] > 0]
        paired_norm = [(cer[fn], math.log(ppl[fn]["norm"]))
                       for fn in cer if fn in ppl and ppl[fn]["norm"] and ppl[fn]["norm"] > 0]
        if paired_raw:
            corr_raw  = pearson_r([p[0] for p in paired_raw],  [p[1] for p in paired_raw])
        if paired_norm:
            corr_norm = pearson_r([p[0] for p in paired_norm], [p[1] for p in paired_norm])

    summary = {
        "n_cer":            n_cer,
        "n_ppl":            len(ppl_raw_vals),
        "cer_mean":         safe_round(statistics.mean(cer_s), 4),
        "cer_median":       safe_round(statistics.median(cer_s), 4),
        "cer_p25":          _pct(cer_s, 0),
        "cer_p75":          _pct(cer_s, 2),
        "ppl_raw_mean":     _mean(ppl_raw_vals),
        "ppl_raw_median":   _median(ppl_raw_vals),
        "ppl_norm_mean":    _mean(ppl_norm_vals),
        "ppl_norm_median":  _median(ppl_norm_vals),
        "corr_cer_log_ppl_raw":  corr_raw,
        "corr_cer_log_ppl_norm": corr_norm,
    }

    # ── Dimension breakdowns ─────────────────────────────────────────────────
    dim_breakdowns: dict[str, dict] = {}
    for dim_label, col_name in DIMENSIONS:
        cats: dict[str, dict] = {}
        for fn, c in cer.items():
            meta = catalog.get(fn, {})
            cat  = (meta.get(col_name) or "").strip() or "Unknown"
            if cat not in cats:
                cats[cat] = {"cer": [], "ppl_raw": [], "ppl_norm": []}
            cats[cat]["cer"].append(c)
            if ppl and fn in ppl:
                pt = ppl[fn]
                if pt["raw"]  is not None:
                    cats[cat]["ppl_raw"].append(pt["raw"])
                if pt["norm"] is not None:
                    cats[cat]["ppl_norm"].append(pt["norm"])

        dim_breakdowns[dim_label] = {
            cat: {
                "n":            len(d["cer"]),
                "cer_mean":     safe_round(statistics.mean(d["cer"]),    4),
                "cer_median":   safe_round(statistics.median(d["cer"]),  4),
                "ppl_raw_mean":  safe_round(statistics.mean(d["ppl_raw"]),  2) if d["ppl_raw"]  else None,
                "ppl_norm_mean": safe_round(statistics.mean(d["ppl_norm"]), 2) if d["ppl_norm"] else None,
            }
            for cat, d in sorted(cats.items())
        }

    return {**summary, "breakdowns": dim_breakdowns}


# ── Image sample ───────────────────────────────────────────────────────────────

def collect_image_sample(
    cer:     dict[str, float],
    ppl:     dict[str, dict] | None,
    catalog: dict[str, dict],
) -> list[dict]:
    """Return per-image data points for the scatter/heatmap chart."""
    sample = []
    for fn, c in cer.items():
        meta = catalog.get(fn, {})
        pt   = (ppl or {}).get(fn, {})
        row: dict = {
            "cer":      round(c, 4),
            "ppl_raw":  safe_round(pt.get("raw"),  2) if pt else None,
            "ppl_norm": safe_round(pt.get("norm"), 2) if pt else None,
            "script":    (meta.get("script_3 types") or "").strip() or "Unknown",
            "legibility":(meta.get("legibility")     or "").strip() or "Unknown",
            "technology":(meta.get("technology")     or "").strip() or "Unknown",
        }
        sample.append(row)
    return sample


# ── Main data processing ───────────────────────────────────────────────────────

def process() -> dict:
    runs    = load_runs()
    catalog = load_catalog()

    ume_images: set[str] = {
        fn for fn, meta in catalog.items()
        if "Ume" in (meta.get("script_3 types") or "")
    }

    # First pass: load all CER data and build image unions per benchmark
    cer_cache: dict[str, dict[str, float]] = {}
    ppl_cache: dict[str, dict[str, dict]] = {}
    benchmark_images: dict[str, set[str]] = {}
    catalog_fns = set(catalog.keys())

    for run in runs:
        bid, rid = run["benchmark_id"], run["run_id"]
        cer = load_cer(bid, rid)
        if cer is None:
            continue
        cer = {fn: v for fn, v in cer.items() if fn in catalog_fns}
        cer_cache[rid] = cer
        benchmark_images.setdefault(bid, set()).update(cer.keys())

        ppl = load_ppl(bid, rid)
        if ppl is not None:
            ppl_cache[rid] = {fn: v for fn, v in ppl.items() if fn in catalog_fns}

    # Inject CER=1 for missing images (hallucination penalty).
    # Transkribus only ran on Ume images, so it is only penalised for
    # missing Ume images — not the full benchmark set.
    for run in runs:
        rid = run["run_id"]
        if rid not in cer_cache:
            continue
        bid = run["benchmark_id"]
        cer = cer_cache[rid]

        if run["engine"] == "transkribus":
            expected = benchmark_images[bid] & ume_images
        else:
            expected = benchmark_images[bid]

        for fn in expected:
            if fn not in cer:
                cer[fn] = 1.0

    # Second pass: compute per-run stats
    all_runs    = []
    breakdowns  = {}

    # Track best run per engine (by CER median, lower is better)
    best_per_engine: dict[str, tuple[float, str]] = {}  # engine → (cer_median, rid)

    for run in runs:
        rid = run["run_id"]
        if rid not in cer_cache:
            continue
        cer  = cer_cache[rid]
        ppl  = ppl_cache.get(rid)
        bid  = run["benchmark_id"]
        opts = parse_options(run.get("options", ""))
        model_display = make_model_display(run["engine"], run["model"])
        options_str   = make_options_str(run["engine"], opts)
        label = f"{model_display} · {options_str}" if options_str else model_display

        stats = compute_run_stats(cer, ppl, catalog)
        if not stats:
            continue

        all_runs.append({
            "bid":           bid,
            "rid":           rid,
            "engine":        run["engine"],
            "model":         run["model"],
            "model_display": model_display,
            "options_str":   options_str,
            "label":         label,
            **{k: v for k, v in stats.items() if k != "breakdowns"},
        })
        breakdowns[rid] = stats["breakdowns"]

        cer_med = stats.get("cer_median")
        engine  = run["engine"]
        if cer_med is not None:
            prev = best_per_engine.get(engine)
            if prev is None or cer_med < prev[0]:
                best_per_engine[engine] = (cer_med, rid)

    # Per-engine breakdowns (using the best run for each engine)
    engine_breakdowns: dict[str, dict] = {}
    for engine, (_, best_rid) in best_per_engine.items():
        if best_rid in breakdowns:
            engine_breakdowns[engine] = breakdowns[best_rid]

    # Overall Pearson r pooled across all runs and all images
    all_cer_raw, all_log_ppl_raw   = [], []
    all_cer_norm, all_log_ppl_norm = [], []
    for run in runs:
        rid = run["run_id"]
        cer = cer_cache.get(rid)
        ppl = ppl_cache.get(rid)
        if cer is None or ppl is None:
            continue
        for fn, c in cer.items():
            pt = ppl.get(fn)
            if pt and pt["raw"] and pt["raw"] > 0:
                all_cer_raw.append(c)
                all_log_ppl_raw.append(math.log(pt["raw"]))
            if pt and pt["norm"] and pt["norm"] > 0:
                all_cer_norm.append(c)
                all_log_ppl_norm.append(math.log(pt["norm"]))

    overall_corr = {
        "cer_vs_log_ppl_raw":  pearson_r(all_cer_raw,  all_log_ppl_raw),
        "cer_vs_log_ppl_norm": pearson_r(all_cer_norm, all_log_ppl_norm),
    }

    # Sample run: best CER median among runs that also have PPL data (for the image-level scatter)
    sample_run = None
    sample_imgs: list[dict] = []
    if all_runs:
        candidates = [r for r in all_runs if r.get("cer_median") is not None and r.get("n_ppl", 0) > 0]
        best = min(candidates, key=lambda r: r["cer_median"], default=None)
        if best:
            sample_run = best["rid"]
            sample_cer = cer_cache.get(sample_run, {})
            sample_ppl = ppl_cache.get(sample_run)
            sample_imgs = collect_image_sample(sample_cer, sample_ppl, catalog)

    return {
        "runs":              all_runs,
        "breakdowns":        breakdowns,
        "engine_breakdowns": engine_breakdowns,
        "overall_correlation": overall_corr,
        "sample_run":        sample_run,
        "image_sample":      sample_imgs,
        "engine_colors":     ENGINE_COLORS,
        "engine_labels":     ENGINE_LABELS,
        "dimensions":        [d[0] for d in DIMENSIONS],
    }


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print("Processing data …")
    data = process()
    print(f"  {len(data['runs'])} runs processed")
    print(f"  {len(data['image_sample'])} images in sample run")

    print("Reading template …")
    tpl = TEMPLATE.read_text(encoding="utf-8")
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    html = tpl.replace("__DASHBOARD_DATA__", data_json)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html, encoding="utf-8")
    size_kb = len(html.encode("utf-8")) / 1024
    print(f"Dashboard written to {OUTPUT}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
