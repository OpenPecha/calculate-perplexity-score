"""
build_report_data.py

Reads catalog + analysis CSVs, produces:
  - data/analysis/merged.csv   (flat joined table)
  - data/analysis/report_data.json  (summary / breakdown JSON)
"""

import csv
import json
import math
import os
import sys

csv.field_size_limit(sys.maxsize)

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
CATALOG_PATH      = os.path.join(BASE, "runs", "catalog.csv")
FULL_CATALOG_PATH = os.path.join(BASE, "full_catalog.csv")
CER_DIR           = os.path.join(BASE, "analysis", "cer")
PPL_DIR           = os.path.join(BASE, "analysis", "perplexity")
MERGED_PATH       = os.path.join(BASE, "analysis", "merged.csv")
REPORT_PATH       = os.path.join(BASE, "analysis", "report_data.json")
GT_PPL_PATH       = os.path.join(BASE, "analysis", "perplexity", "benchmark_transcriptions.csv")

# ── helpers ────────────────────────────────────────────────────────────────────

def read_csv_dicts(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def mean(vals):
    vs = [v for v in vals if v is not None]
    return sum(vs) / len(vs) if vs else None


def median(vals):
    vs = sorted(v for v in vals if v is not None)
    n = len(vs)
    if n == 0:
        return None
    mid = n // 2
    return vs[mid] if n % 2 else (vs[mid - 1] + vs[mid]) / 2


def percentile(vals, p):
    vs = sorted(v for v in vals if v is not None)
    n = len(vs)
    if n == 0:
        return None
    idx = (p / 100) * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    frac = idx - lo
    return vs[lo] * (1 - frac) + vs[hi] * frac


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    n = len(pairs)
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    num = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    dx  = math.sqrt(sum((p[0] - mx) ** 2 for p in pairs))
    dy  = math.sqrt(sum((p[1] - my) ** 2 for p in pairs))
    if dx == 0 or dy == 0:
        return None
    return round(num / (dx * dy), 4)


def log_ppl(v):
    if v is None:
        return None
    try:
        return math.log(float(v) + 1)
    except (ValueError, TypeError):
        return None


def parse_options_str(options_raw):
    """Return a short human-readable string from the JSON options column."""
    if not options_raw or options_raw.strip() == "":
        return ""
    try:
        opts = json.loads(options_raw)
    except (json.JSONDecodeError, TypeError):
        return options_raw

    parts = []
    if "temperature" in opts:
        parts.append(f"t={opts['temperature']}")
    if "prompt_type" in opts:
        parts.append(f"prompt={opts['prompt_type']}")
    if "use_layout_detection" in opts:
        parts.append(f"layout={int(bool(opts['use_layout_detection']))}")
    if "thinking_budget" in opts and opts["thinking_budget"] is not None:
        parts.append(f"think={opts['thinking_budget']}")
    if "language_hints" in opts:
        hints = opts["language_hints"]
        if hints:
            parts.append(f"hints={','.join(hints)}")
    return " · ".join(parts)


def engine_label(engine):
    labels = {
        "gemini":       "Gemini",
        "google_vision":"Google Vision",
        "transkribus":  "Transkribus",
        "qwen":         "Qwen",
        "dots":         "Dots OCR",
        "dots_mocr":    "Dots MOCR",
        "paddleocr":    "PaddleOCR",
    }
    return labels.get(engine, engine.title())


# ── 1. load catalog ────────────────────────────────────────────────────────────
catalog = read_csv_dicts(CATALOG_PATH)

# ── 2. load full_catalog → keyed by new_filename ──────────────────────────────
full_cat_rows = read_csv_dicts(FULL_CATALOG_PATH)
full_cat = {row["new_filename"]: row for row in full_cat_rows}

# ── 3. load CER + PPL per run, join, attach full_catalog ──────────────────────
MERGED_COLS = [
    "benchmark_id", "run_id", "engine", "model",
    "file_name", "cer", "perplexity_raw", "perplexity_normalized",
    "technology", "format", "legibility",
    "script_popularity on BDRC", "script_period",
    "script_name_phonetics", "script_3 types", "script_8 categories",
]

all_merged = []   # list of dicts
run_rows   = {}   # run_id → list of merged dicts (for per-run stats)

for run in catalog:
    bid    = run["benchmark_id"]
    rid    = run["run_id"]
    engine = run["engine"]
    model  = run["model"]

    cer_path = os.path.join(CER_DIR, bid, f"{rid}.csv")
    ppl_path = os.path.join(PPL_DIR, bid, f"{rid}.csv")

    if not os.path.exists(cer_path):
        print(f"[SKIP] Missing CER file: {cer_path}")
        continue
    if not os.path.exists(ppl_path):
        print(f"[SKIP] Missing PPL file: {ppl_path}")
        continue

    cer_data = {r["file_name"]: safe_float(r["cer"])
                for r in read_csv_dicts(cer_path)}
    ppl_data = {r["file_name"]: (safe_float(r["perplexity_raw"]),
                                  safe_float(r["perplexity_normalized"]))
                for r in read_csv_dicts(ppl_path)}

    common_files = set(cer_data) & set(ppl_data)

    rows_for_run = []
    for fname in sorted(common_files):
        fc = full_cat.get(fname, {})
        row = {
            "benchmark_id":              bid,
            "run_id":                    rid,
            "engine":                    engine,
            "model":                     model,
            "file_name":                 fname,
            "cer":                       cer_data[fname],
            "perplexity_raw":            ppl_data[fname][0],
            "perplexity_normalized":     ppl_data[fname][1],
            "technology":                fc.get("technology", ""),
            "format":                    fc.get("format", ""),
            "legibility":                fc.get("legibility", ""),
            "script_popularity on BDRC": fc.get("script_popularity on BDRC", ""),
            "script_period":             fc.get("script_period", ""),
            "script_name_phonetics":     fc.get("script_name_phonetics", ""),
            "script_3 types":            fc.get("script_3 types", ""),
            "script_8 categories":       fc.get("script_8 categories", ""),
        }
        rows_for_run.append(row)
        all_merged.append(row)

    run_rows[rid] = rows_for_run
    print(f"[OK]   {bid}/{rid}  ({engine})  n={len(rows_for_run)}")

# ── 4. write merged.csv ───────────────────────────────────────────────────────
os.makedirs(os.path.dirname(MERGED_PATH), exist_ok=True)
with open(MERGED_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=MERGED_COLS)
    writer.writeheader()
    for row in all_merged:
        writer.writerow({k: row.get(k, "") for k in MERGED_COLS})

print(f"\nWrote {len(all_merged)} rows → {MERGED_PATH}")

# ── 5. build runs summary ─────────────────────────────────────────────────────
runs_summary = []
for run in catalog:
    bid    = run["benchmark_id"]
    rid    = run["run_id"]
    engine = run["engine"]
    model  = run["model"]
    opts   = run.get("options", "")

    rows = run_rows.get(rid)
    if not rows:
        continue

    cers      = [r["cer"]                  for r in rows]
    ppls_raw  = [r["perplexity_raw"]       for r in rows]
    ppls_norm = [r["perplexity_normalized"] for r in rows]

    log_ppls_raw  = [log_ppl(v) for v in ppls_raw]
    log_ppls_norm = [log_ppl(v) for v in ppls_norm]

    options_str = parse_options_str(opts)
    label = f"{engine_label(engine)} / {model}"

    runs_summary.append({
        "bid":          bid,
        "rid":          rid,
        "engine":       engine,
        "model":        model,
        "model_display": model,
        "options_str":  options_str,
        "label":        label,
        "n_cer":        len([v for v in cers if v is not None]),
        "n_ppl":        len([v for v in ppls_raw if v is not None]),
        "cer_mean":     round(mean(cers),   4) if mean(cers) is not None else None,
        "cer_median":   round(median(cers), 4) if median(cers) is not None else None,
        "cer_p25":      round(percentile(cers, 25), 4) if percentile(cers, 25) is not None else None,
        "cer_p75":      round(percentile(cers, 75), 4) if percentile(cers, 75) is not None else None,
        "ppl_raw_mean":    round(mean(ppls_raw),    2) if mean(ppls_raw) is not None else None,
        "ppl_raw_median":  round(median(ppls_raw),  2) if median(ppls_raw) is not None else None,
        "ppl_norm_mean":   round(mean(ppls_norm),   2) if mean(ppls_norm) is not None else None,
        "ppl_norm_median": round(median(ppls_norm), 2) if median(ppls_norm) is not None else None,
        "corr_cer_log_ppl_raw":  pearson(cers, log_ppls_raw),
        "corr_cer_log_ppl_norm": pearson(cers, log_ppls_norm),
    })

# ── 6. build breakdowns ───────────────────────────────────────────────────────
DIMENSION_MAP = {
    "Script Type": "script_3 types",
    "Technology":  "technology",
    "Legibility":  "legibility",
    "Format":      "format",
}

breakdowns = {}
for run in catalog:
    rid  = run["run_id"]
    rows = run_rows.get(rid)
    if not rows:
        continue

    breakdowns[rid] = {}
    for dim_label, col in DIMENSION_MAP.items():
        groups = {}
        for r in rows:
            cat = (r.get(col) or "").strip()
            if not cat or cat.lower() == "unknown":
                continue
            if cat not in groups:
                groups[cat] = []
            groups[cat].append(r)

        dim_result = {}
        for cat, grp in sorted(groups.items()):
            cers      = [g["cer"]                  for g in grp]
            ppls_raw  = [g["perplexity_raw"]       for g in grp]
            ppls_norm = [g["perplexity_normalized"] for g in grp]
            dim_result[cat] = {
                "n":           len(grp),
                "cer_mean":    round(mean(cers),      4) if mean(cers) is not None else None,
                "cer_median":  round(median(cers),    4) if median(cers) is not None else None,
                "ppl_raw_mean":  round(mean(ppls_raw),  2) if mean(ppls_raw) is not None else None,
                "ppl_norm_mean": round(mean(ppls_norm), 2) if mean(ppls_norm) is not None else None,
            }
        breakdowns[rid][dim_label] = dim_result

# ── 7. image_sample: best google_vision run (lowest median CER) ───────────────
gv_runs = [(rs, run_rows[rs["rid"]])
           for rs in runs_summary
           if rs["engine"] == "google_vision" and rs["rid"] in run_rows]

image_sample = []
sample_run   = None

if gv_runs:
    best = min(gv_runs, key=lambda x: (x[0]["cer_median"] is None, x[0]["cer_median"] or 1e9))
    sample_run = best[0]["rid"]
    for r in best[1]:
        if r["cer"] is None or r["perplexity_raw"] is None:
            continue
        image_sample.append({
            "cer":        r["cer"],
            "ppl_raw":    r["perplexity_raw"],
            "ppl_norm":   r["perplexity_normalized"],
            "script":     (r.get("script_3 types") or "").strip(),
            "legibility": (r.get("legibility") or "").strip(),
            "technology": (r.get("technology") or "").strip(),
        })

# ── 8. overall correlation (across all merged rows) ───────────────────────────
all_cer      = [r["cer"]                  for r in all_merged if r["cer"] is not None]
all_ppl_raw  = [r["perplexity_raw"]       for r in all_merged if r["perplexity_raw"] is not None]
all_ppl_norm = [r["perplexity_normalized"] for r in all_merged if r["perplexity_normalized"] is not None]

# align by index (all_merged rows that have all three values)
valid = [(r["cer"], r["perplexity_raw"], r["perplexity_normalized"])
         for r in all_merged
         if r["cer"] is not None
         and r["perplexity_raw"] is not None
         and r["perplexity_normalized"] is not None]

overall_correlation = {
    "cer_vs_log_ppl_raw":  pearson([v[0] for v in valid], [log_ppl(v[1]) for v in valid]),
    "cer_vs_log_ppl_norm": pearson([v[0] for v in valid], [log_ppl(v[2]) for v in valid]),
}

# ── 9. ground truth PPL stats + engine_breakdowns ────────────────────────────
ground_truth_ppl = {"raw_median": None, "norm_median": None, "raw_mean": None, "norm_mean": None}
if os.path.exists(GT_PPL_PATH):
    gt_rows = read_csv_dicts(GT_PPL_PATH)
    gt_raw  = [safe_float(r["perplexity_raw"])        for r in gt_rows]
    gt_norm = [safe_float(r["perplexity_normalized"]) for r in gt_rows]
    ground_truth_ppl["raw_median"]  = round(median(gt_raw),  2) if median(gt_raw)  is not None else None
    ground_truth_ppl["norm_median"] = round(median(gt_norm), 2) if median(gt_norm) is not None else None
    ground_truth_ppl["raw_mean"]    = round(mean(gt_raw),    2) if mean(gt_raw)    is not None else None
    ground_truth_ppl["norm_mean"]   = round(mean(gt_norm),   2) if mean(gt_norm)   is not None else None
    print(f"Ground truth PPL — raw median: {ground_truth_ppl['raw_median']}, norm median: {ground_truth_ppl['norm_median']}")
    runs_summary.append({
        "bid":              "benchmark",
        "rid":              "ground_truth",
        "engine":           "ground_truth",
        "model":            "Ground Truth",
        "model_display":    "Ground Truth",
        "options_str":      "",
        "label":            "Ground Truth",
        "n_cer":            0,
        "n_ppl":            len([r for r in gt_rows if safe_float(r["perplexity_raw"]) is not None]),
        "cer_mean":         None,
        "cer_median":       None,
        "cer_p25":          None,
        "cer_p75":          None,
        "ppl_raw_mean":     ground_truth_ppl["raw_mean"],
        "ppl_raw_median":   ground_truth_ppl["raw_median"],
        "ppl_norm_mean":    ground_truth_ppl["norm_mean"],
        "ppl_norm_median":  ground_truth_ppl["norm_median"],
        "corr_cer_log_ppl_raw":  None,
        "corr_cer_log_ppl_norm": None,
    })
else:
    print(f"[WARN] Ground truth PPL file not found: {GT_PPL_PATH}")

# ── engine_breakdowns: best run per engine → its dimension breakdown ──────────
engine_breakdowns = {}
best_per_engine = {}
for rs in runs_summary:
    eng = rs["engine"]
    if eng == "ground_truth" or rs.get("cer_median") is None:
        continue
    if eng not in best_per_engine or rs["cer_median"] < best_per_engine[eng]["cer_median"]:
        best_per_engine[eng] = rs

for eng, rs in best_per_engine.items():
    rid = rs["rid"]
    if rid in breakdowns:
        engine_breakdowns[eng] = breakdowns[rid]

# Ground-truth dimension breakdown (join GT PPL with full_catalog)
if os.path.exists(GT_PPL_PATH):
    gt_ppl_map = {r["file_name"]: (safe_float(r["perplexity_raw"]), safe_float(r["perplexity_normalized"]))
                  for r in read_csv_dicts(GT_PPL_PATH)}
    gt_dim_data = {}
    for dim_label, col in DIMENSION_MAP.items():
        groups = {}
        for row in full_cat_rows:
            fname = row.get("new_filename", "")
            cat   = (row.get(col) or "").strip()
            if not cat or cat.lower() == "unknown" or fname not in gt_ppl_map:
                continue
            ppl_raw, ppl_norm = gt_ppl_map[fname]
            if ppl_raw is None:
                continue
            if cat not in groups:
                groups[cat] = {"raw": [], "norm": []}
            groups[cat]["raw"].append(ppl_raw)
            if ppl_norm is not None:
                groups[cat]["norm"].append(ppl_norm)
        gt_dim_data[dim_label] = {
            cat: {
                "n":            len(v["raw"]),
                "ppl_raw_mean":  round(mean(v["raw"]),  2) if v["raw"]  else None,
                "ppl_norm_mean": round(mean(v["norm"]), 2) if v["norm"] else None,
            }
            for cat, v in sorted(groups.items())
        }
    engine_breakdowns["ground_truth"] = gt_dim_data

# ── 10. assemble report_data.json ─────────────────────────────────────────────
report = {
    "runs":        runs_summary,
    "breakdowns":  breakdowns,
    "image_sample": image_sample,
    "sample_run":  sample_run,
    "overall_correlation": overall_correlation,
    "ground_truth_ppl": ground_truth_ppl,
    "engine_breakdowns": engine_breakdowns,
    "engine_colors": {
        "gemini":        "#D4793A",
        "google_vision": "#3B6BCA",
        "transkribus":   "#5AA867",
        "qwen":          "#7B5EA7",
        "dots":          "#D4A843",
        "dots_mocr":     "#C94A4A",
        "paddleocr":     "#E8734A",
        "ground_truth":  "#4ade80",
    },
    "engine_labels": {
        "gemini":        "Gemini",
        "google_vision": "Google Vision",
        "transkribus":   "Transkribus",
        "qwen":          "Qwen",
        "dots":          "Dots OCR",
        "dots_mocr":     "Dots MOCR",
        "paddleocr":     "PaddleOCR",
        "ground_truth":  "Ground Truth",
    },
    "dimensions": ["Script Type", "Technology", "Legibility", "Format"],
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"Wrote report_data.json → {REPORT_PATH}")

# Inject DATA directly into docs/index.html so it stays self-contained
HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "index.html")
if os.path.exists(HTML_PATH):
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()
    marker = "const DATA = "
    start  = html.index(marker)
    end    = html.index(";\n", start) + 2
    html   = html[:start] + "const DATA = " + json.dumps(report, ensure_ascii=False) + ";\n" + html[end:]
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Injected DATA → {HTML_PATH}")
