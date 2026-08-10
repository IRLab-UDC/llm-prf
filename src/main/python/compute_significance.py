#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from test_evaluation import (
    load_dataset_config,
    prf_cname,
    RF_MODELS,
    NARRATIVE_SUPPORT,
    requires_narrative,
    two_call_span_candidates,
    one_call_span_candidates,
    unfiltered_span_candidates,
    BASELINE,
)

TF_SUFFIX_TO_CLI = {
    "none": "none",
    "VLLMSpans2tuned": "vllmspans2",
    "VLLMSpans2-NoNarrtuned": "vllmspans2-nonarr",
    "VLLMJudgeSpanstuned": "vllmjudgespans",
}

DATASETS = [
    ("ap8889", "AP8889"),
    ("robust04", "R04"),
    ("wt10g", "WT10G"),
    ("dl19", "DL-20"),
]

ROWS = [
    "BM25",
    "MonoT5 rerank",
    "Blind PRF",
    "Document filtering",
    "Filtering + probability weight",
    "LLM filtering + narrative",
    "Unfiltered SpanSelection",
    "Filtered SpanSelection (1 call)",
    "Filtered SpanSelection (2 calls)",
]

ROW_LETTERS = {row: chr(ord("a") + i) for i, row in enumerate(ROWS)}


def row_candidates(row, baseline=BASELINE):
    if row == "BM25":
        return [f"BASELINE_{baseline}"]
    if row == "MonoT5 rerank":
        return [f"MONOT5_RERANK_{baseline}"]
    if row == "Blind PRF":
        return [prf_cname("PRF", rf, baseline=baseline) for rf in RF_MODELS]
    if row == "Document filtering":
        return [
            prf_cname(s, rf, baseline=baseline)
            for rf in RF_MODELS
            for s in ("MONOT5", "VLLM")
        ]
    if row == "Filtering + probability weight":
        return [prf_cname(s, "RM3", baseline=baseline) for s in ("MONOT5-PROB", "VLLM-PROB")]
    if row == "LLM filtering + narrative":
        cands = [
            prf_cname("VLLM-NARR", "RM3", baseline=baseline),
            prf_cname("VLLM-NARR-PROB", "RM3", baseline=baseline),
        ]
        cands += [prf_cname("VLLM-NARR", rf, baseline=baseline) for rf in ("DMM", "MEDMM")]
        return cands
    if row == "Unfiltered SpanSelection":
        return [c for rf in RF_MODELS for c in unfiltered_span_candidates(rf)]
    if row == "Filtered SpanSelection (1 call)":
        return [c for rf in RF_MODELS for c in one_call_span_candidates(rf)]
    if row == "Filtered SpanSelection (2 calls)":
        return [c for rf in RF_MODELS for c in two_call_span_candidates(rf)]
    raise ValueError(f"Unknown row: {row}")


def getv(all_results, config_name, dataset_label, metric="map"):
    if not NARRATIVE_SUPPORT.get(dataset_label, True) and requires_narrative(config_name):
        return None
    return all_results.get(config_name, {}).get(dataset_label, {}).get(metric)


def pick_winner(all_results, candidates, dataset_label):
    best_cname, best_val = None, None
    for cname in candidates:
        v = getv(all_results, cname, dataset_label, "map")
        if v is not None and (best_val is None or v > best_val):
            best_cname, best_val = cname, v
    return best_cname, best_val


def format_depth(v):
    if isinstance(v, str):
        return v.upper()
    return f"{v:.0f}"


def run_file_baseline(test_run_folder, baseline):
    if baseline == "LMDirichlet":
        return test_run_folder / "LMDirichlet-2000_content"
    return test_run_folder / "BM25_content"


def run_file_monot5_rerank(test_run_folder, baseline, depth):
    prefix = "LMDirichlet-2000_content" if baseline == "LMDirichlet" else "BM25_content"
    return test_run_folder / f"{prefix}_rerank-monoT5_topK-{format_depth(depth)}"


def run_file_prf(test_run_folder, baseline, strategy, rf, depth, lam, e, term_filter):
    prefix = "LMDirichlet-2000_content" if baseline == "LMDirichlet" else "BM25_content"
    lam_str = f"{lam:.2f}" if lam is not None else "0.00"
    tf_str = f"_termFilter-{term_filter.upper()}" if term_filter != "none" else ""
    return test_run_folder / (
        f"{prefix}_prf-true_rfStrategy-{strategy}_rfModel-{rf}"
        f"_prfSmoothing-Additive-0.1000_topK-{format_depth(depth)}"
        f"_lambda-{lam_str}_e-{e:.0f}{tf_str}"
    )


def get_run_file(cname, entry, test_run_folder):
    depth, e, lam = entry.get("depth"), entry.get("e"), entry.get("lambda")
    if cname.startswith("BASELINE_"):
        baseline = cname.split("_", 1)[1]
        return run_file_baseline(test_run_folder, baseline)
    if cname.startswith("MONOT5_RERANK_"):
        baseline = cname.split("_", 2)[2]
        return run_file_monot5_rerank(test_run_folder, baseline, depth)
    _, strategy, rf, baseline, tf_suffix = cname.split("_")
    term_filter = TF_SUFFIX_TO_CLI[tf_suffix]
    return run_file_prf(test_run_folder, baseline, strategy, rf, depth, lam, e, term_filter)


def per_query_ap(qrels_path, run_file):
    cmd = f'trec_eval -q -m map "{qrels_path}" "{run_file}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"trec_eval failed on {run_file}: {result.stderr}")
    pq = {}
    for line in result.stdout.split("\n"):
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "map" and parts[1] != "all":
            pq[parts[1]] = float(parts[2])
    return pq


def bh_correct(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = pvals[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty(n)
    out[order] = adjusted
    return out


def resolve_row(row, all_results, ds_label, test_run_folder, qrels_test_path):
    candidates = row_candidates(row)
    cname, reported_map = pick_winner(all_results, candidates, ds_label)
    if cname is None:
        print(f"  {row}: no candidate found in all_results.json, skipping", file=sys.stderr)
        return None

    entry = all_results[cname][ds_label]
    run_file = get_run_file(cname, entry, test_run_folder)
    if not run_file.exists():
        print(f"  {row} ({cname}): run file not found: {run_file}", file=sys.stderr)
        return None

    pq = per_query_ap(qrels_test_path, run_file)
    recomputed_map = sum(pq.values()) / len(pq) if pq else None
    if recomputed_map is None or abs(recomputed_map - reported_map) > 5e-4:
        print(
            f"  WARNING [{row}] {cname}: run file MAP={recomputed_map} "
            f"!= all_results.json MAP={reported_map} -- stale run file?",
            file=sys.stderr,
        )

    return {"row": row, "config": cname, "map_reported": reported_map, "per_query_ap": pq}


def paired_wilcoxon(pq_x, pq_y):
    common = sorted(set(pq_x) & set(pq_y))
    x = np.array([pq_x[q] for q in common])
    y = np.array([pq_y[q] for q in common])
    diffs = x - y
    n_nonzero = int(np.count_nonzero(diffs))
    try:
        if n_nonzero == 0:
            pval = 1.0
        else:
            _, pval = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        pval = 1.0
    return float(pval), len(common), n_nonzero, float(np.mean(diffs)) if common else 0.0


def main():
    all_results_path = None
    per_dataset_results = {}

    for ds_key, ds_label in DATASETS:
        print(f"=== {ds_label} ({ds_key}) ===", file=sys.stderr)
        cfg = load_dataset_config(ds_key)
        folder = Path(cfg["FOLDER"])
        index = cfg["INDEX"]
        qrels_test_path = folder / "qrels" / cfg["QRELS_TEST"]
        test_run_folder = folder / "runs" / f"{index}_test"
        all_results_path = folder / "test_results" / "all_results.json"

        with open(all_results_path) as f:
            all_results = json.load(f)

        rows_here = [
            r for r in ROWS
            if not (ds_label == "DL-20" and r == "LLM filtering + narrative")
        ]

        resolved = {}
        for row in rows_here:
            r = resolve_row(row, all_results, ds_label, test_run_folder, qrels_test_path)
            if r is not None:
                resolved[row] = r

        active_rows = list(resolved.keys())

        pairs = [
            (a, b)
            for i, a in enumerate(active_rows)
            for b in active_rows[i + 1:]
        ]

        pair_results = []
        for row_a, row_b in pairs:
            pval, n_common, n_nonzero, mean_diff = paired_wilcoxon(
                resolved[row_a]["per_query_ap"], resolved[row_b]["per_query_ap"]
            )
            winner = row_a if mean_diff > 0 else (row_b if mean_diff < 0 else None)
            pair_results.append({
                "a": row_a, "b": row_b, "pval": pval,
                "n_queries": n_common, "n_nonzero_diff": n_nonzero,
                "winner": winner,
            })

        pvals = [p["pval"] for p in pair_results]
        adj = bh_correct(pvals) if pvals else []
        for p, a in zip(pair_results, adj):
            p["pval_bh"] = float(a)
            p["significant"] = bool(a < 0.05)

        beats = {row: set() for row in active_rows}
        for p in pair_results:
            if p["significant"] and p["winner"] is not None:
                loser = p["b"] if p["winner"] == p["a"] else p["a"]
                beats[p["winner"]].add(loser)

        row_summaries = []
        for row in active_rows:
            row_summaries.append({
                "row": row,
                "letter": ROW_LETTERS[row],
                "config": resolved[row]["config"],
                "map": resolved[row]["map_reported"],
                "beats_letters": sorted(ROW_LETTERS[r] for r in beats[row]),
            })

        per_dataset_results[ds_label] = {
            "rows": row_summaries,
            "pairs": pair_results,
        }

    print()
    print("Row letters:")
    for row, letter in ROW_LETTERS.items():
        print(f"  ({letter}) {row}")
    print()

    header = f"{'Collection':<9} {'Row':<34} {'MAP':>7}  beats"
    print(header)
    print("-" * len(header))
    for ds_label, data in per_dataset_results.items():
        for r in data["rows"]:
            beats_str = ",".join(f"({l})" for l in r["beats_letters"])
            print(f"{ds_label:<9} ({r['letter']}) {r['row']:<30} {r['map']:>7.4f}  {beats_str}")
        print()

    if all_results_path is not None:
        out_path = all_results_path.parent / "significance_table5.json"
        with open(out_path, "w") as f:
            json.dump(per_dataset_results, f, indent=2)
        print(f"Saved machine-readable results to {out_path}")


if __name__ == "__main__":
    main()
