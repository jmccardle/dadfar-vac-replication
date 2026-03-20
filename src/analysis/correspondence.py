"""Vocabulary-activation correspondence analysis.

Computes Pearson/Spearman correlations, outlier-robust regression,
and Benjamini-Hochberg FDR correction.
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path


def load_runs(results_path: Path, layer: str = "8") -> tuple[list[dict], list[dict]]:
    """Load runs from a results JSON file.

    Returns (layer_metrics_list, vocab_counts_list) for valid runs.
    """
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)
    runs = data.get("runs", [])

    metrics_list = []
    vocab_list = []
    for r in runs:
        lm = r.get("layer_metrics", {})
        if layer in lm:
            metrics_list.append(lm[layer])
            vocab_list.append(r.get("vocab_counts", {}))
        elif "metrics" in r:
            # Llama flat-metrics format
            metrics_list.append(r["metrics"])
            vocab_list.append(r.get("vocab_counts", {}))

    return metrics_list, vocab_list


def compute_correspondence(
    vocab_counts: list[int],
    metric_values: list[float],
    token_counts: list[int] = None,
) -> dict:
    """Compute full correspondence statistics for a vocab-metric pair.

    Args:
        vocab_counts: Per-run vocabulary cluster counts.
        metric_values: Per-run activation metric values.
        token_counts: Per-run token counts for length-normalized analysis.

    Returns:
        Dict with pearson_r, pearson_p, spearman_rho, spearman_p,
        outlier_removed_r, and optionally normalized variants.
    """
    x = np.array(vocab_counts, dtype=float)
    y = np.array(metric_values, dtype=float)

    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return {"error": "constant input", "n": len(x)}

    r, p = stats.pearsonr(x, y)
    rho, rho_p = stats.spearmanr(x, y)
    or_r, or_p = _outlier_removed_pearson(x, y)

    result = {
        "n": len(x),
        "pearson_r": round(float(r), 4),
        "pearson_p": float(p),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": float(rho_p),
        "outlier_removed_r": round(float(or_r), 4) if not np.isnan(or_r) else None,
        "outlier_removed_p": float(or_p) if or_p is not None and not np.isnan(or_p) else None,
    }

    # Per-token normalized (for spectral power)
    if token_counts is not None:
        t = np.array(token_counts, dtype=float)
        y_norm = np.where(t > 0, y / t, 0)
        if np.std(y_norm) > 1e-10:
            r_n, p_n = stats.pearsonr(x, y_norm)
            result["normalized_r"] = round(float(r_n), 4)
            result["normalized_p"] = float(p_n)

    return result


def _outlier_removed_pearson(x, y):
    """Remove single most influential point, return new (r, p)."""
    base_r, _ = stats.pearsonr(x, y)
    max_abs_change = 0
    worst_idx = -1
    for i in range(len(x)):
        mask = np.ones(len(x), dtype=bool)
        mask[i] = False
        ri, _ = stats.pearsonr(x[mask], y[mask])
        if abs(ri - base_r) > max_abs_change:
            max_abs_change = abs(ri - base_r)
            worst_idx = i
    if worst_idx >= 0:
        mask = np.ones(len(x), dtype=bool)
        mask[worst_idx] = False
        return stats.pearsonr(x[mask], y[mask])
    return base_r, None


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """Apply Benjamini-Hochberg FDR correction.

    Args:
        p_values: List of (name, p_value) tuples.
        alpha: FDR threshold.

    Returns:
        List of dicts with name, p_value, rank, q_value, significant.
    """
    names, pvals = zip(*p_values) if p_values else ([], [])
    pvals = np.array(pvals)
    m = len(pvals)
    if m == 0:
        return []

    sorted_idx = np.argsort(pvals)
    sorted_p = pvals[sorted_idx]
    sorted_names = [names[i] for i in sorted_idx]

    qvals = np.zeros(m)
    for i in range(m):
        qvals[i] = sorted_p[i] * m / (i + 1)

    # Enforce monotonicity
    for i in range(m - 2, -1, -1):
        qvals[i] = min(qvals[i], qvals[i + 1])

    results = []
    for i in range(m):
        results.append({
            "name": sorted_names[i],
            "p_value": float(sorted_p[i]),
            "rank": i + 1,
            "q_value": float(qvals[i]),
            "significant": bool(qvals[i] < alpha),
        })
    return results


def run_full_correspondence(
    results_path: Path,
    layer: str = "8",
    vocab_pairs: dict[str, str] = None,
) -> dict:
    """Run full correspondence analysis on a results file.

    Args:
        results_path: Path to JSON results file.
        layer: Layer key.
        vocab_pairs: Dict of {vocab_cluster: metric_name} to test.
                     If None, tests the 3 Dadfar Qwen pairs.

    Returns:
        Dict with per-pair results and FDR correction.
    """
    if vocab_pairs is None:
        vocab_pairs = {
            "mirror": "spectral_power_low",
            "expand": "spectral_power_low",
            "resonance": "max_norm",
        }

    metrics_list, vocab_list = load_runs(results_path, layer)
    print(f"Loaded {len(metrics_list)} valid runs from {results_path.name}")

    # Extract token counts for normalization
    with open(results_path) as f:
        data = json.load(f)
    runs = data.get("runs", [])
    token_counts = [r.get("n_tokens", r.get("text_length", 1)) for r in runs
                    if r.get("layer_metrics", {}).get(layer) or "metrics" in r]

    pair_results = {}
    fdr_inputs = []

    for vocab_name, metric_name in vocab_pairs.items():
        v = [vc.get(vocab_name, 0) for vc in vocab_list]
        m = [ml.get(metric_name, 0) for ml in metrics_list]

        corr = compute_correspondence(v, m, token_counts if "spectral" in metric_name else None)
        pair_results[f"{vocab_name} <-> {metric_name}"] = corr

        if "pearson_p" in corr:
            fdr_inputs.append((f"{vocab_name} <-> {metric_name}", corr["pearson_p"]))

        # Print summary
        if "error" in corr:
            print(f"  {vocab_name} <-> {metric_name}: {corr['error']}")
        else:
            sig = "***" if corr["pearson_p"] < 0.001 else "**" if corr["pearson_p"] < 0.01 else "*" if corr["pearson_p"] < 0.05 else "ns"
            print(f"  {vocab_name} <-> {metric_name}: r={corr['pearson_r']:.4f} p={corr['pearson_p']:.6f} {sig} | rho={corr['spearman_rho']:.4f}")

    fdr_results = benjamini_hochberg(fdr_inputs)

    return {
        "pairs": pair_results,
        "fdr": fdr_results,
        "n_runs": len(metrics_list),
        "layer": layer,
    }
