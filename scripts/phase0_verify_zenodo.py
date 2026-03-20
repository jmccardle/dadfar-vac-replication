#!/usr/bin/env python3
"""
Phase 0: Independent statistical verification of Dadfar (2026) claims.

Loads all Zenodo data and recomputes every correlation reported in the paper.
Compares computed values to claimed values and reports pass/fail.

Note: The Zenodo public release (record 18614770) does not include raw text files.
The "expand" vocabulary cluster was re-counted from text in the paper's analysis
but is absent from the pre-computed vocab_counts in qwen_baseline_n50.json.
Expand correlations can only be verified from the descriptive control data or
after generating our own data in Phase 1.

Usage:
    python scripts/phase0_verify_zenodo.py
"""

import json
import sys
from pathlib import Path
from scipy import stats
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "zenodo" / "data"

TOLERANCE = 0.015  # r tolerance for matching claimed values
TOLERANCE_DESC = 0.10  # wider tolerance for descriptive controls (different data version)


def load_json(name: str) -> dict:
    path = DATA_DIR / name
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def safe_pearson(x, y):
    """Pearson correlation, returning (NaN, NaN) for constant inputs."""
    ax, ay = np.array(x, dtype=float), np.array(y, dtype=float)
    if np.std(ax) < 1e-10 or np.std(ay) < 1e-10:
        return float("nan"), float("nan")
    return stats.pearsonr(ax, ay)


def safe_spearman(x, y):
    ax, ay = np.array(x, dtype=float), np.array(y, dtype=float)
    if np.std(ax) < 1e-10 or np.std(ay) < 1e-10:
        return float("nan"), float("nan")
    return stats.spearmanr(ax, ay)


def outlier_removed_pearson(x, y):
    """Remove the single most influential point (max |r_change|) per Dadfar's method.

    Finds the point whose removal causes the largest absolute change in r,
    then reports the r without that point. If the most influential point was
    helping the correlation, r decreases. If it was hurting, r increases.
    """
    ax, ay = np.array(x, dtype=float), np.array(y, dtype=float)
    if np.std(ax) < 1e-10 or np.std(ay) < 1e-10:
        return float("nan"), float("nan")
    base_r, _ = stats.pearsonr(ax, ay)
    max_abs_change = 0
    worst_idx = -1
    for i in range(len(ax)):
        mask = np.ones(len(ax), dtype=bool)
        mask[i] = False
        ri, _ = stats.pearsonr(ax[mask], ay[mask])
        abs_change = abs(ri - base_r)
        if abs_change > max_abs_change:
            max_abs_change = abs_change
            worst_idx = i
    if worst_idx >= 0:
        mask = np.ones(len(ax), dtype=bool)
        mask[worst_idx] = False
        return stats.pearsonr(ax[mask], ay[mask])
    return base_r, None


def check(label: str, computed_r: float, claimed_r: float, computed_p: float = None,
          claimed_p: float = None, tol: float = TOLERANCE) -> bool:
    if np.isnan(computed_r):
        print(f"  [SKIP] {label}: computed r=NaN (constant input)")
        return None  # Neither pass nor fail
    delta = abs(computed_r - claimed_r)
    status = "PASS" if delta <= tol else "FAIL"
    p_str = f"p={computed_p:.6f}" if computed_p is not None and not np.isnan(computed_p) else ""
    claimed_p_str = f"(claimed p={claimed_p})" if claimed_p is not None else ""
    print(f"  [{status}] {label}: r={computed_r:.4f} (claimed {claimed_r:.2f}, delta={delta:.4f}) {p_str} {claimed_p_str}")
    return delta <= tol


def verify_qwen_baseline():
    """Verify Table 2: Qwen 2.5-32B correspondence results."""
    print("\n" + "=" * 70)
    print("TABLE 2: Qwen 2.5-32B Baseline Correspondence (N=50)")
    print("=" * 70)

    data = load_json("qwen_baseline_n50.json")
    runs = [r for r in data["runs"] if r.get("layer_metrics")]
    layer = "8"
    print(f"Valid runs: {len(runs)}")

    mirrors = [r["vocab_counts"].get("mirror", 0) for r in runs]
    expands = [r["vocab_counts"].get("expand", 0) for r in runs]
    resonances = [r["vocab_counts"].get("resonance", 0) for r in runs]
    spec_low = [r["layer_metrics"][layer]["spectral_power_low"] for r in runs]
    max_norms = [r["layer_metrics"][layer]["max_norm"] for r in runs]
    tokens = [r.get("n_tokens", r.get("text_length", 1)) for r in runs]
    spec_low_norm = [s / t if t > 0 else 0 for s, t in zip(spec_low, tokens)]

    results = []

    # mirror <-> spectral_power_low (raw)
    r_val, p_val = safe_pearson(mirrors, spec_low)
    results.append(check("mirror <-> spectral (raw)", r_val, 0.62, p_val))

    # mirror <-> spectral_power_low (normalized)
    r_val, p_val = safe_pearson(mirrors, spec_low_norm)
    results.append(check("mirror <-> spectral (norm)", r_val, 0.60, p_val))

    # mirror Spearman
    rho, rho_p = safe_spearman(mirrors, spec_low)
    results.append(check("mirror <-> spectral (Spearman)", rho, 0.57, rho_p))

    # expand — not in Zenodo pre-computed vocab_counts
    if all(v == 0 for v in expands):
        print("\n  [NOTE] 'expand' not in pre-computed vocab_counts (all zeros).")
        print("  Expand correlations require raw text files (restricted Zenodo 18567446).")
        print("  Will be verifiable after Phase 1 generates our own data.")
    else:
        r_val, p_val = safe_pearson(expands, spec_low)
        results.append(check("expand <-> spectral (raw)", r_val, 0.58, p_val))

    # resonance <-> max_norm
    r_val, p_val = safe_pearson(resonances, max_norms)
    results.append(check("resonance <-> max_norm", r_val, 0.54, p_val))

    # resonance Spearman
    rho, rho_p = safe_spearman(resonances, max_norms)
    results.append(check("resonance <-> max_norm (Spearman)", rho, 0.31, rho_p))

    # Outlier-removed (Dadfar method: remove most helpful point)
    print("\n  Outlier-removed (most-helpful-point removal):")
    or_r, or_p = outlier_removed_pearson(mirrors, spec_low)
    results.append(check("mirror <-> spectral (outlier-removed)", or_r, 0.54, or_p))

    or_r, or_p = outlier_removed_pearson(resonances, max_norms)
    results.append(check("resonance <-> max_norm (outlier-removed)", or_r, 0.30, or_p))

    return [r for r in results if r is not None]


def verify_qwen_descriptive_control():
    """Verify Table 2: Qwen descriptive control results."""
    print("\n" + "=" * 70)
    print("QWEN DESCRIPTIVE CONTROL (N=75)")
    print("=" * 70)

    data = load_json("qwen_descriptive_control.json")
    runs = data["runs"]
    layer = "8"
    print(f"Runs: {len(runs)}")

    mirrors = [r["vocab_counts"].get("mirror", 0) for r in runs]
    expands = [r["vocab_counts"].get("expand", 0) for r in runs]
    resonances = [r["vocab_counts"].get("resonance", 0) for r in runs]

    spec_low = [r["layer_metrics"][layer]["spectral_power_low"] for r in runs]
    max_norms = [r["layer_metrics"][layer]["max_norm"] for r in runs]
    tokens = [r.get("n_tokens", r.get("text_length", 1)) for r in runs]
    spec_low_norm = [s / t if t > 0 else 0 for s, t in zip(spec_low, tokens)]

    results = []

    # mirror <-> spectral (desc, normalized) — should vanish
    r_val, p_val = safe_pearson(mirrors, spec_low_norm)
    results.append(check("mirror <-> spectral (desc, norm)", r_val, -0.09, p_val, tol=TOLERANCE_DESC))

    # expand <-> spectral (desc, normalized) — desc file DOES have expand counts
    r_val, p_val = safe_pearson(expands, spec_low_norm)
    results.append(check("expand <-> spectral (desc, norm)", r_val, -0.14, p_val, tol=TOLERANCE_DESC))

    # resonance <-> max_norm (desc)
    r_val, p_val = safe_pearson(resonances, max_norms)
    results.append(check("resonance <-> max_norm (desc)", r_val, 0.16, p_val, tol=TOLERANCE_DESC))

    # Key test: all three should be non-significant
    print("\n  Non-significance checks (the critical test):")
    for name, x, y in [
        ("mirror desc", mirrors, spec_low_norm),
        ("expand desc", expands, spec_low_norm),
        ("resonance desc", resonances, max_norms),
    ]:
        _, p_val_check = safe_pearson(x, y)
        sig = "SIG" if (not np.isnan(p_val_check) and p_val_check < 0.05) else "non-sig"
        expected = "non-sig"
        status = "PASS" if sig == expected else "FAIL"
        print(f"  [{status}] {name} significance: {sig} (p={p_val_check:.4f}, expected {expected})")
        results.append(sig == expected)

    return [r for r in results if r is not None]


def verify_llama_baseline():
    """Verify Table 1: Llama 70B correspondence results."""
    print("\n" + "=" * 70)
    print("TABLE 1: Llama 70B Baseline Correspondence (N=50)")
    print("=" * 70)

    data = load_json("llama_baseline_n50.json")
    runs = data["runs"]
    print(f"Runs: {len(runs)}")

    loops = [r["vocab_counts"]["loop"] for r in runs]
    surges = [r["vocab_counts"].get("surge", 0) for r in runs]
    autocorrs = [r["metrics"]["autocorr_lag1"] for r in runs]
    max_norms = [r["metrics"]["max_norm"] for r in runs]

    results = []

    # loop <-> autocorrelation
    r_val, p_val = safe_pearson(loops, autocorrs)
    results.append(check("loop <-> autocorr", r_val, 0.44, p_val, 0.002))

    rho, rho_p = safe_spearman(loops, autocorrs)
    results.append(check("loop <-> autocorr (Spearman)", rho, 0.36, rho_p))

    or_r, or_p = outlier_removed_pearson(loops, autocorrs)
    results.append(check("loop <-> autocorr (outlier-removed)", or_r, 0.38, or_p))

    # surge <-> max_norm
    r_val, p_val = safe_pearson(surges, max_norms)
    results.append(check("surge <-> max_norm", r_val, 0.44, p_val, 0.002))

    rho, rho_p = safe_spearman(surges, max_norms)
    results.append(check("surge <-> max_norm (Spearman)", rho, 0.34, rho_p))

    or_r, or_p = outlier_removed_pearson(surges, max_norms)
    results.append(check("surge <-> max_norm (outlier-removed)", or_r, 0.32, or_p))

    return [r for r in results if r is not None]


def verify_llama_paired():
    """Verify Table 1: Llama shimmer paired results."""
    print("\n" + "=" * 70)
    print("TABLE 1: Llama 70B Shimmer Paired (N=70)")
    print("=" * 70)

    data = load_json("llama_paired_n70.json")
    runs = data["runs"]
    print(f"Paired runs: {len(runs)}")

    shimmers_s, norm_stds_s = [], []
    shimmer_d, normstd_d = [], []
    for run in runs:
        conds = run["conditions"]
        if "baseline" in conds and "steered" in conds:
            sb = conds["baseline"]["vocab_counts"].get("shimmer", 0)
            ss = conds["steered"]["vocab_counts"].get("shimmer", 0)
            nb = conds["baseline"]["metrics"]["norm_std"]
            ns = conds["steered"]["metrics"]["norm_std"]
            shimmers_s.append(ss)
            norm_stds_s.append(ns)
            shimmer_d.append(ss - sb)
            normstd_d.append(ns - nb)

    results = []
    print(f"  Valid pairs: {len(shimmers_s)}")

    # shimmer <-> norm_std (steered)
    r_val, p_val = safe_pearson(shimmers_s, norm_stds_s)
    results.append(check("shimmer <-> norm_std (steered)", r_val, 0.33, p_val, 0.005))

    rho, rho_p = safe_spearman(shimmers_s, norm_stds_s)
    results.append(check("shimmer <-> norm_std (steered, Spearman)", rho, 0.31, rho_p))

    # shimmer delta
    r_val, p_val = safe_pearson(shimmer_d, normstd_d)
    results.append(check("shimmer delta", r_val, 0.36, p_val, 0.002))

    rho, rho_p = safe_spearman(shimmer_d, normstd_d)
    results.append(check("shimmer delta (Spearman)", rho, 0.39, rho_p))

    # Outlier-removed
    or_r, or_p = outlier_removed_pearson(shimmers_s, norm_stds_s)
    results.append(check("shimmer steered (outlier-removed)", or_r, 0.42, or_p))

    or_r, or_p = outlier_removed_pearson(shimmer_d, normstd_d)
    results.append(check("shimmer delta (outlier-removed)", or_r, 0.43, or_p))

    return [r for r in results if r is not None]


def verify_llama_descriptive_control():
    """Verify Llama descriptive control — loop correspondence should vanish."""
    print("\n" + "=" * 70)
    print("LLAMA DESCRIPTIVE CONTROL (N=50)")
    print("=" * 70)

    data = load_json("llama_descriptive_control.json")
    runs = data["runs"]
    layer = "5"
    print(f"Runs: {len(runs)}")

    if "layer_metrics" in runs[0] and layer in runs[0]["layer_metrics"]:
        loops = [r["vocab_counts"].get("loop", 0) for r in runs]
        autocorrs = [r["layer_metrics"][layer]["autocorr_lag1"] for r in runs]
        surges = [r["vocab_counts"].get("surge", 0) for r in runs]
        max_norms = [r["layer_metrics"][layer]["max_norm"] for r in runs]
    else:
        print("  ERROR: Cannot find layer_metrics[5] in descriptive control data")
        return [False]

    results = []

    # loop <-> autocorrelation (should vanish)
    # Note: exact r may differ between Zenodo data versions; key test is non-significance
    r_val, p_val = safe_pearson(loops, autocorrs)
    results.append(check("loop <-> autocorr (desc)", r_val, 0.05, p_val, 0.82, tol=TOLERANCE_DESC))

    # Key test: loop must be non-significant
    sig = "SIG" if (not np.isnan(p_val) and p_val < 0.05) else "non-sig"
    status = "PASS" if sig == "non-sig" else "FAIL"
    print(f"  [{status}] loop desc significance: {sig} (p={p_val:.4f}, expected non-sig)")
    results.append(sig == "non-sig")

    # surge <-> max_norm (descriptive — paper notes this persists as non-specific)
    r_val, p_val = safe_pearson(surges, max_norms)
    results.append(check("surge <-> max_norm (desc)", r_val, 0.60, p_val, 0.0015, tol=TOLERANCE_DESC))

    return [r for r in results if r is not None]


def verify_fdr():
    """Verify Benjamini-Hochberg FDR correction across all 9 pre-specified tests."""
    print("\n" + "=" * 70)
    print("BENJAMINI-HOCHBERG FDR CORRECTION (9 tests)")
    print("=" * 70)

    tests = [
        ("loop <-> autocorr (baseline)", 0.002),
        ("surge <-> max_norm (baseline)", 0.002),
        ("shimmer <-> norm_std (steered)", 0.005),
        ("shimmer paired delta", 0.002),
        ("surge <-> max_norm (steered)", 0.0005),
        ("surge paired delta", 0.001),
        ("mirror <-> spectral (Qwen)", 0.0001),
        ("expand <-> spectral (Qwen)", 0.0001),
        ("resonance <-> max_norm (Qwen)", 0.0001),
    ]

    names = [t[0] for t in tests]
    pvals = np.array([t[1] for t in tests])
    m = len(pvals)

    sorted_idx = np.argsort(pvals)
    sorted_p = pvals[sorted_idx]
    sorted_names = [names[i] for i in sorted_idx]

    qvals = np.zeros(m)
    for i in range(m):
        rank = i + 1
        qvals[i] = sorted_p[i] * m / rank

    for i in range(m - 2, -1, -1):
        qvals[i] = min(qvals[i], qvals[i + 1])

    all_sig = True
    for i in range(m):
        sig = "YES" if qvals[i] < 0.05 else "NO"
        if qvals[i] >= 0.05:
            all_sig = False
        print(f"  {sorted_names[i]:<45} p={sorted_p[i]:.4f}  rank={i + 1}  q={qvals[i]:.4f}  sig={sig}")

    status = "PASS" if all_sig else "FAIL"
    print(f"\n  [{status}] All 9 tests significant at q<0.05: {all_sig}")
    return [all_sig]


def main():
    print("Phase 0: Independent Verification of Dadfar (2026)")
    print("=" * 70)

    all_results = []

    all_results.extend(verify_llama_baseline())
    all_results.extend(verify_llama_paired())
    all_results.extend(verify_llama_descriptive_control())
    all_results.extend(verify_qwen_baseline())
    all_results.extend(verify_qwen_descriptive_control())
    all_results.extend(verify_fdr())

    n_pass = sum(1 for r in all_results if r)
    n_fail = sum(1 for r in all_results if not r)
    n_total = len(all_results)
    print("\n" + "=" * 70)
    print(f"SUMMARY: {n_pass}/{n_total} passed, {n_fail} failed")
    print("=" * 70)

    if n_fail == 0:
        print("All verifiable statistical claims confirmed. Proceed to Phase 1.")
    else:
        print(f"\n{n_fail} checks outside tolerance. Review details above.")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
