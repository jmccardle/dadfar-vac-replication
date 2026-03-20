#!/usr/bin/env python3
"""R3 + R12: VAC Pair Count Permutation Test and CV-Significance Correlation.

Addresses reviewer Point 3: the "26 vs 20 significant pairs" comparison is
never formally tested. Also computes Spearman correlation between per-condition
significant pair counts and within-condition token CV (Point 3 subpart / R12).

Permutation test: shuffle condition labels between pairs of conditions,
recompute # significant VAC pairs per condition, repeat 10,000 times.

Data: outputs/runs/phase_d_controls/phase_d_results.json (raw run data)
      outputs/analysis/phase_e2_vac_controlled.json (pre-computed VAC results)
Output: outputs/analysis/revision_r03_vac_permutation.json
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
DATA_RUNS = PROJECT / "outputs" / "runs" / "phase_d_controls" / "phase_d_results.json"
DATA_VAC = PROJECT / "outputs" / "analysis" / "phase_e2_vac_controlled.json"
OUT = PROJECT / "outputs" / "analysis" / "revision_r03_vac_permutation.json"

CONDITIONS = [
    "baseline", "abstract_philosophical", "factual_iterative",
    "procedural_self", "descriptive_forest", "descriptive_math",
    "descriptive_music", "nonsense_control",
]

VOCAB_KEYS = [
    "mirror", "expand", "resonance", "loop", "surge",
    "shimmer", "pulse", "void", "depth", "shift",
    "ctrl_the", "ctrl_and", "ctrl_question", "ctrl_what", "ctrl_that",
    "ctrl_processing", "ctrl_system", "ctrl_pull", "ctrl_word", "ctrl_observe",
]

METRIC_KEYS = [
    "mean_norm", "max_norm", "norm_std", "autocorr_lag1",
    "spectral_power_low", "spectral_power_mid",
    "mean_derivative", "mean_token_similarity", "convergence_ratio",
]

N_PERMUTATIONS = 10000
ALPHA = 0.05
LAYER = "8"


def load_runs():
    """Load raw run data grouped by condition."""
    with open(DATA_RUNS) as f:
        data = json.load(f)

    grouped = defaultdict(list)
    for run in data["runs"]:
        grouped[run["condition"]].append(run)
    return grouped


def _fast_pearson_p(x, y):
    """Fast Pearson r and two-sided p-value using numpy (avoids scipy overhead)."""
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    xm = x - x.mean()
    ym = y - y.mean()
    sx = np.sqrt((xm * xm).sum())
    sy = np.sqrt((ym * ym).sum())
    if sx == 0 or sy == 0:
        return 0.0, 1.0
    r = (xm * ym).sum() / (sx * sy)
    r = max(-1.0, min(1.0, r))
    if abs(r) >= 1.0:
        return float(r), 0.0
    t = r * np.sqrt((n - 2) / (1.0 - r * r))
    p = 2.0 * stats.t.sf(abs(t), n - 2)
    return float(r), float(p)


def count_significant_pairs(runs, alpha=ALPHA):
    """Count significant Pearson correlations (vocab × metric) for a set of runs.

    Returns (n_significant, n_tested, list of significant pairs).
    """
    n = len(runs)
    if n < 5:
        return 0, 0, []

    # Extract arrays
    vocab_arrays = {}
    for vk in VOCAB_KEYS:
        vals = [r.get("vocab_counts", {}).get(vk, 0) for r in runs]
        vocab_arrays[vk] = np.array(vals, dtype=float)

    metric_arrays = {}
    for mk in METRIC_KEYS:
        vals = []
        for r in runs:
            lm = r.get("layer_metrics", {}).get(LAYER, {})
            vals.append(lm.get(mk, float("nan")))
        metric_arrays[mk] = np.array(vals, dtype=float)

    n_sig = 0
    n_tested = 0
    sig_pairs = []

    for vk in VOCAB_KEYS:
        for mk in METRIC_KEYS:
            v = vocab_arrays[vk]
            m = metric_arrays[mk]
            mask = np.isfinite(v) & np.isfinite(m)
            if mask.sum() < 5:
                continue
            n_tested += 1
            r, p = _fast_pearson_p(v[mask], m[mask])
            if p < alpha:
                n_sig += 1
                sig_pairs.append({"vocab": vk, "metric": mk, "r": float(r), "p": float(p)})

    return n_sig, n_tested, sig_pairs


def _extract_arrays(runs):
    """Pre-extract vocab and metric arrays from runs for fast permutation."""
    vocab = np.zeros((len(runs), len(VOCAB_KEYS)))
    metric = np.full((len(runs), len(METRIC_KEYS)), np.nan)
    for i, r in enumerate(runs):
        for j, vk in enumerate(VOCAB_KEYS):
            vocab[i, j] = r.get("vocab_counts", {}).get(vk, 0)
        lm = r.get("layer_metrics", {}).get(LAYER, {})
        for j, mk in enumerate(METRIC_KEYS):
            metric[i, j] = lm.get(mk, float("nan"))
    return vocab, metric


def _count_sig_from_arrays(vocab, metric, alpha=ALPHA):
    """Count significant pairs from pre-extracted arrays."""
    n = vocab.shape[0]
    if n < 5:
        return 0
    n_sig = 0
    for vi in range(vocab.shape[1]):
        for mi in range(metric.shape[1]):
            v = vocab[:, vi]
            m = metric[:, mi]
            mask = np.isfinite(v) & np.isfinite(m)
            if mask.sum() < 5:
                continue
            _, p = _fast_pearson_p(v[mask], m[mask])
            if p < alpha:
                n_sig += 1
    return n_sig


def permutation_test_pair_counts(runs_a, runs_b, n_perm=N_PERMUTATIONS):
    """Permutation test: is the difference in significant pair counts significant?

    Shuffles run labels between two conditions, recomputes pair counts.
    """
    obs_a, n_tested_a, _ = count_significant_pairs(runs_a)
    obs_b, n_tested_b, _ = count_significant_pairs(runs_b)
    obs_diff = obs_a - obs_b

    # Pre-extract arrays for speed
    combined = runs_a + runs_b
    na = len(runs_a)
    v_all, m_all = _extract_arrays(combined)

    rng = np.random.default_rng(42)
    count_ge = 0

    for _ in range(n_perm):
        perm_idx = rng.permutation(len(combined))
        sig_a = _count_sig_from_arrays(v_all[perm_idx[:na]], m_all[perm_idx[:na]])
        sig_b = _count_sig_from_arrays(v_all[perm_idx[na:]], m_all[perm_idx[na:]])
        if abs(sig_a - sig_b) >= abs(obs_diff):
            count_ge += 1

    perm_p = (count_ge + 1) / (n_perm + 1)
    return obs_a, obs_b, obs_diff, float(perm_p)


def compute_token_cv(runs):
    """Compute coefficient of variation of token counts."""
    tokens = [r["n_tokens"] for r in runs if "n_tokens" in r]
    if len(tokens) < 2:
        return float("nan")
    arr = np.array(tokens, dtype=float)
    return float(arr.std(ddof=1) / arr.mean()) if arr.mean() > 0 else float("nan")


def main():
    grouped = load_runs()

    results = {
        "description": "R3+R12: VAC pair count permutation test and CV correlation",
        "n_permutations": N_PERMUTATIONS,
        "alpha": ALPHA,
    }

    print("=" * 72)
    print("R3+R12: VAC PAIR COUNT PERMUTATION TEST AND CV CORRELATION")
    print("=" * 72)

    # ── Per-condition significant pair counts and token CV ──────────
    per_condition = {}
    for cond in CONDITIONS:
        runs = grouped.get(cond, [])
        n_sig, n_tested, sig_pairs = count_significant_pairs(runs)
        cv = compute_token_cv(runs)
        per_condition[cond] = {
            "n_runs": len(runs),
            "n_significant_pairs": n_sig,
            "n_tested": n_tested,
            "token_cv": cv,
            "significant_pairs": sig_pairs,
        }

    results["per_condition"] = per_condition

    expected_fp = 0.05 * max(pc["n_tested"] for pc in per_condition.values())
    results["expected_false_positives"] = expected_fp

    print(f"\n── Per-Condition Significant VAC Pairs (α={ALPHA}) ──")
    print(f"Expected false positives at α=0.05: {expected_fp:.1f}")
    print(f"{'Condition':<28} {'N':>3} {'Sig':>4} {'Tested':>6} {'CV':>6}")
    for cond in CONDITIONS:
        pc = per_condition[cond]
        print(f"{cond:<28} {pc['n_runs']:>3} {pc['n_significant_pairs']:>4} "
              f"{pc['n_tested']:>6} {pc['token_cv']:>6.3f}")

    # ── Permutation test: nonsense vs baseline ──────────────────────
    print(f"\n── Permutation Test: Nonsense vs Baseline ──")
    nonsense_runs = grouped.get("nonsense_control", [])
    baseline_runs = grouped.get("baseline", [])

    if nonsense_runs and baseline_runs:
        n_non, n_base, diff, perm_p = permutation_test_pair_counts(
            nonsense_runs, baseline_runs)
        results["permutation_nonsense_vs_baseline"] = {
            "nonsense_sig_pairs": n_non,
            "baseline_sig_pairs": n_base,
            "difference": diff,
            "permutation_p": perm_p,
            "significant": bool(perm_p < 0.05),
        }
        print(f"  Nonsense: {n_non} significant pairs")
        print(f"  Baseline: {n_base} significant pairs")
        print(f"  Difference: {diff}")
        print(f"  Permutation p = {perm_p:.4f}")
        if perm_p < 0.05:
            print("  ** Difference IS significant")
        else:
            print("  Difference is NOT significant — consistent with both being")
            print("  driven by the same residual length variation mechanism")

    # ── All pairwise permutation tests (200 permutations for speed) ──
    N_PAIRWISE_PERM = 200
    print(f"\n── All Pairwise Permutation Tests (n_perm={N_PAIRWISE_PERM}) ──")
    pairwise = []
    for i, c1 in enumerate(CONDITIONS):
        for c2 in CONDITIONS[i + 1:]:
            r1 = grouped.get(c1, [])
            r2 = grouped.get(c2, [])
            if r1 and r2:
                n1, n2, d, p = permutation_test_pair_counts(r1, r2, n_perm=N_PAIRWISE_PERM)
                pair_result = {
                    "condition_1": c1,
                    "condition_2": c2,
                    "sig_pairs_1": n1,
                    "sig_pairs_2": n2,
                    "difference": d,
                    "permutation_p": p,
                    "significant": bool(p < 0.05),
                }
                pairwise.append(pair_result)
                flag = "*" if p < 0.05 else " "
                print(f"  {c1:<24} ({n1:>2}) vs {c2:<24} ({n2:>2})  "
                      f"Δ={d:>+3}  p={p:.4f} {flag}")

    results["pairwise_permutation"] = pairwise
    n_sig_pairs = sum(1 for p in pairwise if p["significant"])
    print(f"\n  {n_sig_pairs}/{len(pairwise)} pairwise differences significant at α=0.05")

    # ── R12: Spearman correlation of sig pairs vs token CV ──────────
    print(f"\n── R12: Spearman ρ (# Sig Pairs vs Token CV) ──")
    sig_counts = [per_condition[c]["n_significant_pairs"] for c in CONDITIONS]
    cvs = [per_condition[c]["token_cv"] for c in CONDITIONS]

    # Filter NaN
    valid = [(s, c) for s, c in zip(sig_counts, cvs) if np.isfinite(c)]
    if len(valid) >= 4:
        s_vals, c_vals = zip(*valid)
        rho, p_spearman = stats.spearmanr(s_vals, c_vals)
        results["cv_correlation"] = {
            "spearman_rho": float(rho),
            "p_value": float(p_spearman),
            "n": len(valid),
            "significant": bool(p_spearman < 0.05),
        }
        print(f"  Spearman ρ = {rho:.3f}, p = {p_spearman:.4f} (N = {len(valid)})")
        if p_spearman < 0.05:
            print("  * Significant: token CV predicts number of significant pairs")
        else:
            print(f"  Not significant at N={len(valid)}, but direction "
                  f"{'positive' if rho > 0 else 'negative'}")
    else:
        print("  Insufficient valid data for Spearman correlation")
        results["cv_correlation"] = {"error": "insufficient data"}

    # ── Save ────────────────────────────────────────────────────────
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT}")


if __name__ == "__main__":
    main()
