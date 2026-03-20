#!/usr/bin/env python3
"""R2: F-statistic Exact P-values and Permutation Tests.

Addresses reviewer Point 2: F_norm=4.17 at Layer 6 labeled "Weak" but may
exceed F(7,16) critical value at α=0.01. Recomputes exact p-values and
runs distribution-free permutation tests at each layer.

Data: outputs/runs/phase_d_layer_sweep/layer_sweep_results.json
Output: outputs/analysis/revision_r02_f_permutation.json
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "outputs" / "runs" / "phase_d_layer_sweep" / "layer_sweep_results.json"
OUT = PROJECT / "outputs" / "analysis" / "revision_r02_f_permutation.json"

LAYERS = [2, 3, 4, 5, 6, 8, 16, 32]
CONDITIONS = [
    "baseline", "abstract_philosophical", "factual_iterative",
    "procedural_self", "descriptive_forest", "descriptive_math",
    "descriptive_music", "nonsense_control",
]

N_PERMUTATIONS = 10000


def load_layer_data():
    """Load per-run activation metrics grouped by condition and layer.

    Returns dict: layer -> condition -> list of metric dicts.
    """
    with open(DATA) as f:
        data = json.load(f)

    grouped = defaultdict(lambda: defaultdict(list))
    for run in data["runs"]:
        cond = run["condition"]
        for layer_str, metrics in run.get("layer_metrics", {}).items():
            layer = int(layer_str)
            grouped[layer][cond].append(metrics)

    return grouped


def compute_f_statistic(groups):
    """One-way F-statistic for list of arrays (groups).

    Returns (F, p, df_between, df_within).
    """
    k = len(groups)
    N = sum(len(g) for g in groups)
    df_between = k - 1
    df_within = N - k

    grand_mean = np.mean(np.concatenate(groups))

    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)

    if df_within == 0 or ss_within == 0:
        return float("nan"), float("nan"), df_between, df_within

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    F = ms_between / ms_within
    p = float(stats.f.sf(F, df_between, df_within))

    return float(F), p, df_between, df_within


def permutation_f_test(groups, n_perm=N_PERMUTATIONS):
    """Permutation test for one-way F-statistic.

    Shuffles condition labels, recomputes F, builds null distribution.
    Returns (observed_F, permutation_p, null_f_percentiles).
    """
    observed_F, _, _, _ = compute_f_statistic(groups)

    # Concatenate all values
    all_vals = np.concatenate(groups)
    group_sizes = [len(g) for g in groups]
    N = len(all_vals)

    rng = np.random.default_rng(42)
    count_ge = 0

    for _ in range(n_perm):
        perm = rng.permutation(all_vals)
        # Split into groups of same sizes
        perm_groups = []
        start = 0
        for size in group_sizes:
            perm_groups.append(perm[start:start + size])
            start += size
        F_perm, _, _, _ = compute_f_statistic(perm_groups)
        if F_perm >= observed_F:
            count_ge += 1

    perm_p = (count_ge + 1) / (n_perm + 1)  # +1 for observed

    return float(observed_F), float(perm_p)


def main():
    grouped = load_layer_data()

    results = {
        "description": "R2: F-statistic exact p-values and permutation tests",
        "data_source": str(DATA),
        "n_permutations": N_PERMUTATIONS,
        "layers": {},
    }

    print("=" * 72)
    print("R2: F-STATISTIC EXACT P-VALUES AND PERMUTATION TESTS")
    print("=" * 72)
    print(f"Permutations: {N_PERMUTATIONS}")

    # ── Critical values for reference ───────────────────────────────
    # F(7, 16) critical values
    for alpha in [0.05, 0.01, 0.001]:
        f_crit = float(stats.f.ppf(1 - alpha, 7, 16))
        print(f"  F(7,16) critical at α={alpha}: {f_crit:.3f}")

    print()
    print(f"{'Layer':>5} {'Depth%':>6} │ {'F_norm':>7} {'p_param':>8} {'p_perm':>8} │ "
          f"{'F_cos':>7} {'p_param':>8} {'p_perm':>8} │ {'df_b':>4} {'df_w':>4}")
    print("─" * 90)

    for layer in LAYERS:
        cond_data = grouped[layer]
        depth_pct = layer / 64 * 100

        # ── F for norm (mean_norm) ──────────────────────────────────
        norm_groups = []
        cos_groups = []
        for cond in CONDITIONS:
            runs = cond_data.get(cond, [])
            norms = [r["mean_norm"] for r in runs if "mean_norm" in r]
            if norms:
                norm_groups.append(np.array(norms))
            # Use mean_token_similarity as proxy for cosine direction metric
            cos_vals = [r.get("mean_token_similarity", r.get("convergence_ratio", 0))
                        for r in runs]
            if cos_vals:
                cos_groups.append(np.array(cos_vals))

        layer_result = {"layer": layer, "depth_pct": depth_pct}

        if norm_groups:
            F_norm, p_norm_param, df_b, df_w = compute_f_statistic(norm_groups)
            F_norm_obs, p_norm_perm = permutation_f_test(norm_groups)
            layer_result["f_norm"] = {
                "F": F_norm,
                "p_parametric": p_norm_param,
                "p_permutation": p_norm_perm,
                "df_between": df_b,
                "df_within": df_w,
                "significant_005": p_norm_param < 0.05,
                "significant_001": p_norm_param < 0.01,
                "perm_significant_005": p_norm_perm < 0.05,
                "perm_significant_001": p_norm_perm < 0.01,
            }
        else:
            F_norm, p_norm_param, p_norm_perm, df_b, df_w = (
                float("nan"), float("nan"), float("nan"), 0, 0)
            layer_result["f_norm"] = {"F": None, "error": "no data"}

        if cos_groups:
            F_cos, p_cos_param, df_b_c, df_w_c = compute_f_statistic(cos_groups)
            F_cos_obs, p_cos_perm = permutation_f_test(cos_groups)
            layer_result["f_cosine"] = {
                "F": F_cos,
                "p_parametric": p_cos_param,
                "p_permutation": p_cos_perm,
                "df_between": df_b_c,
                "df_within": df_w_c,
                "significant_005": p_cos_param < 0.05,
                "significant_001": p_cos_param < 0.01,
                "perm_significant_005": p_cos_perm < 0.05,
                "perm_significant_001": p_cos_perm < 0.01,
            }
        else:
            F_cos, p_cos_param, p_cos_perm = float("nan"), float("nan"), float("nan")
            layer_result["f_cosine"] = {"F": None, "error": "no data"}

        results["layers"][str(layer)] = layer_result

        # Print row
        sig_norm = ""
        if not np.isnan(p_norm_param):
            if p_norm_param < 0.001:
                sig_norm = "***"
            elif p_norm_param < 0.01:
                sig_norm = "**"
            elif p_norm_param < 0.05:
                sig_norm = "*"

        sig_cos = ""
        if not np.isnan(p_cos_param):
            if p_cos_param < 0.001:
                sig_cos = "***"
            elif p_cos_param < 0.01:
                sig_cos = "**"
            elif p_cos_param < 0.05:
                sig_cos = "*"

        print(f"{layer:>5} {depth_pct:>5.1f}% │ {F_norm:>7.3f} {p_norm_param:>8.4f} "
              f"{p_norm_perm:>8.4f} │ {F_cos:>7.3f} {p_cos_param:>8.4f} "
              f"{p_cos_perm:>8.4f} │ {df_b:>4} {df_w:>4}  {sig_norm}{sig_cos}")

    # ── Interpretation ──────────────────────────────────────────────
    print("\n── Interpretation ──")
    layer6 = results["layers"].get("6", {})
    f6 = layer6.get("f_norm", {})
    if f6.get("F") is not None:
        print(f"Layer 6 (9.4% depth): F_norm = {f6['F']:.3f}")
        print(f"  Parametric p = {f6['p_parametric']:.4f}")
        print(f"  Permutation p = {f6['p_permutation']:.4f}")
        if f6.get("significant_001"):
            print("  ** SIGNIFICANT at α=0.01 (parametric)")
            print("  Aaron's concern confirmed: this should NOT be labeled 'Weak'")
        elif f6.get("significant_005"):
            print("  * Significant at α=0.05 but not α=0.01")
        else:
            print("  Not significant at α=0.05")
        if f6.get("perm_significant_005"):
            print("  Permutation test also significant at α=0.05")
        else:
            print("  Permutation test NOT significant — parametric result unreliable at N=3")

    # ── Save ────────────────────────────────────────────────────────
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT}")


if __name__ == "__main__":
    main()
