#!/usr/bin/env python3
"""R2 v2: Corrected F-statistics from activation vectors.

Fixes two bugs in the original revision_r02_f_statistics_permutation.py:
1. Uses norm(mean_vector_per_run) instead of mean_norm from JSON
2. Uses cosine(run_mean, grand_mean) instead of mean_token_similarity

Computes standard one-way ANOVA F-statistics (with n-per-group multiplier)
and 10K permutation tests at each layer.

Data: outputs/runs/phase_d_layer_sweep/vectors/*.npy
Output: outputs/analysis/revision_r02_f_statistics_v2.json
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
VEC_DIR = PROJECT / "outputs" / "runs" / "phase_d_layer_sweep" / "vectors"
RESULTS_PATH = PROJECT / "outputs" / "runs" / "phase_d_layer_sweep" / "layer_sweep_results.json"
OUT = PROJECT / "outputs" / "analysis" / "revision_r02_f_statistics_v2.json"

LAYERS = [2, 3, 4, 5, 6, 8, 16, 32]
CONDITIONS = [
    "baseline", "abstract_philosophical", "factual_iterative",
    "procedural_self", "descriptive_forest", "descriptive_math",
    "descriptive_music", "nonsense_control",
]

N_PERMUTATIONS = 10000


def load_run_mean_vectors():
    """Load per-run mean activation vectors from .npy files.

    Returns dict: layer -> condition -> list of mean vectors.
    """
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    # Build index of runs per condition
    by_cond = defaultdict(list)
    for r in data["runs"]:
        by_cond[r["condition"]].append(r)

    cond_run_means = defaultdict(lambda: defaultdict(list))

    for cond in CONDITIONS:
        runs = by_cond.get(cond, [])
        for layer in LAYERS:
            for r in runs:
                if r["n_tokens"] < 100:
                    continue
                npy_path = VEC_DIR / f"{cond}_{r['run']:03d}_layer{layer}.npy"
                if not npy_path.exists():
                    continue
                vecs = np.load(npy_path, mmap_mode="r")
                run_mean = np.mean(vecs, axis=0).astype(np.float64)
                cond_run_means[layer][cond].append(run_mean)

    return cond_run_means


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def compute_f_statistic(groups):
    """Standard one-way ANOVA F-statistic.

    groups: list of 1D arrays (one per condition).
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
    """Permutation test for one-way F-statistic."""
    observed_F, _, _, _ = compute_f_statistic(groups)
    if np.isnan(observed_F):
        return float("nan"), float("nan")

    all_vals = np.concatenate(groups)
    group_sizes = [len(g) for g in groups]

    rng = np.random.default_rng(42)
    count_ge = 0

    for _ in range(n_perm):
        perm = rng.permutation(all_vals)
        perm_groups = []
        start = 0
        for size in group_sizes:
            perm_groups.append(perm[start : start + size])
            start += size
        F_perm, _, _, _ = compute_f_statistic(perm_groups)
        if F_perm >= observed_F:
            count_ge += 1

    perm_p = (count_ge + 1) / (n_perm + 1)
    return float(observed_F), float(perm_p)


def main():
    print("Loading activation vectors from .npy files...")
    cond_run_means = load_run_mean_vectors()

    # Report what we loaded
    for layer in LAYERS:
        counts = {c: len(cond_run_means[layer][c]) for c in CONDITIONS}
        total = sum(counts.values())
        print(f"  Layer {layer}: {total} runs ({counts})")

    results = {
        "description": "R2 v2: Corrected F-statistics from activation vectors",
        "data_source": str(VEC_DIR),
        "n_permutations": N_PERMUTATIONS,
        "metrics": {
            "f_norm": "norm(mean_vector_per_run) — activation magnitude",
            "f_cosine": "cosine(run_mean, grand_mean) — directional alignment",
        },
        "layers": {},
    }

    print()
    print("=" * 80)
    print("R2 v2: CORRECTED F-STATISTICS FROM ACTIVATION VECTORS")
    print("=" * 80)
    print(f"Permutations: {N_PERMUTATIONS}")

    # Critical values
    for alpha in [0.05, 0.01, 0.001]:
        f_crit = float(stats.f.ppf(1 - alpha, 7, 16))
        print(f"  F(7,16) critical at alpha={alpha}: {f_crit:.3f}")

    print()
    print(
        f"{'Layer':>5} {'Depth%':>6} | {'F_norm':>7} {'p_param':>8} {'p_perm':>8} | "
        f"{'F_cos':>7} {'p_param':>8} {'p_perm':>8} | {'df_b':>4} {'df_w':>4}"
    )
    print("-" * 95)

    for layer in LAYERS:
        depth_pct = layer / 64 * 100
        layer_data = cond_run_means[layer]

        # Compute grand mean vector for cosine metric
        all_vectors = []
        for cond in CONDITIONS:
            all_vectors.extend(layer_data.get(cond, []))
        if not all_vectors:
            print(f"{layer:>5} {depth_pct:>5.1f}% | NO DATA")
            continue
        grand_mean_vec = np.mean(all_vectors, axis=0)

        # Build groups for F_norm: norm of each run's mean vector
        norm_groups = []
        cos_groups = []
        for cond in CONDITIONS:
            vecs = layer_data.get(cond, [])
            if len(vecs) >= 2:
                norms = np.array([float(np.linalg.norm(v)) for v in vecs])
                norm_groups.append(norms)
                cos_vals = np.array([cosine_sim(v, grand_mean_vec) for v in vecs])
                cos_groups.append(cos_vals)

        layer_result = {"layer": layer, "depth_pct": float(depth_pct)}

        # F_norm
        if len(norm_groups) >= 3:
            F_norm, p_norm_param, df_b, df_w = compute_f_statistic(norm_groups)
            _, p_norm_perm = permutation_f_test(norm_groups)
            layer_result["f_norm"] = {
                "F": float(F_norm),
                "p_parametric": float(p_norm_param),
                "p_permutation": float(p_norm_perm),
                "df_between": int(df_b),
                "df_within": int(df_w),
                "significant_005": bool(p_norm_param < 0.05),
                "significant_001": bool(p_norm_param < 0.01),
                "perm_significant_005": bool(p_norm_perm < 0.05),
                "perm_significant_001": bool(p_norm_perm < 0.01),
            }
        else:
            F_norm = p_norm_param = p_norm_perm = float("nan")
            df_b = df_w = 0
            layer_result["f_norm"] = {"F": None, "error": "insufficient groups"}

        # F_cosine
        if len(cos_groups) >= 3:
            F_cos, p_cos_param, df_b_c, df_w_c = compute_f_statistic(cos_groups)
            _, p_cos_perm = permutation_f_test(cos_groups)
            layer_result["f_cosine"] = {
                "F": float(F_cos),
                "p_parametric": float(p_cos_param),
                "p_permutation": float(p_cos_perm),
                "df_between": int(df_b_c),
                "df_within": int(df_w_c),
                "significant_005": bool(p_cos_param < 0.05),
                "significant_001": bool(p_cos_param < 0.01),
                "perm_significant_005": bool(p_cos_perm < 0.05),
                "perm_significant_001": bool(p_cos_perm < 0.01),
            }
        else:
            F_cos = p_cos_param = p_cos_perm = float("nan")
            layer_result["f_cosine"] = {"F": None, "error": "insufficient groups"}

        results["layers"][str(layer)] = layer_result

        # Significance markers
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

        print(
            f"{layer:>5} {depth_pct:>5.1f}% | {F_norm:>7.3f} {p_norm_param:>8.4f} "
            f"{p_norm_perm:>8.4f} | {F_cos:>7.3f} {p_cos_param:>8.4f} "
            f"{p_cos_perm:>8.4f} | {df_b:>4} {df_w:>4}  {sig_norm} {sig_cos}"
        )

    # Interpretation
    print("\n-- Interpretation --")
    layer6 = results["layers"].get("6", {})
    f6_norm = layer6.get("f_norm", {})
    f6_cos = layer6.get("f_cosine", {})
    if f6_norm.get("F") is not None:
        print(f"Layer 6 (9.4% depth): F_norm = {f6_norm['F']:.3f}, p = {f6_norm['p_parametric']:.4f}")
        print(f"  Permutation p = {f6_norm['p_permutation']:.4f}")
    if f6_cos.get("F") is not None:
        print(f"Layer 6 (9.4% depth): F_cosine = {f6_cos['F']:.3f}, p = {f6_cos['p_parametric']:.4f}")
        print(f"  Permutation p = {f6_cos['p_permutation']:.4f}")

    # Save
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT}")


if __name__ == "__main__":
    main()
