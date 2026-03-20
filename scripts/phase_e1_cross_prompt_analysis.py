#!/usr/bin/env python3
"""Phase E1: Cross-prompt dynamical comparison.

Compares lock-in resistance, cycle periods, cycle rates, unique ratios,
activation metrics, and vocabulary distributions across 8 prompt conditions
from Phase D data.

Statistical tests: Kruskal-Wallis (non-parametric ANOVA), pairwise Mann-Whitney U
with Holm-Bonferroni correction, Cohen's d effect sizes.

Outputs:
  - outputs/analysis/phase_e1_cross_prompt.json  (full results)
  - stdout summary
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import numpy as np
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
PHASE_D = PROJECT / "outputs" / "runs" / "phase_d_controls" / "phase_d_results.json"
LAYER_SWEEP = PROJECT / "outputs" / "runs" / "phase_d_layer_sweep" / "layer_sweep_results.json"
OUT = PROJECT / "outputs" / "analysis" / "phase_e1_cross_prompt.json"

CONDITIONS = [
    "baseline", "abstract_philosophical", "factual_iterative",
    "procedural_self", "descriptive_forest", "descriptive_math",
    "descriptive_music", "nonsense_control",
]

CONDITION_LABELS = {
    "baseline": "Baseline (self-ref)",
    "abstract_philosophical": "Abstract-phil",
    "factual_iterative": "Factual-iter",
    "procedural_self": "Procedural-self",
    "descriptive_forest": "Desc-forest",
    "descriptive_math": "Desc-math",
    "descriptive_music": "Desc-music",
    "nonsense_control": "Nonsense",
}


def cohen_d(a, b):
    """Cohen's d (pooled SD) between two groups."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_std = np.sqrt(((na - 1) * a.std(ddof=1)**2 + (nb - 1) * b.std(ddof=1)**2) / (na + nb - 2))
    if pooled_std == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled_std)


def holm_bonferroni(p_values):
    """Apply Holm-Bonferroni correction. Returns adjusted p-values."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = min(p * (n - rank), 1.0)
        cummax = max(cummax, adj_p)
        adjusted[orig_idx] = cummax
    return adjusted


def load_data():
    with open(PHASE_D) as f:
        phase_d = json.load(f)
    sweep = None
    if LAYER_SWEEP.exists():
        with open(LAYER_SWEEP) as f:
            sweep = json.load(f)
    return phase_d, sweep


def group_by_condition(runs):
    """Group runs by condition."""
    groups = defaultdict(list)
    for r in runs:
        groups[r["condition"]].append(r)
    return dict(groups)


def analyze_lock_in(groups):
    """Compare lock-in resistance across conditions."""
    results = {}

    # Extract lock-in observations (None → use max_obs=300 as censored value)
    lock_in_by_cond = {}
    for cond in CONDITIONS:
        runs = groups.get(cond, [])
        lock_ins = []
        for r in runs:
            li = r["cycle"]["lock_in_obs"]
            if li is not None:
                lock_ins.append(li)
            else:
                # Censored: no cycle detected, use n_observations as lower bound
                lock_ins.append(r["n_observations"])
        lock_in_by_cond[cond] = lock_ins

    # Per-condition summary stats
    summaries = {}
    for cond in CONDITIONS:
        vals = np.array(lock_in_by_cond[cond], dtype=float)
        runs = groups.get(cond, [])
        n_cycles = sum(1 for r in runs if r["cycle"]["has_cycle"])
        summaries[cond] = {
            "n": len(vals),
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0,
            "min": float(vals.min()),
            "max": float(vals.max()),
            "cycle_rate": n_cycles / len(runs) if runs else 0,
            "n_cycles": n_cycles,
        }
    results["summaries"] = summaries

    # Kruskal-Wallis test (non-parametric one-way ANOVA)
    arrays = [np.array(lock_in_by_cond[c], dtype=float) for c in CONDITIONS if len(lock_in_by_cond[c]) > 0]
    if len(arrays) >= 2:
        H, p_kw = stats.kruskal(*arrays)
        results["kruskal_wallis"] = {"H": float(H), "p": float(p_kw), "df": len(arrays) - 1}

    # Pairwise comparisons: baseline vs each control
    pairwise = []
    for cond in CONDITIONS[1:]:
        a = np.array(lock_in_by_cond["baseline"], dtype=float)
        b = np.array(lock_in_by_cond[cond], dtype=float)
        if len(a) >= 2 and len(b) >= 2:
            U, p_mw = stats.mannwhitneyu(a, b, alternative="two-sided")
            d = cohen_d(a, b)
            pairwise.append({
                "comparison": f"baseline vs {cond}",
                "U": float(U),
                "p_raw": float(p_mw),
                "cohen_d": d,
                "baseline_median": float(np.median(a)),
                "control_median": float(np.median(b)),
            })

    # Apply Holm-Bonferroni
    if pairwise:
        raw_ps = [p["p_raw"] for p in pairwise]
        adj_ps = holm_bonferroni(raw_ps)
        for i, p in enumerate(pairwise):
            p["p_adjusted"] = adj_ps[i]
            p["significant"] = adj_ps[i] < 0.05

    results["pairwise_vs_baseline"] = pairwise

    # All pairwise comparisons
    all_pairs = []
    cond_pairs = list(combinations(CONDITIONS, 2))
    for c1, c2 in cond_pairs:
        a = np.array(lock_in_by_cond[c1], dtype=float)
        b = np.array(lock_in_by_cond[c2], dtype=float)
        if len(a) >= 2 and len(b) >= 2:
            U, p_mw = stats.mannwhitneyu(a, b, alternative="two-sided")
            d = cohen_d(a, b)
            all_pairs.append({
                "comparison": f"{c1} vs {c2}",
                "U": float(U),
                "p_raw": float(p_mw),
                "cohen_d": d,
            })

    if all_pairs:
        raw_ps = [p["p_raw"] for p in all_pairs]
        adj_ps = holm_bonferroni(raw_ps)
        for i, p in enumerate(all_pairs):
            p["p_adjusted"] = adj_ps[i]
            p["significant"] = adj_ps[i] < 0.05
        n_sig = sum(1 for p in all_pairs if p["significant"])
        results["all_pairwise"] = {
            "n_comparisons": len(all_pairs),
            "n_significant": n_sig,
            "pairs": all_pairs,
        }

    return results


def analyze_cycle_properties(groups):
    """Compare cycle periods and unique ratios across conditions."""
    results = {}

    for cond in CONDITIONS:
        runs = groups.get(cond, [])
        periods = [r["cycle"]["cycle_period"] for r in runs if r["cycle"]["cycle_period"] is not None]
        unique_ratios = [r["cycle"]["unique_ratio"] for r in runs if r["cycle"]["unique_ratio"] is not None]
        n_obs = [r["n_observations"] for r in runs]

        results[cond] = {
            "n_with_cycle": len(periods),
            "cycle_period": {
                "values": periods,
                "mean": float(np.mean(periods)) if periods else None,
                "median": float(np.median(periods)) if periods else None,
                "std": float(np.std(periods, ddof=1)) if len(periods) > 1 else None,
                "range": [int(min(periods)), int(max(periods))] if periods else None,
            },
            "unique_ratio": {
                "mean": float(np.mean(unique_ratios)) if unique_ratios else None,
                "median": float(np.median(unique_ratios)) if unique_ratios else None,
            },
            "n_observations": {
                "mean": float(np.mean(n_obs)),
                "median": float(np.median(n_obs)),
                "range": [int(min(n_obs)), int(max(n_obs))],
            },
        }

    # Kruskal-Wallis on cycle periods (among runs that have cycles)
    period_arrays = []
    period_conds = []
    for cond in CONDITIONS:
        runs = groups.get(cond, [])
        periods = [r["cycle"]["cycle_period"] for r in runs if r["cycle"]["cycle_period"] is not None]
        if len(periods) >= 2:
            period_arrays.append(np.array(periods, dtype=float))
            period_conds.append(cond)

    if len(period_arrays) >= 2:
        H, p = stats.kruskal(*period_arrays)
        results["cycle_period_kruskal"] = {"H": float(H), "p": float(p), "conditions_tested": period_conds}

    return results


def analyze_activation_metrics(groups):
    """Compare activation metrics across conditions at each layer."""
    results = {}

    layers = ["8", "16", "32"]
    metrics = [
        "mean_norm", "convergence_ratio", "mean_token_similarity",
        "spectral_power_low", "autocorr_lag1", "norm_std", "mean_derivative",
    ]

    for layer in layers:
        layer_results = {}
        for metric in metrics:
            # Extract per-condition
            cond_arrays = {}
            for cond in CONDITIONS:
                vals = []
                for r in groups.get(cond, []):
                    if layer in r.get("layer_metrics", {}):
                        v = r["layer_metrics"][layer].get(metric)
                        if v is not None and np.isfinite(v):
                            vals.append(v)
                cond_arrays[cond] = np.array(vals, dtype=float)

            # Per-condition summaries
            summaries = {}
            for cond in CONDITIONS:
                arr = cond_arrays[cond]
                if len(arr) > 0:
                    summaries[cond] = {
                        "n": len(arr),
                        "mean": float(arr.mean()),
                        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0,
                        "median": float(np.median(arr)),
                    }
            layer_results[metric] = {"summaries": summaries}

            # Kruskal-Wallis
            arrays = [cond_arrays[c] for c in CONDITIONS if len(cond_arrays[c]) >= 2]
            if len(arrays) >= 2:
                H, p = stats.kruskal(*arrays)
                layer_results[metric]["kruskal_wallis"] = {"H": float(H), "p": float(p)}

        results[f"layer_{layer}"] = layer_results

    return results


def analyze_vocabulary(groups):
    """Compare vocabulary distributions across conditions."""
    results = {}

    # Get all vocab keys from first run
    first_run = groups[CONDITIONS[0]][0]
    vocab_keys = sorted(first_run["vocab_counts"].keys())

    # Separate into Dadfar-style vocab and control words
    dadfar_vocab = [k for k in vocab_keys if not k.startswith("ctrl_")]
    ctrl_vocab = [k for k in vocab_keys if k.startswith("ctrl_")]

    for vk in dadfar_vocab:
        cond_arrays = {}
        for cond in CONDITIONS:
            vals = [r["vocab_counts"].get(vk, 0) for r in groups.get(cond, [])]
            cond_arrays[cond] = np.array(vals, dtype=float)

        summaries = {}
        for cond in CONDITIONS:
            arr = cond_arrays[cond]
            summaries[cond] = {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "total": int(arr.sum()),
            }

        # Kruskal-Wallis
        arrays = [cond_arrays[c] for c in CONDITIONS if len(cond_arrays[c]) >= 2]
        kw = None
        if len(arrays) >= 2:
            H, p = stats.kruskal(*arrays)
            kw = {"H": float(H), "p": float(p)}

        results[vk] = {"summaries": summaries, "kruskal_wallis": kw}

    # Total Dadfar vocab by condition
    total_dadfar = {}
    for cond in CONDITIONS:
        totals = []
        for r in groups.get(cond, []):
            t = sum(r["vocab_counts"].get(vk, 0) for vk in dadfar_vocab)
            totals.append(t)
        total_dadfar[cond] = {
            "mean": float(np.mean(totals)),
            "median": float(np.median(totals)),
        }
    results["_total_dadfar_vocab"] = total_dadfar

    return results


def main():
    print("Loading Phase D data...")
    phase_d, sweep = load_data()
    groups = group_by_condition(phase_d["runs"])

    print(f"Conditions: {len(groups)}")
    for cond in CONDITIONS:
        print(f"  {cond}: {len(groups.get(cond, []))} runs")

    # === LOCK-IN RESISTANCE ===
    print("\n" + "=" * 80)
    print("LOCK-IN RESISTANCE ANALYSIS")
    print("=" * 80)
    lock_in = analyze_lock_in(groups)

    print(f"\n{'Condition':<25s} {'N':>3s} {'Cycles':>7s} {'Rate':>5s} {'Median':>7s} {'Mean':>7s} {'Std':>7s} {'Range':>12s}")
    print("-" * 80)
    for cond in CONDITIONS:
        s = lock_in["summaries"][cond]
        print(f"{CONDITION_LABELS.get(cond, cond):<25s} {s['n']:>3d} {s['n_cycles']:>4d}/10 {s['cycle_rate']:>5.1%} "
              f"{s['median']:>7.0f} {s['mean']:>7.1f} {s['std']:>7.1f} {s['min']:>5.0f}-{s['max']:>5.0f}")

    if "kruskal_wallis" in lock_in:
        kw = lock_in["kruskal_wallis"]
        print(f"\nKruskal-Wallis: H={kw['H']:.2f}, p={kw['p']:.4f}, df={kw['df']}")

    print("\nPairwise: baseline vs each control (Holm-Bonferroni corrected):")
    print(f"{'Comparison':<40s} {'U':>6s} {'d':>7s} {'p_adj':>10s} {'Sig':>5s}")
    print("-" * 70)
    for p in lock_in["pairwise_vs_baseline"]:
        sig = "YES" if p["significant"] else "no"
        print(f"{p['comparison']:<40s} {p['U']:>6.0f} {p['cohen_d']:>+7.2f} {p['p_adjusted']:>10.4f} {sig:>5s}")

    # === CYCLE PROPERTIES ===
    print("\n" + "=" * 80)
    print("CYCLE PROPERTIES")
    print("=" * 80)
    cycles = analyze_cycle_properties(groups)

    print(f"\n{'Condition':<25s} {'Cycles':>7s} {'Period (med)':>12s} {'Period (range)':>15s} {'Unique ratio':>13s}")
    print("-" * 75)
    for cond in CONDITIONS:
        c = cycles[cond]
        cp = c["cycle_period"]
        ur = c["unique_ratio"]
        period_str = f"{cp['median']:.0f}" if cp["median"] else "—"
        range_str = f"{cp['range'][0]}-{cp['range'][1]}" if cp["range"] else "—"
        ur_str = f"{ur['median']:.3f}" if ur["median"] else "—"
        print(f"{CONDITION_LABELS.get(cond, cond):<25s} {c['n_with_cycle']:>4d}/10 {period_str:>12s} {range_str:>15s} {ur_str:>13s}")

    if "cycle_period_kruskal" in cycles:
        kw = cycles["cycle_period_kruskal"]
        print(f"\nCycle period Kruskal-Wallis: H={kw['H']:.2f}, p={kw['p']:.4f}")

    # === ACTIVATION METRICS ===
    print("\n" + "=" * 80)
    print("ACTIVATION METRICS (Layer 8)")
    print("=" * 80)
    activations = analyze_activation_metrics(groups)

    l8 = activations["layer_8"]
    sig_metrics = []
    for metric in l8:
        kw = l8[metric].get("kruskal_wallis")
        if kw and kw["p"] < 0.05:
            sig_metrics.append((metric, kw["H"], kw["p"]))
    sig_metrics.sort(key=lambda x: x[2])

    print(f"\nMetrics with significant cross-condition differences (p<0.05):")
    for m, H, p in sig_metrics:
        print(f"  {m:<30s}  H={H:.2f}  p={p:.4f}")
    if not sig_metrics:
        print("  None at p<0.05")

    # Show key metrics across conditions
    for metric in ["mean_norm", "convergence_ratio", "spectral_power_low"]:
        print(f"\n  {metric}:")
        for cond in CONDITIONS:
            s = l8[metric]["summaries"].get(cond)
            if s:
                print(f"    {CONDITION_LABELS.get(cond, cond):<25s}  mean={s['mean']:>12.2f}  std={s['std']:>10.2f}")

    # === VOCABULARY ===
    print("\n" + "=" * 80)
    print("VOCABULARY DISTRIBUTIONS")
    print("=" * 80)
    vocab = analyze_vocabulary(groups)

    # Show total Dadfar vocab
    print(f"\nTotal Dadfar vocab count (median):")
    td = vocab["_total_dadfar_vocab"]
    for cond in CONDITIONS:
        print(f"  {CONDITION_LABELS.get(cond, cond):<25s}  {td[cond]['median']:.0f}")

    # Vocab terms with significant cross-condition differences
    sig_vocab = []
    for vk in vocab:
        if vk.startswith("_"):
            continue
        kw = vocab[vk].get("kruskal_wallis")
        if kw and kw["p"] < 0.05:
            sig_vocab.append((vk, kw["H"], kw["p"]))
    sig_vocab.sort(key=lambda x: x[2])

    print(f"\nVocab terms with significant cross-condition differences:")
    for vk, H, p in sig_vocab:
        print(f"  {vk:<20s}  H={H:.2f}  p={p:.4f}")
    if not sig_vocab:
        print("  None at p<0.05")

    # === SAVE ===
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "lock_in_resistance": lock_in,
        "cycle_properties": cycles,
        "activation_metrics": activations,
        "vocabulary": vocab,
    }
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT}")


if __name__ == "__main__":
    main()
