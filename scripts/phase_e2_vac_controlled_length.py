#!/usr/bin/env python3
"""Phase E2: VAC test on controlled-length Phase D data.

Tests whether vocabulary-activation correlations survive when generation
length is controlled by design (early termination) and prompt content
is varied across 8 conditions.

Key questions:
1. Do any VAC correlations exist in the pooled Phase D data?
2. Are they prompt-specific (stronger for self-referential baseline)?
3. Do they survive partial correlation controlling for n_tokens?

Outputs:
  - outputs/analysis/phase_e2_vac_controlled.json
  - stdout summary
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
PHASE_D = PROJECT / "outputs" / "runs" / "phase_d_controls" / "phase_d_results.json"
OUT = PROJECT / "outputs" / "analysis" / "phase_e2_vac_controlled.json"

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


def partial_corr(x, y, z):
    """Partial correlation of x and y controlling for z."""
    x, y, z = np.array(x), np.array(y), np.array(z)
    if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0 or np.std(z) == 0:
        return float("nan"), float("nan")
    sl_xz, int_xz, _, _, _ = stats.linregress(z, x)
    sl_yz, int_yz, _, _, _ = stats.linregress(z, y)
    x_resid = x - (sl_xz * z + int_xz)
    y_resid = y - (sl_yz * z + int_yz)
    if np.std(x_resid) == 0 or np.std(y_resid) == 0:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(x_resid, y_resid)
    return float(r), float(p)


def load_data():
    with open(PHASE_D) as f:
        return json.load(f)


def compute_vac(runs, label="all"):
    """Compute all vocab × metric correlations for a set of runs."""
    if len(runs) < 5:
        return {"n": len(runs), "note": "too few runs", "correlations": []}

    vocab_keys = sorted(k for k in runs[0]["vocab_counts"].keys() if not k.startswith("ctrl_"))
    ctrl_keys = sorted(k for k in runs[0]["vocab_counts"].keys() if k.startswith("ctrl_"))
    all_vocab = vocab_keys + ctrl_keys

    metrics = [
        "mean_norm", "max_norm", "norm_std", "convergence_ratio",
        "mean_token_similarity", "spectral_power_low", "spectral_power_mid",
        "autocorr_lag1", "mean_derivative",
    ]
    layer = "8"

    n_tokens = np.array([r["n_tokens"] for r in runs], dtype=float)

    # Extract arrays
    vocab_arrays = {}
    for vk in all_vocab:
        vocab_arrays[vk] = np.array([r["vocab_counts"].get(vk, 0) for r in runs], dtype=float)

    metric_arrays = {}
    for mk in metrics:
        vals = []
        for r in runs:
            v = r.get("layer_metrics", {}).get(layer, {}).get(mk)
            vals.append(v if v is not None else np.nan)
        metric_arrays[mk] = np.array(vals, dtype=float)

    correlations = []
    for vk in all_vocab:
        for mk in metrics:
            v = vocab_arrays[vk]
            m = metric_arrays[mk]
            mask = np.isfinite(v) & np.isfinite(m) & np.isfinite(n_tokens)
            n = int(mask.sum())
            if n < 5:
                continue
            if np.std(v[mask]) == 0 or np.std(m[mask]) == 0:
                continue

            r_raw, p_raw = stats.pearsonr(v[mask], m[mask])
            r_part, p_part = partial_corr(v[mask], m[mask], n_tokens[mask])

            correlations.append({
                "vocab": vk,
                "metric": mk,
                "r_raw": float(r_raw),
                "p_raw": float(p_raw),
                "r_partial": float(r_part),
                "p_partial": float(p_part),
                "n": n,
            })

    sig_raw = [c for c in correlations if c["p_raw"] < 0.05]
    sig_partial = [c for c in correlations if not np.isnan(c["p_partial"]) and c["p_partial"] < 0.05]
    n_tests = len(correlations)
    bonf = 0.05 / n_tests if n_tests > 0 else 0.05
    sig_bonf = [c for c in correlations if c["p_raw"] < bonf]

    token_stats = {
        "mean": float(n_tokens.mean()),
        "std": float(n_tokens.std()),
        "min": int(n_tokens.min()),
        "max": int(n_tokens.max()),
        "cv": float(n_tokens.std() / n_tokens.mean()) if n_tokens.mean() > 0 else 0,
    }

    return {
        "label": label,
        "n": len(runs),
        "n_tests": n_tests,
        "token_stats": token_stats,
        "n_sig_raw": len(sig_raw),
        "n_sig_partial": len(sig_partial),
        "n_sig_bonferroni": len(sig_bonf),
        "sig_raw": sorted(sig_raw, key=lambda x: -abs(x["r_raw"])),
        "sig_partial": sorted(sig_partial, key=lambda x: -abs(x["r_partial"])),
        "all_correlations": correlations,
    }


def compare_conditions(groups):
    """Test whether specific VAC pairs differ in strength across conditions.
    For each pair that's significant in baseline, test if the correlation
    is significantly different in other conditions (Fisher z-test)."""
    results = []

    baseline_vac = compute_vac(groups["baseline"], "baseline")
    if not baseline_vac["sig_raw"]:
        return {"note": "no significant baseline correlations to compare", "pairs": []}

    for pair in baseline_vac["sig_raw"][:10]:
        vk, mk = pair["vocab"], pair["metric"]
        r_base = pair["r_raw"]
        n_base = pair["n"]

        condition_rs = {"baseline": {"r": r_base, "n": n_base}}
        for cond in CONDITIONS[1:]:
            cond_runs = groups[cond]
            # Find this pair in condition data
            vocab_vals = np.array([r["vocab_counts"].get(vk, 0) for r in cond_runs], dtype=float)
            metric_vals = []
            for r in cond_runs:
                v = r.get("layer_metrics", {}).get("8", {}).get(mk)
                metric_vals.append(v if v is not None else np.nan)
            metric_vals = np.array(metric_vals, dtype=float)
            mask = np.isfinite(vocab_vals) & np.isfinite(metric_vals)
            n = int(mask.sum())
            if n >= 5 and np.std(vocab_vals[mask]) > 0 and np.std(metric_vals[mask]) > 0:
                r, p = stats.pearsonr(vocab_vals[mask], metric_vals[mask])
                condition_rs[cond] = {"r": float(r), "n": n, "p": float(p)}
            else:
                condition_rs[cond] = {"r": None, "n": n, "p": None}

        results.append({
            "vocab": vk,
            "metric": mk,
            "baseline_r": r_base,
            "condition_correlations": condition_rs,
        })

    return {"pairs": results}


def main():
    print("Loading Phase D data...")
    data = load_data()
    runs = data["runs"]
    groups = defaultdict(list)
    for r in runs:
        groups[r["condition"]].append(r)

    print(f"Total runs: {len(runs)}")
    n_tokens = [r["n_tokens"] for r in runs]
    print(f"Token range: {min(n_tokens)}-{max(n_tokens)} (mean={np.mean(n_tokens):.0f}, CV={np.std(n_tokens)/np.mean(n_tokens):.2f})")

    # === 1. POOLED ANALYSIS (all 80 runs) ===
    print("\n" + "=" * 80)
    print("POOLED VAC ANALYSIS (all 80 runs, 8 conditions)")
    print("=" * 80)
    pooled = compute_vac(runs, "pooled_all")

    ts = pooled["token_stats"]
    print(f"Token stats: mean={ts['mean']:.0f}, std={ts['std']:.0f}, CV={ts['cv']:.2f}")
    print(f"Tests performed: {pooled['n_tests']}")
    print(f"Significant raw (p<0.05): {pooled['n_sig_raw']}")
    print(f"Significant partial (p<0.05): {pooled['n_sig_partial']}")
    print(f"Significant Bonferroni: {pooled['n_sig_bonferroni']}")

    if pooled["sig_raw"]:
        print(f"\n{'Vocab':<20s} {'Metric':<25s} {'r_raw':>7s} {'p_raw':>10s} {'r_part':>7s} {'p_part':>10s}")
        print("-" * 85)
        for c in pooled["sig_raw"][:15]:
            print(f"{c['vocab']:<20s} {c['metric']:<25s} {c['r_raw']:+7.3f} {c['p_raw']:10.3e} "
                  f"{c['r_partial']:+7.3f} {c['p_partial']:10.3e}")

    # === 2. PER-CONDITION ANALYSIS ===
    print("\n" + "=" * 80)
    print("PER-CONDITION VAC ANALYSIS")
    print("=" * 80)
    per_condition = {}
    for cond in CONDITIONS:
        cond_vac = compute_vac(groups[cond], cond)
        per_condition[cond] = cond_vac
        label = CONDITION_LABELS.get(cond, cond)
        print(f"\n  {label:<25s}  N={cond_vac['n']:>2d}  "
              f"sig_raw={cond_vac['n_sig_raw']:>2d}  "
              f"sig_partial={cond_vac['n_sig_partial']:>2d}  "
              f"token_CV={cond_vac['token_stats']['cv']:.2f}")
        if cond_vac["sig_raw"]:
            for c in cond_vac["sig_raw"][:3]:
                print(f"    {c['vocab']:<18s} × {c['metric']:<22s}  r={c['r_raw']:+.3f}  p={c['p_raw']:.3e}")

    # === 3. BASELINE-SPECIFIC vs UNIVERSAL ===
    print("\n" + "=" * 80)
    print("PROMPT-SPECIFICITY TEST")
    print("=" * 80)
    print("Are baseline-significant VAC pairs also significant in other conditions?")
    comparison = compare_conditions(groups)

    if comparison["pairs"]:
        for pair in comparison["pairs"]:
            print(f"\n  {pair['vocab']} × {pair['metric']}:")
            for cond in CONDITIONS:
                cr = pair["condition_correlations"].get(cond)
                if cr and cr["r"] is not None:
                    label = CONDITION_LABELS.get(cond, cond)
                    sig = "*" if cr.get("p") and cr["p"] < 0.05 else ""
                    print(f"    {label:<25s}  r={cr['r']:+.3f}  {sig}")
                else:
                    label = CONDITION_LABELS.get(cond, cond)
                    print(f"    {label:<25s}  insufficient data")
    else:
        print("  No significant baseline correlations to compare.")

    # === 4. SELF-REF vs NON-SELF-REF SPLIT ===
    print("\n" + "=" * 80)
    print("SELF-REFERENTIAL vs NON-SELF-REFERENTIAL")
    print("=" * 80)
    self_ref_conds = ["baseline", "procedural_self"]
    non_self_ref_conds = [c for c in CONDITIONS if c not in self_ref_conds]

    self_ref_runs = [r for r in runs if r["condition"] in self_ref_conds]
    non_self_ref_runs = [r for r in runs if r["condition"] in non_self_ref_conds]

    self_ref_vac = compute_vac(self_ref_runs, "self_referential")
    non_self_ref_vac = compute_vac(non_self_ref_runs, "non_self_referential")

    print(f"\nSelf-referential (N={self_ref_vac['n']}): "
          f"sig_raw={self_ref_vac['n_sig_raw']}, sig_partial={self_ref_vac['n_sig_partial']}")
    print(f"Non-self-referential (N={non_self_ref_vac['n']}): "
          f"sig_raw={non_self_ref_vac['n_sig_raw']}, sig_partial={non_self_ref_vac['n_sig_partial']}")

    if self_ref_vac["sig_raw"]:
        print(f"\n  Self-ref top correlations:")
        for c in self_ref_vac["sig_raw"][:5]:
            print(f"    {c['vocab']:<18s} × {c['metric']:<22s}  r={c['r_raw']:+.3f}  p={c['p_raw']:.3e}")

    if non_self_ref_vac["sig_raw"]:
        print(f"\n  Non-self-ref top correlations:")
        for c in non_self_ref_vac["sig_raw"][:5]:
            print(f"    {c['vocab']:<18s} × {c['metric']:<22s}  r={c['r_raw']:+.3f}  p={c['p_raw']:.3e}")

    # === SAVE ===
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "pooled": pooled,
        "per_condition": per_condition,
        "prompt_specificity": comparison,
        "self_ref_vs_non": {
            "self_referential": self_ref_vac,
            "non_self_referential": non_self_ref_vac,
        },
    }
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT}")


if __name__ == "__main__":
    main()
