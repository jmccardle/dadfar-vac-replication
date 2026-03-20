#!/usr/bin/env python3
"""Phase B3/B4: Spectral power length confound and VAC partial correlation analysis.

B3: Demonstrates superlinear scaling of spectral_power_low with generation length.
B4: Replicates Dadfar's VAC correlations from Zenodo data, then tests whether
    they survive partial correlation controlling for n_tokens.

Outputs:
  - outputs/analysis/phase_b_spectral_vac.json  (full results)
  - stdout summary
"""

import json
import sys
from pathlib import Path
import numpy as np
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
ZENODO = PROJECT / "zenodo" / "data" / "qwen_baseline_n50.json"
EXTENDED = PROJECT / "outputs" / "runs" / "extended_baseline" / "extended_baseline_results.json"
OUT = PROJECT / "outputs" / "analysis" / "phase_b_spectral_vac.json"


def load_data():
    with open(ZENODO) as f:
        zenodo = json.load(f)
    with open(EXTENDED) as f:
        extended = json.load(f)
    return zenodo, extended


def partial_corr(x, y, z):
    """Partial correlation of x and y controlling for z.
    Returns (r_partial, p_value)."""
    x, y, z = np.array(x), np.array(y), np.array(z)
    # Residualise x and y on z
    slope_xz, intercept_xz, _, _, _ = stats.linregress(z, x)
    slope_yz, intercept_yz, _, _, _ = stats.linregress(z, y)
    x_resid = x - (slope_xz * z + intercept_xz)
    y_resid = y - (slope_yz * z + intercept_yz)
    r, p = stats.pearsonr(x_resid, y_resid)
    return float(r), float(p)


def b3_spectral_scaling(zenodo_runs, extended_runs):
    """B3: Regression of spectral_power_low ~ n_tokens across datasets."""
    results = {}

    # --- Zenodo data (all 50 runs, Layer 8) ---
    z_tokens = []
    z_spectral = []
    z_spectral_per_tok = []
    for r in zenodo_runs:
        nt = r["n_tokens"]
        sp = r["layer_metrics"]["8"]["spectral_power_low"]
        z_tokens.append(nt)
        z_spectral.append(sp)
        z_spectral_per_tok.append(sp / nt if nt > 0 else 0)

    z_tokens = np.array(z_tokens, dtype=float)
    z_spectral = np.array(z_spectral, dtype=float)

    # Log-log regression: log(spectral) = alpha * log(tokens) + beta
    mask = (z_tokens > 0) & (z_spectral > 0)
    log_tok = np.log10(z_tokens[mask])
    log_sp = np.log10(z_spectral[mask])
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_tok, log_sp)

    results["zenodo_scaling"] = {
        "n": int(mask.sum()),
        "alpha": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value**2),
        "p_value": float(p_value),
        "std_err": float(std_err),
        "interpretation": (
            f"spectral_power_low ~ n_tokens^{slope:.2f} "
            f"(R²={r_value**2:.3f}, p={p_value:.2e}). "
            f"Alpha>1 confirms superlinear scaling."
        ),
    }

    # Token count range
    results["zenodo_token_range"] = {
        "min": int(z_tokens.min()),
        "max": int(z_tokens.max()),
        "ratio": float(z_tokens.max() / z_tokens.min()),
    }

    # Pearson correlation: raw spectral_power_low vs n_tokens
    r_raw, p_raw = stats.pearsonr(z_tokens[mask], z_spectral[mask])
    results["zenodo_spectral_vs_tokens"] = {
        "r": float(r_raw),
        "p": float(p_raw),
    }

    # Per-token spectral vs tokens (should still correlate if alpha > 1)
    z_spt = z_spectral[mask] / z_tokens[mask]
    r_pt, p_pt = stats.pearsonr(z_tokens[mask], z_spt)
    results["zenodo_spectral_per_token_vs_tokens"] = {
        "r": float(r_pt),
        "p": float(p_pt),
        "interpretation": "Per-token normalisation does not remove length dependence" if p_pt < 0.05 else "Per-token normalisation appears sufficient",
    }

    # --- Extended baseline (Mode A only, Layer 8) ---
    e_tokens = []
    e_spectral = []
    for r in extended_runs:
        nt = r["n_tokens"]
        sp = r["layer_metrics"]["8"]["spectral_power_low"]
        # Mode A = long runs (>2000 tokens)
        if nt > 2000:
            e_tokens.append(nt)
            e_spectral.append(sp)

    e_tokens = np.array(e_tokens, dtype=float)
    e_spectral = np.array(e_spectral, dtype=float)

    if len(e_tokens) > 5:
        mask_e = (e_tokens > 0) & (e_spectral > 0)
        log_tok_e = np.log10(e_tokens[mask_e])
        log_sp_e = np.log10(e_spectral[mask_e])
        sl_e, int_e, rv_e, pv_e, se_e = stats.linregress(log_tok_e, log_sp_e)

        results["extended_scaling_mode_a"] = {
            "n": int(mask_e.sum()),
            "alpha": float(sl_e),
            "r_squared": float(rv_e**2),
            "p_value": float(pv_e),
            "token_range": [int(e_tokens.min()), int(e_tokens.max())],
        }

    return results


def b4_vac_correlations(zenodo_runs, extended_runs):
    """B4: Replicate and test VAC correlations with partial correlation control."""
    results = {}

    # Dadfar's three significant Qwen pairs (from paper):
    # mirror ↔ spectral_power_low (r=0.62)
    # expand ↔ spectral_power_low (r=0.58) — "expand" not in Zenodo vocab; closest is "surge" or check
    # resonance ↔ max_norm (r=0.54)
    # Note: Zenodo vocab keys include the Dadfar vocabulary categories

    # Extract arrays from Zenodo
    vocab_keys = sorted(zenodo_runs[0]["vocab_counts"].keys())
    n_tokens_z = np.array([r["n_tokens"] for r in zenodo_runs], dtype=float)

    # Layer 8 metrics
    layer = "8"
    metrics_of_interest = [
        "spectral_power_low", "spectral_power_mid",
        "mean_norm", "max_norm", "norm_std",
        "convergence_ratio", "mean_token_similarity",
        "autocorr_lag1", "mean_derivative",
    ]

    layer_metrics = {}
    for m in metrics_of_interest:
        layer_metrics[m] = np.array(
            [r["layer_metrics"][layer][m] for r in zenodo_runs], dtype=float
        )

    vocab_arrays = {}
    for k in vocab_keys:
        vocab_arrays[k] = np.array(
            [r["vocab_counts"][k] for r in zenodo_runs], dtype=float
        )

    # --- Full correlation matrix: all vocab × all metrics ---
    all_correlations = []
    for vk in vocab_keys:
        for mk in metrics_of_interest:
            v = vocab_arrays[vk]
            m = layer_metrics[mk]
            mask = np.isfinite(v) & np.isfinite(m) & np.isfinite(n_tokens_z)
            if mask.sum() < 5:
                continue
            r_raw, p_raw = stats.pearsonr(v[mask], m[mask])
            r_part, p_part = partial_corr(v[mask], m[mask], n_tokens_z[mask])
            all_correlations.append({
                "vocab": vk,
                "metric": mk,
                "r_raw": float(r_raw),
                "p_raw": float(p_raw),
                "r_partial": float(r_part),
                "p_partial": float(p_part),
                "n": int(mask.sum()),
                "survives_bonferroni": float(p_part) < 0.05 / (len(vocab_keys) * len(metrics_of_interest)),
            })

    results["all_correlations"] = all_correlations

    # --- Highlight: significant raw correlations and their partial results ---
    sig_raw = [c for c in all_correlations if c["p_raw"] < 0.05]
    sig_raw.sort(key=lambda x: -abs(x["r_raw"]))
    results["significant_raw"] = sig_raw

    sig_partial = [c for c in all_correlations if c["p_partial"] < 0.05]
    sig_partial.sort(key=lambda x: -abs(x["r_partial"]))
    results["significant_partial"] = sig_partial

    # --- Vocab counts vs n_tokens (confound check) ---
    vocab_vs_length = {}
    for vk in vocab_keys:
        v = vocab_arrays[vk]
        mask = np.isfinite(v)
        r, p = stats.pearsonr(v[mask], n_tokens_z[mask])
        vocab_vs_length[vk] = {"r": float(r), "p": float(p)}
    results["vocab_vs_n_tokens"] = vocab_vs_length

    # --- Metrics vs n_tokens ---
    metric_vs_length = {}
    for mk in metrics_of_interest:
        m = layer_metrics[mk]
        mask = np.isfinite(m)
        r, p = stats.pearsonr(m[mask], n_tokens_z[mask])
        metric_vs_length[mk] = {"r": float(r), "p": float(p)}
    results["metric_vs_n_tokens"] = metric_vs_length

    # --- Extended baseline: Mode A only (uniform length, no confound) ---
    ext_mode_a = [r for r in extended_runs if r["n_tokens"] > 2000]
    if len(ext_mode_a) >= 10:
        e_n_tokens = np.array([r["n_tokens"] for r in ext_mode_a], dtype=float)
        e_layer_metrics = {}
        for mk in metrics_of_interest:
            e_layer_metrics[mk] = np.array(
                [r["layer_metrics"][layer].get(mk, np.nan) for r in ext_mode_a], dtype=float
            )

        # Extended baseline uses potentially different vocab keys
        ext_vocab_keys = sorted(ext_mode_a[0]["vocab_counts"].keys())
        e_vocab = {}
        for vk in ext_vocab_keys:
            e_vocab[vk] = np.array([r["vocab_counts"][vk] for r in ext_mode_a], dtype=float)

        # Common vocab keys
        common_vocab = sorted(set(vocab_keys) & set(ext_vocab_keys))
        common_metrics = [mk for mk in metrics_of_interest if mk in e_layer_metrics]

        ext_correlations = []
        for vk in common_vocab:
            for mk in common_metrics:
                v = e_vocab[vk]
                m = e_layer_metrics[mk]
                mask = np.isfinite(v) & np.isfinite(m)
                if mask.sum() < 5:
                    continue
                r_raw, p_raw = stats.pearsonr(v[mask], m[mask])
                r_part, p_part = partial_corr(v[mask], m[mask], e_n_tokens[mask])
                ext_correlations.append({
                    "vocab": vk,
                    "metric": mk,
                    "r_raw": float(r_raw),
                    "p_raw": float(p_raw),
                    "r_partial": float(r_part),
                    "p_partial": float(p_part),
                    "n": int(mask.sum()),
                })

        results["extended_mode_a"] = {
            "n_runs": len(ext_mode_a),
            "token_range": [int(e_n_tokens.min()), int(e_n_tokens.max())],
            "token_cv": float(e_n_tokens.std() / e_n_tokens.mean()),
            "correlations": ext_correlations,
            "sig_raw": [c for c in ext_correlations if c["p_raw"] < 0.05],
            "sig_partial": [c for c in ext_correlations if c["p_partial"] < 0.05],
        }

    return results


def main():
    print("Loading data...")
    zenodo, extended = load_data()
    zenodo_runs = zenodo["runs"]
    extended_runs = extended["runs"]

    print(f"Zenodo: {len(zenodo_runs)} runs")
    print(f"Extended: {len(extended_runs)} runs")

    print("\n" + "="*70)
    print("B3: SPECTRAL POWER LENGTH CONFOUND")
    print("="*70)
    b3 = b3_spectral_scaling(zenodo_runs, extended_runs)

    zs = b3["zenodo_scaling"]
    print(f"\nZenodo log-log regression (N={zs['n']}):")
    print(f"  spectral_power_low ~ n_tokens^{zs['alpha']:.2f}")
    print(f"  R² = {zs['r_squared']:.3f}, p = {zs['p_value']:.2e}")
    print(f"  Token range: {b3['zenodo_token_range']['min']}-{b3['zenodo_token_range']['max']} ({b3['zenodo_token_range']['ratio']:.0f}× ratio)")

    zst = b3["zenodo_spectral_vs_tokens"]
    print(f"\nRaw spectral_power_low vs n_tokens: r={zst['r']:.3f}, p={zst['p']:.2e}")

    zpt = b3["zenodo_spectral_per_token_vs_tokens"]
    print(f"Per-token spectral vs n_tokens:      r={zpt['r']:.3f}, p={zpt['p']:.2e}")

    if "extended_scaling_mode_a" in b3:
        es = b3["extended_scaling_mode_a"]
        print(f"\nExtended Mode A log-log regression (N={es['n']}):")
        print(f"  alpha = {es['alpha']:.2f}, R² = {es['r_squared']:.3f}, p = {es['p_value']:.2e}")
        print(f"  Token range: {es['token_range'][0]}-{es['token_range'][1]}")

    print("\n" + "="*70)
    print("B4: VAC CORRELATIONS AND PARTIAL CORRELATIONS")
    print("="*70)
    b4 = b4_vac_correlations(zenodo_runs, extended_runs)

    # Vocab vs length confound
    print("\nVocab counts correlated with n_tokens:")
    vvl = b4["vocab_vs_n_tokens"]
    for vk in sorted(vvl.keys(), key=lambda k: -abs(vvl[k]["r"])):
        v = vvl[vk]
        sig = "***" if v["p"] < 0.001 else "**" if v["p"] < 0.01 else "*" if v["p"] < 0.05 else ""
        if abs(v["r"]) > 0.2 or v["p"] < 0.05:
            print(f"  {vk:20s}  r={v['r']:+.3f}  p={v['p']:.3e} {sig}")

    # Metric vs length confound
    print("\nMetrics correlated with n_tokens:")
    mvl = b4["metric_vs_n_tokens"]
    for mk in sorted(mvl.keys(), key=lambda k: -abs(mvl[k]["r"])):
        v = mvl[mk]
        sig = "***" if v["p"] < 0.001 else "**" if v["p"] < 0.01 else "*" if v["p"] < 0.05 else ""
        print(f"  {mk:25s}  r={v['r']:+.3f}  p={v['p']:.3e} {sig}")

    # Top raw correlations
    print(f"\nSignificant raw correlations (p<0.05): {len(b4['significant_raw'])}")
    print(f"{'Vocab':<20s} {'Metric':<25s} {'r_raw':>7s} {'p_raw':>10s} {'r_partial':>9s} {'p_partial':>10s} {'Survives?':>10s}")
    print("-" * 95)
    for c in b4["significant_raw"][:20]:
        surv = "YES" if c["p_partial"] < 0.05 else "no"
        print(f"{c['vocab']:<20s} {c['metric']:<25s} {c['r_raw']:+7.3f} {c['p_raw']:10.3e} {c['r_partial']:+9.3f} {c['p_partial']:10.3e} {surv:>10s}")

    # Summary
    n_sig_raw = len(b4["significant_raw"])
    n_sig_partial = len(b4["significant_partial"])
    print(f"\n** SUMMARY: {n_sig_raw} correlations significant raw -> {n_sig_partial} survive partial correlation controlling for n_tokens **")

    # Extended baseline
    if "extended_mode_a" in b4:
        ea = b4["extended_mode_a"]
        print(f"\nExtended baseline Mode A (N={ea['n_runs']}, token CV={ea['token_cv']:.3f}):")
        print(f"  Significant raw correlations: {len(ea['sig_raw'])}")
        print(f"  Significant partial correlations: {len(ea['sig_partial'])}")
        if ea["sig_raw"]:
            print(f"  Top raw:")
            for c in ea["sig_raw"][:5]:
                print(f"    {c['vocab']:<20s} × {c['metric']:<25s}  r={c['r_raw']:+.3f} p={c['p_raw']:.3e}")

    # Save
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results = {"b3_scaling": b3, "b4_vac": b4}
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT}")


if __name__ == "__main__":
    main()
