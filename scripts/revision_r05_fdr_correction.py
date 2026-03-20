#!/usr/bin/env python3
"""R5: Benjamini-Hochberg FDR Correction on 153 Partial Correlations.

Addresses reviewer Point 5: Section 5.6 tests 153 vocab-metric pairs without
multiple comparison correction. Expected false positives: 7.65.

Also validates the 7 suppressor effects with bootstrap CIs.

Data: outputs/analysis/phase_b_spectral_vac.json
      zenodo/data/qwen_baseline_n50.json (for bootstrap recomputation)
Output: outputs/analysis/revision_r05_fdr.json
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
DATA_VAC = PROJECT / "outputs" / "analysis" / "phase_b_spectral_vac.json"
DATA_ZENODO = PROJECT / "zenodo" / "data" / "qwen_baseline_n50.json"
OUT = PROJECT / "outputs" / "analysis" / "revision_r05_fdr.json"

N_BOOTSTRAP = 1000


def benjamini_hochberg(p_values, q=0.05):
    """Apply Benjamini-Hochberg FDR correction.

    Returns list of (original_index, p_raw, p_adjusted, rejected).
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    rejected = [False] * n

    # Step-up procedure
    prev_adj = 1.0
    for rank_idx in range(n - 1, -1, -1):
        orig_idx, p = indexed[rank_idx]
        rank = rank_idx + 1  # 1-based rank
        adj_p = min(p * n / rank, prev_adj)
        adjusted[orig_idx] = adj_p
        rejected[orig_idx] = adj_p < q
        prev_adj = adj_p

    return [(i, p_values[i], adjusted[i], rejected[i]) for i in range(n)]


def partial_corr(x, y, z):
    """Partial correlation of x and y controlling for z."""
    x, y, z = np.array(x), np.array(y), np.array(z)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    if len(x) < 5:
        return float("nan"), float("nan")
    slope_xz, intercept_xz, _, _, _ = stats.linregress(z, x)
    slope_yz, intercept_yz, _, _, _ = stats.linregress(z, y)
    x_resid = x - (slope_xz * z + intercept_xz)
    y_resid = y - (slope_yz * z + intercept_yz)
    r, p = stats.pearsonr(x_resid, y_resid)
    return float(r), float(p)


def bootstrap_partial_corr(x, y, z, n_boot=N_BOOTSTRAP):
    """Bootstrap CI for partial correlation."""
    x, y, z = np.array(x), np.array(y), np.array(z)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    n = len(x)
    if n < 5:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(42)
    boot_rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r, _ = partial_corr(x[idx], y[idx], z[idx])
        if np.isfinite(r):
            boot_rs.append(r)

    if len(boot_rs) < 100:
        return float("nan"), float("nan"), float("nan")

    boot_rs = np.array(boot_rs)
    ci_low = float(np.percentile(boot_rs, 2.5))
    ci_high = float(np.percentile(boot_rs, 97.5))
    boot_mean = float(np.mean(boot_rs))
    return boot_mean, ci_low, ci_high


def main():
    # ── Load pre-computed VAC results ───────────────────────────────
    with open(DATA_VAC) as f:
        vac_data = json.load(f)

    # ── Load Zenodo data for bootstrap recomputation ────────────────
    with open(DATA_ZENODO) as f:
        zenodo = json.load(f)
    zenodo_runs = zenodo.get("runs", zenodo.get("data", []))

    results = {
        "description": "R5: BH-FDR correction on 153 partial correlations + bootstrap validation of suppressors",
        "n_bootstrap": N_BOOTSTRAP,
        "fdr_q": 0.05,
    }

    print("=" * 72)
    print("R5: BH-FDR ON 153 PARTIAL CORRELATIONS + BOOTSTRAP VALIDATION")
    print("=" * 72)

    # ── Extract all partial correlation p-values ────────────────────
    # Look for the B4 section in the VAC data
    b4 = vac_data.get("b4_vac_partial_correlations", vac_data.get("b4", {}))

    # Try to find the correlation list
    all_correlations = b4.get("all_correlations", b4.get("correlations", []))

    if not all_correlations:
        # Try alternative structure: look through all keys for correlation data
        for key in vac_data:
            section = vac_data[key]
            if isinstance(section, dict):
                for subkey in section:
                    if "correlation" in subkey.lower() or "pair" in subkey.lower():
                        candidate = section[subkey]
                        if isinstance(candidate, list) and len(candidate) > 100:
                            all_correlations = candidate
                            break

    if not all_correlations:
        print("WARNING: Could not find correlation data in phase_b_spectral_vac.json")
        print("Available top-level keys:", list(vac_data.keys()))
        # Try to reconstruct from Zenodo data
        print("Reconstructing correlations from Zenodo data...")
        all_correlations = reconstruct_correlations(zenodo_runs)

    print(f"Total correlations found: {len(all_correlations)}")

    # ── Separate raw-significant, partial-significant, suppressors ──
    raw_sig = [c for c in all_correlations if c.get("p_raw", 1) < 0.05]
    partial_sig = [c for c in all_correlations if c.get("p_partial", 1) < 0.05]
    suppressors = [c for c in all_correlations
                   if c.get("p_raw", 1) >= 0.05 and c.get("p_partial", 1) < 0.05]

    print(f"Raw significant (p<0.05): {len(raw_sig)}")
    print(f"Partial significant (p<0.05): {len(partial_sig)}")
    print(f"Suppressors (raw NS, partial sig): {len(suppressors)}")

    # ── BH-FDR on partial correlation p-values ──────────────────────
    partial_ps = [c.get("p_partial", 1.0) for c in all_correlations]
    fdr_results = benjamini_hochberg(partial_ps, q=0.05)

    n_fdr_survivors = sum(1 for _, _, _, rej in fdr_results if rej)
    results["fdr_n_tested"] = len(all_correlations)
    results["fdr_n_surviving"] = n_fdr_survivors
    results["expected_false_positives"] = 0.05 * len(all_correlations)

    print(f"\n── BH-FDR Results (q=0.05) ──")
    print(f"  Tested: {len(all_correlations)}")
    print(f"  Expected false positives (uncorrected): {0.05 * len(all_correlations):.1f}")
    print(f"  Partial significant (uncorrected): {len(partial_sig)}")
    print(f"  Surviving FDR: {n_fdr_survivors}")

    # Build detailed survivor list
    survivors = []
    for i, (orig_idx, p_raw, p_adj, rejected) in enumerate(fdr_results):
        corr = all_correlations[orig_idx]
        corr_info = {
            "vocab": corr.get("vocab", ""),
            "metric": corr.get("metric", ""),
            "r_raw": corr.get("r_raw", float("nan")),
            "p_raw": corr.get("p_raw", float("nan")),
            "r_partial": corr.get("r_partial", float("nan")),
            "p_partial": corr.get("p_partial", float("nan")),
            "p_fdr_adjusted": p_adj,
            "survives_fdr": rejected,
            "is_suppressor": corr.get("p_raw", 1) >= 0.05 and corr.get("p_partial", 1) < 0.05,
        }
        if rejected:
            survivors.append(corr_info)

    results["fdr_survivors"] = sorted(survivors, key=lambda x: x["p_fdr_adjusted"])

    print(f"\n  FDR Survivors:")
    print(f"  {'Vocab':<16} {'Metric':<24} {'r_raw':>6} {'r_part':>7} "
          f"{'p_part':>8} {'p_adj':>8} {'Suppr?'}")
    for s in results["fdr_survivors"]:
        suppr = "SUPPR" if s["is_suppressor"] else ""
        print(f"  {s['vocab']:<16} {s['metric']:<24} {s['r_raw']:>6.3f} "
              f"{s['r_partial']:>7.3f} {s['p_partial']:>8.4f} "
              f"{s['p_fdr_adjusted']:>8.4f} {suppr}")

    # ── Bootstrap validation of suppressors ─────────────────────────
    print(f"\n── Bootstrap Validation of Suppressors (N={N_BOOTSTRAP}) ──")
    bootstrap_results = []

    if zenodo_runs and suppressors:
        # Extract vocab and metric arrays from Zenodo
        for sup in suppressors:
            vk = sup.get("vocab", "")
            mk = sup.get("metric", "")

            vocab_vals = [r.get("vocab_counts", {}).get(vk, 0) for r in zenodo_runs]
            n_tokens = [r.get("n_tokens", r.get("text_length", 0)) for r in zenodo_runs]

            # Find metric in layer 8
            metric_vals = []
            for r in zenodo_runs:
                lm = r.get("layer_metrics", {}).get("8", r.get("layer_8", {}))
                if isinstance(lm, dict):
                    metric_vals.append(lm.get(mk, float("nan")))
                else:
                    metric_vals.append(float("nan"))

            if len(vocab_vals) > 0 and len(metric_vals) > 0:
                boot_mean, ci_lo, ci_hi = bootstrap_partial_corr(
                    vocab_vals, metric_vals, n_tokens)

                boot_result = {
                    "vocab": vk,
                    "metric": mk,
                    "r_partial": sup.get("r_partial", float("nan")),
                    "bootstrap_mean": boot_mean,
                    "bootstrap_ci_low": ci_lo,
                    "bootstrap_ci_high": ci_hi,
                    "ci_includes_zero": ci_lo <= 0 <= ci_hi if np.isfinite(ci_lo) else True,
                    "validated": not (ci_lo <= 0 <= ci_hi) if np.isfinite(ci_lo) else False,
                }
                bootstrap_results.append(boot_result)

                flag = "VALID" if boot_result["validated"] else "FAIL"
                print(f"  {vk:<16} × {mk:<24}: r_partial={sup.get('r_partial', 0):.3f}  "
                      f"boot CI=[{ci_lo:.3f}, {ci_hi:.3f}]  [{flag}]")

    results["suppressor_bootstrap"] = bootstrap_results
    n_validated = sum(1 for b in bootstrap_results if b["validated"])
    print(f"\n  Suppressors validated: {n_validated}/{len(bootstrap_results)}")

    # ── Final count ─────────────────────────────────────────────────
    # Non-suppressor FDR survivors + validated suppressors
    non_supp_survivors = sum(1 for s in results["fdr_survivors"] if not s["is_suppressor"])
    supp_fdr_survivors = sum(1 for s in results["fdr_survivors"] if s["is_suppressor"])

    results["final_count"] = {
        "non_suppressor_fdr_survivors": non_supp_survivors,
        "suppressor_fdr_survivors": supp_fdr_survivors,
        "suppressors_bootstrap_validated": n_validated,
        "total_robust": non_supp_survivors + n_validated,
    }

    print(f"\n── Final Count ──")
    print(f"  Non-suppressor FDR survivors: {non_supp_survivors}")
    print(f"  Suppressor FDR survivors: {supp_fdr_survivors}")
    print(f"  Suppressors bootstrap-validated: {n_validated}")
    print(f"  Total robust partial correlations: {non_supp_survivors + n_validated}")

    # ── Save ────────────────────────────────────────────────────────
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT}")


def reconstruct_correlations(zenodo_runs):
    """Reconstruct all 153 vocab-metric correlations from Zenodo raw data."""
    VOCAB_KEYS = [
        "mirror", "expand", "resonance", "loop", "surge",
        "shimmer", "pulse", "void", "depth", "shift",
        "ctrl_the", "ctrl_and", "ctrl_question", "ctrl_what", "ctrl_that",
        "ctrl_processing", "ctrl_system",
    ]
    METRIC_KEYS = [
        "mean_norm", "max_norm", "norm_std", "autocorr_lag1",
        "spectral_power_low", "spectral_power_mid",
        "mean_derivative", "mean_token_similarity", "convergence_ratio",
    ]

    correlations = []
    n_tokens_vals = np.array([r.get("n_tokens", r.get("text_length", 0))
                              for r in zenodo_runs], dtype=float)

    for vk in VOCAB_KEYS:
        vocab_vals = np.array([r.get("vocab_counts", {}).get(vk, 0)
                               for r in zenodo_runs], dtype=float)
        for mk in METRIC_KEYS:
            metric_vals = []
            for r in zenodo_runs:
                lm = r.get("layer_metrics", {}).get("8", r.get("layer_8", {}))
                if isinstance(lm, dict):
                    metric_vals.append(lm.get(mk, float("nan")))
                else:
                    metric_vals.append(float("nan"))
            metric_vals = np.array(metric_vals, dtype=float)

            mask = np.isfinite(vocab_vals) & np.isfinite(metric_vals) & np.isfinite(n_tokens_vals)
            n = int(mask.sum())
            if n < 5:
                continue

            v, m, z = vocab_vals[mask], metric_vals[mask], n_tokens_vals[mask]
            r_raw, p_raw = stats.pearsonr(v, m)
            r_part, p_part = partial_corr(v.tolist(), m.tolist(), z.tolist())

            correlations.append({
                "vocab": vk,
                "metric": mk,
                "n": n,
                "r_raw": float(r_raw),
                "p_raw": float(p_raw),
                "r_partial": float(r_part),
                "p_partial": float(p_part),
            })

    return correlations


if __name__ == "__main__":
    main()
