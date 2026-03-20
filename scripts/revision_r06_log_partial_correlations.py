#!/usr/bin/env python3
"""R6: Log-Corrected Partial Correlations.

Addresses reviewer Point 6: spectral power scales as n_tokens^1.46, but
partial correlations use linear n_tokens control. This under-corrects.

Recomputes all partial correlations using log10(n_tokens) and compares
with linear correction.

Data: zenodo/data/qwen_baseline_n50.json
Output: outputs/analysis/revision_r06_log_partials.json
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "zenodo" / "data" / "qwen_baseline_n50.json"
OUT = PROJECT / "outputs" / "analysis" / "revision_r06_log_partials.json"

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


def partial_corr(x, y, z):
    """Partial correlation of x and y controlling for z."""
    x, y, z = np.array(x), np.array(y), np.array(z)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    if len(x) < 5:
        return float("nan"), float("nan"), int(mask.sum())
    slope_xz, intercept_xz, _, _, _ = stats.linregress(z, x)
    slope_yz, intercept_yz, _, _, _ = stats.linregress(z, y)
    x_resid = x - (slope_xz * z + intercept_xz)
    y_resid = y - (slope_yz * z + intercept_yz)
    r, p = stats.pearsonr(x_resid, y_resid)
    return float(r), float(p), int(mask.sum())


def main():
    with open(DATA) as f:
        zenodo = json.load(f)

    runs = zenodo.get("runs", zenodo.get("data", []))
    print(f"Loaded {len(runs)} Zenodo runs")

    # Extract n_tokens
    n_tokens = np.array([r.get("n_tokens", r.get("text_length", 0))
                         for r in runs], dtype=float)
    log_n_tokens = np.log10(np.maximum(n_tokens, 1))  # avoid log(0)

    results = {
        "description": "R6: Linear vs log-corrected partial correlations",
        "n_runs": len(runs),
        "comparisons": [],
    }

    print("=" * 72)
    print("R6: LOG-CORRECTED PARTIAL CORRELATIONS")
    print("=" * 72)

    print(f"\n{'Vocab':<16} {'Metric':<24} │ {'r_raw':>6} │ "
          f"{'r_lin':>6} {'p_lin':>8} │ {'r_log':>6} {'p_log':>8} │ {'Δr':>6} {'Verdict'}")
    print("─" * 110)

    key_pairs = []  # Track the paper's highlighted pairs

    for vk in VOCAB_KEYS:
        vocab_vals = np.array([r.get("vocab_counts", {}).get(vk, 0)
                               for r in runs], dtype=float)

        for mk in METRIC_KEYS:
            metric_vals = []
            for r in runs:
                lm = r.get("layer_metrics", {}).get("8", r.get("layer_8", {}))
                if isinstance(lm, dict):
                    metric_vals.append(lm.get(mk, float("nan")))
                else:
                    metric_vals.append(float("nan"))
            metric_vals = np.array(metric_vals, dtype=float)

            mask = np.isfinite(vocab_vals) & np.isfinite(metric_vals) & np.isfinite(n_tokens)
            if mask.sum() < 5:
                continue

            # Raw correlation
            r_raw, p_raw = stats.pearsonr(vocab_vals[mask], metric_vals[mask])

            # Linear partial
            r_lin, p_lin, n_lin = partial_corr(vocab_vals, metric_vals, n_tokens)

            # Log partial
            r_log, p_log, n_log = partial_corr(vocab_vals, metric_vals, log_n_tokens)

            delta_r = r_log - r_lin if np.isfinite(r_lin) and np.isfinite(r_log) else float("nan")

            # Determine if this is a key pair
            is_key = (vk in ["mirror", "resonance", "whisper", "pulse"] and
                      mk in ["spectral_power_low", "max_norm", "spectral_power_mid"])
            # Or if it was a highlighted paper pair
            if (vk == "mirror" and mk == "spectral_power_low") or \
               (vk == "resonance" and mk == "max_norm") or \
               (vk == "resonance" and mk == "spectral_power_mid"):
                is_key = True

            # Verdict
            if np.isnan(r_lin) or np.isnan(r_log):
                verdict = "---"
            elif p_lin < 0.05 and p_log >= 0.05:
                verdict = "LOST"
            elif p_lin < 0.05 and p_log < 0.05:
                verdict = "SURVIVES"
            elif p_lin >= 0.05 and p_log < 0.05:
                verdict = "GAINED"
            else:
                verdict = "ns"

            comp = {
                "vocab": vk,
                "metric": mk,
                "n": int(mask.sum()),
                "r_raw": float(r_raw),
                "p_raw": float(p_raw),
                "r_partial_linear": r_lin,
                "p_partial_linear": p_lin,
                "r_partial_log": r_log,
                "p_partial_log": p_log,
                "delta_r": delta_r,
                "linear_significant": p_lin < 0.05 if np.isfinite(p_lin) else False,
                "log_significant": p_log < 0.05 if np.isfinite(p_log) else False,
                "verdict": verdict,
            }
            results["comparisons"].append(comp)

            # Print key pairs or those that change status
            if is_key or verdict in ["LOST", "GAINED"]:
                marker = "***" if is_key else "   "
                print(f"{vk:<16} {mk:<24} │ {r_raw:>6.3f} │ "
                      f"{r_lin:>6.3f} {p_lin:>8.4f} │ {r_log:>6.3f} {p_log:>8.4f} │ "
                      f"{delta_r:>+6.3f} {verdict:<10} {marker}")

                if is_key:
                    key_pairs.append(comp)

    # ── Summary ─────────────────────────────────────────────────────
    all_comps = results["comparisons"]
    lin_sig = sum(1 for c in all_comps if c["linear_significant"])
    log_sig = sum(1 for c in all_comps if c["log_significant"])
    lost = sum(1 for c in all_comps if c["verdict"] == "LOST")
    gained = sum(1 for c in all_comps if c["verdict"] == "GAINED")

    results["summary"] = {
        "n_tested": len(all_comps),
        "n_linear_significant": lin_sig,
        "n_log_significant": log_sig,
        "n_lost": lost,
        "n_gained": gained,
        "n_status_changed": lost + gained,
    }

    print(f"\n── Summary ──")
    print(f"  Total tested: {len(all_comps)}")
    print(f"  Linear partial significant: {lin_sig}")
    print(f"  Log partial significant: {log_sig}")
    print(f"  Lost significance with log: {lost}")
    print(f"  Gained significance with log: {gained}")

    print(f"\n── Key Paper Pairs ──")
    for kp in key_pairs:
        print(f"  {kp['vocab']} × {kp['metric']}:")
        print(f"    Raw: r={kp['r_raw']:.3f}")
        print(f"    Linear partial: r={kp['r_partial_linear']:.3f} (p={kp['p_partial_linear']:.4f})")
        print(f"    Log partial:    r={kp['r_partial_log']:.3f} (p={kp['p_partial_log']:.4f})")
        print(f"    Verdict: {kp['verdict']}")

    # ── Save ────────────────────────────────────────────────────────
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT}")


if __name__ == "__main__":
    main()
