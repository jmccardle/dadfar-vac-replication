#!/usr/bin/env python3
"""B3: Spectral power vs length confound analysis.

Demonstrates that Dadfar's VAC correlations are confounded with generation
length and evaluates whether they survive correction.

Reads Zenodo published data only — no GPU required.

Outputs:
  - Console: correlation tables, partial correlations, subgroup analyses
  - JSON: full results for paper tables
"""

import json
import sys
from pathlib import Path
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


ZENODO_PATH = Path("zenodo/data/qwen_baseline_n50.json")
OUTPUT_PATH = Path("outputs/analysis/b3_spectral_confound.json")
LAYER = "8"


def load_zenodo(path: Path) -> list[dict]:
    """Load Zenodo runs with layer metrics, vocab counts, and metadata."""
    with open(path) as f:
        data = json.load(f)

    runs = []
    for r in data["runs"]:
        lm = r.get("layer_metrics", {}).get(LAYER, {})
        if not lm:
            continue
        runs.append({
            "run": r["run"],
            "n_tokens": lm.get("n_tokens", r.get("n_tokens", 0)),
            "terminal": r.get("terminal"),
            "spectral_power_low": lm.get("spectral_power_low", 0),
            "spectral_power_mid": lm.get("spectral_power_mid", 0),
            "max_norm": lm.get("max_norm", 0),
            "mean_norm": lm.get("mean_norm", 0),
            "convergence_ratio": lm.get("convergence_ratio", 0),
            "mirror": r.get("vocab_counts", {}).get("mirror", 0),
            "expand": r.get("vocab_counts", {}).get("expand", 0),
            "resonance": r.get("vocab_counts", {}).get("resonance", 0),
        })
    return runs


def partial_correlation(x, y, z):
    """Partial correlation of x and y, controlling for z.

    Uses linear regression to residualize both x and y on z.
    """
    x, y, z = np.array(x), np.array(y), np.array(z)

    # Residualize x on z
    slope_xz = np.polyfit(z, x, 1)
    x_resid = x - np.polyval(slope_xz, z)

    # Residualize y on z
    slope_yz = np.polyfit(z, y, 1)
    y_resid = y - np.polyval(slope_yz, z)

    r, p = stats.pearsonr(x_resid, y_resid)
    return float(r), float(p)


def run_analysis():
    runs = load_zenodo(ZENODO_PATH)
    print(f"Loaded {len(runs)} runs from Zenodo\n")

    # Extract arrays
    n_tokens = np.array([r["n_tokens"] for r in runs])
    sp_low = np.array([r["spectral_power_low"] for r in runs])
    max_norm = np.array([r["max_norm"] for r in runs])
    mirror = np.array([r["mirror"] for r in runs])
    expand = np.array([r["expand"] for r in runs])
    resonance = np.array([r["resonance"] for r in runs])
    terminals = [r["terminal"] for r in runs]

    # --- Section 1: Length distribution ---
    print("=" * 60)
    print("1. GENERATION LENGTH DISTRIBUTION")
    print("=" * 60)
    short = n_tokens[n_tokens < 1000]
    long = n_tokens[n_tokens >= 1000]
    print(f"  Total runs: {len(n_tokens)}")
    print(f"  Short runs (< 1000 tokens): {len(short)}")
    print(f"    Range: {short.min()}-{short.max()}")
    print(f"  Long runs (>= 1000 tokens): {len(long)}")
    print(f"    Range: {long.min()}-{long.max()}")
    print(f"  Length ratio (max/min): {n_tokens.max() / n_tokens.min():.1f}x")
    print(f"  Runs with terminal word: {sum(1 for t in terminals if t)}")
    print(f"  Runs without terminal word: {sum(1 for t in terminals if not t)}")
    print()

    # --- Section 2: Length confounds ---
    print("=" * 60)
    print("2. CORRELATIONS WITH GENERATION LENGTH (n_tokens)")
    print("=" * 60)
    for name, arr in [("spectral_power_low", sp_low), ("mirror", mirror),
                      ("max_norm", max_norm), ("resonance", resonance)]:
        r, p = stats.pearsonr(n_tokens, arr)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  n_tokens vs {name:25s}: r={r:.4f}  p={p:.2e}  {sig}")
    print()

    # --- Section 3: Dadfar's VAC correlations (raw) ---
    print("=" * 60)
    print("3. DADFAR'S VAC CORRELATIONS (RAW, ALL 50 RUNS)")
    print("=" * 60)
    results = {}

    pairs = [
        ("mirror", "spectral_power_low", mirror, sp_low),
        ("expand", "spectral_power_low", expand, sp_low),
        ("resonance", "max_norm", resonance, max_norm),
    ]

    for vname, mname, v, m in pairs:
        key = f"{vname}_vs_{mname}"
        if np.std(v) < 1e-10:
            print(f"  {vname} vs {mname}: CONSTANT INPUT (all {vname} = {v[0]})")
            results[key] = {"error": "constant_input"}
            continue

        r, p = stats.pearsonr(v, m)
        rho, rho_p = stats.spearmanr(v, m)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {vname:12s} vs {mname:20s}: r={r:.4f}  p={p:.2e}  {sig}  rho={rho:.4f}")
        results[key] = {
            "raw_pearson_r": round(float(r), 4),
            "raw_pearson_p": float(p),
            "raw_spearman_rho": round(float(rho), 4),
            "raw_spearman_p": float(rho_p),
            "n": len(v),
        }
    print()

    # --- Section 4: Partial correlations controlling for length ---
    print("=" * 60)
    print("4. PARTIAL CORRELATIONS (CONTROLLING FOR n_tokens)")
    print("=" * 60)
    for vname, mname, v, m in pairs:
        key = f"{vname}_vs_{mname}"
        if key not in results or "error" in results[key]:
            print(f"  {vname} vs {mname}: SKIPPED (constant input)")
            continue

        r_part, p_part = partial_correlation(v, m, n_tokens)
        sig = "***" if p_part < 0.001 else "**" if p_part < 0.01 else "*" if p_part < 0.05 else "ns"
        print(f"  {vname:12s} vs {mname:20s}: r_partial={r_part:.4f}  p={p_part:.2e}  {sig}")
        results[key]["partial_r"] = round(r_part, 4)
        results[key]["partial_p"] = float(p_part)
    print()

    # --- Section 5: Double-rate normalization ---
    print("=" * 60)
    print("5. DOUBLE-RATE NORMALIZATION (both vocab and metric / n_tokens)")
    print("=" * 60)
    for vname, mname, v, m in pairs:
        key = f"{vname}_vs_{mname}"
        if key not in results or "error" in results[key]:
            continue

        v_rate = v / n_tokens
        m_rate = m / n_tokens

        if np.std(v_rate) < 1e-10 or np.std(m_rate) < 1e-10:
            print(f"  {vname}/N vs {mname}/N: CONSTANT after normalization")
            continue

        r_dr, p_dr = stats.pearsonr(v_rate, m_rate)
        sig = "***" if p_dr < 0.001 else "**" if p_dr < 0.01 else "*" if p_dr < 0.05 else "ns"
        print(f"  {vname}/N vs {mname}/N: r={r_dr:.4f}  p={p_dr:.2e}  {sig}")
        results[key]["double_rate_r"] = round(float(r_dr), 4)
        results[key]["double_rate_p"] = float(p_dr)
    print()

    # --- Section 6: Long-runs-only subgroup ---
    print("=" * 60)
    print("6. LONG-RUNS-ONLY SUBGROUP (n_tokens >= 1000, eliminates length variation)")
    print("=" * 60)
    long_mask = n_tokens >= 1000
    n_long = long_mask.sum()
    print(f"  N = {n_long} runs (all at {n_tokens[long_mask].min()}-{n_tokens[long_mask].max()} tokens)")
    print()

    for vname, mname, v, m in pairs:
        key = f"{vname}_vs_{mname}"
        if key not in results or "error" in results[key]:
            continue

        v_long = v[long_mask]
        m_long = m[long_mask]

        if np.std(v_long) < 1e-10 or np.std(m_long) < 1e-10:
            print(f"  {vname} vs {mname}: CONSTANT in long subgroup")
            continue

        r_long, p_long = stats.pearsonr(v_long, m_long)
        rho_long, rho_p_long = stats.spearmanr(v_long, m_long)
        sig = "***" if p_long < 0.001 else "**" if p_long < 0.01 else "*" if p_long < 0.05 else "ns"
        print(f"  {vname:12s} vs {mname:20s}: r={r_long:.4f}  p={p_long:.2e}  {sig}  rho={rho_long:.4f}")
        results[key]["long_only_r"] = round(float(r_long), 4)
        results[key]["long_only_p"] = float(p_long)
        results[key]["long_only_rho"] = round(float(rho_long), 4)
        results[key]["long_only_n"] = int(n_long)
    print()

    # --- Section 7: Dadfar's normalization (metric only, not vocab) ---
    print("=" * 60)
    print("7. DADFAR'S NORMALIZATION (metric/N only, vocab raw — the code's approach)")
    print("=" * 60)
    for vname, mname, v, m in pairs:
        key = f"{vname}_vs_{mname}"
        if key not in results or "error" in results[key]:
            continue

        m_norm = m / n_tokens
        if np.std(m_norm) < 1e-10:
            continue

        r_dn, p_dn = stats.pearsonr(v, m_norm)
        sig = "***" if p_dn < 0.001 else "**" if p_dn < 0.01 else "*" if p_dn < 0.05 else "ns"
        print(f"  {vname} vs {mname}/N: r={r_dn:.4f}  p={p_dn:.2e}  {sig}")
        results[key]["dadfar_norm_r"] = round(float(r_dn), 4)
        results[key]["dadfar_norm_p"] = float(p_dn)
    print()

    # --- Section 8: Summary table ---
    print("=" * 60)
    print("8. SUMMARY: mirror vs spectral_power_low ACROSS CORRECTIONS")
    print("=" * 60)
    m_key = "mirror_vs_spectral_power_low"
    if m_key in results and "error" not in results[m_key]:
        r = results[m_key]
        print(f"  {'Method':<40s} {'r':>8s} {'p':>12s} {'sig':>5s}")
        print(f"  {'-'*65}")
        for label, rk, pk in [
            ("Raw Pearson (all 50)", "raw_pearson_r", "raw_pearson_p"),
            ("Partial (controlling n_tokens)", "partial_r", "partial_p"),
            ("Dadfar's norm (metric/N, vocab raw)", "dadfar_norm_r", "dadfar_norm_p"),
            ("Double-rate (both/N)", "double_rate_r", "double_rate_p"),
            ("Long-runs-only (N=36)", "long_only_r", "long_only_p"),
        ]:
            if rk in r:
                sig = "***" if r[pk] < 0.001 else "**" if r[pk] < 0.01 else "*" if r[pk] < 0.05 else "ns"
                print(f"  {label:<40s} {r[rk]:>8.4f} {r[pk]:>12.2e} {sig:>5s}")
    print()

    # --- Section 9: resonance vs max_norm (the surviving pair) ---
    print("=" * 60)
    print("9. SUMMARY: resonance vs max_norm ACROSS CORRECTIONS")
    print("=" * 60)
    r_key = "resonance_vs_max_norm"
    if r_key in results and "error" not in results[r_key]:
        r = results[r_key]
        print(f"  {'Method':<40s} {'r':>8s} {'p':>12s} {'sig':>5s}")
        print(f"  {'-'*65}")
        for label, rk, pk in [
            ("Raw Pearson (all 50)", "raw_pearson_r", "raw_pearson_p"),
            ("Partial (controlling n_tokens)", "partial_r", "partial_p"),
            ("Long-runs-only (N=36)", "long_only_r", "long_only_p"),
        ]:
            if rk in r:
                sig = "***" if r[pk] < 0.001 else "**" if r[pk] < 0.01 else "*" if r[pk] < 0.05 else "ns"
                print(f"  {label:<40s} {r[rk]:>8.4f} {r[pk]:>12.2e} {sig:>5s}")
    print()

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_analysis()
