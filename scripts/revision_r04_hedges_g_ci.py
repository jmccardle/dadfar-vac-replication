#!/usr/bin/env python3
"""R4: Hedges' g and Confidence Intervals for Effect Sizes.

Addresses reviewer Point 4: Cohen's d at N=3 per group has 37.5% upward bias.
Recomputes all reported effect sizes as Hedges' g with 95% CIs.

Key corrections:
  - d=8.02 (introspection direction, Llama 8B) → g with CI
  - d=4.48 (forest-vs-math direction) → g with CI
  - d=4.27 (Dadfar's transfer test, N=5 per group) → g with CI
  - d=4.05/4.10 (Llama 70B directions) → g with CI

Data: Various cross-model result files
Output: outputs/analysis/revision_r04_hedges_g.json
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "outputs" / "analysis" / "revision_r04_hedges_g.json"


def cohens_d(x, y):
    """Pooled-variance Cohen's d."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    pooled_var = ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2)
    pooled_sd = np.sqrt(pooled_var)
    if pooled_sd == 0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / pooled_sd)


def hedges_correction_factor(n1, n2):
    """Hedges' correction factor J.

    J = 1 - 3 / (4*(n1+n2-2) - 1)
    Corrects for upward bias in d at small N.
    """
    df = n1 + n2 - 2
    if df <= 0:
        return float("nan")
    return 1 - 3 / (4 * df - 1)


def hedges_g(d, n1, n2):
    """Convert Cohen's d to Hedges' g."""
    J = hedges_correction_factor(n1, n2)
    return float(d * J)


def d_se(d, n1, n2):
    """Standard error of Cohen's d (or Hedges' g with correction).

    SE(d) = sqrt(1/n1 + 1/n2 + d^2/(2*(n1+n2)))
    """
    return float(np.sqrt(1/n1 + 1/n2 + d**2 / (2 * (n1 + n2))))


def ci_nct(d, n1, n2, alpha=0.05):
    """Confidence interval for d using noncentral t distribution.

    Returns (ci_low, ci_high).
    """
    df = n1 + n2 - 2
    ncp = d * np.sqrt(n1 * n2 / (n1 + n2))

    try:
        ncp_low = stats.nct.ppf(alpha / 2, df, ncp)
        ncp_high = stats.nct.ppf(1 - alpha / 2, df, ncp)
        # Convert back to d scale
        ci_low = ncp_low / np.sqrt(n1 * n2 / (n1 + n2))
        ci_high = ncp_high / np.sqrt(n1 * n2 / (n1 + n2))
        return float(ci_low), float(ci_high)
    except Exception:
        # Fallback to normal approximation
        se = d_se(d, n1, n2)
        z = stats.norm.ppf(1 - alpha / 2)
        return float(d - z * se), float(d + z * se)


def compute_full_correction(label, d_value, n1, n2):
    """Compute full Hedges' g correction with CI for a reported effect size."""
    J = hedges_correction_factor(n1, n2)
    g = hedges_g(d_value, n1, n2)
    se = d_se(g, n1, n2)
    ci_low, ci_high = ci_nct(g, n1, n2)
    bias_pct = (1 - J) * 100

    return {
        "label": label,
        "n1": n1,
        "n2": n2,
        "n_total": n1 + n2,
        "cohens_d": d_value,
        "hedges_J": float(J),
        "hedges_g": g,
        "bias_correction_pct": float(bias_pct),
        "se_g": se,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "ci_width": ci_high - ci_low,
    }


def main():
    results = {
        "description": "R4: Hedges' g corrections and confidence intervals",
        "corrections": [],
    }

    print("=" * 72)
    print("R4: HEDGES' g AND CONFIDENCE INTERVALS FOR EFFECT SIZES")
    print("=" * 72)

    # ── Define all effect sizes to correct ──────────────────────────
    corrections = [
        ("Introspection direction (Llama 8B)", 8.02, 3, 3),
        ("Forest-vs-Math direction (Llama 8B)", 4.48, 3, 3),
        ("Dadfar transfer test (reported)", 4.27, 5, 5),
        ("Introspection direction (Llama 70B)", 4.05, 3, 3),
        ("Forest-vs-Math direction (Llama 70B)", 4.10, 3, 3),
    ]

    # Also include lock-in pairwise effect sizes (from Phase D, typical N~5-9)
    # These will be computed from data if available, but for now from paper values
    lockin_corrections = [
        ("Lock-in baseline vs nonsense (cycling)", 0.59, 6, 4),
        ("Lock-in baseline vs desc_math (cycling)", 0.44, 6, 9),
    ]

    print(f"\n{'Label':<45} {'d':>6} {'n1':>3} {'n2':>3} │ {'g':>6} "
          f"{'J':>5} {'Bias%':>6} │ {'CI_low':>7} {'CI_high':>7} {'Width':>7}")
    print("─" * 110)

    all_corrections = corrections + lockin_corrections
    for label, d, n1, n2 in all_corrections:
        c = compute_full_correction(label, d, n1, n2)
        results["corrections"].append(c)
        print(f"{label:<45} {d:>6.2f} {n1:>3} {n2:>3} │ {c['hedges_g']:>6.2f} "
              f"{c['hedges_J']:>5.3f} {c['bias_correction_pct']:>5.1f}% │ "
              f"{c['ci_95_low']:>7.2f} {c['ci_95_high']:>7.2f} {c['ci_width']:>7.2f}")

    # ── Interpretation summary ──────────────────────────────────────
    print("\n── Key Corrections ──")
    for c in results["corrections"][:5]:
        print(f"  {c['label']}:")
        print(f"    d = {c['cohens_d']:.2f} → g = {c['hedges_g']:.2f} "
              f"(correction: {c['bias_correction_pct']:.1f}% reduction)")
        print(f"    95% CI: [{c['ci_95_low']:.2f}, {c['ci_95_high']:.2f}] "
              f"(width: {c['ci_width']:.2f})")

    # ── Comparison table for paper ──────────────────────────────────
    print("\n── Table for Paper (comparing our vs Dadfar) ──")
    our_intro = results["corrections"][0]  # d=8.02
    dadfar = results["corrections"][2]     # d=4.27
    print(f"  Our introspection:   g = {our_intro['hedges_g']:.2f} "
          f"[{our_intro['ci_95_low']:.2f}, {our_intro['ci_95_high']:.2f}]  (N={our_intro['n_total']})")
    print(f"  Dadfar's transfer:   g = {dadfar['hedges_g']:.2f} "
          f"[{dadfar['ci_95_low']:.2f}, {dadfar['ci_95_high']:.2f}]  (N={dadfar['n_total']})")
    print(f"  CIs overlap: {'Yes' if our_intro['ci_95_low'] < dadfar['ci_95_high'] and dadfar['ci_95_low'] < our_intro['ci_95_high'] else 'No'}")

    # ── Hedges' correction factor table ─────────────────────────────
    print("\n── Reference: Hedges' J by Sample Size ──")
    for n in [3, 4, 5, 6, 8, 10, 15, 20, 30, 50]:
        J = hedges_correction_factor(n, n)
        bias = (1 - J) * 100
        print(f"  N per group = {n:>3}: J = {J:.4f}, bias correction = {bias:.1f}%")

    # ── Save ────────────────────────────────────────────────────────
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT}")


if __name__ == "__main__":
    main()
