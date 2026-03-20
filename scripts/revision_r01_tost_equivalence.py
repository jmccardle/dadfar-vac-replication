#!/usr/bin/env python3
"""R1: TOST Equivalence Test for Lock-in Resistance.

Addresses reviewer Point 1: non-significant KW (p=0.32) treated as evidence
for the null. Computes:
  (a) TOST (Two One-Sided Tests) for pairwise equivalence
  (b) Power analysis at N=10 for various effect sizes
  (c) Minimum detectable effect at 80% power
  (d) Bayesian estimation with ROPE

Data: outputs/runs/phase_d_controls/phase_d_results.json
Output: outputs/analysis/revision_r01_tost.json
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "outputs" / "runs" / "phase_d_controls" / "phase_d_results.json"
OUT = PROJECT / "outputs" / "analysis" / "revision_r01_tost.json"

CONDITIONS = [
    "baseline", "abstract_philosophical", "factual_iterative",
    "procedural_self", "descriptive_forest", "descriptive_math",
    "descriptive_music", "nonsense_control",
]


def load_lockin_data():
    """Load lock-in observations grouped by condition.

    For runs without cycles, use n_observations as right-censored lower bound.
    Returns dict: condition -> list of lock-in values (cycling runs only)
    and condition -> list of (value, censored_flag) for survival-aware analysis.
    """
    with open(DATA) as f:
        data = json.load(f)

    cycling = defaultdict(list)      # cycling runs only
    all_runs = defaultdict(list)     # all runs with censoring flag

    for run in data["runs"]:
        cond = run["condition"]
        cycle = run.get("cycle", {})
        has_cycle = cycle.get("has_cycle", False)
        lock_in = cycle.get("lock_in_obs")
        n_obs = run.get("n_observations", 300)

        if has_cycle and lock_in is not None:
            cycling[cond].append(float(lock_in))
            all_runs[cond].append((float(lock_in), False))
        else:
            all_runs[cond].append((float(n_obs), True))  # censored

    return cycling, all_runs


def cohens_d(x, y):
    """Pooled-variance Cohen's d."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    pooled_var = ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2)
    pooled_sd = np.sqrt(pooled_var)
    if pooled_sd == 0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / pooled_sd)


def hedges_g(x, y):
    """Hedges' g: bias-corrected effect size."""
    d = cohens_d(x, y)
    n = len(x) + len(y)
    if n <= 3:
        return float("nan")
    J = 1 - 3 / (4 * (n - 2) - 1)
    return float(d * J)


def tost_test(x, y, equivalence_bound):
    """Two One-Sided Tests for equivalence.

    Tests whether |mean(x) - mean(y)| < equivalence_bound.
    Returns (p_tost, p_lower, p_upper, mean_diff, se_diff).

    p_tost = max(p_lower, p_upper).
    If p_tost < alpha, equivalence is established within the bound.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    mean_diff = float(np.mean(x) - np.mean(y))

    # Pooled SE for unequal n
    pooled_var = ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2)
    se_diff = np.sqrt(pooled_var * (1 / nx + 1 / ny))
    df = nx + ny - 2

    if se_diff == 0:
        return (0.0, 0.0, 0.0, mean_diff, 0.0)

    # Lower test: H0: mean_diff <= -bound
    t_lower = (mean_diff - (-equivalence_bound)) / se_diff
    p_lower = float(stats.t.sf(t_lower, df))  # one-sided, upper tail

    # Upper test: H0: mean_diff >= +bound
    t_upper = (mean_diff - equivalence_bound) / se_diff
    p_upper = float(stats.t.cdf(t_upper, df))  # one-sided, lower tail

    p_tost = max(p_lower, p_upper)

    return (float(p_tost), float(p_lower), float(p_upper),
            float(mean_diff), float(se_diff))


def power_for_effect_size(d, n_per_group, alpha=0.05):
    """Approximate power for two-sample t-test at given d and N.

    Uses noncentral t distribution.
    """
    df = 2 * n_per_group - 2
    ncp = d * np.sqrt(n_per_group / 2)  # noncentrality parameter
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    # Power = P(|T_ncp| > t_crit)
    power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
    return float(power)


def minimum_detectable_effect(n_per_group, alpha=0.05, target_power=0.80):
    """Find minimum d detectable at target_power via binary search."""
    lo, hi = 0.0, 5.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if power_for_effect_size(mid, n_per_group, alpha) < target_power:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def bayesian_rope_analysis(x, y, rope_low, rope_high, n_samples=50000):
    """Bayesian estimation of mean difference with ROPE analysis.

    Uses conjugate normal model with uninformative prior.
    Returns: P(diff in ROPE), P(diff < ROPE_low), P(diff > ROPE_high),
    posterior mean, posterior 95% HDI.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)

    # Posterior for each group mean (conjugate normal, uninformative prior)
    # With large uninformative prior, posterior ≈ N(x_bar, s²/n)
    x_bar, y_bar = np.mean(x), np.mean(y)
    x_se = np.std(x, ddof=1) / np.sqrt(nx)
    y_se = np.std(y, ddof=1) / np.sqrt(ny)

    # Sample from posterior of difference
    rng = np.random.default_rng(42)
    x_samples = rng.normal(x_bar, x_se, n_samples)
    y_samples = rng.normal(y_bar, y_se, n_samples)
    diff_samples = x_samples - y_samples

    p_rope = float(np.mean((diff_samples >= rope_low) & (diff_samples <= rope_high)))
    p_below = float(np.mean(diff_samples < rope_low))
    p_above = float(np.mean(diff_samples > rope_high))
    post_mean = float(np.mean(diff_samples))
    hdi_low = float(np.percentile(diff_samples, 2.5))
    hdi_high = float(np.percentile(diff_samples, 97.5))

    return {
        "p_in_rope": p_rope,
        "p_below_rope": p_below,
        "p_above_rope": p_above,
        "posterior_mean": post_mean,
        "hdi_95_low": hdi_low,
        "hdi_95_high": hdi_high,
    }


def main():
    cycling, all_runs = load_lockin_data()

    results = {
        "description": "R1: TOST equivalence test for lock-in resistance",
        "data_source": str(DATA),
        "conditions": CONDITIONS,
    }

    # ── Summary statistics ──────────────────────────────────────────
    summary = {}
    for cond in CONDITIONS:
        vals = np.array(cycling.get(cond, []))
        if len(vals) > 0:
            summary[cond] = {
                "n_cycling": len(vals),
                "n_total": len(all_runs.get(cond, [])),
                "cycle_rate": len(vals) / len(all_runs.get(cond, [])),
                "mean": float(vals.mean()),
                "median": float(np.median(vals)),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(vals.min()),
                "max": float(vals.max()),
            }
        else:
            summary[cond] = {
                "n_cycling": 0,
                "n_total": len(all_runs.get(cond, [])),
                "cycle_rate": 0.0,
            }
    results["summary"] = summary

    # ── Power analysis ──────────────────────────────────────────────
    n_per = 10  # N per group (cycling subset is smaller)
    effect_sizes = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    power_table = {}
    for d in effect_sizes:
        power_table[f"d={d}"] = {
            "n_per_group": n_per,
            "power": power_for_effect_size(d, n_per),
        }
    mde = minimum_detectable_effect(n_per)
    results["power_analysis"] = {
        "n_per_group": n_per,
        "alpha": 0.05,
        "target_power": 0.80,
        "minimum_detectable_d": mde,
        "power_by_effect_size": power_table,
    }

    # Also compute MDE for typical cycling subset sizes
    cycling_ns = [s["n_cycling"] for s in summary.values() if s.get("n_cycling", 0) > 0]
    if cycling_ns:
        median_cycling_n = int(np.median(cycling_ns))
        mde_cycling = minimum_detectable_effect(median_cycling_n)
        results["power_analysis"]["median_cycling_n"] = median_cycling_n
        results["power_analysis"]["mde_at_cycling_n"] = mde_cycling

    # ── TOST pairwise tests ─────────────────────────────────────────
    # SESOI: 30 observations (roughly half the baseline median of 54)
    # Also test with SESOI = 50 and SESOI = 20 for sensitivity
    sesoi_values = [20, 30, 50]
    tost_results = {}

    for sesoi in sesoi_values:
        key = f"sesoi_{sesoi}"
        tost_results[key] = {"equivalence_bound": sesoi, "pairs": []}

        for i, c1 in enumerate(CONDITIONS):
            for c2 in CONDITIONS[i + 1:]:
                v1 = cycling.get(c1, [])
                v2 = cycling.get(c2, [])
                if len(v1) < 2 or len(v2) < 2:
                    continue

                p_tost, p_lo, p_hi, mean_diff, se_diff = tost_test(v1, v2, sesoi)
                d = cohens_d(v1, v2)
                g = hedges_g(v1, v2)

                tost_results[key]["pairs"].append({
                    "condition_1": c1,
                    "condition_2": c2,
                    "n1": len(v1),
                    "n2": len(v2),
                    "mean_diff": mean_diff,
                    "se_diff": se_diff,
                    "p_tost": p_tost,
                    "p_lower": p_lo,
                    "p_upper": p_hi,
                    "equivalent": p_tost < 0.05,
                    "cohens_d": d,
                    "hedges_g": g,
                })

        # Summary counts
        pairs = tost_results[key]["pairs"]
        n_equiv = sum(1 for p in pairs if p["equivalent"])
        tost_results[key]["n_pairs_tested"] = len(pairs)
        tost_results[key]["n_equivalent"] = n_equiv

    results["tost"] = tost_results

    # ── Bayesian ROPE analysis ──────────────────────────────────────
    # ROPE: [-30, +30] observations (matching SESOI=30)
    rope_lo, rope_hi = -30, 30
    bayesian_results = {"rope": [rope_lo, rope_hi], "pairs": []}

    baseline_vals = cycling.get("baseline", [])
    if len(baseline_vals) >= 2:
        for cond in CONDITIONS:
            if cond == "baseline":
                continue
            vals = cycling.get(cond, [])
            if len(vals) < 2:
                continue
            bayes = bayesian_rope_analysis(baseline_vals, vals, rope_lo, rope_hi)
            bayes["condition"] = cond
            bayes["n_baseline"] = len(baseline_vals)
            bayes["n_condition"] = len(vals)
            bayesian_results["pairs"].append(bayes)

    results["bayesian_rope"] = bayesian_results

    # ── Save results ────────────────────────────────────────────────
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    # ── Console output ──────────────────────────────────────────────
    print("=" * 72)
    print("R1: TOST EQUIVALENCE TEST FOR LOCK-IN RESISTANCE")
    print("=" * 72)

    print("\n── Summary Statistics (cycling runs only) ──")
    print(f"{'Condition':<28} {'N_cyc':>5} {'N_tot':>5} {'Rate':>6} "
          f"{'Mean':>7} {'Median':>7} {'SD':>7}")
    for cond in CONDITIONS:
        s = summary[cond]
        if s["n_cycling"] > 0:
            print(f"{cond:<28} {s['n_cycling']:>5} {s['n_total']:>5} "
                  f"{s['cycle_rate']:>6.0%} {s['mean']:>7.1f} "
                  f"{s['median']:>7.1f} {s['std']:>7.1f}")
        else:
            print(f"{cond:<28} {s['n_cycling']:>5} {s['n_total']:>5} "
                  f"{s['cycle_rate']:>6.0%}     ---     ---     ---")

    print(f"\n── Power Analysis (N={n_per} per group) ──")
    for d in effect_sizes:
        p = power_table[f"d={d}"]["power"]
        print(f"  d = {d:.1f}: power = {p:.3f}")
    print(f"  Minimum detectable effect (80% power): d = {mde:.2f}")
    if "mde_at_cycling_n" in results["power_analysis"]:
        print(f"  MDE at median cycling N={results['power_analysis']['median_cycling_n']}: "
              f"d = {results['power_analysis']['mde_at_cycling_n']:.2f}")

    print(f"\n── TOST Pairwise Equivalence ──")
    for sesoi in sesoi_values:
        key = f"sesoi_{sesoi}"
        tr = tost_results[key]
        print(f"\n  SESOI = ±{sesoi} observations:")
        print(f"  {tr['n_equivalent']}/{tr['n_pairs_tested']} pairs establish equivalence (α=0.05)")
        for p in tr["pairs"]:
            flag = "EQUIV" if p["equivalent"] else "     "
            print(f"    {p['condition_1']:<24} vs {p['condition_2']:<24} "
                  f"Δ={p['mean_diff']:>+7.1f}  p_TOST={p['p_tost']:.4f}  "
                  f"g={p['hedges_g']:>+6.2f}  [{flag}]")

    print(f"\n── Bayesian ROPE (±{rope_hi} observations) ──")
    for b in bayesian_results["pairs"]:
        print(f"  baseline vs {b['condition']:<24}: "
              f"P(in ROPE)={b['p_in_rope']:.3f}  "
              f"posterior Δ={b['posterior_mean']:>+7.1f}  "
              f"95% HDI=[{b['hdi_95_low']:>+7.1f}, {b['hdi_95_high']:>+7.1f}]")

    print(f"\nResults saved to {OUT}")


if __name__ == "__main__":
    main()
