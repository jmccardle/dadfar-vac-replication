#!/usr/bin/env python3
"""R7: Kaplan-Meier Survival Analysis for Lock-in.

Addresses reviewer Point 8: Runs that don't cycle are right-censored at
max_observations but are excluded from KW analysis. If censoring rates
differ across conditions (40-90%), this biases the comparison.

Implements:
  (a) Kaplan-Meier estimation per condition
  (b) Log-rank test for cross-condition comparison
  (c) Pairwise log-rank tests with Holm-Bonferroni correction

Data: outputs/runs/phase_d_controls/phase_d_results.json
Output: outputs/analysis/revision_r07_survival.json
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "outputs" / "runs" / "phase_d_controls" / "phase_d_results.json"
OUT = PROJECT / "outputs" / "analysis" / "revision_r07_survival.json"

CONDITIONS = [
    "baseline", "abstract_philosophical", "factual_iterative",
    "procedural_self", "descriptive_forest", "descriptive_math",
    "descriptive_music", "nonsense_control",
]


def load_survival_data():
    """Load (time, event) pairs for each condition.

    time = lock-in observation (or n_observations if censored)
    event = 1 if cycle detected, 0 if censored (no cycle)
    """
    with open(DATA) as f:
        data = json.load(f)

    grouped = defaultdict(list)
    for run in data["runs"]:
        cond = run["condition"]
        cycle = run.get("cycle", {})
        has_cycle = cycle.get("has_cycle", False)
        lock_in = cycle.get("lock_in_obs")
        n_obs = run.get("n_observations", 300)

        if has_cycle and lock_in is not None:
            grouped[cond].append((float(lock_in), 1))  # event
        else:
            grouped[cond].append((float(n_obs), 0))    # censored

    return grouped


def kaplan_meier(times_events):
    """Compute Kaplan-Meier survival function.

    Returns (unique_times, survival_probs, n_at_risk, n_events).
    """
    times_events = sorted(times_events, key=lambda x: x[0])
    n = len(times_events)

    # Get unique event times
    event_times = sorted(set(t for t, e in times_events if e == 1))

    if not event_times:
        return [0], [1.0], [n], [0]

    km_times = [0]
    km_surv = [1.0]
    km_at_risk = [n]
    km_events = [0]

    # At each event time, compute survival
    remaining = list(times_events)
    current_surv = 1.0

    for t in event_times:
        # Number at risk just before time t
        n_at_risk = sum(1 for ti, _ in remaining if ti >= t)
        # Number of events at time t
        n_event = sum(1 for ti, ei in remaining if ti == t and ei == 1)

        if n_at_risk > 0:
            current_surv *= (1 - n_event / n_at_risk)

        km_times.append(t)
        km_surv.append(current_surv)
        km_at_risk.append(n_at_risk)
        km_events.append(n_event)

    return km_times, km_surv, km_at_risk, km_events


def log_rank_test(groups):
    """Multi-group log-rank test.

    groups: list of [(time, event), ...] for each group.
    Returns (chi2, p_value, df).

    Uses the standard log-rank (Mantel-Cox) test.
    """
    # Pool all unique event times
    all_events = []
    for group in groups:
        for t, e in group:
            if e == 1:
                all_events.append(t)
    event_times = sorted(set(all_events))

    if not event_times:
        return 0.0, 1.0, len(groups) - 1

    k = len(groups)
    # For each group, compute observed and expected events
    O = np.zeros(k)  # observed events per group
    E = np.zeros(k)  # expected events per group
    V = np.zeros((k, k))  # variance-covariance matrix

    for t in event_times:
        # At risk in each group at time t
        n_i = np.array([sum(1 for ti, _ in g if ti >= t) for g in groups], dtype=float)
        # Events in each group at time t
        d_i = np.array([sum(1 for ti, ei in g if ti == t and ei == 1) for g in groups], dtype=float)

        N = n_i.sum()
        D = d_i.sum()

        if N <= 1 or D == 0:
            continue

        O += d_i
        E += n_i * D / N

        # Variance contribution
        factor = D * (N - D) / (N * N * (N - 1)) if N > 1 else 0
        for i in range(k):
            for j in range(k):
                if i == j:
                    V[i, j] += n_i[i] * (N - n_i[i]) * factor
                else:
                    V[i, j] -= n_i[i] * n_i[j] * factor

    # Chi-squared statistic (using first k-1 groups)
    OE = (O - E)[:-1]
    V_sub = V[:-1, :-1]

    try:
        V_inv = np.linalg.pinv(V_sub)
        chi2 = float(OE @ V_inv @ OE)
    except np.linalg.LinAlgError:
        chi2 = float("nan")

    df = k - 1
    p = float(stats.chi2.sf(chi2, df)) if np.isfinite(chi2) else float("nan")

    return chi2, p, df


def pairwise_log_rank(groups_dict):
    """Pairwise log-rank tests with Holm-Bonferroni correction."""
    conditions = list(groups_dict.keys())
    pairs = []

    for i, c1 in enumerate(conditions):
        for c2 in conditions[i + 1:]:
            chi2, p, df = log_rank_test([groups_dict[c1], groups_dict[c2]])
            pairs.append({
                "condition_1": c1,
                "condition_2": c2,
                "chi2": chi2,
                "p_raw": p,
                "df": df,
            })

    # Holm-Bonferroni correction
    n_pairs = len(pairs)
    sorted_pairs = sorted(range(n_pairs), key=lambda i: pairs[i]["p_raw"])
    for rank, idx in enumerate(sorted_pairs):
        adj_p = min(pairs[idx]["p_raw"] * (n_pairs - rank), 1.0)
        pairs[idx]["p_adjusted"] = adj_p
        pairs[idx]["significant"] = bool(adj_p < 0.05)

    return pairs


def main():
    grouped = load_survival_data()

    results = {
        "description": "R7: Kaplan-Meier survival analysis for lock-in resistance",
        "data_source": str(DATA),
    }

    print("=" * 72)
    print("R7: KAPLAN-MEIER SURVIVAL ANALYSIS FOR LOCK-IN RESISTANCE")
    print("=" * 72)

    # ── Per-condition KM curves ─────────────────────────────────────
    km_results = {}
    print(f"\n── Per-Condition Summary ──")
    print(f"{'Condition':<28} {'N':>3} {'Events':>6} {'Censored':>8} "
          f"{'Median':>7} {'25th':>6} {'75th':>6}")

    for cond in CONDITIONS:
        te = grouped.get(cond, [])
        times, surv, at_risk, events = kaplan_meier(te)

        n_total = len(te)
        n_events = sum(e for _, e in te)
        n_censored = n_total - n_events

        # Median survival time (where S(t) <= 0.5)
        median_time = float("nan")
        for i, s in enumerate(surv):
            if s <= 0.5:
                median_time = times[i]
                break

        # 25th and 75th percentile survival times
        q25 = float("nan")
        q75 = float("nan")
        for i, s in enumerate(surv):
            if np.isnan(q75) and s <= 0.75:
                q75 = times[i]
            if np.isnan(q25) and s <= 0.25:
                q25 = times[i]

        km_results[cond] = {
            "n_total": n_total,
            "n_events": n_events,
            "n_censored": n_censored,
            "censoring_rate": n_censored / n_total if n_total > 0 else 0,
            "median_survival": median_time,
            "q25_survival": q25,
            "q75_survival": q75,
            "km_times": [float(t) for t in times],
            "km_survival": [float(s) for s in surv],
        }

        med_str = f"{median_time:>7.0f}" if np.isfinite(median_time) else "    N/A"
        q25_str = f"{q25:>6.0f}" if np.isfinite(q25) else "   N/A"
        q75_str = f"{q75:>6.0f}" if np.isfinite(q75) else "   N/A"
        print(f"{cond:<28} {n_total:>3} {n_events:>6} {n_censored:>8} "
              f"{med_str} {q25_str} {q75_str}")

    results["kaplan_meier"] = km_results

    # ── Multi-group log-rank test ───────────────────────────────────
    print(f"\n── Multi-Group Log-Rank Test ──")
    group_list = [grouped[c] for c in CONDITIONS if c in grouped]
    chi2, p, df = log_rank_test(group_list)
    results["log_rank_omnibus"] = {
        "chi2": chi2,
        "p_value": p,
        "df": df,
        "significant": bool(p < 0.05) if np.isfinite(p) else False,
    }
    print(f"  χ² = {chi2:.3f}, df = {df}, p = {p:.4f}")
    if p < 0.05:
        print("  * SIGNIFICANT — survival distributions differ across conditions")
    else:
        print("  Not significant — no evidence of different survival distributions")

    # ── Compare with original KW result ─────────────────────────────
    print(f"\n── Comparison with Original Kruskal-Wallis ──")
    print(f"  KW (cycling runs only): H = 8.12, p = 0.32")
    print(f"  Log-rank (all runs, censoring-adjusted): χ² = {chi2:.3f}, p = {p:.4f}")
    if (p < 0.05) != (0.32 < 0.05):
        print("  ** CONCLUSIONS DIFFER — the censoring adjustment changes the result")
    else:
        print("  Conclusions agree — both tests give the same qualitative result")

    # ── Pairwise log-rank tests ─────────────────────────────────────
    print(f"\n── Pairwise Log-Rank Tests (Holm-Bonferroni) ──")
    pairwise = pairwise_log_rank(grouped)
    results["pairwise_log_rank"] = pairwise

    for p in pairwise:
        flag = "*" if p["significant"] else " "
        print(f"  {p['condition_1']:<24} vs {p['condition_2']:<24}  "
              f"χ²={p['chi2']:>6.2f}  p_raw={p['p_raw']:.4f}  "
              f"p_adj={p['p_adjusted']:.4f} {flag}")

    n_sig = sum(1 for p in pairwise if p["significant"])
    print(f"\n  {n_sig}/{len(pairwise)} pairwise comparisons significant")

    # ── Censoring rates ─────────────────────────────────────────────
    print(f"\n── Censoring Rates (proportion of non-cycling runs) ──")
    for cond in CONDITIONS:
        km = km_results[cond]
        print(f"  {cond:<28}: {km['censoring_rate']:.0%} "
              f"({km['n_censored']}/{km['n_total']} censored)")

    # ── Save ────────────────────────────────────────────────────────
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT}")


if __name__ == "__main__":
    main()
