#!/usr/bin/env python3
"""R9: Pairwise Temperature Follow-up.

Addresses reviewer Point 24: The only significant KW result (cycle period
H=6.59, p=0.037 across 3 temperatures) lacks post-hoc pairwise comparisons.

Data: outputs/runs/phase_d3_temperature/temperature_ablation_results.json
      or outputs/analysis/phase_d3_temperature.json
Output: outputs/analysis/revision_r09_temperature_pairwise.json
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
# Try both possible locations
DATA_PRIMARY = PROJECT / "outputs" / "runs" / "phase_d3_temperature" / "temperature_ablation_results.json"
DATA_SECONDARY = PROJECT / "outputs" / "analysis" / "phase_d3_temperature.json"
OUT = PROJECT / "outputs" / "analysis" / "revision_r09_temperature_pairwise.json"

TEMPERATURES = [0.3, 0.7, 1.0]


def load_temperature_data():
    """Load cycle period data grouped by temperature."""
    # Try primary data source
    data_path = DATA_PRIMARY if DATA_PRIMARY.exists() else DATA_SECONDARY

    with open(data_path) as f:
        data = json.load(f)

    # Structure depends on data source
    grouped = defaultdict(list)

    if "runs" in data:
        for run in data["runs"]:
            temp = run.get("temperature", run.get("temp"))
            cycle = run.get("cycle", {})
            if cycle.get("has_cycle", False) and cycle.get("cycle_period") is not None:
                grouped[float(temp)].append({
                    "cycle_period": cycle["cycle_period"],
                    "lock_in_obs": cycle.get("lock_in_obs"),
                    "n_tokens": run.get("n_tokens"),
                })
    else:
        # May be pre-analyzed data with different structure
        for key, section in data.items():
            if isinstance(section, dict):
                for temp_key, temp_data in section.items():
                    if isinstance(temp_data, dict) and "cycle_periods" in temp_data:
                        temp = float(temp_key.replace("T=", "").replace("t_", ""))
                        for p in temp_data["cycle_periods"]:
                            grouped[temp].append({"cycle_period": p})

    return grouped, str(data_path)


def holm_bonferroni(p_values):
    """Apply Holm-Bonferroni correction."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = min(p * (n - rank), 1.0)
        cummax = max(cummax, adj_p)
        adjusted[orig_idx] = cummax
    return adjusted


def main():
    grouped, data_path = load_temperature_data()

    results = {
        "description": "R9: Pairwise temperature comparisons for cycle period",
        "data_source": data_path,
    }

    print("=" * 72)
    print("R9: PAIRWISE TEMPERATURE FOLLOW-UP FOR CYCLE PERIOD")
    print("=" * 72)

    # ── Summary per temperature ─────────────────────────────────────
    print(f"\n── Summary ──")
    print(f"{'Temp':>6} {'N_cycling':>9} {'Median':>7} {'Mean':>7} {'SD':>7} {'Range'}")

    summaries = {}
    for temp in TEMPERATURES:
        runs = grouped.get(temp, [])
        periods = np.array([r["cycle_period"] for r in runs], dtype=float)

        if len(periods) > 0:
            summaries[str(temp)] = {
                "n": len(periods),
                "mean": float(periods.mean()),
                "median": float(np.median(periods)),
                "std": float(periods.std(ddof=1)) if len(periods) > 1 else 0,
                "min": float(periods.min()),
                "max": float(periods.max()),
                "values": [float(v) for v in periods],
            }
            print(f"{temp:>6.1f} {len(periods):>9} {np.median(periods):>7.1f} "
                  f"{periods.mean():>7.1f} {periods.std(ddof=1):>7.1f} "
                  f"{periods.min():.0f}-{periods.max():.0f}")
        else:
            summaries[str(temp)] = {"n": 0}
            print(f"{temp:>6.1f} {0:>9}   no cycling runs")

    results["summaries"] = summaries

    # ── Re-verify omnibus KW ────────────────────────────────────────
    groups = []
    for temp in TEMPERATURES:
        runs = grouped.get(temp, [])
        periods = [r["cycle_period"] for r in runs]
        if periods:
            groups.append(np.array(periods, dtype=float))

    if len(groups) >= 2:
        H, p_kw = stats.kruskal(*groups)
        results["omnibus_kw"] = {
            "H": float(H),
            "p": float(p_kw),
            "significant": bool(p_kw < 0.05),
        }
        print(f"\n  Omnibus Kruskal-Wallis: H = {H:.3f}, p = {p_kw:.4f}")
    else:
        print("\n  Insufficient groups for KW test")
        results["omnibus_kw"] = {"error": "insufficient groups"}

    # ── Pairwise Mann-Whitney U tests ───────────────────────────────
    print(f"\n── Pairwise Mann-Whitney U (Holm-Bonferroni corrected) ──")
    pairs = []
    raw_ps = []

    for i, t1 in enumerate(TEMPERATURES):
        for t2 in TEMPERATURES[i + 1:]:
            runs1 = grouped.get(t1, [])
            runs2 = grouped.get(t2, [])
            p1 = np.array([r["cycle_period"] for r in runs1], dtype=float)
            p2 = np.array([r["cycle_period"] for r in runs2], dtype=float)

            if len(p1) >= 2 and len(p2) >= 2:
                U, p_mw = stats.mannwhitneyu(p1, p2, alternative="two-sided")
                # Effect size r = Z / sqrt(N)
                Z = stats.norm.ppf(1 - p_mw / 2) if p_mw < 1 else 0
                r_effect = Z / np.sqrt(len(p1) + len(p2))

                pair_result = {
                    "temp_1": t1,
                    "temp_2": t2,
                    "n_1": len(p1),
                    "n_2": len(p2),
                    "U": float(U),
                    "p_raw": float(p_mw),
                    "r_effect": float(r_effect),
                    "median_1": float(np.median(p1)),
                    "median_2": float(np.median(p2)),
                    "median_diff": float(np.median(p1) - np.median(p2)),
                }
                pairs.append(pair_result)
                raw_ps.append(float(p_mw))

    # Apply Holm-Bonferroni
    if raw_ps:
        adj_ps = holm_bonferroni(raw_ps)
        for i, pair in enumerate(pairs):
            pair["p_adjusted"] = adj_ps[i]
            pair["significant"] = bool(adj_ps[i] < 0.05)

    results["pairwise"] = pairs

    for pair in pairs:
        sig = "*" if pair.get("significant", False) else " "
        print(f"  T={pair['temp_1']:.1f} (med={pair['median_1']:.0f}) vs "
              f"T={pair['temp_2']:.1f} (med={pair['median_2']:.0f}): "
              f"U={pair['U']:.1f}, p_raw={pair['p_raw']:.4f}, "
              f"p_adj={pair.get('p_adjusted', pair['p_raw']):.4f}, "
              f"r={pair['r_effect']:.3f} {sig}")

    # ── Which pair drives the effect? ───────────────────────────────
    print(f"\n── Interpretation ──")
    if pairs:
        min_p_pair = min(pairs, key=lambda x: x["p_raw"])
        print(f"  Strongest pairwise effect: T={min_p_pair['temp_1']:.1f} vs "
              f"T={min_p_pair['temp_2']:.1f}")
        print(f"    p_raw = {min_p_pair['p_raw']:.4f}, "
              f"p_adjusted = {min_p_pair.get('p_adjusted', 'N/A')}")
        print(f"    Median periods: {min_p_pair['median_1']:.0f} vs "
              f"{min_p_pair['median_2']:.0f}")

        n_sig = sum(1 for p in pairs if p.get("significant", False))
        print(f"\n  {n_sig}/{len(pairs)} pairs significant after correction")

        # Check non-monotonicity
        meds = {t: summaries[str(t)].get("median", 0) for t in TEMPERATURES
                if summaries[str(t)].get("n", 0) > 0}
        if len(meds) == 3:
            temps = sorted(meds.keys())
            monotonic = (meds[temps[0]] <= meds[temps[1]] <= meds[temps[2]] or
                         meds[temps[0]] >= meds[temps[1]] >= meds[temps[2]])
            print(f"  Pattern: {'Monotonic' if monotonic else 'Non-monotonic'}")
            print(f"    T={temps[0]}: {meds[temps[0]]:.0f}, "
                  f"T={temps[1]}: {meds[temps[1]]:.0f}, "
                  f"T={temps[2]}: {meds[temps[2]]:.0f}")

    # ── Save ────────────────────────────────────────────────────────
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT}")


if __name__ == "__main__":
    main()
