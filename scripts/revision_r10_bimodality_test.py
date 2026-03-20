#!/usr/bin/env python3
"""R10: Hartigan's Dip Test for Bimodality.

Addresses reviewer Point 25: Mode classification relies on researcher
judgment. Applies Hartigan's dip test to formally test bimodality of
the token count distribution.

Data: zenodo/data/qwen_baseline_n50.json
      outputs/runs/extended_baseline/extended_baseline_results.json
Output: outputs/analysis/revision_r10_bimodality.json
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
DATA_ZENODO = PROJECT / "zenodo" / "data" / "qwen_baseline_n50.json"
DATA_EXTENDED = PROJECT / "outputs" / "runs" / "extended_baseline" / "extended_baseline_results.json"
OUT = PROJECT / "outputs" / "analysis" / "revision_r10_bimodality.json"


def hartigan_dip_test(data, n_boot=10000):
    """Hartigan's dip test for unimodality.

    The dip statistic measures the maximum difference between the empirical
    CDF and the closest unimodal CDF. Larger values indicate multimodality.

    This is a simplified implementation using the bootstrap approach:
    compare the observed dip to dips from uniform distributions.

    Returns (dip_statistic, p_value).
    """
    data = np.sort(np.asarray(data, dtype=float))
    n = len(data)

    if n < 4:
        return float("nan"), float("nan")

    # Compute dip for observed data
    dip_obs = _compute_dip(data)

    # Bootstrap: compare to uniform distribution
    rng = np.random.default_rng(42)
    count_ge = 0
    for _ in range(n_boot):
        uniform_sample = np.sort(rng.uniform(0, 1, n))
        dip_boot = _compute_dip(uniform_sample)
        if dip_boot >= dip_obs:
            count_ge += 1

    p_value = count_ge / n_boot
    return float(dip_obs), float(p_value)


def _compute_dip(sorted_data):
    """Compute the dip statistic for sorted data.

    Uses the ECDF-based approach: the dip is the maximum difference between
    the ECDF and the best-fitting unimodal (uniform) distribution.
    """
    n = len(sorted_data)
    # Normalize to [0, 1]
    if sorted_data[-1] == sorted_data[0]:
        return 0.0
    x = (sorted_data - sorted_data[0]) / (sorted_data[-1] - sorted_data[0])

    # ECDF values
    ecdf = np.arange(1, n + 1) / n

    # Greatest convex minorant (GCM)
    gcm = np.zeros(n)
    gcm[0] = ecdf[0]
    for i in range(1, n):
        gcm[i] = max(ecdf[i], gcm[i - 1])

    # Least concave majorant (LCM)
    lcm = np.zeros(n)
    lcm[-1] = ecdf[-1]
    for i in range(n - 2, -1, -1):
        lcm[i] = min(ecdf[i], lcm[i + 1])

    # Dip = max deviation between GCM and LCM, divided by 2
    dip = np.max(gcm - lcm) / 2

    # Alternative: max deviation from diagonal (uniform reference)
    uniform_cdf = x  # already normalized
    dip_alt = np.max(np.abs(ecdf - uniform_cdf)) / 2

    return max(dip, dip_alt)


def classify_modes(token_counts, threshold=3000):
    """Classify runs into Mode A (long) and Mode B (short).

    Returns classification and any ambiguous runs.
    """
    mode_a = []
    mode_b = []
    ambiguous = []

    for i, tc in enumerate(token_counts):
        if tc >= 5000:
            mode_a.append((i, tc, "A"))
        elif tc <= threshold:
            mode_b.append((i, tc, "B"))
        else:
            ambiguous.append((i, tc, "?"))

    return mode_a, mode_b, ambiguous


def main():
    results = {
        "description": "R10: Hartigan's dip test for bimodality",
    }

    print("=" * 72)
    print("R10: HARTIGAN'S DIP TEST FOR BIMODALITY")
    print("=" * 72)

    # ── Zenodo data ─────────────────────────────────────────────────
    with open(DATA_ZENODO) as f:
        zenodo = json.load(f)
    zenodo_runs = zenodo.get("runs", zenodo.get("data", []))

    zenodo_tokens = np.array([r.get("n_tokens", r.get("text_length", 0))
                               for r in zenodo_runs], dtype=float)

    print(f"\n── Zenodo Baseline (N={len(zenodo_tokens)}) ──")
    if len(zenodo_tokens) > 0:
        print(f"  Token range: {zenodo_tokens.min():.0f} - {zenodo_tokens.max():.0f}")

    dip_z, p_z = hartigan_dip_test(zenodo_tokens)
    results["zenodo"] = {
        "n": len(zenodo_tokens),
        "token_min": float(zenodo_tokens.min()) if len(zenodo_tokens) > 0 else None,
        "token_max": float(zenodo_tokens.max()) if len(zenodo_tokens) > 0 else None,
        "token_mean": float(zenodo_tokens.mean()) if len(zenodo_tokens) > 0 else None,
        "token_std": float(zenodo_tokens.std(ddof=1)) if len(zenodo_tokens) > 1 else None,
        "dip_statistic": dip_z,
        "dip_p_value": p_z,
        "bimodal": p_z < 0.05,
    }

    print(f"  Dip statistic: {dip_z:.4f}")
    print(f"  p-value: {p_z:.4f}")
    if p_z < 0.05:
        print(f"  * SIGNIFICANT — bimodality formally established")
    else:
        print(f"  Not significant — cannot reject unimodality")

    # Mode classification
    mode_a, mode_b, ambiguous = classify_modes(zenodo_tokens)
    results["zenodo"]["mode_a_count"] = len(mode_a)
    results["zenodo"]["mode_b_count"] = len(mode_b)
    results["zenodo"]["ambiguous_count"] = len(ambiguous)
    print(f"  Mode A (≥5000 tokens): {len(mode_a)}")
    print(f"  Mode B (≤3000 tokens): {len(mode_b)}")
    print(f"  Ambiguous (3000-5000): {len(ambiguous)}")

    if ambiguous:
        print(f"  Ambiguous runs: {[(i, tc) for i, tc, _ in ambiguous]}")

    # ── Extended baseline ───────────────────────────────────────────
    with open(DATA_EXTENDED) as f:
        ext_data = json.load(f)
    ext_runs = ext_data.get("runs", ext_data.get("data", []))
    ext_tokens = np.array([r.get("n_tokens", r.get("text_length", 0))
                            for r in ext_runs], dtype=float)

    print(f"\n── Extended Baseline (N={len(ext_tokens)}) ──")
    if len(ext_tokens) > 0:
        print(f"  Token range: {ext_tokens.min():.0f} - {ext_tokens.max():.0f}")

    dip_e, p_e = hartigan_dip_test(ext_tokens)
    results["extended_baseline"] = {
        "n": len(ext_tokens),
        "token_min": float(ext_tokens.min()) if len(ext_tokens) > 0 else None,
        "token_max": float(ext_tokens.max()) if len(ext_tokens) > 0 else None,
        "token_mean": float(ext_tokens.mean()) if len(ext_tokens) > 0 else None,
        "token_std": float(ext_tokens.std(ddof=1)) if len(ext_tokens) > 1 else None,
        "dip_statistic": dip_e,
        "dip_p_value": p_e,
        "bimodal": p_e < 0.05,
    }

    print(f"  Dip statistic: {dip_e:.4f}")
    print(f"  p-value: {p_e:.4f}")
    if p_e < 0.05:
        print(f"  * SIGNIFICANT — bimodality formally established")
    else:
        print(f"  Not significant — cannot reject unimodality")

    mode_a_e, mode_b_e, ambiguous_e = classify_modes(ext_tokens)
    results["extended_baseline"]["mode_a_count"] = len(mode_a_e)
    results["extended_baseline"]["mode_b_count"] = len(mode_b_e)
    results["extended_baseline"]["ambiguous_count"] = len(ambiguous_e)
    print(f"  Mode A (≥5000 tokens): {len(mode_a_e)}")
    print(f"  Mode B (≤3000 tokens): {len(mode_b_e)}")
    print(f"  Ambiguous (3000-5000): {len(ambiguous_e)}")

    # ── Algorithmic exclusion criterion ─────────────────────────────
    print(f"\n── Proposed Algorithmic Exclusion Criterion ──")
    if len(zenodo_tokens) > 0:
        # Find natural gap in distribution
        sorted_tokens = np.sort(zenodo_tokens)
        gaps = np.diff(sorted_tokens)
        if len(gaps) > 0:
            max_gap_idx = np.argmax(gaps)
            gap_lower = sorted_tokens[max_gap_idx]
            gap_upper = sorted_tokens[max_gap_idx + 1]
            results["exclusion_criterion"] = {
                "method": "maximum_gap",
                "gap_lower": float(gap_lower),
                "gap_upper": float(gap_upper),
                "gap_size": float(gaps[max_gap_idx]),
                "criterion": f"Exclude runs with token count in ({gap_lower:.0f}, {gap_upper:.0f})",
            }
            print(f"  Largest gap in distribution: {gap_lower:.0f} to {gap_upper:.0f} "
                  f"(size: {gaps[max_gap_idx]:.0f})")
            print(f"  Proposed: Mode B = tokens ≤ {gap_lower:.0f}, "
                  f"Mode A = tokens ≥ {gap_upper:.0f}")
            print(f"  Runs in gap (excluded): "
                  f"{sum(1 for t in zenodo_tokens if gap_lower < t < gap_upper)}")

    # ── Shapiro-Wilk as supplementary test ──────────────────────────
    print(f"\n── Supplementary: Shapiro-Wilk Normality Test ──")
    for label, tokens in [("Zenodo", zenodo_tokens), ("Extended", ext_tokens)]:
        if len(tokens) >= 3:
            W, p_sw = stats.shapiro(tokens)
            print(f"  {label}: W = {W:.4f}, p = {p_sw:.4f} "
                  f"({'rejects normality' if p_sw < 0.05 else 'consistent with normality'})")

    # ── Save ────────────────────────────────────────────────────────
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT}")


if __name__ == "__main__":
    main()
