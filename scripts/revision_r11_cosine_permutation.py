#!/usr/bin/env python3
"""R11: Permutation Test on Cosine Similarity Difference.

Addresses reviewer Point 26: Section 9.7.4 dismisses a 0.08 cosine difference
(baseline-mean 0.64 vs others-mean 0.56) without testing it.

Permutation test: randomly designate one of the 7 directions as "baseline",
compute mean baseline-vs-others and others-vs-others cosines, repeat 10,000
times.

Data: outputs/analysis/layer_sweep_geometric_analysis.json
Output: outputs/analysis/revision_r11_cosine_perm.json
"""

import json
from pathlib import Path
from itertools import combinations
import numpy as np
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "outputs" / "analysis" / "layer_sweep_geometric_analysis.json"
OUT = PROJECT / "outputs" / "analysis" / "revision_r11_cosine_perm.json"

N_PERMUTATIONS = 10000


def extract_direction_cosines(data, layer="8"):
    """Extract pairwise cosine similarities between directions at a given layer.

    Returns (condition_names, cosine_matrix) where cosine_matrix[i,j] is
    the cosine similarity between direction i and direction j.
    """
    # Try to find direction data in the geometric analysis
    layer_data = data.get(f"layer_{layer}", data.get(str(layer), {}))

    # Look for direction comparisons
    directions = layer_data.get("directions", layer_data.get("direction_cosines", {}))

    if isinstance(directions, dict) and "pairwise" in directions:
        # Structured format
        pairs = directions["pairwise"]
        conditions = set()
        for p in pairs:
            conditions.add(p.get("condition_1", p.get("dir_1", "")))
            conditions.add(p.get("condition_2", p.get("dir_2", "")))
        conditions = sorted(conditions)

        n = len(conditions)
        cond_idx = {c: i for i, c in enumerate(conditions)}
        matrix = np.eye(n)

        for p in pairs:
            c1 = p.get("condition_1", p.get("dir_1", ""))
            c2 = p.get("condition_2", p.get("dir_2", ""))
            cos = p.get("cosine", p.get("cosine_similarity", 0))
            if c1 in cond_idx and c2 in cond_idx:
                i, j = cond_idx[c1], cond_idx[c2]
                matrix[i, j] = cos
                matrix[j, i] = cos

        return conditions, matrix

    # Try alternative: look for centroid_cosines or similar
    for key in data:
        section = data[key]
        if isinstance(section, dict):
            for subkey in section:
                if "cosine" in subkey.lower() and "direction" in subkey.lower():
                    return extract_from_nested(section[subkey])

    return None, None


def extract_from_nested(cosine_data):
    """Try to extract cosine matrix from various data formats."""
    if isinstance(cosine_data, list):
        conditions = set()
        for item in cosine_data:
            if isinstance(item, dict):
                conditions.add(item.get("condition_1", item.get("dir_1", "")))
                conditions.add(item.get("condition_2", item.get("dir_2", "")))

        conditions = sorted(c for c in conditions if c)
        n = len(conditions)
        if n < 3:
            return None, None

        cond_idx = {c: i for i, c in enumerate(conditions)}
        matrix = np.eye(n)
        for item in cosine_data:
            c1 = item.get("condition_1", item.get("dir_1", ""))
            c2 = item.get("condition_2", item.get("dir_2", ""))
            cos = item.get("cosine", item.get("cosine_similarity", 0))
            if c1 in cond_idx and c2 in cond_idx:
                i, j = cond_idx[c1], cond_idx[c2]
                matrix[i, j] = cos
                matrix[j, i] = cos
        return conditions, matrix

    return None, None


def compute_baseline_vs_others_diff(cosine_matrix, baseline_idx):
    """Compute the difference between mean baseline-vs-others and
    mean others-vs-others cosine similarities."""
    n = len(cosine_matrix)

    # Baseline vs others
    baseline_others = []
    for j in range(n):
        if j != baseline_idx:
            baseline_others.append(cosine_matrix[baseline_idx, j])

    # Others vs others (excluding baseline)
    others = [i for i in range(n) if i != baseline_idx]
    others_pairs = []
    for i, j in combinations(others, 2):
        others_pairs.append(cosine_matrix[i, j])

    if not baseline_others or not others_pairs:
        return float("nan"), float("nan"), float("nan")

    mean_baseline = np.mean(baseline_others)
    mean_others = np.mean(others_pairs)
    diff = mean_baseline - mean_others

    return float(mean_baseline), float(mean_others), float(diff)


def permutation_test_cosine(cosine_matrix, baseline_idx, n_perm=N_PERMUTATIONS):
    """Permutation test: is the baseline direction more central than expected?

    Randomly designate each direction as "baseline" and compute the difference.
    """
    n = len(cosine_matrix)
    _, _, obs_diff = compute_baseline_vs_others_diff(cosine_matrix, baseline_idx)

    rng = np.random.default_rng(42)
    count_ge = 0

    for _ in range(n_perm):
        perm_baseline = rng.integers(0, n)
        _, _, perm_diff = compute_baseline_vs_others_diff(cosine_matrix, perm_baseline)
        if perm_diff >= obs_diff:
            count_ge += 1

    perm_p = (count_ge + 1) / (n_perm + 1)
    return float(perm_p)


def main():
    results = {
        "description": "R11: Permutation test on cosine similarity difference",
        "n_permutations": N_PERMUTATIONS,
    }

    print("=" * 72)
    print("R11: PERMUTATION TEST ON COSINE SIMILARITY DIFFERENCE")
    print("=" * 72)

    # ── Load data ───────────────────────────────────────────────────
    if not DATA.exists():
        print(f"Data file not found: {DATA}")
        print("Attempting to reconstruct from paper values...")

        # Use the values reported in the paper (Table 34, Section 9.7.4)
        conditions = [
            "baseline", "procedural_self", "nonsense_control",
            "abstract_philosophical", "descriptive_math",
            "descriptive_music", "factual_iterative",
        ]
        # From Table 34 in the paper (baseline vs each)
        baseline_cosines = {
            "procedural_self": 0.83,
            "nonsense_control": 0.77,
            "abstract_philosophical": 0.67,
            "descriptive_math": 0.64,
            "descriptive_music": 0.49,
            "factual_iterative": 0.45,
        }
        # Paper reports: mean baseline-vs-others = 0.64, mean others-vs-others = 0.56

        # We need the full matrix for permutation. Without it, we can still do
        # an approximate test using the reported summary statistics.
        print("\n  NOTE: Full cosine matrix not available in pre-computed data.")
        print("  Using paper-reported values for approximate analysis.")
        print("  For full permutation test, need layer_sweep_geometric_analysis.json")

        mean_baseline = 0.64
        mean_others = 0.56
        obs_diff = 0.08

        results["from_paper_values"] = {
            "mean_baseline_vs_others": mean_baseline,
            "mean_others_vs_others": mean_others,
            "observed_difference": obs_diff,
            "note": "Full permutation test requires cosine matrix data",
        }

        # Approximate significance via z-test on Fisher-z transformed cosines
        # This is rough but gives a direction
        n_baseline_pairs = 6  # baseline vs 6 others
        n_other_pairs = 15   # C(6,2) = 15 other-other pairs

        z_b = np.arctanh(mean_baseline)
        z_o = np.arctanh(mean_others)
        se_diff = np.sqrt(1 / (n_baseline_pairs - 3) + 1 / (n_other_pairs - 3))
        z_test = (z_b - z_o) / se_diff
        p_approx = 2 * stats.norm.sf(abs(z_test))

        results["approximate_z_test"] = {
            "z_baseline": float(z_b),
            "z_others": float(z_o),
            "z_test": float(z_test),
            "p_value": float(p_approx),
            "significant": p_approx < 0.05,
            "note": "Approximate using Fisher z-transformation; full permutation preferred",
        }

        print(f"\n  Approximate z-test (Fisher z-transformation):")
        print(f"    z = {z_test:.3f}, p = {p_approx:.4f}")
        if p_approx < 0.05:
            print("    * Significant: baseline IS slightly more central")
        else:
            print("    Not significant: 0.08 difference not reliably different from chance")

        OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {OUT}")
        return

    # ── Full analysis with cosine matrix ────────────────────────────
    with open(DATA) as f:
        data = json.load(f)

    # Extract from actual data format: topic_directions_layer8.pairwise_cosines
    td = data.get("topic_directions_layer8", {})
    pairwise_cosines = td.get("pairwise_cosines", {})

    if pairwise_cosines:
        # Parse condition names from keys like "abstract_philosophical_vs_baseline"
        conditions_set = set()
        for key in pairwise_cosines:
            parts = key.split("_vs_")
            if len(parts) == 2:
                conditions_set.add(parts[0])
                conditions_set.add(parts[1])
        conditions = sorted(conditions_set)
        n = len(conditions)
        cond_idx = {c: i for i, c in enumerate(conditions)}
        cosine_matrix = np.eye(n)
        for key, cos in pairwise_cosines.items():
            parts = key.split("_vs_")
            if len(parts) == 2 and parts[0] in cond_idx and parts[1] in cond_idx:
                i, j = cond_idx[parts[0]], cond_idx[parts[1]]
                cosine_matrix[i, j] = cos
                cosine_matrix[j, i] = cos
    else:
        conditions, cosine_matrix = extract_direction_cosines(data)

    if conditions is None or cosine_matrix is None or len(conditions) < 3:
        print("Could not extract direction cosines from data file")
        print("Available keys:", list(data.keys()))
        results["error"] = "could not extract cosine matrix"
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)
        return

    n_dirs = len(conditions)
    baseline_idx = None
    for i, c in enumerate(conditions):
        if "baseline" in c.lower():
            baseline_idx = i
            break

    if baseline_idx is None:
        print("Could not identify baseline condition")
        return

    print(f"\n  Conditions: {conditions}")
    print(f"  Baseline index: {baseline_idx} ({conditions[baseline_idx]})")

    # Compute observed difference
    mean_b, mean_o, obs_diff = compute_baseline_vs_others_diff(cosine_matrix, baseline_idx)
    print(f"\n  Mean baseline-vs-others cosine: {mean_b:.4f}")
    print(f"  Mean others-vs-others cosine:   {mean_o:.4f}")
    print(f"  Difference: {obs_diff:.4f}")

    # Permutation test
    perm_p = permutation_test_cosine(cosine_matrix, baseline_idx)

    results["full_permutation"] = {
        "conditions": conditions,
        "baseline": conditions[baseline_idx],
        "n_directions": n_dirs,
        "mean_baseline_vs_others": mean_b,
        "mean_others_vs_others": mean_o,
        "observed_difference": obs_diff,
        "permutation_p": perm_p,
        "significant": bool(perm_p < 0.05),
    }

    print(f"\n  Permutation p-value: {perm_p:.4f}")
    if perm_p < 0.05:
        print("  * SIGNIFICANT — baseline direction is more central than expected by chance")
        print("    This doesn't rescue the 'introspection mode' claim but requires discussion:")
        print("    self-referential language may overlap with multiple domains")
    else:
        print("  NOT significant — the 0.08 difference is within chance variation")
        print("  The dismissal in the paper is now formally justified")

    # ── Per-direction analysis ──────────────────────────────────────
    print(f"\n  Per-direction mean cosine (each treated as 'baseline'):")
    for i, cond in enumerate(conditions):
        mb, mo, d = compute_baseline_vs_others_diff(cosine_matrix, i)
        marker = " <-- actual baseline" if i == baseline_idx else ""
        print(f"    {cond:<28}: mean_vs_others={mb:.4f}  diff={d:>+.4f}{marker}")

    # ── Save ────────────────────────────────────────────────────────
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT}")


if __name__ == "__main__":
    main()
