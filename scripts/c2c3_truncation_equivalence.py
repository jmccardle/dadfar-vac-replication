#!/usr/bin/env python3
"""C2/C3: Truncation equivalence and transition matrix convergence.

For each Mode A run, computes dynamical metrics at multiple observation
cutoffs (50, 100, 150, 200, 300, 500, 1000) and measures:

C2: Do text-level dynamical metrics stabilize by observation 200?
C3: Does the Markov transition matrix converge by observation 200?

Uses existing text-only data (CPU only, no GPU required).

Outputs:
  - Console: per-cutoff metric summaries
  - JSON: full results for paper figures/tables
"""

import json
import sys
import math
from pathlib import Path
from collections import Counter
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.loop_detection import (
    parse_observations, normalize_observation, analyze_run,
    build_transition_matrix, assign_states, extract_cycle_vocabulary,
    detect_cycle_exact, detect_cycle_similarity,
)


TEXT_DIR = Path("outputs/runs/baseline/activations")
RESULTS_PATH = Path("outputs/runs/baseline/baseline_results.json")
OUTPUT_PATH = Path("outputs/analysis/c2c3_truncation_equivalence.json")

CUTOFFS = [50, 100, 150, 200, 300, 500, 1000]


def identify_mode_a_runs() -> list[int]:
    """Identify Mode A runs from baseline results.

    Mode A: text_length > 5000 (individual observations, long) AND
    reached observation 1000 (has terminal or n_tokens suggests completion).
    """
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    mode_a = []
    for r in data["runs"]:
        # Mode A: long text AND actually reached 1000 observations
        # Check by reading the text file and counting observations
        run_id = r["run"]
        text_path = TEXT_DIR / f"run_{run_id:03d}_text.txt"
        if not text_path.exists():
            continue

        text = text_path.read_text()
        obs = parse_observations(text)
        n_tokens = r.get("n_tokens", 0)

        # Mode A: >= 900 observations AND > 5000 tokens
        if len(obs) >= 900 and n_tokens > 5000:
            mode_a.append(run_id)

    return mode_a


def frobenius_distance(m1: list[list[float]], m2: list[list[float]],
                       states1: list[str], states2: list[str]) -> float:
    """Frobenius distance between two transition matrices.

    Aligns matrices on shared states. States present in only one matrix
    are treated as zero rows/columns in the other.
    """
    all_states = list(dict.fromkeys(states1 + states2))  # Preserve order
    n = len(all_states)

    def to_array(states, matrix):
        arr = np.zeros((n, n))
        state_idx = {s: i for i, s in enumerate(states)}
        for i, s_from in enumerate(all_states):
            if s_from not in state_idx:
                continue
            si = state_idx[s_from]
            for j, s_to in enumerate(all_states):
                if s_to not in state_idx:
                    continue
                sj = state_idx[s_to]
                if si < len(matrix) and sj < len(matrix[si]):
                    arr[i][j] = matrix[si][sj]
        return arr

    a1 = to_array(states1, m1)
    a2 = to_array(states2, m2)
    return float(np.linalg.norm(a1 - a2, 'fro'))


def analyze_at_cutoff(observations: list[str], cutoff: int,
                      full_result=None) -> dict:
    """Compute metrics using only the first `cutoff` observations."""
    obs_subset = observations[:cutoff]
    norms = [normalize_observation(o) for o in obs_subset]
    unique = set(norms)

    # Cycle detection on subset
    lock_in, period = detect_cycle_exact(obs_subset, min_period=3, max_period=50, min_confirmations=3)
    if lock_in is None:
        lock_in, period = detect_cycle_similarity(obs_subset, min_period=3, max_period=50,
                                                   threshold=0.85, min_confirmations=3)

    # Vocabulary distribution
    word_counts = Counter(norms)
    top_5 = word_counts.most_common(5)

    # Build transition matrix using the full run's cycle vocabulary for state assignment
    if full_result and full_result.cycle_vocabulary:
        states = assign_states(obs_subset, full_result.cycle_vocabulary)
    else:
        states = [normalize_observation(o) for o in obs_subset]

    state_names, matrix = build_transition_matrix(states)

    # Cycle state fraction
    n_cycle_states = sum(1 for s in state_names if s.startswith("cycle_"))
    cycle_obs_count = sum(1 for s in states if s.startswith("cycle_"))

    return {
        "cutoff": cutoff,
        "n_observations": len(obs_subset),
        "n_unique": len(unique),
        "unique_ratio": len(unique) / len(obs_subset),
        "cycle_detected": lock_in is not None,
        "lock_in_obs": lock_in,
        "cycle_period": period,
        "top_5_vocabulary": [(w, c) for w, c in top_5],
        "n_states": len(state_names),
        "n_cycle_states": n_cycle_states,
        "cycle_obs_fraction": cycle_obs_count / len(obs_subset),
        # For matrix convergence analysis
        "_state_names": state_names,
        "_matrix": matrix,
    }


def analyze_run_truncation(run_id: int) -> dict:
    """Full truncation analysis for a single Mode A run."""
    text_path = TEXT_DIR / f"run_{run_id:03d}_text.txt"
    text = text_path.read_text()

    # Full analysis first (for reference cycle vocabulary)
    full_result = analyze_run(text)

    results_at_cutoffs = []
    for cutoff in CUTOFFS:
        r = analyze_at_cutoff(full_result.observations, cutoff, full_result)
        results_at_cutoffs.append(r)

    # Transition matrix convergence: Frobenius distance between consecutive cutoffs
    matrix_distances = []
    for i in range(1, len(results_at_cutoffs)):
        prev = results_at_cutoffs[i - 1]
        curr = results_at_cutoffs[i]
        dist = frobenius_distance(
            prev["_matrix"], curr["_matrix"],
            prev["_state_names"], curr["_state_names"]
        )
        matrix_distances.append({
            "from_cutoff": prev["cutoff"],
            "to_cutoff": curr["cutoff"],
            "frobenius_distance": round(dist, 6),
        })

    # Distance from each cutoff's matrix to the final (1000-obs) matrix
    final = results_at_cutoffs[-1]
    distances_to_final = []
    for r in results_at_cutoffs[:-1]:
        dist = frobenius_distance(
            r["_matrix"], final["_matrix"],
            r["_state_names"], final["_state_names"]
        )
        distances_to_final.append({
            "cutoff": r["cutoff"],
            "frobenius_to_final": round(dist, 6),
        })

    # Clean up internal fields before returning
    clean_cutoffs = []
    for r in results_at_cutoffs:
        clean = {k: v for k, v in r.items() if not k.startswith("_")}
        clean["unique_ratio"] = round(clean["unique_ratio"], 4)
        clean["cycle_obs_fraction"] = round(clean["cycle_obs_fraction"], 4)
        clean_cutoffs.append(clean)

    return {
        "run_id": run_id,
        "full_analysis": {
            "n_observations": full_result.n_observations,
            "n_unique": full_result.n_unique,
            "unique_ratio": round(full_result.unique_ratio, 4),
            "has_cycle": full_result.has_cycle,
            "lock_in_obs": full_result.lock_in_obs,
            "cycle_period": full_result.cycle_period,
        },
        "cutoff_metrics": clean_cutoffs,
        "matrix_convergence": matrix_distances,
        "distances_to_final": distances_to_final,
    }


def main():
    mode_a_runs = identify_mode_a_runs()
    print(f"Found {len(mode_a_runs)} Mode A runs: {mode_a_runs}\n")

    all_results = []
    for run_id in mode_a_runs:
        print(f"Analyzing Run {run_id}...", end=" ", flush=True)
        result = analyze_run_truncation(run_id)
        all_results.append(result)

        fa = result["full_analysis"]
        print(f"lock-in={fa['lock_in_obs']}, period={fa['cycle_period']}, "
              f"unique={fa['n_unique']}/{fa['n_observations']}")

    # --- Aggregate analysis ---
    print(f"\n{'='*70}")
    print(f"AGGREGATE TRUNCATION EQUIVALENCE (N={len(all_results)} Mode A runs)")
    print(f"{'='*70}\n")

    # C2: Metric stability across cutoffs
    print("C2: METRIC STABILITY ACROSS CUTOFFS")
    print("-" * 70)
    print(f"{'Cutoff':>8s} | {'N unique':>10s} | {'Unique %':>10s} | "
          f"{'Cycle?':>8s} | {'Period':>8s} | {'Cycle frac':>10s}")
    print("-" * 70)

    for ci, cutoff in enumerate(CUTOFFS):
        uniques = [r["cutoff_metrics"][ci]["n_unique"] for r in all_results]
        ratios = [r["cutoff_metrics"][ci]["unique_ratio"] for r in all_results]
        cycle_detected = [r["cutoff_metrics"][ci]["cycle_detected"] for r in all_results]
        periods = [r["cutoff_metrics"][ci]["cycle_period"] for r in all_results
                   if r["cutoff_metrics"][ci]["cycle_period"] is not None]
        cycle_fracs = [r["cutoff_metrics"][ci]["cycle_obs_fraction"] for r in all_results]

        n_with_cycle = sum(cycle_detected)
        mean_period = np.mean(periods) if periods else float('nan')

        print(f"{cutoff:>8d} | {np.mean(uniques):>10.1f} | {np.mean(ratios):>10.1%} | "
              f"{n_with_cycle:>5d}/{len(all_results):<2d} | "
              f"{mean_period:>8.1f} | {np.mean(cycle_fracs):>10.1%}")

    # C2: How close is unique_ratio at each cutoff to the final value?
    print(f"\nC2: UNIQUE RATIO RELATIVE ERROR vs FINAL (cutoff 1000)")
    print("-" * 50)
    for ci, cutoff in enumerate(CUTOFFS[:-1]):
        errors = []
        for r in all_results:
            ratio_at_cutoff = r["cutoff_metrics"][ci]["unique_ratio"]
            ratio_at_final = r["cutoff_metrics"][-1]["unique_ratio"]
            if ratio_at_final > 0:
                rel_error = abs(ratio_at_cutoff - ratio_at_final) / ratio_at_final
                errors.append(rel_error)
        if errors:
            print(f"  Cutoff {cutoff:>4d}: mean relative error = {np.mean(errors):.1%}, "
                  f"median = {np.median(errors):.1%}, max = {np.max(errors):.1%}")

    # C2: Period detection accuracy at each cutoff
    print(f"\nC2: CYCLE PERIOD DETECTION ACCURACY vs FINAL")
    print("-" * 50)
    for ci, cutoff in enumerate(CUTOFFS[:-1]):
        correct = 0
        detected = 0
        total = 0
        for r in all_results:
            final_period = r["full_analysis"]["cycle_period"]
            if final_period is None:
                continue
            total += 1
            cutoff_period = r["cutoff_metrics"][ci]["cycle_period"]
            if cutoff_period is not None:
                detected += 1
                if cutoff_period == final_period:
                    correct += 1
        print(f"  Cutoff {cutoff:>4d}: detected={detected}/{total}, "
              f"correct period={correct}/{total} "
              f"({correct/total:.0%})" if total > 0 else "")

    # C3: Transition matrix convergence
    print(f"\n{'='*70}")
    print(f"C3: TRANSITION MATRIX CONVERGENCE")
    print(f"{'='*70}\n")

    print("Frobenius distance to final matrix (1000 obs):")
    print("-" * 50)
    for ci, cutoff in enumerate(CUTOFFS[:-1]):
        dists = [r["distances_to_final"][ci]["frobenius_to_final"]
                 for r in all_results if ci < len(r["distances_to_final"])]
        if dists:
            print(f"  Cutoff {cutoff:>4d}: mean={np.mean(dists):.4f}, "
                  f"median={np.median(dists):.4f}, max={np.max(dists):.4f}")

    print("\nConsecutive matrix distances:")
    print("-" * 50)
    for ci in range(len(CUTOFFS) - 1):
        dists = [r["matrix_convergence"][ci]["frobenius_distance"]
                 for r in all_results if ci < len(r["matrix_convergence"])]
        if dists:
            print(f"  {CUTOFFS[ci]:>4d} -> {CUTOFFS[ci+1]:>4d}: mean={np.mean(dists):.4f}, "
                  f"median={np.median(dists):.4f}")

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Clean for JSON serialization
    output = {
        "n_runs": len(all_results),
        "cutoffs": CUTOFFS,
        "runs": all_results,
    }
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
