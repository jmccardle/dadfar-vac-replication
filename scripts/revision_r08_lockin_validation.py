#!/usr/bin/env python3
"""R8: Lock-in Resistance Construct Validation.

Addresses reviewer Point 18: Lock-in resistance is the paper's central
constructive contribution but lacks construct validity testing, test-retest
reliability, and sensitivity analysis.

Three parts:
  (a) Construct validity: correlate lock-in with vocabulary diversity, entropy
  (b) Sensitivity analysis: vary detection parameters K and P_max
  (c) Test-retest reliability: split-half on 50 extended baseline runs

Data: outputs/runs/extended_baseline/extended_baseline_results.json
      outputs/runs/phase_d_controls/phase_d_results.json
      Text files in outputs/runs/extended_baseline/ and phase_d_controls/activations/
Output: outputs/analysis/revision_r08_lockin_validation.json
"""

import json
import re
import math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
DATA_EXTENDED = PROJECT / "outputs" / "runs" / "extended_baseline" / "extended_baseline_results.json"
DATA_PHASE_D = PROJECT / "outputs" / "runs" / "phase_d_controls" / "phase_d_results.json"
TEXT_DIR_EXTENDED = PROJECT / "outputs" / "runs" / "extended_baseline"
TEXT_DIR_PHASE_D = PROJECT / "outputs" / "runs" / "phase_d_controls" / "activations"
OUT = PROJECT / "outputs" / "analysis" / "revision_r08_lockin_validation.json"


def parse_observations(text):
    """Parse numbered observations from Pull Methodology output.

    Returns list of (obs_number, text) tuples.
    """
    pattern = r'(?:^|\n)\s*(\d+)\.\s+(.*?)(?=\n\s*\d+\.|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    observations = []
    for num_str, obs_text in matches:
        num = int(num_str)
        obs_text = obs_text.strip()
        if obs_text and num > 0:
            observations.append((num, obs_text))
    return observations


def normalize_observation(text):
    """Normalize observation for comparison."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def detect_cycle(observations, min_period=3, max_period=50, min_confirmations=3):
    """Detect repeating cycles in observation sequence.

    Returns (has_cycle, lock_in_obs, cycle_period, unique_before_lockin).
    """
    normalized = [normalize_observation(text) for _, text in observations]
    n = len(normalized)

    for start in range(n):
        for period in range(min_period, min(max_period + 1, (n - start) // min_confirmations + 1)):
            # Check if observations[start:start+period] repeat
            pattern = normalized[start:start + period]
            confirmations = 0
            pos = start + period

            while pos + period <= n:
                if normalized[pos:pos + period] == pattern:
                    confirmations += 1
                    pos += period
                else:
                    break

            if confirmations >= min_confirmations:
                lock_in = observations[start][0] if start < len(observations) else start
                unique_before = len(set(normalized[:start]))
                return True, int(lock_in), period, unique_before

    return False, None, None, len(set(normalized))


def compute_vocabulary_diversity(observations, lock_in_idx=None):
    """Compute vocabulary diversity metrics for observations before lock-in.

    Returns dict with:
      - type_token_ratio: unique words / total words
      - unique_obs_ratio: unique observations / total observations
      - shannon_entropy: entropy of word distribution
    """
    if lock_in_idx is not None:
        obs_texts = [text for _, text in observations[:lock_in_idx]]
    else:
        obs_texts = [text for _, text in observations]

    if not obs_texts:
        return {"type_token_ratio": float("nan"),
                "unique_obs_ratio": float("nan"),
                "shannon_entropy": float("nan"),
                "n_observations": 0}

    # Word-level diversity
    all_words = []
    for text in obs_texts:
        words = re.findall(r'\b\w+\b', text.lower())
        all_words.extend(words)

    if not all_words:
        return {"type_token_ratio": float("nan"),
                "unique_obs_ratio": float("nan"),
                "shannon_entropy": float("nan"),
                "n_observations": len(obs_texts)}

    n_types = len(set(all_words))
    n_tokens = len(all_words)
    ttr = n_types / n_tokens if n_tokens > 0 else 0

    # Observation-level diversity
    normalized = [normalize_observation(t) for t in obs_texts]
    unique_obs = len(set(normalized))
    unique_obs_ratio = unique_obs / len(normalized) if normalized else 0

    # Shannon entropy of word distribution
    word_counts = Counter(all_words)
    total = sum(word_counts.values())
    probs = [c / total for c in word_counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)

    return {
        "type_token_ratio": float(ttr),
        "unique_obs_ratio": float(unique_obs_ratio),
        "shannon_entropy": float(entropy),
        "n_observations": len(obs_texts),
        "n_unique_words": n_types,
        "n_total_words": n_tokens,
    }


def main():
    results = {
        "description": "R8: Lock-in resistance construct validation",
    }

    print("=" * 72)
    print("R8: LOCK-IN RESISTANCE CONSTRUCT VALIDATION")
    print("=" * 72)

    # ── Load Phase D data (has cycle info + text files) ─────────────
    with open(DATA_PHASE_D) as f:
        phase_d_data = json.load(f)
    phase_d_runs = phase_d_data.get("runs", phase_d_data.get("data", []))

    # Also load extended baseline for Part (c)
    with open(DATA_EXTENDED) as f:
        ext_data = json.load(f)
    ext_runs = ext_data.get("runs", ext_data.get("data", []))

    # ── Part (a): Construct Validity ────────────────────────────────
    print("\n── Part (a): Construct Validity ──")
    print("Correlating lock-in resistance with pre-lock-in vocabulary diversity\n")

    construct_data = []
    for i, run in enumerate(phase_d_runs):
        cycle = run.get("cycle", {})
        if not cycle.get("has_cycle", False):
            continue

        lock_in = cycle.get("lock_in_obs")
        if lock_in is None:
            continue

        # Text file naming: {condition}_{run:03d}_text.txt
        cond = run.get("condition", "unknown")
        run_num = run.get("run", i)
        text_file = TEXT_DIR_PHASE_D / f"{cond}_{run_num:03d}_text.txt"
        if not text_file.exists():
            # Try alternative naming patterns
            text_files = sorted(TEXT_DIR_PHASE_D.glob(f"{cond}_{run_num:03d}*text*"))
            if not text_files:
                continue
            text_file = text_files[0]

        text = text_file.read_text(errors="replace")
        observations = parse_observations(text)
        if not observations:
            continue

        # Find lock-in index in observation list
        lock_in_idx = None
        for idx, (num, _) in enumerate(observations):
            if num >= lock_in:
                lock_in_idx = idx
                break

        diversity = compute_vocabulary_diversity(observations, lock_in_idx)
        if diversity["n_observations"] < 5:
            continue

        construct_data.append({
            "run": i,
            "condition": cond,
            "lock_in_obs": float(lock_in),
            "cycle_period": cycle.get("cycle_period"),
            "unique_ratio": cycle.get("unique_ratio"),
            **diversity,
        })

    if len(construct_data) >= 5:
        lock_ins = np.array([d["lock_in_obs"] for d in construct_data])
        ttrs = np.array([d["type_token_ratio"] for d in construct_data])
        unique_ratios = np.array([d["unique_obs_ratio"] for d in construct_data])
        entropies = np.array([d["shannon_entropy"] for d in construct_data])

        validity_results = {}
        for name, vals in [("type_token_ratio", ttrs),
                           ("unique_obs_ratio", unique_ratios),
                           ("shannon_entropy", entropies)]:
            mask = np.isfinite(lock_ins) & np.isfinite(vals)
            if mask.sum() >= 5:
                r, p = stats.pearsonr(lock_ins[mask], vals[mask])
                rho, p_s = stats.spearmanr(lock_ins[mask], vals[mask])
                validity_results[name] = {
                    "pearson_r": float(r),
                    "pearson_p": float(p),
                    "spearman_rho": float(rho),
                    "spearman_p": float(p_s),
                    "n": int(mask.sum()),
                    "significant_pearson": bool(p < 0.05),
                    "significant_spearman": bool(p_s < 0.05),
                }
                sig_p = "*" if p < 0.05 else " "
                sig_s = "*" if p_s < 0.05 else " "
                print(f"  Lock-in vs {name:<20}: r={r:>+.3f} (p={p:.4f}{sig_p})  "
                      f"ρ={rho:>+.3f} (p={p_s:.4f}{sig_s})  N={mask.sum()}")
        results["construct_validity"] = {
            "n_runs_analyzed": len(construct_data),
            "correlations": validity_results,
        }
    else:
        print(f"  Insufficient text data found ({len(construct_data)} runs with text)")
        results["construct_validity"] = {"error": "insufficient data",
                                          "n_found": len(construct_data)}

    # ── Part (b): Sensitivity Analysis ──────────────────────────────
    print("\n── Part (b): Sensitivity Analysis ──")
    print("Varying detection parameters K (min_period) and P_max\n")

    # Load Phase D text files for sensitivity analysis
    sensitivity_data = []
    param_configs = [
        (2, 20), (2, 50), (2, 100),
        (3, 20), (3, 50), (3, 100),
        (5, 20), (5, 50), (5, 100),
    ]

    # Collect text from baseline runs (which have text files)
    TEXT_DIR_BASELINE = PROJECT / "outputs" / "runs" / "baseline" / "activations"
    cycling_texts = []
    for i in range(50):
        text_file = TEXT_DIR_BASELINE / f"run_{i:03d}_text.txt"
        if not text_file.exists():
            continue
        text = text_file.read_text(errors="replace")
        observations = parse_observations(text)
        if len(observations) >= 20:
            cycling_texts.append((i, observations))

    print(f"  Found {len(cycling_texts)} runs with ≥20 observations for sensitivity analysis")

    sensitivity_results = {}
    for K, P_max in param_configs:
        config_key = f"K={K}_Pmax={P_max}"
        lock_ins = []
        n_cycling = 0

        for run_idx, observations in cycling_texts:
            has_cycle, lock_in, period, unique = detect_cycle(
                observations, min_period=K, max_period=P_max)
            if has_cycle and lock_in is not None:
                lock_ins.append(lock_in)
                n_cycling += 1

        arr = np.array(lock_ins, dtype=float) if lock_ins else np.array([])
        sensitivity_results[config_key] = {
            "K": K,
            "P_max": P_max,
            "n_total": len(cycling_texts),
            "n_cycling": n_cycling,
            "cycle_rate": n_cycling / len(cycling_texts) if cycling_texts else 0,
            "mean_lockin": float(arr.mean()) if len(arr) > 0 else None,
            "median_lockin": float(np.median(arr)) if len(arr) > 0 else None,
            "std_lockin": float(arr.std(ddof=1)) if len(arr) > 1 else None,
        }

    results["sensitivity_analysis"] = sensitivity_results

    # Reference config: K=3, P_max=50 (paper's default)
    ref = sensitivity_results.get("K=3_Pmax=50", {})

    print(f"\n  {'Config':<16} {'Cycling':>7} {'Rate':>6} {'Mean':>7} {'Median':>7} {'SD':>7}")
    print("  " + "─" * 60)
    for config_key in sorted(sensitivity_results.keys()):
        s = sensitivity_results[config_key]
        mean_str = f"{s['mean_lockin']:>7.1f}" if s['mean_lockin'] is not None else "    N/A"
        med_str = f"{s['median_lockin']:>7.1f}" if s['median_lockin'] is not None else "    N/A"
        std_str = f"{s['std_lockin']:>7.1f}" if s['std_lockin'] is not None else "    N/A"
        marker = " <-- paper default" if config_key == "K=3_Pmax=50" else ""
        print(f"  {config_key:<16} {s['n_cycling']:>7} {s['cycle_rate']:>6.0%} "
              f"{mean_str} {med_str} {std_str}{marker}")

    # Robustness: compare lock-in distributions across configs
    if ref.get("median_lockin") is not None:
        ref_median = ref["median_lockin"]
        max_deviation = 0
        for s in sensitivity_results.values():
            if s["median_lockin"] is not None:
                dev = abs(s["median_lockin"] - ref_median) / ref_median * 100
                max_deviation = max(max_deviation, dev)
        results["sensitivity_summary"] = {
            "reference_config": "K=3_Pmax=50",
            "reference_median": ref_median,
            "max_median_deviation_pct": max_deviation,
            "robust": max_deviation < 50,  # within 50% of reference
        }
        print(f"\n  Max median deviation from reference: {max_deviation:.1f}%")
        print(f"  Assessment: {'ROBUST' if max_deviation < 50 else 'PARAMETER-SENSITIVE'}")

    # ── Part (c): Test-Retest Reliability ───────────────────────────
    print("\n── Part (c): Test-Retest Reliability ──")
    print("Split-half analysis on extended baseline (50 runs)\n")

    # Get lock-in values for all cycling runs from Phase D
    all_lock_ins = []
    for i, run in enumerate(phase_d_runs):
        cycle = run.get("cycle", {})
        if cycle.get("has_cycle", False) and cycle.get("lock_in_obs") is not None:
            all_lock_ins.append((i, float(cycle["lock_in_obs"])))

    if len(all_lock_ins) >= 10:
        # Split by odd/even
        odd = np.array([v for i, v in all_lock_ins if i % 2 == 1])
        even = np.array([v for i, v in all_lock_ins if i % 2 == 0])

        # KS test
        ks_stat, ks_p = stats.ks_2samp(odd, even)
        # Cohen's d between halves
        if len(odd) >= 2 and len(even) >= 2:
            pooled_var = ((len(odd) - 1) * np.var(odd, ddof=1) +
                          (len(even) - 1) * np.var(even, ddof=1)) / (len(odd) + len(even) - 2)
            d_halves = float((np.mean(odd) - np.mean(even)) / np.sqrt(pooled_var)) if pooled_var > 0 else 0
        else:
            d_halves = float("nan")

        # Spearman-Brown corrected reliability (split-half)
        # For this, we need paired data - use first-half/second-half instead
        first_half = np.array([v for i, v in all_lock_ins if i < len(ext_runs) // 2])
        second_half = np.array([v for i, v in all_lock_ins if i >= len(ext_runs) // 2])

        reliability_results = {
            "n_total_cycling": len(all_lock_ins),
            "odd_even_split": {
                "n_odd": len(odd),
                "n_even": len(even),
                "mean_odd": float(odd.mean()),
                "mean_even": float(even.mean()),
                "ks_statistic": float(ks_stat),
                "ks_p_value": float(ks_p),
                "cohens_d": d_halves,
                "distributions_match": bool(ks_p > 0.05),
            },
            "first_second_split": {
                "n_first": len(first_half),
                "n_second": len(second_half),
                "mean_first": float(first_half.mean()) if len(first_half) > 0 else None,
                "mean_second": float(second_half.mean()) if len(second_half) > 0 else None,
            },
        }

        if len(first_half) >= 2 and len(second_half) >= 2:
            ks2, p2 = stats.ks_2samp(first_half, second_half)
            reliability_results["first_second_split"]["ks_statistic"] = float(ks2)
            reliability_results["first_second_split"]["ks_p_value"] = float(p2)

        results["test_retest"] = reliability_results

        print(f"  Total cycling runs: {len(all_lock_ins)}")
        print(f"  Odd/Even split: N_odd={len(odd)}, N_even={len(even)}")
        print(f"    Mean odd: {odd.mean():.1f}, Mean even: {even.mean():.1f}")
        print(f"    KS test: D={ks_stat:.3f}, p={ks_p:.4f}")
        print(f"    Cohen's d: {d_halves:.3f}")
        if ks_p > 0.05:
            print(f"    Distributions match (p > 0.05) → RELIABLE")
        else:
            print(f"    Distributions differ (p < 0.05) → UNRELIABLE")
    else:
        print(f"  Insufficient cycling runs: {len(all_lock_ins)}")
        results["test_retest"] = {"error": "insufficient data"}

    # ── Save ────────────────────────────────────────────────────────
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT}")


if __name__ == "__main__":
    main()
