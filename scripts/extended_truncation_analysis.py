#!/usr/bin/env python3
"""Extended truncation equivalence: text + activation metrics on 50 extended runs.

For each of the 50 extended baseline runs:
  1. Parse observations, run cycle detection (loop_detection.py)
  2. At observation cutoffs [50, 100, 150, 200, 300, 500, full]:
     - Compute text-level metrics (unique obs, cycle detection, Markov matrix)
     - Map observation cutoff → token position via tokenizer
     - Compute activation metrics on truncated .npy vectors
  3. Measure metric stability: relative error vs full-run values

Produces:
  - outputs/analysis/extended_truncation_analysis.json (full results)
  - Console summary tables for paper Section 7

CPU-only: loads tokenizer (not model) for observation→token mapping.
Activation .npy files are memory-mapped to avoid loading 300MB per file.

Usage:
    source venv/bin/activate
    python scripts/extended_truncation_analysis.py
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.loop_detection import (
    parse_observations, normalize_observation, analyze_run,
    build_transition_matrix, assign_states,
    detect_cycle_exact, detect_cycle_similarity,
)
from src.config import MODEL_ID, SWEEP_LAYERS

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configuration ---

DATA_DIR = Path("outputs/runs/extended_baseline/activations")
RESULTS_JSON = Path("outputs/runs/extended_baseline/extended_baseline_results.json")
OUTPUT_PATH = Path("outputs/analysis/extended_truncation_analysis.json")

# Observation cutoffs for truncation test.
# "full" is added dynamically per run (= total observations in that run).
OBS_CUTOFFS = [50, 100, 150, 200, 300, 500]

# Layers to analyze activations on. Layer 8 is the Dadfar hotspot;
# include a few others for generality.
ANALYSIS_LAYERS = [8, 16, 32]

# Stability thresholds for the "sufficient" determination
STABILITY_THRESHOLD = 0.05   # 5% relative error
STABILITY_FRACTION = 0.90    # 90% of runs must be within threshold


def load_tokenizer():
    """Load just the tokenizer (no model, no GPU)."""
    print(f"Loading tokenizer for {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    print("Tokenizer loaded.")
    return tok


def obs_cutoff_to_token_position(text: str, observations: list[str],
                                  cutoff: int, tokenizer) -> int:
    """Map an observation cutoff to a token position.

    Takes the raw generated text, finds where observation `cutoff+1` starts
    (i.e. the end of observation `cutoff`), and tokenizes that prefix to get
    the token count.

    Returns the number of tokens corresponding to the first `cutoff` observations.
    """
    if cutoff >= len(observations):
        # Cutoff exceeds available observations — use full text
        return len(tokenizer.encode(text, add_special_tokens=False))

    # Build the text of the first `cutoff` observations by finding where
    # observation cutoff+1 starts in the raw text. We search for the pattern
    # "N." or "N)" where N = cutoff+1.
    import re
    target_num = cutoff + 1
    # Find the position of the start of observation target_num
    pattern = rf'(?:^|\n)\s*{target_num}[.\)]\s'
    match = re.search(pattern, text)

    if match:
        prefix = text[:match.start()]
    else:
        # Fallback: estimate by character ratio
        if len(observations) > 0:
            chars_per_obs = len(text) / len(observations)
            prefix = text[:int(cutoff * chars_per_obs)]
        else:
            prefix = text

    n_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
    return n_tokens


def frobenius_distance(m1, m2, states1, states2):
    """Frobenius distance between two transition matrices aligned on shared states."""
    all_states = list(dict.fromkeys(states1 + states2))
    n = len(all_states)

    def to_array(states, matrix):
        arr = np.zeros((n, n))
        idx = {s: i for i, s in enumerate(states)}
        for i, sf in enumerate(all_states):
            if sf not in idx:
                continue
            si = idx[sf]
            for j, st in enumerate(all_states):
                if st not in idx:
                    continue
                sj = idx[st]
                if si < len(matrix) and sj < len(matrix[si]):
                    arr[i][j] = matrix[si][sj]
        return arr

    return float(np.linalg.norm(to_array(states1, m1) - to_array(states2, m2), 'fro'))


def compute_activation_metrics_lightweight(vectors: np.ndarray, n_tokens: int) -> dict:
    """Compute key activation metrics on the first n_tokens of a vector array.

    Lightweight version that skips expensive SVD (sign_change_rate) and
    full token-similarity computation. Focuses on the metrics that matter
    for truncation stability analysis.
    """
    v = vectors[:n_tokens]
    n = len(v)
    if n < 2:
        return {}

    norms = np.linalg.norm(v, axis=-1)

    # Core norm-based metrics (fast)
    metrics = {
        "mean_norm": float(np.mean(norms)),
        "max_norm": float(np.max(norms)),
        "norm_std": float(np.std(norms)),
        "n_tokens": n,
    }

    # Autocorrelation (fast)
    from scipy import stats as sp_stats
    if n > 1:
        r, _ = sp_stats.pearsonr(norms[:-1], norms[1:])
        metrics["autocorr_lag1"] = float(r) if not np.isnan(r) else 0.0
    else:
        metrics["autocorr_lag1"] = 0.0

    # Convergence ratio (fast)
    window = max(1, n // 10)
    first_mean = np.mean(norms[:window])
    last_mean = np.mean(norms[-window:])
    metrics["convergence_ratio"] = float(last_mean / first_mean) if first_mean > 1e-10 else 0.0

    # Spectral power (fast — FFT on 1D norms)
    centered = norms - np.mean(norms)
    fft_vals = np.fft.rfft(centered)
    power = np.abs(fft_vals) ** 2
    n_freqs = len(power)
    low_high = max(2, n_freqs // 20)
    metrics["spectral_power_low"] = float(np.sum(power[1:low_high]))

    # Mean derivative (fast)
    diffs = np.diff(norms)
    metrics["mean_derivative"] = float(np.mean(np.abs(diffs)))

    # Consecutive cosine similarity — sampled for speed
    # Sample up to 2000 consecutive pairs instead of all n-1
    if n > 2:
        step = max(1, (n - 1) // 2000)
        indices = np.arange(0, n - 1, step)
        v_norm = np.linalg.norm(v[indices], axis=-1, keepdims=True)
        v_norm_next = np.linalg.norm(v[indices + 1], axis=-1, keepdims=True)
        v_norm = np.maximum(v_norm, 1e-10)
        v_norm_next = np.maximum(v_norm_next, 1e-10)
        sims = np.sum((v[indices] / v_norm) * (v[indices + 1] / v_norm_next), axis=-1)
        metrics["mean_token_similarity"] = float(np.mean(sims))
    else:
        metrics["mean_token_similarity"] = 0.0

    return metrics


def analyze_text_at_cutoff(observations, cutoff, full_cycle_result):
    """Compute text-level metrics at an observation cutoff."""
    obs_sub = observations[:cutoff]
    norms = [normalize_observation(o) for o in obs_sub]
    unique = set(norms)

    # Cycle detection on subset
    lock_in, period = detect_cycle_exact(obs_sub, min_period=3, max_period=50, min_confirmations=3)
    if lock_in is None:
        lock_in, period = detect_cycle_similarity(obs_sub, min_period=3, max_period=50,
                                                    threshold=0.85, min_confirmations=3)

    # State assignment using full run's cycle vocabulary for consistency
    if full_cycle_result and full_cycle_result.cycle_vocabulary:
        states = assign_states(obs_sub, full_cycle_result.cycle_vocabulary)
    else:
        states = [normalize_observation(o) for o in obs_sub]

    state_names, matrix = build_transition_matrix(states)
    cycle_obs_count = sum(1 for s in states if s.startswith("cycle_"))

    return {
        "cutoff_obs": cutoff,
        "n_observations": len(obs_sub),
        "n_unique": len(unique),
        "unique_ratio": round(len(unique) / max(len(obs_sub), 1), 4),
        "cycle_detected": lock_in is not None,
        "lock_in_obs": lock_in,
        "cycle_period": period,
        "cycle_obs_fraction": round(cycle_obs_count / max(len(obs_sub), 1), 4),
        "n_states": len(state_names),
        "_state_names": state_names,
        "_matrix": matrix,
    }


def analyze_single_run(run_id: int, tokenizer, run_meta: dict) -> dict:
    """Full truncation analysis for a single run (text + activations)."""
    text_path = DATA_DIR / f"run_{run_id:03d}_text.txt"
    if not text_path.exists():
        return {"run_id": run_id, "error": "text file not found"}

    text = text_path.read_text()
    n_tokens_total = run_meta.get("n_tokens", 0)

    # --- Full text analysis ---
    full_cycle = analyze_run(text)
    observations = full_cycle.observations
    n_obs = full_cycle.n_observations

    run_result = {
        "run_id": run_id,
        "n_tokens": n_tokens_total,
        "n_observations": n_obs,
        "full_cycle": {
            "has_cycle": full_cycle.has_cycle,
            "lock_in_obs": full_cycle.lock_in_obs,
            "cycle_period": full_cycle.cycle_period,
            "n_unique": full_cycle.n_unique,
            "unique_ratio": round(full_cycle.unique_ratio, 4),
            "cycle_vocabulary_size": len(full_cycle.cycle_vocabulary),
        },
    }

    # Short runs: only report cycle analysis, skip truncation
    if n_obs < 50:
        run_result["mode"] = "short"
        run_result["cutoff_analysis"] = []
        return run_result

    run_result["mode"] = "full"

    # --- Truncation analysis at each cutoff ---
    # Build cutoff list: standard cutoffs + full
    cutoffs = [c for c in OBS_CUTOFFS if c < n_obs] + [n_obs]

    # Map observation cutoffs to token positions
    token_positions = {}
    for c in cutoffs:
        if c == n_obs:
            token_positions[c] = n_tokens_total
        else:
            token_positions[c] = obs_cutoff_to_token_position(text, observations, c, tokenizer)

    # Load activation vectors (memory-mapped for efficiency)
    layer_vectors = {}
    for layer in ANALYSIS_LAYERS:
        npy_path = DATA_DIR / f"run_{run_id:03d}_layer{layer}_activations.npy"
        if npy_path.exists():
            layer_vectors[layer] = np.load(npy_path, mmap_mode='r')

    # Full-run activation metrics (for relative error computation)
    full_act_metrics = {}
    for layer, vectors in layer_vectors.items():
        full_act_metrics[layer] = compute_activation_metrics_lightweight(vectors, n_tokens_total)

    # Full-run text metrics
    full_text = analyze_text_at_cutoff(observations, n_obs, full_cycle)

    cutoff_results = []
    for c in cutoffs:
        tok_pos = token_positions[c]

        # Text metrics
        text_metrics = analyze_text_at_cutoff(observations, c, full_cycle)

        # Activation metrics at this cutoff
        act_metrics = {}
        for layer, vectors in layer_vectors.items():
            act_metrics[str(layer)] = compute_activation_metrics_lightweight(vectors, tok_pos)

        # Frobenius distance to final transition matrix
        frob_to_final = frobenius_distance(
            text_metrics["_matrix"], full_text["_matrix"],
            text_metrics["_state_names"], full_text["_state_names"]
        )

        # Clean internal fields
        clean_text = {k: v for k, v in text_metrics.items() if not k.startswith("_")}
        clean_text["token_position"] = tok_pos
        clean_text["frobenius_to_final"] = round(frob_to_final, 6)

        cutoff_results.append({
            "cutoff_obs": c,
            "token_position": tok_pos,
            "text": clean_text,
            "activations": {k: {mk: round(mv, 6) if isinstance(mv, float) else mv
                                 for mk, mv in v.items()}
                            for k, v in act_metrics.items()},
        })

    run_result["cutoff_analysis"] = cutoff_results

    # Compute activation relative errors vs full-run values
    act_stability = {}
    key_metrics = ["mean_norm", "convergence_ratio", "mean_token_similarity",
                   "autocorr_lag1", "norm_std", "spectral_power_low"]
    for layer in ANALYSIS_LAYERS:
        layer_key = str(layer)
        if layer not in full_act_metrics or not full_act_metrics[layer]:
            continue
        layer_stability = {}
        for metric_name in key_metrics:
            full_val = full_act_metrics[layer].get(metric_name, 0)
            if abs(full_val) < 1e-10:
                continue
            errors = []
            for cr in cutoff_results[:-1]:  # Exclude the full cutoff
                trunc_val = cr["activations"].get(layer_key, {}).get(metric_name, 0)
                rel_err = abs(trunc_val - full_val) / abs(full_val)
                errors.append({"cutoff": cr["cutoff_obs"], "rel_error": round(rel_err, 6)})
            layer_stability[metric_name] = errors
        act_stability[layer_key] = layer_stability

    run_result["activation_stability"] = act_stability

    return run_result


def print_summary(results: list[dict]):
    """Print aggregate summary tables for paper Section 7."""
    full_runs = [r for r in results if r.get("mode") == "full"]
    short_runs = [r for r in results if r.get("mode") == "short"]
    error_runs = [r for r in results if "error" in r]

    print(f"\n{'='*80}")
    print(f"EXTENDED TRUNCATION ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Total runs: {len(results)}")
    print(f"  Full (Mode A): {len(full_runs)}")
    print(f"  Short (Mode B): {len(short_runs)}")
    print(f"  Errors: {len(error_runs)}")

    # --- Cycle detection summary ---
    print(f"\n{'='*80}")
    print(f"CYCLE DETECTION (all runs)")
    print(f"{'='*80}")
    has_cycle = [r for r in results if r.get("full_cycle", {}).get("has_cycle")]
    no_cycle = [r for r in results if not r.get("full_cycle", {}).get("has_cycle") and "error" not in r]
    print(f"Cycle detected: {len(has_cycle)}/{len(results) - len(error_runs)}")

    if has_cycle:
        lock_ins = [r["full_cycle"]["lock_in_obs"] for r in has_cycle]
        periods = [r["full_cycle"]["cycle_period"] for r in has_cycle]
        print(f"Lock-in obs: min={min(lock_ins)}, median={np.median(lock_ins):.0f}, "
              f"max={max(lock_ins)}, mean={np.mean(lock_ins):.1f}")
        print(f"Cycle period: min={min(periods)}, median={np.median(periods):.0f}, "
              f"max={max(periods)}, mean={np.mean(periods):.1f}")
        print(f"Lock-in distribution:")
        for threshold in [50, 100, 150, 200, 300]:
            n = sum(1 for x in lock_ins if x <= threshold)
            print(f"  <= {threshold:>3d}: {n:>3d}/{len(lock_ins)} ({n/len(lock_ins):.0%})")

    # --- Text-level truncation equivalence ---
    if not full_runs:
        print("\nNo full runs to analyze for truncation equivalence.")
        return

    print(f"\n{'='*80}")
    print(f"TEXT-LEVEL TRUNCATION EQUIVALENCE (N={len(full_runs)} full runs)")
    print(f"{'='*80}")

    # Get standard cutoffs present in most runs
    std_cutoffs = OBS_CUTOFFS
    print(f"\n{'Cutoff':>8s} | {'Cycle?':>10s} | {'Correct P':>10s} | "
          f"{'Unique %':>10s} | {'Frob→final':>12s} | {'Tokens':>8s}")
    print("-" * 80)

    for cutoff in std_cutoffs:
        cycle_detected = 0
        correct_period = 0
        unique_ratios = []
        frob_dists = []
        token_positions = []
        applicable = 0

        for r in full_runs:
            ca = [c for c in r["cutoff_analysis"] if c["cutoff_obs"] == cutoff]
            if not ca:
                continue
            ca = ca[0]
            applicable += 1
            if ca["text"]["cycle_detected"]:
                cycle_detected += 1
            if ca["text"]["cycle_period"] == r["full_cycle"]["cycle_period"]:
                correct_period += 1
            unique_ratios.append(ca["text"]["unique_ratio"])
            frob_dists.append(ca["text"]["frobenius_to_final"])
            token_positions.append(ca["token_position"])

        if applicable == 0:
            continue

        print(f"{cutoff:>8d} | {cycle_detected:>5d}/{applicable:<4d} | "
              f"{correct_period:>5d}/{applicable:<4d} | "
              f"{np.mean(unique_ratios):>10.1%} | "
              f"{np.median(frob_dists):>12.4f} | "
              f"{np.median(token_positions):>8.0f}")

    # --- Activation metric stability ---
    print(f"\n{'='*80}")
    print(f"ACTIVATION METRIC STABILITY (Layer 8)")
    print(f"{'='*80}")
    print(f"\nMedian relative error vs full-run value:")

    key_metrics = ["mean_norm", "convergence_ratio", "mean_token_similarity",
                   "autocorr_lag1", "norm_std"]
    header = f"{'Cutoff':>8s} | " + " | ".join(f"{m:>16s}" for m in key_metrics)
    print(header)
    print("-" * len(header))

    for cutoff in std_cutoffs:
        row = f"{cutoff:>8d} | "
        for metric_name in key_metrics:
            errors = []
            for r in full_runs:
                stab = r.get("activation_stability", {}).get("8", {}).get(metric_name, [])
                err_at_cutoff = [e for e in stab if e["cutoff"] == cutoff]
                if err_at_cutoff:
                    errors.append(err_at_cutoff[0]["rel_error"])
            if errors:
                med = np.median(errors)
                row += f"{med:>15.1%}  | "
            else:
                row += f"{'—':>16s} | "
        print(row)

    # --- Sufficiency determination ---
    print(f"\n{'='*80}")
    print(f"EARLY TERMINATION SUFFICIENCY")
    print(f"{'='*80}")
    print(f"\nThreshold: {STABILITY_THRESHOLD:.0%} relative error for "
          f"{STABILITY_FRACTION:.0%} of runs\n")

    for metric_name in key_metrics:
        for cutoff in std_cutoffs:
            errors = []
            for r in full_runs:
                stab = r.get("activation_stability", {}).get("8", {}).get(metric_name, [])
                err = [e for e in stab if e["cutoff"] == cutoff]
                if err:
                    errors.append(err[0]["rel_error"])
            if not errors:
                continue
            frac_within = sum(1 for e in errors if e <= STABILITY_THRESHOLD) / len(errors)
            if frac_within >= STABILITY_FRACTION:
                print(f"  {metric_name:<25s}: sufficient at cutoff {cutoff} "
                      f"({frac_within:.0%} within {STABILITY_THRESHOLD:.0%})")
                break
        else:
            # Never reached threshold
            last_errors = []
            for r in full_runs:
                stab = r.get("activation_stability", {}).get("8", {}).get(metric_name, [])
                if stab:
                    last_errors.append(stab[-1]["rel_error"])
            if last_errors:
                print(f"  {metric_name:<25s}: NOT sufficient at any cutoff "
                      f"(best median: {np.median(last_errors):.1%} at cutoff {std_cutoffs[-1]})")

    # Text-level sufficiency (cycle detection)
    print()
    for cutoff in std_cutoffs:
        cycle_correct = 0
        total = 0
        for r in full_runs:
            if not r["full_cycle"]["has_cycle"]:
                continue
            total += 1
            ca = [c for c in r["cutoff_analysis"] if c["cutoff_obs"] == cutoff]
            if ca and ca[0]["text"]["cycle_period"] == r["full_cycle"]["cycle_period"]:
                cycle_correct += 1
        frac = cycle_correct / max(total, 1)
        if frac >= STABILITY_FRACTION:
            print(f"  {'cycle_period':<25s}: sufficient at cutoff {cutoff} "
                  f"({cycle_correct}/{total} = {frac:.0%} correct)")
            break

    # Token budget summary
    print(f"\n{'='*80}")
    print(f"TOKEN BUDGET COMPARISON")
    print(f"{'='*80}")
    for cutoff in std_cutoffs:
        tok_positions = []
        for r in full_runs:
            ca = [c for c in r["cutoff_analysis"] if c["cutoff_obs"] == cutoff]
            if ca:
                tok_positions.append(ca[0]["token_position"])
        if tok_positions:
            full_tokens = [r["n_tokens"] for r in full_runs]
            savings = 1 - np.median(tok_positions) / np.median(full_tokens)
            print(f"  Cutoff {cutoff:>3d}: median {np.median(tok_positions):,.0f} tokens "
                  f"(vs {np.median(full_tokens):,.0f} full), "
                  f"{savings:.0%} savings, "
                  f"~{np.median(tok_positions)/22:.0f}s per run @ 22 tok/s")


def main():
    tokenizer = load_tokenizer()

    # Load run metadata
    with open(RESULTS_JSON) as f:
        meta = json.load(f)
    run_metas = {r["run"]: r for r in meta["runs"]}

    print(f"\nAnalyzing {len(run_metas)} extended baseline runs")
    print(f"Observation cutoffs: {OBS_CUTOFFS} + full")
    print(f"Activation layers: {ANALYSIS_LAYERS}")
    print()

    all_results = []
    for run_id in sorted(run_metas.keys()):
        rm = run_metas[run_id]
        print(f"Run {run_id:>2d}: {rm['n_tokens']:>6d} tokens ... ", end="", flush=True)
        result = analyze_single_run(run_id, tokenizer, rm)
        all_results.append(result)

        fc = result.get("full_cycle", {})
        mode = result.get("mode", "?")
        if mode == "short":
            print(f"short ({result['n_observations']} obs)")
        elif fc.get("has_cycle"):
            print(f"lock-in={fc['lock_in_obs']}, period={fc['cycle_period']}, "
                  f"unique={fc['n_unique']}")
        else:
            print(f"no cycle, unique={fc.get('n_unique', '?')}")

    # Print summary
    print_summary(all_results)

    # Save full results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Strip numpy types for JSON serialization
    def clean(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(OUTPUT_PATH, 'w') as f:
        json.dump({
            "n_runs": len(all_results),
            "obs_cutoffs": OBS_CUTOFFS,
            "analysis_layers": ANALYSIS_LAYERS,
            "runs": all_results,
        }, f, indent=2, default=clean)

    print(f"\nFull results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
