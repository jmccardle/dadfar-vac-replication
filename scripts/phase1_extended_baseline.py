#!/usr/bin/env python3
"""
Phase 1 Extended: Run N=50 Pull Methodology baseline with full activation capture
and a token cap sufficient to eliminate truncation (Mode C).

DIFFERENCES FROM REPLICATION (phase1_generate_baseline.py):
  - max_new_tokens raised from 16000 to 28000 to ensure all runs reach obs 1000
    (16K cap caused 40% truncation; worst-case run needed ~25K tokens)
  - Activation hooks verified working (transformers 5.x tuple/tensor fix)
  - Saves .npy activation vectors for all SWEEP_LAYERS

This script represents our first methodological extension beyond Dadfar (2026).
The higher token cap means our results are NOT directly comparable to Zenodo data
for truncation-sensitive metrics (spectral power, etc.), but all runs will have
complete 1000-observation trajectories for dynamical systems analysis.

For strict replication with Dadfar's parameters, use phase1_generate_baseline.py.

Usage:
    python scripts/phase1_extended_baseline.py [--n-runs 50] [--output-dir outputs/runs/extended_baseline]

Estimated runtime: ~50 hours on RTX 4090 (verbose runs produce more tokens)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROMPT_NEUTRAL, SWEEP_LAYERS, GENERATION_CONFIG
from src.generation.pull_runner import load_model
from src.generation.batch_runner import run_baseline_batch


# Override: raise token cap to eliminate Mode C truncation.
# Analysis of 50 text-only runs showed:
#   - 20/50 runs truncated at 16K tokens (Mode C)
#   - Most verbose run (Run 13): 636 obs at 16K, estimated 25.2 tok/obs → ~25K needed
#   - 28K provides ~12% headroom over worst case
EXTENDED_GENERATION_CONFIG = {**GENERATION_CONFIG, "max_new_tokens": 28000}


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 Extended: Baseline with higher token cap + activation capture"
    )
    parser.add_argument("--n-runs", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="outputs/runs/extended_baseline")
    parser.add_argument("--capture-layers", type=int, nargs="+", default=SWEEP_LAYERS)
    parser.add_argument("--no-save-vectors", action="store_true")
    parser.add_argument("--results-file", type=str, default="extended_baseline_results.json")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    model, tokenizer = load_model()

    # Temporarily patch generation config for extended runs
    import src.config
    original_config = src.config.GENERATION_CONFIG.copy()
    src.config.GENERATION_CONFIG.update(EXTENDED_GENERATION_CONFIG)

    print(f"\nPhase 1 Extended: {args.n_runs} baseline Pull Methodology sessions")
    print(f"Token cap: {EXTENDED_GENERATION_CONFIG['max_new_tokens']} (extended from {original_config['max_new_tokens']})")
    print(f"Capturing layers: {args.capture_layers}")
    print(f"Output: {output_dir}")
    print(f"Save vectors: {not args.no_save_vectors}")
    print()

    try:
        results = run_baseline_batch(
            model=model,
            tokenizer=tokenizer,
            prompt=PROMPT_NEUTRAL,
            n_runs=args.n_runs,
            output_dir=output_dir,
            capture_layers=args.capture_layers,
            save_vectors=not args.no_save_vectors,
            results_file=args.results_file,
        )
    finally:
        # Restore original config
        src.config.GENERATION_CONFIG.update(original_config)

    print(f"\nDone. {len(results['runs'])} runs completed.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
