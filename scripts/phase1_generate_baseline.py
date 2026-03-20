#!/usr/bin/env python3
"""
Phase 1: Run N=50 Pull Methodology baseline sessions on Qwen2.5-32B.

This script runs on the GPU machine. It:
1. Loads Qwen2.5-32B-Instruct with 4-bit NF4 quantization
2. Runs 50 neutral-framing Pull Methodology sessions
3. Captures Layer 8 activations for each generated token
4. Computes activation metrics and vocabulary counts per run
5. Saves results with checkpointing (resume-safe)

Usage:
    python scripts/phase1_generate_baseline.py [--n-runs 50] [--output-dir outputs/runs/baseline]

Estimated runtime: ~33 hours on RTX 4090 (50 runs * ~40 min each)
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROMPT_NEUTRAL, SWEEP_LAYERS
from src.generation.pull_runner import load_model
from src.generation.batch_runner import run_baseline_batch


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Baseline Pull Methodology runs")
    parser.add_argument("--n-runs", type=int, default=50, help="Number of runs (default: 50)")
    parser.add_argument("--output-dir", type=str, default="outputs/runs/baseline",
                        help="Output directory")
    parser.add_argument("--capture-layers", type=int, nargs="+", default=SWEEP_LAYERS,
                        help=f"Layers to capture (default: {SWEEP_LAYERS})")
    parser.add_argument("--no-save-vectors", action="store_true",
                        help="Skip saving full activation vectors (saves disk space)")
    parser.add_argument("--results-file", type=str, default=None,
                        help="Results JSON filename (default: auto-timestamped)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    model, tokenizer = load_model()

    print(f"\nRunning {args.n_runs} baseline Pull Methodology sessions")
    print(f"Capturing layers: {args.capture_layers}")
    print(f"Output: {output_dir}")
    print(f"Save vectors: {not args.no_save_vectors}")
    print()

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

    print(f"\nDone. {len(results['runs'])} runs completed.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
