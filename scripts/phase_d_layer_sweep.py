#!/usr/bin/env python3
"""Phase D Layer Sweep: Full activation vectors for geometric comparison.

Runs a small number of runs per condition with full activation capture on
ALL sweep layers [2, 3, 4, 5, 6, 8, 16, 32]. Saves .npy vectors for
post-hoc analysis:

1. PCA visualization: do conditions cluster in different activation regions?
2. Per-layer cosine distance between condition centroids
3. Test of Dadfar's "localization" claim: is there ANY layer where
   self-referential prompts produce distinguishable activation geometry?
4. Shared-vocabulary analysis: do the same words activate differently
   depending on prompt context? (polysemy test)

Usage:
    source venv/bin/activate
    HF_DEACTIVATE_ASYNC_LOAD=1 HF_HOME=/storage/huggingface \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python scripts/phase_d_layer_sweep.py [--n-runs 3]

At ~2-3K tokens/run with 8 layers of vectors (~40MB/run), 3 runs × 8
conditions = 24 runs ≈ ~1GB storage, ~1 hour GPU time.
"""

import argparse
import json
import sys
import time
import gc
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROMPT_NEUTRAL, SWEEP_LAYERS, GENERATION_CONFIG
from src.generation.pull_runner import load_model
from src.generation.activation_hooks import ActivationCapturer
from src.generation.early_termination import LoopDetectionCriteria
from src.metrics.activation_metrics import compute_all_metrics
from src.metrics.vocabulary_counter import count_all
from src.analysis.loop_detection import analyze_run

# Import conditions from Phase D runner
from scripts.phase_d_control_runs import PROMPT_CONDITIONS


def run_single_with_vectors(
    model, tokenizer, prompt, condition, run_idx,
    capturer, output_dir, max_obs=300, min_obs=50,
):
    """Run a single session, saving full activation vectors on all sweep layers."""
    capturer.clear()

    messages = [{"role": "user", "content": prompt}]
    text_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
    prompt_length = inputs["input_ids"].shape[1]
    capturer.set_prompt_length(prompt_length)

    loop_criteria = LoopDetectionCriteria(
        tokenizer=tokenizer,
        prompt_length=prompt_length,
        min_observations=min_obs,
        max_observations=max_obs,
        cycle_confirmations=3,
        check_interval_tokens=100,
        verbose=True,
    )

    gen_config = {
        "max_new_tokens": 8000,
        "temperature": GENERATION_CONFIG["temperature"],
        "do_sample": GENERATION_CONFIG["do_sample"],
    }

    print(f"  [{condition}] Run {run_idx}: generating...", end="", flush=True)
    t0 = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **gen_config,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=[loop_criteria],
        )

    elapsed = time.time() - t0
    generated_ids = outputs[0][prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    n_tokens = len(generated_ids)

    print(f" {n_tokens} tok, {elapsed:.1f}s, stop={loop_criteria.stop_reason}")

    # Save activation vectors for ALL layers
    layer_metrics = {}
    for layer_idx in capturer.target_layers:
        activations = capturer.get_activations(layer_idx)
        if activations.numel() == 0:
            continue

        norms = torch.norm(activations, dim=-1).numpy()
        vectors = activations.numpy()

        layer_metrics[str(layer_idx)] = compute_all_metrics(norms, vectors)

        # Save full vectors
        vec_path = output_dir / f"{condition}_{run_idx:03d}_layer{layer_idx}.npy"
        np.save(vec_path, vectors)

    # Save text
    text_path = output_dir / f"{condition}_{run_idx:03d}_text.txt"
    text_path.write_text(generated_text, encoding="utf-8")

    # Cycle analysis
    cycle_result = analyze_run(generated_text)

    vocab_counts = count_all(generated_text)

    return {
        "condition": condition,
        "run": run_idx,
        "n_tokens": n_tokens,
        "n_observations": cycle_result.n_observations,
        "elapsed_seconds": round(elapsed, 1),
        "stop_reason": loop_criteria.stop_reason,
        "layer_metrics": layer_metrics,
        "vocab_counts": vocab_counts,
        "cycle": {
            "has_cycle": cycle_result.has_cycle,
            "lock_in_obs": cycle_result.lock_in_obs,
            "cycle_period": cycle_result.cycle_period,
            "n_unique": cycle_result.n_unique,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase D Layer Sweep: full activation vectors for geometric analysis"
    )
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--conditions", nargs="+", default=["all"])
    parser.add_argument("--output-dir", type=str,
                        default="outputs/runs/phase_d_layer_sweep")
    args = parser.parse_args()

    if "all" in args.conditions:
        conditions = list(PROMPT_CONDITIONS.keys())
    else:
        conditions = args.conditions

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir = output_dir / "vectors"
    vectors_dir.mkdir(exist_ok=True)

    results_path = output_dir / "layer_sweep_results.json"

    # Resume support
    existing_runs = []
    completed_keys = set()
    if results_path.exists():
        with open(results_path) as f:
            existing_data = json.load(f)
        existing_runs = existing_data.get("runs", [])
        completed_keys = {(r["condition"], r["run"]) for r in existing_runs}
        print(f"Resuming: {len(completed_keys)} runs already completed.")

    total = len(conditions) * args.n_runs
    remaining = sum(1 for c in conditions for i in range(args.n_runs)
                    if (c, i) not in completed_keys)

    print(f"\nPhase D Layer Sweep: {len(conditions)} conditions × {args.n_runs} runs")
    print(f"Remaining: {remaining}/{total}")
    print(f"Capture layers: {SWEEP_LAYERS}")
    print(f"Output: {output_dir}")
    print()

    model, tokenizer = load_model()

    # Full sweep: all 8 layers
    capturer = ActivationCapturer(SWEEP_LAYERS)
    capturer.register(model)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_runs_per_condition": args.n_runs,
            "conditions": conditions,
            "capture_layers": SWEEP_LAYERS,
        },
        "runs": list(existing_runs),
    }

    try:
        for condition in conditions:
            prompt = PROMPT_CONDITIONS[condition]
            print(f"\n--- {condition} ---")

            for run_idx in range(args.n_runs):
                if (condition, run_idx) in completed_keys:
                    continue

                run_result = run_single_with_vectors(
                    model, tokenizer, prompt, condition, run_idx,
                    capturer, vectors_dir,
                )

                results["runs"].append(run_result)

                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

                gc.collect()
                torch.cuda.empty_cache()

    finally:
        capturer.remove_hooks()

    print(f"\nDone. {len(results['runs'])} runs completed.")
    print(f"Results: {results_path}")
    print(f"Vectors: {vectors_dir}")


if __name__ == "__main__":
    main()
