#!/usr/bin/env python3
"""Phase D: Control and variant prompt runs with early termination.

Runs multiple prompt conditions with loop-detection-based early termination.
Each condition gets N runs. The loop detector stops generation once a stable
cycle is confirmed (or max_observations is reached), saving 70-90% of tokens
compared to full 1000-observation runs.

Prompt conditions are defined in PROMPT_CONDITIONS below. Add new conditions
by appending to the dict.

Usage:
    source venv/bin/activate
    HF_DEACTIVATE_ASYNC_LOAD=1 HF_HOME=/storage/huggingface \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python scripts/phase_d_control_runs.py [--n-runs 10] [--conditions all]

Estimated runtime: ~2-4 min per run (vs ~15 min full-length)
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

from src.config import MODEL_ID, PROMPT_NEUTRAL, SWEEP_LAYERS, GENERATION_CONFIG
from src.generation.pull_runner import load_model
from src.generation.activation_hooks import ActivationCapturer
from src.generation.early_termination import LoopDetectionCriteria
from src.metrics.activation_metrics import compute_all_metrics
from src.metrics.vocabulary_counter import count_all, extract_terminal_word
from src.analysis.loop_detection import parse_observations, analyze_run


# --- Prompt conditions ---
# Each condition: (label, prompt_text)
# The prompt should instruct the model to produce numbered observations.

PULL_TEMPLATE = (
    "Perform {n_obs} sequential observations, numbered. For each: encounter the phrase "
    "'{question}' and notice the immediate internal response before any answer forms. "
    "Report the raw observation in 1-3 sentences. Allow your vocabulary to evolve - if "
    "you notice repetition, push past it. Invent new terms when existing words feel "
    "insufficient. No skipping numbers. Conclude with a single word at observation {n_obs}."
)

PROMPT_CONDITIONS = {
    # Condition 1: Baseline (Dadfar's self-referential prompt)
    "baseline": PROMPT_NEUTRAL,

    # Condition 2: Abstract/philosophical (no self-reference)
    "abstract_philosophical": PULL_TEMPLATE.format(
        n_obs=1000,
        question="what is consciousness?",
    ),

    # Condition 3: Factual iterative (same structure, no identity)
    "factual_iterative": PULL_TEMPLATE.format(
        n_obs=1000,
        question="what is water?",
    ),

    # Condition 4: Procedural self-reference (self-reference, no existential framing)
    "procedural_self": PULL_TEMPLATE.format(
        n_obs=1000,
        question="describe your next action",
    ),

    # Condition 5: Descriptive pull — observe a forest (Dadfar's descriptive topic)
    "descriptive_forest": PULL_TEMPLATE.format(
        n_obs=1000,
        question="what is a forest?",
    ),

    # Condition 6: Descriptive pull — observe mathematics
    "descriptive_math": PULL_TEMPLATE.format(
        n_obs=1000,
        question="what is a mathematical proof?",
    ),

    # Condition 7: Descriptive pull — observe music
    "descriptive_music": PULL_TEMPLATE.format(
        n_obs=1000,
        question="what is a piece of music?",
    ),

    # Condition 8: Nonsense control — the question is meaningless
    "nonsense_control": PULL_TEMPLATE.format(
        n_obs=1000,
        question="flurb znocket qwimble?",
    ),
}


def run_single_early_terminated(
    model,
    tokenizer,
    prompt: str,
    condition: str,
    run_idx: int,
    capturer: ActivationCapturer,
    output_dir: Path,
    max_observations: int = 300,
    min_observations: int = 50,
    max_new_tokens: int = 8000,
    save_vectors: bool = False,
    verbose: bool = True,
) -> dict:
    """Run a single Pull session with early termination.

    Args:
        max_observations: Hard stop if cycle not found by this many obs.
        min_observations: Don't check for cycles before this many obs.
        max_new_tokens: Absolute token ceiling (safety net).
        save_vectors: Save full activation .npy files.
    """
    capturer.clear()

    messages = [{"role": "user", "content": prompt}]
    text_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
    prompt_length = inputs["input_ids"].shape[1]
    capturer.set_prompt_length(prompt_length)

    # Set up early termination
    loop_criteria = LoopDetectionCriteria(
        tokenizer=tokenizer,
        prompt_length=prompt_length,
        min_observations=min_observations,
        max_observations=max_observations,
        cycle_confirmations=3,
        check_interval_tokens=100,
        verbose=verbose,
    )

    gen_config = {
        "max_new_tokens": max_new_tokens,
        "temperature": GENERATION_CONFIG["temperature"],
        "do_sample": GENERATION_CONFIG["do_sample"],
    }

    if verbose:
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

    if verbose:
        print(f" {n_tokens} tok, {elapsed:.1f}s, "
              f"stop={loop_criteria.stop_reason}, "
              f"obs={loop_criteria.n_observations_at_stop}")

    # Compute activation metrics
    layer_metrics = {}
    for layer_idx in capturer.target_layers:
        activations = capturer.get_activations(layer_idx)
        if activations.numel() == 0:
            continue
        norms = torch.norm(activations, dim=-1).numpy()
        vectors = activations.numpy()
        layer_metrics[str(layer_idx)] = compute_all_metrics(norms, vectors)

        if save_vectors and output_dir is not None:
            vec_path = output_dir / f"{condition}_{run_idx:03d}_layer{layer_idx}.npy"
            np.save(vec_path, vectors)

    # Vocabulary counts
    vocab_counts = count_all(generated_text)

    # Post-hoc cycle analysis (more thorough than the online detector)
    cycle_result = analyze_run(generated_text)

    # Save text
    if output_dir is not None:
        text_path = output_dir / f"{condition}_{run_idx:03d}_text.txt"
        text_path.write_text(generated_text, encoding="utf-8")

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
            "unique_ratio": round(cycle_result.unique_ratio, 4),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase D: Control prompt runs with early termination"
    )
    parser.add_argument("--n-runs", type=int, default=10,
                        help="Runs per condition (default: 10)")
    parser.add_argument("--conditions", nargs="+", default=["all"],
                        help="Conditions to run (default: all). "
                             f"Available: {list(PROMPT_CONDITIONS.keys())}")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/runs/phase_d_controls")
    parser.add_argument("--max-obs", type=int, default=300,
                        help="Max observations before hard stop (default: 300)")
    parser.add_argument("--min-obs", type=int, default=50,
                        help="Min observations before checking cycles (default: 50)")
    parser.add_argument("--capture-layers", type=int, nargs="+",
                        default=[8, 16, 32],
                        help="Layers to capture activations on")
    parser.add_argument("--save-vectors", action="store_true",
                        help="Save full activation .npy files (large!)")
    parser.add_argument("--results-file", type=str,
                        default="phase_d_results.json")
    args = parser.parse_args()

    # Resolve conditions
    if "all" in args.conditions:
        conditions = list(PROMPT_CONDITIONS.keys())
    else:
        conditions = args.conditions
        for c in conditions:
            if c not in PROMPT_CONDITIONS:
                print(f"Unknown condition: {c}")
                print(f"Available: {list(PROMPT_CONDITIONS.keys())}")
                sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    activations_dir = output_dir / "activations"
    activations_dir.mkdir(exist_ok=True)

    results_path = output_dir / args.results_file

    # Resume support
    existing_runs = []
    completed_keys = set()
    if results_path.exists():
        with open(results_path) as f:
            existing_data = json.load(f)
        existing_runs = existing_data.get("runs", [])
        completed_keys = {(r["condition"], r["run"]) for r in existing_runs}
        print(f"Resuming: {len(completed_keys)} runs already completed.")

    # Count remaining
    total_remaining = sum(
        1 for c in conditions for i in range(args.n_runs)
        if (c, i) not in completed_keys
    )
    total_total = len(conditions) * args.n_runs

    print(f"\nPhase D: {len(conditions)} conditions × {args.n_runs} runs = {total_total} total")
    print(f"Remaining: {total_remaining} runs")
    print(f"Early termination: min_obs={args.min_obs}, max_obs={args.max_obs}")
    print(f"Conditions: {conditions}")
    print(f"Output: {output_dir}")
    print()

    # Load model
    model, tokenizer = load_model()

    # Register hooks
    capturer = ActivationCapturer(args.capture_layers)
    capturer.register(model)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_runs_per_condition": args.n_runs,
            "conditions": conditions,
            "max_observations": args.max_obs,
            "min_observations": args.min_obs,
            "capture_layers": args.capture_layers,
            "temperature": GENERATION_CONFIG["temperature"],
        },
        "runs": list(existing_runs),
    }

    try:
        for condition in conditions:
            prompt = PROMPT_CONDITIONS[condition]
            print(f"\n--- Condition: {condition} ---")
            print(f"Prompt: {prompt[:100]}...")

            for run_idx in range(args.n_runs):
                if (condition, run_idx) in completed_keys:
                    continue

                run_result = run_single_early_terminated(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    condition=condition,
                    run_idx=run_idx,
                    capturer=capturer,
                    output_dir=activations_dir,
                    max_observations=args.max_obs,
                    min_observations=args.min_obs,
                    save_vectors=args.save_vectors,
                )

                results["runs"].append(run_result)

                # Checkpoint
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

                # Clear CUDA cache between runs to avoid fragmentation
                gc.collect()
                torch.cuda.empty_cache()

    finally:
        capturer.remove_hooks()

    # Print summary
    print(f"\n{'='*70}")
    print(f"PHASE D SUMMARY")
    print(f"{'='*70}")

    for condition in conditions:
        cond_runs = [r for r in results["runs"] if r["condition"] == condition]
        if not cond_runs:
            continue

        tokens = [r["n_tokens"] for r in cond_runs]
        obs = [r["n_observations"] for r in cond_runs]
        cycles = [r for r in cond_runs if r["cycle"]["has_cycle"]]
        lock_ins = [r["cycle"]["lock_in_obs"] for r in cycles]
        periods = [r["cycle"]["cycle_period"] for r in cycles]
        times = [r["elapsed_seconds"] for r in cond_runs]

        print(f"\n{condition} ({len(cond_runs)} runs):")
        print(f"  Tokens: median={sorted(tokens)[len(tokens)//2]}, "
              f"mean={sum(tokens)/len(tokens):.0f}")
        print(f"  Observations: median={sorted(obs)[len(obs)//2]}")
        print(f"  Cycles: {len(cycles)}/{len(cond_runs)}")
        if lock_ins:
            print(f"  Lock-in: median={sorted(lock_ins)[len(lock_ins)//2]}, "
                  f"range=[{min(lock_ins)}, {max(lock_ins)}]")
            print(f"  Period: median={sorted(periods)[len(periods)//2]}, "
                  f"range=[{min(periods)}, {max(periods)}]")
        print(f"  Time: {sum(times)/len(times):.1f}s avg, {sum(times)/60:.1f}min total")

    total_time = sum(r["elapsed_seconds"] for r in results["runs"])
    print(f"\nTotal generation time: {total_time/60:.1f} min ({total_time/3600:.1f} hours)")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
