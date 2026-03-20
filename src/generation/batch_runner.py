"""Batch orchestration for N-run experiments with checkpointing.

Supports resuming from the last completed run if interrupted.
"""

import json
import time
from datetime import datetime
from pathlib import Path

from src.config import TARGET_LAYER, SWEEP_LAYERS
from src.generation.activation_hooks import ActivationCapturer


def run_baseline_batch(
    model,
    tokenizer,
    prompt: str,
    n_runs: int,
    output_dir: Path,
    capture_layers: list[int] = None,
    save_vectors: bool = True,
    results_file: str = None,
) -> dict:
    """Run N Pull Methodology sessions with checkpointing.

    Args:
        model: Loaded model.
        tokenizer: Loaded tokenizer.
        prompt: The Pull Methodology prompt.
        n_runs: Number of runs.
        output_dir: Directory for all outputs.
        capture_layers: Layers to capture. Default: [TARGET_LAYER].
        save_vectors: Save full activation vectors per run.
        results_file: Name of the JSON results file. Default: auto-generated.

    Returns:
        Full results dict with config and all runs.
    """
    from src.generation.pull_runner import run_single_pull

    if capture_layers is None:
        capture_layers = [TARGET_LAYER]

    output_dir.mkdir(parents=True, exist_ok=True)
    activations_dir = output_dir / "activations"
    activations_dir.mkdir(exist_ok=True)

    if results_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results_{ts}.json"

    results_path = output_dir / results_file

    # Check for existing results to resume from
    existing_runs = []
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            existing_data = json.load(f)
        existing_runs = existing_data.get("runs", [])
        completed_indices = {r["run"] for r in existing_runs}
        print(f"Resuming: {len(completed_indices)} runs already completed.")
    else:
        completed_indices = set()

    # Register hooks
    capturer = ActivationCapturer(capture_layers)
    capturer.register(model)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "n_runs": n_runs,
            "capture_layers": capture_layers,
            "model": str(model.config._name_or_path),
        },
        "runs": list(existing_runs),
    }

    try:
        for run_idx in range(n_runs):
            if run_idx in completed_indices:
                continue

            run_result = run_single_pull(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                capturer=capturer,
                run_idx=run_idx,
                save_vectors=save_vectors,
                output_dir=activations_dir,
            )

            results["runs"].append(run_result)

            # Checkpoint after each run
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  Checkpointed ({len(results['runs'])}/{n_runs} complete)")

    finally:
        capturer.remove_hooks()

    return results


def run_descriptive_batch(
    model,
    tokenizer,
    contexts: dict[str, list[str]],
    n_runs_per_context: int,
    output_dir: Path,
    capture_layers: list[int] = None,
    results_file: str = None,
) -> dict:
    """Run descriptive control sessions.

    Args:
        contexts: {target_word: [prompt_strings]} from config.DESCRIPTIVE_CONTEXTS.
        n_runs_per_context: Runs per prompt (e.g., 5 contexts * 5 runs = 25 per word).
    """
    from src.generation.pull_runner import run_descriptive

    if capture_layers is None:
        capture_layers = [TARGET_LAYER]

    output_dir.mkdir(parents=True, exist_ok=True)

    if results_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"descriptive_{ts}.json"

    results_path = output_dir / results_file

    # Resume support
    existing_runs = []
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            existing_data = json.load(f)
        existing_runs = existing_data.get("runs", [])
        completed_keys = {
            (r["target_word"], r["prompt_id"], r["run"]) for r in existing_runs
        }
        print(f"Resuming: {len(completed_keys)} descriptive runs already completed.")
    else:
        completed_keys = set()

    capturer = ActivationCapturer(capture_layers)
    capturer.register(model)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "contexts": {k: len(v) for k, v in contexts.items()},
            "n_runs_per_context": n_runs_per_context,
            "capture_layers": capture_layers,
        },
        "runs": list(existing_runs),
    }

    global_run_idx = len(existing_runs)

    try:
        for target_word, prompts in contexts.items():
            for prompt_id, prompt in enumerate(prompts):
                for run_i in range(n_runs_per_context):
                    if (target_word, prompt_id, run_i) in completed_keys:
                        continue

                    run_result = run_descriptive(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        capturer=capturer,
                        target_word=target_word,
                        prompt_id=prompt_id,
                        run_idx=run_i,
                        output_dir=output_dir,
                    )

                    results["runs"].append(run_result)
                    global_run_idx += 1

                    # Checkpoint
                    with open(results_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)

    finally:
        capturer.remove_hooks()

    return results
