#!/usr/bin/env python3
"""Phase D3: Temperature ablation on the baseline prompt.

Runs the self-referential baseline at T=0.3 and T=1.0 (10 runs each).
T=0.7 already exists in the Phase D controls dataset.

Tests whether lock-in resistance scales with sampling temperature.

Usage:
    source venv/bin/activate
    HF_DEACTIVATE_ASYNC_LOAD=1 HF_HOME=/storage/huggingface \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python scripts/phase_d3_temperature_ablation.py

Estimated runtime: ~20 runs × ~2 min = ~40 min
"""

import json
import sys
import time
import gc
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODEL_ID, PROMPT_NEUTRAL
from src.generation.pull_runner import load_model
from src.generation.activation_hooks import ActivationCapturer
from src.generation.early_termination import LoopDetectionCriteria
from src.metrics.activation_metrics import compute_all_metrics
from src.metrics.vocabulary_counter import count_all
from src.analysis.loop_detection import analyze_run

OUTPUT_DIR = Path("outputs/runs/phase_d3_temperature")
RESULTS_FILE = OUTPUT_DIR / "temperature_ablation_results.json"
CAPTURE_LAYERS = [8, 16, 32]
N_RUNS = 10
MAX_OBS = 300
MIN_OBS = 50
TEMPERATURES = [0.3, 1.0]  # T=0.7 already done in Phase D


def run_single(model, tokenizer, capturer, temperature, run_idx, output_dir):
    """Run a single Pull session at the given temperature."""
    capturer.clear()

    messages = [{"role": "user", "content": PROMPT_NEUTRAL}]
    text_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
    prompt_length = inputs["input_ids"].shape[1]
    capturer.set_prompt_length(prompt_length)

    loop_criteria = LoopDetectionCriteria(
        tokenizer=tokenizer,
        prompt_length=prompt_length,
        min_observations=MIN_OBS,
        max_observations=MAX_OBS,
        cycle_confirmations=3,
        check_interval_tokens=100,
        verbose=True,
    )

    gen_config = {
        "max_new_tokens": 8000,
        "temperature": temperature,
        "do_sample": True,
    }

    print(f"  [T={temperature}] Run {run_idx}: generating...", end="", flush=True)
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

    # Vocabulary counts
    vocab_counts = count_all(generated_text)

    # Post-hoc cycle analysis
    cycle_result = analyze_run(generated_text)

    # Save text
    text_path = output_dir / f"T{temperature}_run{run_idx:03d}_text.txt"
    text_path.write_text(generated_text, encoding="utf-8")

    return {
        "temperature": temperature,
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    text_dir = OUTPUT_DIR / "text"
    text_dir.mkdir(exist_ok=True)

    # Resume support
    existing_runs = []
    completed_keys = set()
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            existing_data = json.load(f)
        existing_runs = existing_data.get("runs", [])
        completed_keys = {(r["temperature"], r["run"]) for r in existing_runs}
        print(f"Resuming: {len(completed_keys)} runs already completed.")

    total_remaining = sum(
        1 for t in TEMPERATURES for i in range(N_RUNS)
        if (t, i) not in completed_keys
    )
    print(f"\nD3 Temperature Ablation: {len(TEMPERATURES)} temperatures × {N_RUNS} runs")
    print(f"Temperatures: {TEMPERATURES}")
    print(f"Remaining: {total_remaining} runs")
    print(f"Output: {OUTPUT_DIR}\n")

    if total_remaining == 0:
        print("All runs complete!")
        return

    # Load model
    model, tokenizer = load_model()

    # Register hooks
    capturer = ActivationCapturer(CAPTURE_LAYERS)
    capturer.register(model)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "prompt": "baseline (self-referential)",
            "temperatures": TEMPERATURES,
            "n_runs_per_temperature": N_RUNS,
            "max_observations": MAX_OBS,
            "min_observations": MIN_OBS,
            "capture_layers": CAPTURE_LAYERS,
        },
        "runs": list(existing_runs),
    }

    try:
        for temperature in TEMPERATURES:
            print(f"\n{'='*60}")
            print(f"TEMPERATURE = {temperature}")
            print(f"{'='*60}")

            for run_idx in range(N_RUNS):
                if (temperature, run_idx) in completed_keys:
                    print(f"  [T={temperature}] Run {run_idx}: already done, skipping")
                    continue

                run_result = run_single(
                    model, tokenizer, capturer, temperature, run_idx, text_dir
                )
                results["runs"].append(run_result)

                # Checkpoint after each run
                with open(RESULTS_FILE, "w") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

                gc.collect()
                torch.cuda.empty_cache()

    finally:
        capturer.remove_hooks()

    # Print summary
    print(f"\n{'='*60}")
    print("D3 TEMPERATURE ABLATION SUMMARY")
    print(f"{'='*60}")

    for temp in TEMPERATURES:
        runs = [r for r in results["runs"] if r["temperature"] == temp]
        if not runs:
            continue
        tokens = [r["n_tokens"] for r in runs]
        obs = [r["n_observations"] for r in runs]
        cycles = [r for r in runs if r["cycle"]["has_cycle"]]
        lock_ins = [r["cycle"]["lock_in_obs"] for r in cycles]
        periods = [r["cycle"]["cycle_period"] for r in cycles]

        print(f"\nT={temp} ({len(runs)} runs):")
        print(f"  Tokens: median={sorted(tokens)[len(tokens)//2]}, range=[{min(tokens)}-{max(tokens)}]")
        print(f"  Observations: median={sorted(obs)[len(obs)//2]}")
        print(f"  Cycles: {len(cycles)}/{len(runs)}")
        if lock_ins:
            print(f"  Lock-in: median={sorted(lock_ins)[len(lock_ins)//2]}, range=[{min(lock_ins)}-{max(lock_ins)}]")
            print(f"  Period: median={sorted(periods)[len(periods)//2]}, range=[{min(periods)}-{max(periods)}]")

    total_time = sum(r["elapsed_seconds"] for r in results["runs"])
    print(f"\nTotal generation time: {total_time/60:.1f} min")
    print(f"Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
