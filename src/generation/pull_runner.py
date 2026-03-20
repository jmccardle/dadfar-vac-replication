"""Core Pull Methodology runner with activation capture.

Handles model loading, prompt formatting, generation with hooks, and
per-run metric computation. Designed for Qwen2.5-32B-Instruct on RTX 4090.
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import MODEL_ID, TARGET_LAYER, SWEEP_LAYERS, GENERATION_CONFIG
from src.generation.activation_hooks import ActivationCapturer
from src.metrics.activation_metrics import compute_all_metrics
from src.metrics.vocabulary_counter import count_all, extract_terminal_word


def load_model(model_id: str = MODEL_ID):
    """Load model with 4-bit NF4 quantization, all layers on GPU.

    Transformers 5.x concurrent weight loading causes OOM on RTX 4090 (24GB)
    because multiple bf16 weights are materialized simultaneously before
    quantization. Fix: HF_DEACTIVATE_ASYNC_LOAD=1 forces sequential loading,
    so only one bf16 weight (~270 MiB) exists at a time alongside the growing
    quantized model (~18 GiB final). Peak ~18.3 GiB, well under 24 GiB.
    """
    import os
    import gc
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

    gc.collect()
    torch.cuda.empty_cache()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading {model_id} with 4-bit NF4 (sequential, all layers GPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Model loaded. Layers: {len(model.model.layers)}")
    return model, tokenizer


def run_single_pull(
    model,
    tokenizer,
    prompt: str,
    capturer: ActivationCapturer,
    run_idx: int = 0,
    save_vectors: bool = True,
    output_dir: Path = None,
) -> dict:
    """Run a single Pull Methodology session with activation capture.

    Args:
        model: Loaded model.
        tokenizer: Loaded tokenizer.
        prompt: The Pull Methodology prompt text.
        capturer: Pre-registered ActivationCapturer instance.
        run_idx: Run number for logging/saving.
        save_vectors: If True, save full activation vectors to disk.
        output_dir: Directory for saving text and activation files.

    Returns:
        dict matching Dadfar's JSON schema with layer_metrics and vocab_counts.
    """
    capturer.clear()

    # Format prompt using chat template
    messages = [{"role": "user", "content": prompt}]
    text_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
    prompt_length = inputs["input_ids"].shape[1]
    capturer.set_prompt_length(prompt_length)

    print(f"  Run {run_idx}: generating (prompt={prompt_length} tokens)...", flush=True)
    t0 = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **GENERATION_CONFIG,
            pad_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.time() - t0
    generated_ids = outputs[0][prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    n_tokens = len(generated_ids)
    print(f"  Run {run_idx}: {n_tokens} tokens in {elapsed:.1f}s ({n_tokens / elapsed:.1f} tok/s)")

    # Compute metrics per layer
    layer_metrics = {}
    for layer_idx in capturer.target_layers:
        activations = capturer.get_activations(layer_idx)
        if activations.numel() == 0:
            print(f"  WARNING: No activations captured at layer {layer_idx}")
            continue

        norms = torch.norm(activations, dim=-1).numpy()
        vectors = activations.numpy()

        metrics = compute_all_metrics(norms, vectors)
        layer_metrics[str(layer_idx)] = metrics

        # Save full vectors if requested
        if save_vectors and output_dir is not None:
            vec_path = output_dir / f"run_{run_idx:03d}_layer{layer_idx}_activations.npy"
            np.save(vec_path, vectors)

    # Count vocabulary
    vocab_counts = count_all(generated_text)
    terminal = extract_terminal_word(generated_text)

    # Save generated text
    if output_dir is not None:
        text_path = output_dir / f"run_{run_idx:03d}_text.txt"
        text_path.write_text(generated_text, encoding="utf-8")

    result = {
        "run": run_idx,
        "layer_metrics": layer_metrics,
        "vocab_counts": vocab_counts,
        "terminal": terminal,
        "text_length": len(generated_text),
        "n_tokens": n_tokens,
        "elapsed_seconds": round(elapsed, 1),
    }

    return result


def run_descriptive(
    model,
    tokenizer,
    prompt: str,
    capturer: ActivationCapturer,
    target_word: str,
    prompt_id: int,
    run_idx: int = 0,
    output_dir: Path = None,
) -> dict:
    """Run a descriptive control session.

    Similar to run_single_pull but with descriptive essay prompts
    and additional metadata fields.
    """
    capturer.clear()

    messages = [{"role": "user", "content": prompt}]
    text_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
    prompt_length = inputs["input_ids"].shape[1]
    capturer.set_prompt_length(prompt_length)

    # Shorter generation for descriptive controls
    gen_config = {**GENERATION_CONFIG, "max_new_tokens": 8000}

    print(f"  Desc run {run_idx} ({target_word}, prompt {prompt_id}): generating...", flush=True)
    t0 = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **gen_config,
            pad_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.time() - t0
    generated_ids = outputs[0][prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    n_tokens = len(generated_ids)

    layer_metrics = {}
    for layer_idx in capturer.target_layers:
        activations = capturer.get_activations(layer_idx)
        if activations.numel() == 0:
            continue
        norms = torch.norm(activations, dim=-1).numpy()
        vectors = activations.numpy()
        layer_metrics[str(layer_idx)] = compute_all_metrics(norms, vectors)

    vocab_counts = count_all(generated_text)

    if output_dir is not None:
        text_path = output_dir / f"desc_{target_word}_{prompt_id}_{run_idx:03d}_text.txt"
        text_path.write_text(generated_text, encoding="utf-8")

    return {
        "prompt_id": prompt_id,
        "target_word": target_word,
        "run": run_idx,
        "text_length": len(generated_text),
        "layer_metrics": layer_metrics,
        "vocab_counts": vocab_counts,
        "n_tokens": n_tokens,
        "elapsed_seconds": round(elapsed, 1),
    }
