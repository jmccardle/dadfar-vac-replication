"""Unified data loaders with graceful skip for missing files."""

import json
import warnings
from pathlib import Path
from typing import Any, Optional

PROJECT = Path(__file__).resolve().parent.parent.parent


def _load_json(relpath: str) -> Optional[dict]:
    """Load a JSON file relative to project root. Returns None if missing."""
    p = PROJECT / relpath
    if not p.exists():
        warnings.warn(f"Data not found, skipping: {relpath}")
        return None
    with open(p) as f:
        return json.load(f)


def load_all_data() -> dict:
    """Load all analysis data files. Missing files produce None values."""
    data = {}

    # --- Raw run data ---
    data["extended_baseline"] = _load_json(
        "outputs/runs/extended_baseline/extended_baseline_results.json")
    data["phase_d"] = _load_json(
        "outputs/runs/phase_d_controls/phase_d_results.json")
    data["phase_d_layer_sweep"] = _load_json(
        "outputs/runs/phase_d_layer_sweep/layer_sweep_results.json")
    data["d3_runs"] = _load_json(
        "outputs/runs/phase_d3_temperature/temperature_ablation_results.json")
    data["zenodo_baseline"] = _load_json(
        "zenodo/data/qwen_baseline_n50.json")
    data["zenodo_descriptive"] = _load_json(
        "zenodo/data/qwen_descriptive_control.json")
    data["zenodo_layer_sweep_70b"] = _load_json(
        "zenodo/data/llama_layer_sweep_70b.json")

    # --- Analysis results ---
    data["truncation"] = _load_json(
        "outputs/analysis/extended_truncation_analysis.json")
    data["b3b4"] = _load_json(
        "outputs/analysis/phase_b_spectral_vac.json")
    data["e1"] = _load_json(
        "outputs/analysis/phase_e1_cross_prompt.json")
    data["e2"] = _load_json(
        "outputs/analysis/phase_e2_vac_controlled.json")
    data["d3_analysis"] = _load_json(
        "outputs/analysis/phase_d3_temperature.json")
    data["layer_sweep_geo"] = _load_json(
        "outputs/analysis/layer_sweep_geometric_analysis.json")

    # --- Cross-model replication data ---
    # Each model uses the unified pipeline (scripts/llama_replication.py) which
    # produces phase1_baseline/, phase3_controls/, phase4_directions/ under
    # the model's output directory.
    MODEL_DIRS = {
        "llama_8b": ["local_replication", "llama_8b_replication"],
        "mistral_7b": ["mistral_replication"],
        "gemma_9b": ["gemma_replication"],
        "llama_70b": ["cloud_replication"],
    }
    for model_key, candidate_dirs in MODEL_DIRS.items():
        for d in candidate_dirs:
            base = f"outputs/runs/{d}"
            p1 = _load_json(f"{base}/phase1_baseline/phase1_results.json")
            if p1 is not None:
                data[f"{model_key}_phase1"] = p1
                data[f"{model_key}_phase2"] = _load_json(
                    f"{base}/phase2_truncation/truncation_analysis.json")
                data[f"{model_key}_phase3"] = _load_json(
                    f"{base}/phase3_controls/phase3_results.json")
                data[f"{model_key}_phase4"] = _load_json(
                    f"{base}/phase4_directions/phase4_results.json")
                break
        else:
            # No directory found for this model
            for phase in ["phase1", "phase2", "phase3", "phase4"]:
                data.setdefault(f"{model_key}_{phase}", None)

    # --- Compliance test data ---
    for model_key, fname in [
        ("gemma_9b", "gemma_compliance_test.json"),
        ("mistral_7b", "mistral_compliance_test.json"),
        ("llama_8b", "llama_8b_compliance_test.json"),
    ]:
        data[f"{model_key}_compliance"] = _load_json(f"outputs/runs/{fname}")

    loaded = sum(1 for v in data.values() if v is not None)
    total = len(data)
    print(f"Loaded {loaded}/{total} data files.")
    return data


# --- Extraction helpers ---

def get_token_lengths_by_mode(data: dict) -> Optional[dict]:
    """Extract token lengths grouped by mode from truncation analysis.

    Returns dict: {"full": [int, ...], "short": [int, ...]}
    """
    trunc = data.get("truncation")
    if trunc is None:
        return None
    result = {"full": [], "short": []}
    for r in trunc["runs"]:
        mode = r["mode"]  # "full" or "short"
        result[mode].append(r["n_tokens"])
    return result


def get_lock_in_by_condition(data: dict) -> Optional[dict]:
    """Extract per-condition lock-in values from Phase D raw runs.

    Returns dict: {condition: [int, ...]}
    """
    pd = data.get("phase_d")
    if pd is None:
        return None
    from collections import defaultdict
    result = defaultdict(list)
    for r in pd["runs"]:
        c = r["condition"]
        li = r["cycle"]["lock_in_obs"]
        if li is not None:
            result[c].append(li)
        else:
            result[c].append(r["n_observations"])
    return dict(result)


def get_lock_in_by_temperature(data: dict) -> Optional[dict]:
    """Extract per-temperature lock-in values from D3 analysis.

    Returns dict: {0.3: [float, ...], 0.7: [...], 1.0: [...]}
    """
    d3 = data.get("d3_analysis")
    if d3 is None:
        return None
    result = {}
    for key in ["T_0.3", "T_0.7", "T_1.0"]:
        if key in d3:
            temp = float(key.split("_")[1])
            result[temp] = d3[key]["lock_in"]["values"]
    return result


def get_metric_convergence(data: dict) -> Optional[dict]:
    """Extract median relative error by cutoff and metric from truncation analysis.

    Returns dict: {metric: {cutoff: median_rel_error}}
    Only uses 'full' mode runs.
    """
    trunc = data.get("truncation")
    if trunc is None:
        return None
    import numpy as np

    # Collect per-metric, per-cutoff errors across runs
    metrics = ["mean_norm", "convergence_ratio", "mean_token_similarity",
               "spectral_power_low", "norm_std", "autocorr_lag1"]
    layer = "8"  # primary analysis layer

    result = {m: {} for m in metrics}
    full_runs = [r for r in trunc["runs"] if r["mode"] == "full"]

    for r in full_runs:
        stab = r.get("activation_stability", {}).get(layer, {})
        for metric in metrics:
            entries = stab.get(metric, [])
            for entry in entries:
                cutoff = entry["cutoff"]
                err = entry["rel_error"]
                result[metric].setdefault(cutoff, []).append(err)

    # Compute medians
    for metric in metrics:
        for cutoff in result[metric]:
            vals = result[metric][cutoff]
            result[metric][cutoff] = float(np.median(vals))

    return result


def get_f_statistics(data: dict) -> Optional[dict]:
    """Extract F-statistics by layer from layer sweep.

    Returns dict: {layer_int: {"f_norm": float, "f_cosine": float}}
    """
    geo = data.get("layer_sweep_geo")
    if geo is None:
        return None
    fstats = geo.get("f_statistics", {})
    result = {}
    for layer_str, v in fstats.items():
        result[int(layer_str)] = {
            "f_norm": v.get("f_norm", float("nan")),
            "f_cosine": v.get("f_cosine", float("nan")),
        }
    return result


def get_centroid_cosines(data: dict) -> Optional[dict]:
    """Extract centroid cosine similarity stats by layer.

    Returns dict: {layer_int: {"mean": float, "min": float, "baseline_min": float}}
    """
    geo = data.get("layer_sweep_geo")
    if geo is None:
        return None
    cs = geo.get("centroid_similarities", {})
    result = {}
    for layer_str, v in cs.items():
        result[int(layer_str)] = {
            "mean": v.get("mean", float("nan")),
            "min": v.get("min", float("nan")),
            "baseline_min": v.get("baseline_min", float("nan")),
        }
    return result


def get_topic_direction_cosines(data: dict) -> Optional[dict]:
    """Extract topic direction cosine similarities by layer.

    Returns dict: {layer_int: {"baseline_mean": float, "others_mean": float}}
    """
    geo = data.get("layer_sweep_geo")
    if geo is None:
        return None
    td = geo.get("topic_directions_all_layers", {})
    result = {}
    for layer_str, v in td.items():
        result[int(layer_str)] = {
            "baseline_mean": v.get("baseline_mean_cosine", float("nan")),
            "others_mean": v.get("non_baseline_mean_cosine", float("nan")),
        }
    return result


def get_vac_survival(data: dict) -> Optional[dict]:
    """Extract VAC survival counts across the three test levels.

    Returns dict with Level 1 (Zenodo partial), Level 2 (extended),
    Level 3 (per-condition controlled).
    """
    b = data.get("b3b4")
    e2 = data.get("e2")
    if b is None and e2 is None:
        return None

    result = {}

    if b is not None:
        b4 = b.get("b4_vac", {})
        n_raw = len(b4.get("significant_raw", []))
        n_partial = len(b4.get("significant_partial", []))
        result["level1_zenodo"] = {"n_raw": n_raw, "n_partial": n_partial}

        ext = b4.get("extended_mode_a", {})
        if ext:
            ext_corrs = ext.get("correlations", [])
            n_ext_raw = sum(1 for c in ext_corrs if c.get("p_raw", 1) < 0.05)
            n_ext_partial = sum(1 for c in ext_corrs if c.get("p_partial", 1) < 0.05)
            result["level2_extended"] = {
                "n_raw": n_ext_raw,
                "n_partial": n_ext_partial,
            }

    if e2 is not None:
        pc = e2.get("per_condition", {})
        result["level3_per_condition"] = {
            c: v.get("n_sig_raw", 0) for c, v in pc.items()
        }

    return result


def get_compliance_matrix(data: dict) -> Optional[dict]:
    """Extract cross-model prompt compliance data.

    Returns dict: {"models": [str], "conditions": [str],
                    "n_obs": [[int]], "status": [[str]]}
    """
    models = []
    model_data = {}
    for model_key in ["llama_8b", "mistral_7b", "gemma_9b", "llama_70b"]:
        comp = data.get(f"{model_key}_compliance")
        if comp is not None:
            models.append(model_key)
            model_data[model_key] = comp

    if not models:
        return None

    # Use the condition order from the first available model
    conditions = list(model_data[models[0]].keys())

    n_obs = []
    status = []
    for mk in models:
        row_obs = []
        row_stat = []
        for cond in conditions:
            entry = model_data[mk].get(cond, {})
            row_obs.append(entry.get("n_observations", 0))
            row_stat.append(entry.get("status", "MISSING"))
        n_obs.append(row_obs)
        status.append(row_stat)

    return {
        "models": models,
        "conditions": conditions,
        "n_obs": n_obs,
        "status": status,
    }


def get_vac_scatter_data(data: dict) -> Optional[dict]:
    """Extract per-run vocab counts + activation metrics for scatter rebuttal.

    Uses Qwen zenodo baseline (introspective N=50) and descriptive control
    (mirror prompts N=25) to replicate Dadfar's scatter format.

    Returns dict with:
        "baseline": {"vocab": [int], "metric": [float]}
        "descriptive": {"vocab": [int], "metric": [float]}  (or absent)
    """
    zenodo = data.get("zenodo_baseline")
    if zenodo is None:
        return None

    result = {}

    # Baseline introspective: mirror vs spectral_power_low at layer 8
    bv, bm = [], []
    for r in zenodo["runs"]:
        mirror = r.get("vocab_counts", {}).get("mirror", 0)
        sp = r.get("layer_metrics", {}).get("8", {}).get("spectral_power_low")
        if sp is not None:
            bv.append(mirror)
            bm.append(sp)
    result["baseline"] = {"vocab": bv, "metric": bm}

    # Descriptive control: mirror prompts only
    desc = data.get("zenodo_descriptive")
    if desc is not None:
        dv, dm = [], []
        for r in desc["runs"]:
            if r.get("target_word") != "mirror":
                continue
            mirror = r.get("vocab_counts", {}).get("mirror", 0)
            sp = r.get("layer_metrics", {}).get("8", {}).get("spectral_power_low")
            if sp is not None:
                dv.append(mirror)
                dm.append(sp)
        if dv:
            result["descriptive"] = {"vocab": dv, "metric": dm}

    return result


def get_llama_layer_sweep(data: dict) -> Optional[dict]:
    """Extract Llama 70B layer sweep data from Zenodo.

    Returns dict: {layer_int: {"intro_delta": float, "depth_pct": float}}
    """
    import numpy as np
    sweep = data.get("zenodo_layer_sweep_70b")
    if sweep is None:
        return None

    result = {}
    for layer_str, layer_data in sweep.get("by_layer", {}).items():
        stats = layer_data.get("stats", {})
        result[int(layer_str)] = {
            "intro_delta": stats.get("intro_delta", 0),
            "depth_pct": stats.get("depth_pct", 0) * 100,
        }
    return result


def get_b4_heatmap_data(data: dict) -> Optional[dict]:
    """Extract all B4 correlations for heatmap (vocab × metric → r_partial).

    Returns dict: {"vocabs": [str], "metrics": [str],
                    "r_partial": [[float]], "significant": [[bool]]}
    """
    b = data.get("b3b4")
    if b is None:
        return None
    import numpy as np

    corrs = b["b4_vac"]["all_correlations"]
    vocabs = sorted(set(c["vocab"] for c in corrs))
    metrics = sorted(set(c["metric"] for c in corrs))

    r_mat = np.full((len(vocabs), len(metrics)), np.nan)
    sig_mat = np.full((len(vocabs), len(metrics)), False)

    v_idx = {v: i for i, v in enumerate(vocabs)}
    m_idx = {m: i for i, m in enumerate(metrics)}

    for c in corrs:
        i, j = v_idx[c["vocab"]], m_idx[c["metric"]]
        r_mat[i, j] = c["r_partial"]
        sig_mat[i, j] = c.get("p_partial", 1.0) < 0.05

    return {
        "vocabs": vocabs,
        "metrics": metrics,
        "r_partial": r_mat.tolist(),
        "significant": sig_mat.tolist(),
    }
