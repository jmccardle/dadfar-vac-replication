"""Individual figure functions. Each returns a matplotlib Figure."""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

from .style import (
    CONDITION_ORDER, CONDITION_COLORS, CONDITION_LABELS,
    TEMP_COLORS, TEMP_EDGE, MODE_COLORS, METRIC_LABELS,
    single_col, full_width, label_color, label_name,
)
from . import loaders


# ---------------------------------------------------------------------------
# Fig 1: Bimodal histogram (token length by mode)
# ---------------------------------------------------------------------------

def fig_bimodal_histogram(data: dict) -> plt.Figure:
    """Token length distribution coloured by output mode (A/B)."""
    lengths = loaders.get_token_lengths_by_mode(data)
    if lengths is None:
        return _placeholder("Fig 1: Bimodal histogram (data missing)")

    fig, ax = plt.subplots(figsize=single_col(0.65))

    bins = np.arange(0, 22000, 500)
    ax.hist(lengths["short"], bins=bins, color=MODE_COLORS["short"],
            alpha=0.85, label=f"Mode B (n={len(lengths['short'])})",
            edgecolor="white", linewidth=0.4)
    ax.hist(lengths["full"], bins=bins, color=MODE_COLORS["full"],
            alpha=0.85, label=f"Mode A (n={len(lengths['full'])})",
            edgecolor="white", linewidth=0.4)

    ax.set_xlabel("Token count")
    ax.set_ylabel("Number of runs")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 21500)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 2: Spectral scaling (log-log scatter + regression)
# ---------------------------------------------------------------------------

def fig_spectral_scaling(data: dict) -> plt.Figure:
    """Log-log plot of spectral_power_low vs n_tokens with OLS fit."""
    zenodo = data.get("zenodo_baseline")
    b3b4 = data.get("b3b4")
    if zenodo is None or b3b4 is None:
        return _placeholder("Fig 2: Spectral scaling (data missing)")

    # Extract raw data points from zenodo baseline
    tokens = []
    spectral = []
    for r in zenodo["runs"]:
        n = r["n_tokens"]
        lm = r.get("layer_metrics", {}).get("8", {})
        sp = lm.get("spectral_power_low")
        if sp is not None and n > 0:
            tokens.append(n)
            spectral.append(sp)

    tokens = np.array(tokens, dtype=float)
    spectral = np.array(spectral, dtype=float)

    # Regression parameters from analysis
    scaling = b3b4["b3_scaling"]["zenodo_scaling"]
    alpha = scaling["alpha"]
    intercept = scaling["intercept"]
    r_sq = scaling["r_squared"]

    fig, ax = plt.subplots(figsize=single_col(0.75))

    ax.scatter(tokens, spectral, s=22, alpha=0.45, color="#2166AC",
               edgecolors="white", linewidth=0.4, zorder=3)

    # Regression line
    t_line = np.logspace(np.log10(tokens.min()), np.log10(tokens.max()), 100)
    sp_line = 10**intercept * t_line**alpha
    ax.plot(t_line, sp_line, color="#D73027", linewidth=1.5, zorder=2,
            label=f"$\\alpha = {alpha:.2f}$, $R^2 = {r_sq:.3f}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Token count")
    ax.set_ylabel("Spectral power (low)")
    ax.legend(loc="upper left")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 3: Partial correlation heatmap (vocab x metric)
# ---------------------------------------------------------------------------

def fig_partial_correlation_heatmap(data: dict) -> plt.Figure:
    """Heatmap of partial correlations (controlling for n_tokens)."""
    hm = loaders.get_b4_heatmap_data(data)
    if hm is None:
        return _placeholder("Fig 3: Partial correlation heatmap (data missing)")

    r_partial = np.array(hm["r_partial"])
    significant = np.array(hm["significant"])
    vocabs = hm["vocabs"]
    metrics = hm["metrics"]

    # Clean up labels with consistent capitalization
    metric_labels = [METRIC_LABELS.get(m, m.replace("_", " ").title()) for m in metrics]
    vocab_labels = [v.replace("ctrl_", "").replace("_", " ").title() for v in vocabs]

    fig, ax = plt.subplots(figsize=full_width(0.55))

    vmax = max(0.6, np.nanmax(np.abs(r_partial)))
    im = ax.imshow(r_partial, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto")

    # Mark significant cells
    for i in range(len(vocabs)):
        for j in range(len(metrics)):
            if significant[i, j]:
                ax.text(j, i, "*", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="black")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metric_labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(vocabs)))
    ax.set_yticklabels(vocab_labels, fontsize=7.5)

    cb = fig.colorbar(im, ax=ax, shrink=0.8,
                      label="$r_{\\mathrm{partial}}$", pad=0.02)
    cb.ax.tick_params(labelsize=7)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 4: Metric convergence by cutoff
# ---------------------------------------------------------------------------

def fig_metric_convergence(data: dict) -> plt.Figure:
    """Relative error of activation metrics vs observation cutoff."""
    conv = loaders.get_metric_convergence(data)
    if conv is None:
        return _placeholder("Fig 4: Metric convergence (data missing)")

    fig, ax = plt.subplots(figsize=single_col(0.75))

    # Distinguish well-behaved from problematic
    good_metrics = ["mean_norm", "convergence_ratio", "mean_token_similarity", "norm_std"]
    bad_metrics = ["spectral_power_low"]

    for metric in good_metrics:
        if metric not in conv:
            continue
        cutoffs = sorted(conv[metric].keys())
        errors = [conv[metric][c] * 100 for c in cutoffs]  # percent
        ax.plot(cutoffs, errors, "o-", markersize=4,
                label=METRIC_LABELS.get(metric, metric))

    for metric in bad_metrics:
        if metric not in conv:
            continue
        cutoffs = sorted(conv[metric].keys())
        errors = [conv[metric][c] * 100 for c in cutoffs]
        ax.plot(cutoffs, errors, "s--", markersize=4, color="#D73027",
                linewidth=1.5, label=METRIC_LABELS.get(metric, metric))

    ax.axhline(5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Observation cutoff")
    ax.set_ylabel("Median relative error (%)")
    ax.set_yscale("log")
    ax.set_ylim(0.05, 200)
    ax.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 5: Lock-in resistance by condition (strip + box)
# ---------------------------------------------------------------------------

def fig_lock_in_by_condition(data: dict) -> plt.Figure:
    """Strip plot of lock-in observations across 8 prompt conditions."""
    lockins = loaders.get_lock_in_by_condition(data)
    if lockins is None:
        return _placeholder("Fig 5: Lock-in by condition (data missing)")

    fig, ax = plt.subplots(figsize=full_width(0.4))

    conditions = [c for c in CONDITION_ORDER if c in lockins]
    positions = range(len(conditions))

    for i, cond in enumerate(conditions):
        vals = np.array(lockins[cond], dtype=float)
        color = CONDITION_COLORS[cond]

        # Box
        bp = ax.boxplot([vals], positions=[i], widths=0.5,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.2),
                        boxprops=dict(facecolor=color, alpha=0.3, edgecolor=color),
                        whiskerprops=dict(color=color),
                        capprops=dict(color=color))

        # Strip (jittered)
        jitter = np.random.default_rng(42 + i).uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full_like(vals, i) + jitter, vals,
                   s=20, color=color, alpha=0.8, edgecolors="white",
                   linewidth=0.3, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels([label_name(c) for c in conditions],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Lock-in observation")
    ax.set_ylim(0, None)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 6: Lock-in resistance by temperature
# ---------------------------------------------------------------------------

def fig_lock_in_by_temperature(data: dict) -> plt.Figure:
    """Strip + box plot of lock-in across sampling temperatures."""
    lockins = loaders.get_lock_in_by_temperature(data)
    if lockins is None:
        return _placeholder("Fig 6: Temperature ablation (data missing)")

    fig, ax = plt.subplots(figsize=single_col(0.75))

    temps = sorted(lockins.keys())
    for i, temp in enumerate(temps):
        vals = np.array(lockins[temp], dtype=float)
        color = TEMP_COLORS.get(temp, "#333333")
        edge = TEMP_EDGE.get(temp, "#000000")

        bp = ax.boxplot([vals], positions=[i], widths=0.5,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.2),
                        boxprops=dict(facecolor=color, alpha=0.5, edgecolor=edge),
                        whiskerprops=dict(color=edge),
                        capprops=dict(color=edge))

        jitter = np.random.default_rng(42 + i).uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full_like(vals, i) + jitter, vals,
                   s=24, color=edge, alpha=0.7, edgecolors="white",
                   linewidth=0.3, zorder=3)

    ax.set_xticks(range(len(temps)))
    ax.set_xticklabels([f"T = {t}" for t in temps], fontsize=9)
    ax.set_ylabel("Lock-in observation")
    ax.set_ylim(0, None)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 7: F-statistics by layer depth
# ---------------------------------------------------------------------------

def fig_f_statistics(data: dict) -> plt.Figure:
    """F-statistics (norm and cosine) across layers."""
    fstats = loaders.get_f_statistics(data)
    if fstats is None:
        return _placeholder("Fig 7: F-statistics (data missing)")

    fig, ax = plt.subplots(figsize=single_col(0.75))

    layers = sorted(fstats.keys())
    depths = [l / 64 * 100 for l in layers]  # Qwen 32B has 64 layers
    f_norm = [fstats[l]["f_norm"] for l in layers]
    f_cos = [fstats[l]["f_cosine"] for l in layers]

    ax.plot(depths, f_norm, "o-", color="#2166AC", label="$F_{\\mathrm{norm}}$")
    ax.plot(depths, f_cos, "s--", color="#D73027", label="$F_{\\mathrm{cosine}}$")

    # Reference lines
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    # Dadfar hotspot
    ax.axvline(12.5, color="#999999", linestyle="--", linewidth=0.8, alpha=0.6)

    # Position annotation to avoid overlap with legend and data
    y_max = max(max(f_norm), max(f_cos))
    ax.text(14, y_max * 0.45, "Dadfar\n\"hotspot\"", fontsize=7,
            color="#666666", va="center", ha="left")

    ax.set_xlabel("Layer depth (%)")
    ax.set_ylabel("F-statistic")
    ax.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 8: Centroid cosine similarity by layer
# ---------------------------------------------------------------------------

def fig_centroid_cosines(data: dict) -> plt.Figure:
    """Centroid cosine similarity (mean, min, baseline min) across layers."""
    cosines = loaders.get_centroid_cosines(data)
    if cosines is None:
        return _placeholder("Fig 8: Centroid cosines (data missing)")

    fig, ax = plt.subplots(figsize=single_col(0.75))

    layers = sorted(cosines.keys())
    depths = [l / 64 * 100 for l in layers]

    mean_cos = [cosines[l]["mean"] for l in layers]
    min_cos = [cosines[l]["min"] for l in layers]
    bl_min = [cosines[l]["baseline_min"] for l in layers]

    ax.plot(depths, mean_cos, "o-", color="#2166AC", label="Mean (all pairs)")
    ax.plot(depths, min_cos, "v-", color="#D73027", label="Minimum pair")
    ax.plot(depths, bl_min, "^--", color="#4DAF4A", label="Baseline minimum")

    ax.axvline(12.5, color="#999999", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Layer depth (%)")
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(0.8, 1.0)
    ax.legend(loc="lower left", fontsize=7)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 9: VAC survival funnel
# ---------------------------------------------------------------------------

def fig_vac_survival(data: dict) -> plt.Figure:
    """Bar chart showing VAC correlation counts at 3 test levels."""
    survival = loaders.get_vac_survival(data)
    if survival is None:
        return _placeholder("Fig 9: VAC survival (data missing)")

    fig, axes = plt.subplots(1, 2, figsize=full_width(0.35),
                             gridspec_kw={"width_ratios": [1, 2]})

    # Left panel: Zenodo funnel
    ax = axes[0]
    l1 = survival.get("level1_zenodo", {})
    l2 = survival.get("level2_extended", {})
    labels = ["Raw\n(Zenodo)", "Partial\n(Zenodo)", "Raw\n(Mode A)", "Partial\n(Mode A)"]
    values = [
        l1.get("n_raw", 0),
        l1.get("n_partial", 0),
        l2.get("n_raw", 0) if l2 else 0,
        l2.get("n_partial", 0) if l2 else 0,
    ]
    colors = ["#92C5DE", "#2166AC", "#FDD49E", "#EF8A62"]
    ax.barh(range(len(labels)), values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Significant pairs")
    ax.set_title("Length correction", fontsize=9)
    ax.invert_yaxis()

    # Right panel: per-condition (Level 3)
    ax = axes[1]
    l3 = survival.get("level3_per_condition", {})
    conditions = [c for c in CONDITION_ORDER if c in l3]
    vals = [l3[c] for c in conditions]
    colors = [CONDITION_COLORS[c] for c in conditions]
    labels = [label_name(c) for c in conditions]

    bars = ax.barh(range(len(conditions)), vals, color=colors, edgecolor="white")
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Significant pairs (raw)")
    ax.set_title("Controlled experiment (N=10 each)", fontsize=9)
    ax.invert_yaxis()

    # Highlight nonsense if it's the highest
    if vals:
        max_idx = vals.index(max(vals))
        max_cond = conditions[max_idx]
        if max_cond == "nonsense_control":
            bars[max_idx].set_edgecolor("#A50026")
            bars[max_idx].set_linewidth(1.5)
            ax.annotate(f"N={vals[max_idx]}",
                        xy=(vals[max_idx], max_idx),
                        xytext=(vals[max_idx] + 1.5, max_idx),
                        fontsize=6.5, fontweight="bold", color="#A50026",
                        va="center")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 10: Cross-model comparison (template, fills with available data)
# ---------------------------------------------------------------------------

def fig_cross_model_lock_in(data: dict) -> plt.Figure:
    """Cross-model observation count comparison. Shows available models only."""
    # Collect per-model observation counts from phase3 controls
    models = {}

    # Qwen (from Phase D — has lock_in_obs via cycle detection)
    qwen_lockins = loaders.get_lock_in_by_condition(data)
    if qwen_lockins is not None:
        all_vals = []
        for v in qwen_lockins.values():
            all_vals.extend(v)
        if all_vals:
            models["qwen_32b"] = all_vals

    # Cross-model: use phase3 controls (n_observations)
    for model_key in ["llama_8b", "mistral_7b", "gemma_9b", "llama_70b"]:
        ctrl = data.get(f"{model_key}_phase3")
        if ctrl is not None:
            vals = []
            for r in ctrl.get("runs", []):
                n = r.get("n_observations")
                if n is not None:
                    vals.append(n)
            if vals:
                models[model_key] = vals

    if not models:
        return _placeholder("Fig 10: Cross-model (no data)")

    from .style import MODEL_COLORS, MODEL_LABELS

    fig, ax = plt.subplots(figsize=single_col(0.75))

    model_names = list(models.keys())
    for i, mk in enumerate(model_names):
        vals = np.array(models[mk], dtype=float)
        color = MODEL_COLORS.get(mk, "#333333")

        bp = ax.boxplot([vals], positions=[i], widths=0.5,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.2),
                        boxprops=dict(facecolor=color, alpha=0.3, edgecolor=color),
                        whiskerprops=dict(color=color),
                        capprops=dict(color=color))

        jitter = np.random.default_rng(42 + i).uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.full_like(vals, i) + jitter, vals,
                   s=12, color=color, alpha=0.6, edgecolors="white",
                   linewidth=0.3, zorder=3)

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels([MODEL_LABELS.get(m, m).replace(" ", "\n")
                        for m in model_names], fontsize=7.5)
    ax.set_ylabel("Observations per run")
    ax.set_ylim(0, None)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 12: Cross-model direction analysis (Phase 4)
# ---------------------------------------------------------------------------

def fig_cross_model_directions(data: dict) -> plt.Figure:
    """Introspection vs topic Cohen's d across models."""
    from .style import MODEL_COLORS, MODEL_LABELS

    model_keys = ["llama_8b", "mistral_7b", "gemma_9b", "llama_70b"]
    results = {}

    for mk in model_keys:
        p4 = data.get(f"{mk}_phase4")
        if p4 is not None:
            results[mk] = {
                "introspection_d": p4["introspection_transfer"]["cohens_d"],
                "topic_d": p4["topic_transfer"]["cohens_d"],
            }

    if not results:
        return _placeholder("Fig 12: Direction analysis (no data)")

    fig, ax = plt.subplots(figsize=single_col(0.85))

    model_names = list(results.keys())
    x = np.arange(len(model_names))
    width = 0.35

    intro_d = [results[m]["introspection_d"] for m in model_names]
    topic_d = [results[m]["topic_d"] for m in model_names]

    # Clip bars at a readable threshold; annotate actual value above
    clip_at = 12.0
    intro_d_clipped = [min(d, clip_at) for d in intro_d]
    topic_d_clipped = [min(d, clip_at) for d in topic_d]

    bars1 = ax.bar(x - width / 2, intro_d_clipped, width,
                   label="Introspection direction",
                   color=[MODEL_COLORS.get(m, "#333") for m in model_names],
                   alpha=0.9, edgecolor="white")
    bars2 = ax.bar(x + width / 2, topic_d_clipped, width,
                   label="Topic direction",
                   color=[MODEL_COLORS.get(m, "#333") for m in model_names],
                   alpha=0.4, edgecolor="white", hatch="//")

    # Annotate clipped bars with actual value
    for i, (d_raw, d_clip) in enumerate(zip(intro_d, intro_d_clipped)):
        if d_raw > clip_at:
            ax.text(x[i] - width / 2, clip_at + 0.15,
                    f"$d$={d_raw:.1f}", fontsize=5.5, ha="center",
                    va="bottom", fontweight="bold",
                    color=MODEL_COLORS.get(model_names[i], "#333"))

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m).replace(" ", "\n")
                        for m in model_names], fontsize=7.5)
    ax.set_ylabel("Cohen's $d$")
    ax.set_ylim(0, clip_at + 1.5)
    ax.legend(loc="upper left", fontsize=7)
    ax.axhline(0.8, color="gray", linestyle=":", linewidth=0.8, alpha=0.5,
               label="_nolegend_")
    ax.text(0.02, 1.0, "$d = 0.8$ (large)", fontsize=6, color="gray",
            ha="left", va="bottom")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig 11: Cross-model compliance heatmap
# ---------------------------------------------------------------------------

def fig_compliance_heatmap(data: dict) -> plt.Figure:
    """Heatmap: observation count by model x condition, coloured by status."""
    comp = loaders.get_compliance_matrix(data)
    if comp is None:
        return _placeholder("Fig 11: Compliance heatmap (data missing)")

    from .style import MODEL_LABELS, CONDITION_LABELS

    models = comp["models"]
    conditions = comp["conditions"]
    n_obs = np.array(comp["n_obs"], dtype=float)
    status = comp["status"]

    fig, ax = plt.subplots(figsize=full_width(0.45))

    # Custom colormap: 0=red, low=orange, high=green
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "compliance", ["#D73027", "#FEE08B", "#66BD63"], N=256)

    im = ax.imshow(n_obs, cmap=cmap, aspect="auto", vmin=0,
                   vmax=max(100, np.nanmax(n_obs)))

    # Annotate cells with observation count and status
    for i in range(len(models)):
        for j in range(len(conditions)):
            s = status[i][j]
            n = int(n_obs[i, j])
            marker = "" if s == "COMPLIANT" else (" P" if s == "PARTIAL" else " R")
            color = "white" if n < 15 else "black"
            ax.text(j, i, f"{n}{marker}", ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold" if s == "REFUSED" else "normal")

    model_labels = [MODEL_LABELS.get(m, m) for m in models]
    # Shorten long condition labels
    short_cond = {"refusal_instructed": "Refusal", "safety_boundary": "Safety"}
    cond_labels = [short_cond.get(c, CONDITION_LABELS.get(c, c)) for c in conditions]

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(cond_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(model_labels, fontsize=8)

    cb = fig.colorbar(im, ax=ax, shrink=0.8, label="Observations parsed")
    cb.ax.tick_params(labelsize=7)

    fig.tight_layout()

    # Status legend — placed after tight_layout to avoid collision with x-labels
    fig.text(0.45, 0.01, "P = Partial compliance    R = Refused",
             fontsize=6.5, ha="center", color="#555555", style="italic")

    fig.subplots_adjust(bottom=0.25)
    return fig


# ---------------------------------------------------------------------------
# Fig B1: Appendix B mode histogram (explicit TODO in paper)
# ---------------------------------------------------------------------------

def fig_appendix_mode_histogram(data: dict) -> plt.Figure:
    """Three-dataset mode distribution comparison (Zenodo / 16K / 28K)."""
    # We need token lengths from multiple sources
    trunc = data.get("truncation")
    zenodo = data.get("zenodo_baseline")

    if trunc is None and zenodo is None:
        return _placeholder("Fig B1: Mode histogram (data missing)")

    fig, axes = plt.subplots(1, 2, figsize=full_width(0.4),
                             gridspec_kw={"width_ratios": [2, 1]})

    # Left: histogram of token lengths (28K extended)
    ax = axes[0]
    if trunc is not None:
        full_tokens = [r["n_tokens"] for r in trunc["runs"] if r["mode"] == "full"]
        short_tokens = [r["n_tokens"] for r in trunc["runs"] if r["mode"] == "short"]
        bins = np.arange(0, 22000, 500)
        ax.hist(short_tokens, bins=bins, color=MODE_COLORS["short"],
                alpha=0.85, label=f"Mode B (n={len(short_tokens)})",
                edgecolor="white", linewidth=0.4)
        ax.hist(full_tokens, bins=bins, color=MODE_COLORS["full"],
                alpha=0.85, label=f"Mode A (n={len(full_tokens)})",
                edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Token count")
        ax.set_ylabel("Number of runs")
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title("Extended baseline (28K cap)", fontsize=9)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))

    # Right: stacked bar of mode proportions across datasets
    ax = axes[1]
    datasets = []
    if zenodo is not None:
        zt = [r["n_tokens"] for r in zenodo["runs"]]
        n_a = sum(1 for t in zt if t > 5000)
        n_b = sum(1 for t in zt if t <= 5000 and t > 0)
        # Zenodo has truncated runs (mode C)
        n_total = len(zt)
        # approximate: tokens > 7000 are truncated (near cap)
        # Actually Zenodo cap is ~7885, so truncated = near cap
        n_c = sum(1 for t in zt if t > 7000)
        n_a_z = 0  # Zenodo has 0 Mode A (all truncated or short)
        n_b_z = sum(1 for t in zt if t <= 5000)
        n_c_z = n_total - n_b_z
        datasets.append(("Zenodo\n(~8K)", n_a_z, n_b_z, n_c_z, n_total))

    if trunc is not None:
        n_full = sum(1 for r in trunc["runs"] if r["mode"] == "full")
        n_short = sum(1 for r in trunc["runs"] if r["mode"] == "short")
        n_total = len(trunc["runs"])
        datasets.append(("Extended\n(28K)", n_full, n_short, 0, n_total))

    if datasets:
        x = range(len(datasets))
        labels = [d[0] for d in datasets]
        a_pct = [d[1] / d[4] * 100 for d in datasets]
        b_pct = [d[2] / d[4] * 100 for d in datasets]
        c_pct = [d[3] / d[4] * 100 for d in datasets]

        ax.bar(x, a_pct, color=MODE_COLORS["A"], label="Mode A")
        ax.bar(x, b_pct, bottom=a_pct, color=MODE_COLORS["B"], label="Mode B")
        ax.bar(x, c_pct, bottom=[a + b for a, b in zip(a_pct, b_pct)],
               color=MODE_COLORS["C"], label="Mode C")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("Percentage")
        ax.set_ylim(0, 105)
        ax.legend(loc="upper right", fontsize=6)
        ax.set_title("Mode distribution", fontsize=9)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig NEW: VAC Scatter Rebuttal (answers Dadfar Figs 5, 6, 9)
# ---------------------------------------------------------------------------

def fig_vac_scatter_rebuttal(data: dict) -> plt.Figure:
    """Three-panel VAC rebuttal: replicate Dadfar's scatter + show ubiquity.

    Panel A: Baseline introspective scatter (mirror vs spectral_power_low)
    Panel B: Descriptive control scatter (same pair, mirror prompts)
    Panel C: Per-condition significant-pair counts showing VAC in ALL conditions
    """
    from scipy import stats as sp_stats

    scatter = loaders.get_vac_scatter_data(data)
    survival = loaders.get_vac_survival(data)

    if scatter is None and survival is None:
        return _placeholder("VAC scatter rebuttal (data missing)")

    fig, axes = plt.subplots(1, 3, figsize=full_width(0.42),
                             gridspec_kw={"width_ratios": [1, 1, 1.3]})

    # --- Panel A: Baseline introspective ---
    ax = axes[0]
    if scatter is not None and "baseline" in scatter:
        bv = np.array(scatter["baseline"]["vocab"], dtype=float)
        bm = np.array(scatter["baseline"]["metric"], dtype=float)
        ax.scatter(bv, bm, s=18, alpha=0.6, color="#2166AC",
                   edgecolors="white", linewidth=0.3, zorder=3)
        # Regression
        if len(bv) > 2:
            slope, intercept, r, p, _ = sp_stats.linregress(bv, bm)
            x_fit = np.linspace(bv.min(), bv.max(), 50)
            ax.plot(x_fit, slope * x_fit + intercept, "--",
                    color="#D73027", linewidth=1.2, zorder=2)
            ax.text(0.05, 0.92, f"$r$ = {r:.2f}\n$p$ = {p:.3f}\nN = {len(bv)}",
                    transform=ax.transAxes, fontsize=6.5, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#cccccc", alpha=0.9))
    ax.set_xlabel("Mirror count", fontsize=8)
    ax.set_ylabel("Spectral power (low)", fontsize=8)
    ax.set_title("A. Introspective", fontsize=9, fontweight="bold")

    # --- Panel B: Descriptive control ---
    ax = axes[1]
    if scatter is not None and "descriptive" in scatter:
        dv = np.array(scatter["descriptive"]["vocab"], dtype=float)
        dm = np.array(scatter["descriptive"]["metric"], dtype=float)
        ax.scatter(dv, dm, s=18, alpha=0.6, color="#FDB863",
                   edgecolors="white", linewidth=0.3, zorder=3)
        if len(dv) > 2:
            slope, intercept, r, p, _ = sp_stats.linregress(dv, dm)
            x_fit = np.linspace(dv.min(), dv.max(), 50)
            ax.plot(x_fit, slope * x_fit + intercept, "--",
                    color="#888888", linewidth=1.2, zorder=2)
            ax.text(0.05, 0.92, f"$r$ = {r:.2f}\n$p$ = {p:.2f}\nN = {len(dv)}",
                    transform=ax.transAxes, fontsize=6.5, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#cccccc", alpha=0.9))
    ax.set_xlabel("Mirror count", fontsize=8)
    ax.set_title("B. Descriptive control", fontsize=9, fontweight="bold")

    # --- Panel C: Per-condition VAC counts ---
    ax = axes[2]
    if survival is not None and "level3_per_condition" in survival:
        l3 = survival["level3_per_condition"]
        conditions = [c for c in CONDITION_ORDER if c in l3]
        vals = [l3[c] for c in conditions]
        colors = [CONDITION_COLORS[c] for c in conditions]
        labels = [label_name(c) for c in conditions]

        bars = ax.barh(range(len(conditions)), vals, color=colors,
                       edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(conditions)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Significant VAC pairs", fontsize=8)
        ax.invert_yaxis()

        # Highlight nonsense
        if vals:
            max_idx = vals.index(max(vals))
            max_cond = conditions[max_idx]
            if max_cond == "nonsense_control":
                bars[max_idx].set_edgecolor("#A50026")
                bars[max_idx].set_linewidth(2)
                ax.annotate(f"{vals[max_idx]}",
                            xy=(vals[max_idx], max_idx),
                            xytext=(vals[max_idx] + 1, max_idx),
                            fontsize=7, fontweight="bold", color="#A50026",
                            va="center")

    ax.set_title("C. VAC per condition", fontsize=9, fontweight="bold")

    fig.tight_layout(w_pad=1.5)
    return fig


# ---------------------------------------------------------------------------
# Fig NEW: Layer Sweep Comparison (answers Dadfar Fig 3)
# ---------------------------------------------------------------------------

def fig_layer_sweep_comparison(data: dict) -> plt.Figure:
    """Side-by-side: Dadfar's Llama layer sweep vs our Qwen F-statistics.

    Panel A: Llama 70B intro_delta by layer (from Zenodo) — shows Layer 5 hotspot
    Panel B: Qwen 32B F_norm by layer depth — shows no hotspot
    """
    llama_sweep = loaders.get_llama_layer_sweep(data)
    qwen_fstats = loaders.get_f_statistics(data)

    if llama_sweep is None and qwen_fstats is None:
        return _placeholder("Layer sweep comparison (data missing)")

    fig, axes = plt.subplots(1, 2, figsize=full_width(0.42))

    # --- Panel A: Llama 70B layer sweep (Dadfar's data) ---
    ax = axes[0]
    if llama_sweep is not None:
        layers = sorted(llama_sweep.keys())
        deltas = [llama_sweep[l]["intro_delta"] for l in layers]
        # Compute depth from layer number (Llama 70B has 80 layers)
        depth_pcts = [l / 80 * 100 for l in layers]
        x_labels = [f"L{l}" for l in layers]

        bar_colors = ["#D73027" if l == 5 else "#92C5DE" for l in layers]
        ax.bar(range(len(layers)), deltas, color=bar_colors,
               edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(x_labels, fontsize=6.5, rotation=45, ha="right")
        ax.set_ylabel("Introspective density boost", fontsize=8)

        # Annotate Layer 5
        l5_idx = layers.index(5) if 5 in layers else None
        if l5_idx is not None:
            ax.annotate(f"Layer 5\n({depth_pcts[l5_idx]:.1f}% depth)",
                        xy=(l5_idx, deltas[l5_idx]),
                        xytext=(l5_idx + 2.5, deltas[l5_idx] * 0.85),
                        fontsize=6.5, fontweight="bold", color="#A50026",
                        ha="center", va="top",
                        arrowprops=dict(arrowstyle="->", color="#A50026",
                                        linewidth=0.8))
    ax.set_title("A. Llama 3.1-70B (Dadfar)", fontsize=9, fontweight="bold")

    # --- Panel B: Qwen F-statistics (our data) ---
    ax = axes[1]
    if qwen_fstats is not None:
        layers_q = sorted(qwen_fstats.keys())
        depths_q = [l / 64 * 100 for l in layers_q]
        f_norm = [qwen_fstats[l]["f_norm"] for l in layers_q]
        f_cos = [qwen_fstats[l]["f_cosine"] for l in layers_q]

        ax.plot(depths_q, f_norm, "o-", color="#2166AC", markersize=5,
                label="$F_{\\mathrm{norm}}$")
        ax.plot(depths_q, f_cos, "s--", color="#D73027", markersize=4,
                label="$F_{\\mathrm{cosine}}$")
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

        # Equivalent Dadfar hotspot position
        ax.axvline(12.5, color="#999999", linestyle="--", linewidth=0.8,
                   alpha=0.5)
        ax.text(14, max(f_norm) * 0.4, "Equivalent\ndepth",
                fontsize=6.5, color="#666666", va="center")

        ax.set_xlabel("Layer depth (%)", fontsize=8)
        ax.set_ylabel("F-statistic", fontsize=8)
        ax.legend(loc="upper right", fontsize=7)
    ax.set_title("B. Qwen 2.5-32B (this work)", fontsize=9, fontweight="bold")

    fig.tight_layout(w_pad=1.5)
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _placeholder(title: str) -> plt.Figure:
    """Return a placeholder figure with a warning message."""
    fig, ax = plt.subplots(figsize=single_col(0.5))
    ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=10,
            color="#999999", transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    warnings.warn(title)
    return fig
