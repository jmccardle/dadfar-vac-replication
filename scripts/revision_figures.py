#!/usr/bin/env python3
"""Generate revision figures from analysis JSONs.

Produces:
1. fig_kaplan_meier.pdf — KM survival curves for 8 conditions
2. fig07_f_statistics.pdf — Corrected F-statistics dual panel
3. fig_effect_sizes.pdf — Hedges' g with 95% CI error bars
"""

import json
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT = Path(__file__).resolve().parent.parent
ANALYSIS = PROJECT / "outputs" / "analysis"
FIG_DIR = PROJECT / "latex" / "figures"

# Consistent colors for 8 conditions
COND_COLORS = {
    "baseline": "#1f77b4",
    "abstract_philosophical": "#ff7f0e",
    "factual_iterative": "#2ca02c",
    "procedural_self": "#d62728",
    "descriptive_forest": "#9467bd",
    "descriptive_math": "#8c564b",
    "descriptive_music": "#e377c2",
    "nonsense_control": "#7f7f7f",
}

COND_LABELS = {
    "baseline": "Baseline (self-ref)",
    "abstract_philosophical": "Abstract phil.",
    "factual_iterative": "Factual iter.",
    "procedural_self": "Procedural self",
    "descriptive_forest": "Desc. forest",
    "descriptive_math": "Desc. math",
    "descriptive_music": "Desc. music",
    "nonsense_control": "Nonsense ctrl",
}


def fig_kaplan_meier():
    """KM survival curves for 8 conditions."""
    with open(ANALYSIS / "revision_r07_survival.json") as f:
        data = json.load(f)

    km = data["kaplan_meier"]
    lr = data["log_rank_omnibus"]

    fig, ax = plt.subplots(figsize=(7, 5))

    for cond, color in COND_COLORS.items():
        if cond not in km:
            continue
        cd = km[cond]
        times = cd["km_times"]
        surv = cd["km_survival"]

        # Step plot for KM curve
        step_times = []
        step_surv = []
        for i in range(len(times)):
            if i > 0:
                step_times.append(times[i])
                step_surv.append(surv[i - 1])
            step_times.append(times[i])
            step_surv.append(surv[i])

        lw = 2.0 if cond in ("baseline", "nonsense_control", "descriptive_math") else 1.2
        ax.plot(step_times, step_surv, color=color, linewidth=lw,
                label=f"{COND_LABELS[cond]} (n={cd['n_total']})")

        # Censoring marks (+ at end if censored)
        if cd["n_censored"] > 0 and surv[-1] > 0:
            ax.plot(times[-1], surv[-1], "+", color=color, markersize=8, markeredgewidth=1.5)

    ax.set_xlabel("Lock-in resistance (observations)", fontsize=11)
    ax.set_ylabel("Survival probability", fontsize=11)
    ax.set_title("Kaplan-Meier Survival Curves for Lock-in Resistance", fontsize=12)
    ax.set_xlim(0, 260)
    ax.set_ylim(0, 1.05)

    # Log-rank annotation
    ax.annotate(
        f"Log-rank: $\\chi^2={lr['chi2']:.1f}$, $p={lr['p_value']:.3f}$",
        xy=(0.98, 0.98), xycoords="axes fraction",
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9,
              bbox_to_anchor=(0.98, 0.88))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / "fig_kaplan_meier.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def fig_f_statistics():
    """Corrected F-statistics dual panel."""
    f_path = ANALYSIS / "revision_r02_f_statistics_v2.json"
    if not f_path.exists():
        print(f"SKIP: {f_path} does not exist yet. Run revision_r02_f_statistics_v2.py first.")
        return

    with open(f_path) as f:
        data = json.load(f)

    layers_data = data["layers"]
    layers = sorted(int(k) for k in layers_data.keys())
    depth_pcts = [layers_data[str(l)]["depth_pct"] for l in layers]

    f_norms = []
    f_cosines = []
    sig_norm = []
    sig_cos = []

    for l in layers:
        ld = layers_data[str(l)]
        fn = ld.get("f_norm", {})
        fc = ld.get("f_cosine", {})
        f_norms.append(fn.get("F", 0) or 0)
        f_cosines.append(fc.get("F", 0) or 0)
        sig_norm.append(fn.get("significant_005", False))
        sig_cos.append(fc.get("significant_005", False))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

    # F_norm panel
    bars1 = ax1.bar(range(len(layers)), f_norms, color=["#d62728" if s else "#4c72b0" for s in sig_norm])
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels([f"L{l}\n({d:.0f}%)" for l, d in zip(layers, depth_pcts)], fontsize=8)
    ax1.set_ylabel("$F_{\\mathrm{norm}}$", fontsize=11)
    ax1.set_title("$F$ for Activation Norm", fontsize=11)
    ax1.axhline(y=2.66, color="gray", linestyle="--", alpha=0.5, label="$F_{crit}$ ($\\alpha=0.05$)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    # Add significance stars
    for i, (v, s) in enumerate(zip(f_norms, sig_norm)):
        if s:
            ax1.text(i, v + 0.1, "*", ha="center", fontsize=12, color="#d62728")

    # F_cosine panel
    bars2 = ax2.bar(range(len(layers)), f_cosines, color=["#d62728" if s else "#4c72b0" for s in sig_cos])
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels([f"L{l}\n({d:.0f}%)" for l, d in zip(layers, depth_pcts)], fontsize=8)
    ax2.set_ylabel("$F_{\\mathrm{cosine}}$", fontsize=11)
    ax2.set_title("$F$ for Directional Alignment", fontsize=11)
    ax2.axhline(y=2.66, color="gray", linestyle="--", alpha=0.5, label="$F_{crit}$ ($\\alpha=0.05$)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    for i, (v, s) in enumerate(zip(f_cosines, sig_cos)):
        if s:
            ax2.text(i, v + 0.05, "*", ha="center", fontsize=12, color="#d62728")

    plt.tight_layout()
    out = FIG_DIR / "fig07_f_statistics.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def fig_effect_sizes():
    """Hedges' g with 95% CI error bars."""
    with open(ANALYSIS / "revision_r04_hedges_g.json") as f:
        data = json.load(f)

    corrections = data["corrections"]

    labels = [c["label"] for c in corrections]
    gs = [c["hedges_g"] for c in corrections]
    ci_lows = [c["ci_95_low"] for c in corrections]
    ci_highs = [c["ci_95_high"] for c in corrections]

    # Compute error bar sizes
    err_low = [g - cl for g, cl in zip(gs, ci_lows)]
    err_high = [ch - g for g, ch in zip(gs, ci_highs)]

    fig, ax = plt.subplots(figsize=(8, 5))

    y_pos = range(len(labels))
    colors = ["#1f77b4" if "Lock-in" not in l else "#7f7f7f" for l in labels]

    ax.barh(y_pos, gs, xerr=[err_low, err_high], height=0.6,
            color=colors, alpha=0.8, capsize=4, ecolor="black")

    # Short labels for y-axis
    short_labels = []
    for l in labels:
        l = l.replace("Introspection direction ", "Introspection ")
        l = l.replace("Forest-vs-Math direction ", "Forest-vs-Math ")
        l = l.replace("Dadfar transfer test (reported)", "Dadfar (reported)")
        l = l.replace("Lock-in baseline vs ", "Lock-in: bl vs ")
        l = l.replace(" (cycling)", "")
        short_labels.append(l)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_labels, fontsize=9)
    ax.set_xlabel("Hedges' $g$ (with 95% CI)", fontsize=11)
    ax.set_title("Effect Sizes: Hedges' $g$ Corrections", fontsize=12)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="x")

    # Annotate values
    for i, (g, cl, ch) in enumerate(zip(gs, ci_lows, ci_highs)):
        ax.text(max(ch + 0.3, g + 0.5), i, f"$g={g:.2f}$", va="center", fontsize=8)

    plt.tight_layout()
    out = FIG_DIR / "fig_effect_sizes.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating revision figures...")
    fig_kaplan_meier()
    fig_effect_sizes()
    fig_f_statistics()
    print("Done.")


if __name__ == "__main__":
    main()
