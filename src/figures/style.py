"""Shared matplotlib configuration for publication-quality figures."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# --- Colour palette ---
# Condition colours: consistent across all figures
CONDITION_COLORS = {
    "baseline":                "#2166AC",  # dark blue
    "abstract_philosophical":  "#4393C3",  # medium blue
    "factual_iterative":       "#92C5DE",  # light blue
    "procedural_self":         "#D1E5F0",  # very light blue
    "descriptive_forest":      "#66BD63",  # green
    "descriptive_math":        "#A6D96A",  # yellow-green
    "descriptive_music":       "#FEE08B",  # gold
    "nonsense_control":        "#D73027",  # red
    "refusal_instructed":      "#762A83",  # purple
    "safety_boundary":         "#C51B7D",  # magenta
}

# Short display labels for conditions
CONDITION_LABELS = {
    "baseline":                "Baseline",
    "abstract_philosophical":  "Abstract",
    "factual_iterative":       "Factual",
    "procedural_self":         "Procedural",
    "descriptive_forest":      "Forest",
    "descriptive_math":        "Math",
    "descriptive_music":       "Music",
    "nonsense_control":        "Nonsense",
    "refusal_instructed":      "Refusal (instructed)",
    "safety_boundary":         "Safety boundary",
}

# Canonical condition order for all plots
CONDITION_ORDER = [
    "baseline",
    "abstract_philosophical",
    "factual_iterative",
    "procedural_self",
    "descriptive_forest",
    "descriptive_math",
    "descriptive_music",
    "nonsense_control",
    "refusal_instructed",
    "safety_boundary",
]

# Temperature colours
TEMP_COLORS = {0.3: "#4575B4", 0.7: "#FFFFBF", 1.0: "#D73027"}
TEMP_EDGE   = {0.3: "#2166AC", 0.7: "#8C510A", 1.0: "#A50026"}

# Mode colours
MODE_COLORS = {"A": "#2166AC", "B": "#D73027", "C": "#999999",
               "full": "#2166AC", "short": "#D73027"}

# Metric display names
METRIC_LABELS = {
    "mean_norm":             "Mean norm",
    "max_norm":              "Max norm",
    "convergence_ratio":     "Convergence ratio",
    "mean_token_similarity": "Token similarity",
    "token_similarity_std":  "Token sim. SD",
    "spectral_power_low":    "Spectral power (low)",
    "spectral_power_mid":    "Spectral power (mid)",
    "norm_std":              "Norm SD",
    "norm_kurtosis":         "Norm kurtosis",
    "autocorr_lag1":         "Autocorrelation (lag 1)",
    "autocorr_lag2":         "Autocorrelation (lag 2)",
    "mean_derivative":       "Mean derivative",
    "max_derivative":        "Max derivative",
    "sign_changes":          "Sign changes",
    "sign_change_rate":      "Sign change rate",
    "sparsity":              "Sparsity",
}

# Model display names and colours
MODEL_COLORS = {
    "qwen_32b":   "#2166AC",
    "llama_8b":   "#D73027",
    "llama_70b":  "#4DAF4A",
    "mistral_7b": "#984EA3",
    "gemma_9b":   "#FF7F00",
}
MODEL_LABELS = {
    "qwen_32b":   "Qwen 2.5-32B",
    "llama_8b":   "Llama 3.1 8B",
    "llama_70b":  "Llama 3.1 70B",
    "mistral_7b": "Mistral 7B",
    "gemma_9b":   "Gemma 2 9B",
}


def apply_style():
    """Apply publication-quality matplotlib defaults."""
    plt.rcParams.update({
        # Font
        "font.family":        "serif",
        "font.serif":         ["DejaVu Serif", "Computer Modern Roman", "CMU Serif", "Times New Roman"],
        "mathtext.fontset":   "cm",
        "font.size":          9,
        "axes.titlesize":     10,
        "axes.labelsize":     9,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    8,
        # Figure
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
        # Axes
        "axes.linewidth":     0.6,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "xtick.major.width":  0.6,
        "ytick.major.width":  0.6,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        # Grid
        "axes.grid":          False,
        # Legend
        "legend.frameon":     False,
        # Lines
        "lines.linewidth":    1.2,
        "lines.markersize":   4,
    })


# --- Layout helpers ---

# Single-column: ~3.3", double-column: ~7.0" (for typical journal)
COL_WIDTH = 3.35   # inches, single column
FULL_WIDTH = 7.0   # inches, full page width


def single_col(aspect=0.75):
    """Return (width, height) for a single-column figure."""
    return (COL_WIDTH, COL_WIDTH * aspect)


def full_width(aspect=0.4):
    """Return (width, height) for a full-width figure."""
    return (FULL_WIDTH, FULL_WIDTH * aspect)


def label_color(condition):
    return CONDITION_COLORS.get(condition, "#333333")


def label_name(condition):
    return CONDITION_LABELS.get(condition, condition)
