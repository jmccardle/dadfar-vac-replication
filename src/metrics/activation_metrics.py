"""Compute activation metrics from per-token hidden state vectors.

Implements all 6 metrics from Dadfar (2026) Section 2.5, plus additional
metrics for vocabulary discovery (Phase 2).
"""

import numpy as np
from scipy import stats


def compute_all_metrics(norms: np.ndarray, vectors: np.ndarray = None) -> dict:
    """Compute all activation metrics from per-token data.

    Args:
        norms: shape (n_tokens,) — L2 norm of hidden state at each token position.
        vectors: shape (n_tokens, hidden_dim) — full activation vectors. Optional;
                 required for sparsity and sign_change_rate.

    Returns:
        dict with all metric values matching Dadfar's JSON schema.
    """
    n = len(norms)
    if n < 2:
        return _empty_metrics(n)

    metrics = {
        # Core metrics from paper
        "mean_norm": float(np.mean(norms)),
        "max_norm": float(np.max(norms)),
        "norm_std": float(np.std(norms)),
        "norm_kurtosis": float(stats.kurtosis(norms)) if n > 3 else 0.0,
        "autocorr_lag1": _autocorr(norms, lag=1),
        "autocorr_lag2": _autocorr(norms, lag=2),
        "spectral_power_low": _spectral_power(norms, band="low"),
        "spectral_power_mid": _spectral_power(norms, band="mid"),
        # Derivative metrics
        "mean_derivative": _mean_derivative(norms),
        "max_derivative": _max_derivative(norms),
        # Sign changes (requires full vectors or just norms-based proxy)
        "sign_changes": 0,
        "sign_change_rate": 0.0,
        # Similarity metrics
        "mean_token_similarity": 0.0,
        "token_similarity_std": 0.0,
        # Sparsity
        "sparsity": 0.0,
        # Convergence
        "convergence_ratio": _convergence_ratio(norms),
        # Sequence length
        "n_tokens": n,
    }

    if vectors is not None and len(vectors) > 1:
        metrics["sparsity"] = _sparsity(vectors)
        sc, scr = _sign_change_rate(vectors)
        metrics["sign_changes"] = sc
        metrics["sign_change_rate"] = scr
        sim_mean, sim_std = _token_similarity(vectors)
        metrics["mean_token_similarity"] = sim_mean
        metrics["token_similarity_std"] = sim_std

    return metrics


def _empty_metrics(n: int) -> dict:
    return {
        "mean_norm": 0.0, "max_norm": 0.0, "norm_std": 0.0,
        "norm_kurtosis": 0.0, "autocorr_lag1": 0.0, "autocorr_lag2": 0.0,
        "spectral_power_low": 0.0, "spectral_power_mid": 0.0,
        "mean_derivative": 0.0, "max_derivative": 0.0,
        "sign_changes": 0, "sign_change_rate": 0.0,
        "mean_token_similarity": 0.0, "token_similarity_std": 0.0,
        "sparsity": 0.0, "convergence_ratio": 0.0, "n_tokens": n,
    }


def _autocorr(norms: np.ndarray, lag: int = 1) -> float:
    """Lag-k autocorrelation of activation norms."""
    if len(norms) <= lag:
        return 0.0
    r, _ = stats.pearsonr(norms[:-lag], norms[lag:])
    return float(r) if not np.isnan(r) else 0.0


def _spectral_power(norms: np.ndarray, band: str = "low") -> float:
    """FFT power in specified frequency band.

    band="low": bottom 5% of frequency bins (excluding DC)
    band="mid": 5%-25% of frequency bins

    Note: NOT per-token normalized. Caller must divide by n_tokens if needed.
    Dadfar normalizes spectral_power_low by token count for some analyses.
    """
    centered = norms - np.mean(norms)
    fft_vals = np.fft.rfft(centered)
    power = np.abs(fft_vals) ** 2
    n_freqs = len(power)

    if band == "low":
        low = 1  # skip DC
        high = max(2, n_freqs // 20)  # bottom 5%
    elif band == "mid":
        low = max(2, n_freqs // 20)
        high = max(low + 1, n_freqs // 4)  # 5%-25%
    else:
        raise ValueError(f"Unknown band: {band}")

    return float(np.sum(power[low:high]))


def _mean_derivative(norms: np.ndarray) -> float:
    diffs = np.diff(norms)
    return float(np.mean(np.abs(diffs)))


def _max_derivative(norms: np.ndarray) -> float:
    diffs = np.diff(norms)
    return float(np.max(np.abs(diffs)))


def _convergence_ratio(norms: np.ndarray) -> float:
    """Ratio of mean norm in last 10% vs first 10% of sequence."""
    n = len(norms)
    window = max(1, n // 10)
    first = np.mean(norms[:window])
    last = np.mean(norms[-window:])
    if first < 1e-10:
        return 0.0
    return float(last / first)


def _sparsity(vectors: np.ndarray, threshold: float = 0.1) -> float:
    """Fraction of dimensions below threshold magnitude, averaged across tokens."""
    return float(np.mean(np.abs(vectors) < threshold))


def _sign_change_rate(vectors: np.ndarray) -> tuple[int, float]:
    """Fraction of first-PC dimensions changing sign between consecutive tokens.

    Returns (sign_changes, sign_change_rate).
    """
    n_tokens, hidden_dim = vectors.shape
    if n_tokens < 3:
        return 0, 0.0

    # Use mean-centered vectors, compute first PC via SVD on a sample if too large
    centered = vectors - np.mean(vectors, axis=0, keepdims=True)

    if n_tokens > 10000:
        # Subsample for PCA efficiency
        indices = np.random.choice(n_tokens, 10000, replace=False)
        sample = centered[indices]
    else:
        sample = centered

    # First PC via truncated SVD
    _, _, Vt = np.linalg.svd(sample, full_matrices=False)
    pc1 = Vt[0]  # (hidden_dim,)

    # Project all tokens onto PC1
    projected = centered @ pc1  # (n_tokens,)
    signs = np.sign(projected)
    changes = int(np.sum(signs[:-1] != signs[1:]))
    rate = changes / (n_tokens - 1)
    return changes, float(rate)


def _token_similarity(vectors: np.ndarray) -> tuple[float, float]:
    """Mean and std of cosine similarity between consecutive tokens."""
    n = len(vectors)
    if n < 2:
        return 0.0, 0.0

    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = vectors / norms

    # Cosine similarity between consecutive tokens
    sims = np.sum(normalized[:-1] * normalized[1:], axis=-1)
    return float(np.mean(sims)), float(np.std(sims))
