"""Loop detection for Pull Methodology observation sequences.

Detects limit cycles in the numbered observation output of the Pull
Methodology, characterizes cycle period and vocabulary, and identifies
the lock-in observation.

Two detection strategies:
1. Exact string matching (fast, works for established cycles)
2. Bag-of-words cosine similarity (robust, detects near-match transitions)
"""

import re
from collections import Counter
from dataclasses import dataclass, field
import math


@dataclass
class CycleResult:
    """Result of cycle detection on a single run."""
    has_cycle: bool = False
    lock_in_obs: int | None = None
    cycle_period: int | None = None
    cycle_vocabulary: list[str] = field(default_factory=list)
    exploration_vocabulary: list[str] = field(default_factory=list)
    n_observations: int = 0
    n_unique: int = 0
    unique_ratio: float = 0.0
    observations: list[str] = field(default_factory=list)
    # Per-observation state labels for Markov analysis
    state_sequence: list[str] = field(default_factory=list)


def parse_observations(text: str) -> list[str]:
    """Parse numbered observations from Pull Methodology output.

    Handles formats:
        "1. observation text"
        "1) observation text"

    Returns list of observation strings (0-indexed: element 0 = observation 1).
    """
    lines = text.strip().split('\n')
    observations = []
    current_obs = None
    current_num = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Match observation number at start of line
        # Handles: "1. text", "1) text", "Pull 1: text", "Observation 1: text"
        m = re.match(r'^(?:(?:Pull|Observation)\s+)?(\d+)[.\):]\s*(.*)', line, re.IGNORECASE)
        if m:
            num = int(m.group(1))
            obs_text = m.group(2).strip()

            # If this is a new observation number, save previous and start new
            if num > current_num:
                if current_obs is not None:
                    observations.append(current_obs)
                current_obs = obs_text
                current_num = num
            elif num == current_num and current_obs is not None:
                # Continuation of same observation
                current_obs += ' ' + obs_text
        elif current_obs is not None:
            # Continuation line (no number prefix)
            current_obs += ' ' + line

    # Don't forget the last observation
    if current_obs is not None:
        observations.append(current_obs)

    return observations


def normalize_observation(obs: str) -> str:
    """Normalize an observation for comparison.

    Lowercases, strips punctuation, collapses whitespace.
    """
    obs = obs.lower()
    obs = re.sub(r'[^\w\s]', '', obs)
    obs = re.sub(r'\s+', ' ', obs).strip()
    return obs


def observation_to_bow(obs: str) -> Counter:
    """Convert observation to bag-of-words Counter."""
    norm = normalize_observation(obs)
    # Filter stopwords for cleaner comparison
    stopwords = {
        'a', 'an', 'the', 'of', 'in', 'to', 'for', 'is', 'on', 'and',
        'or', 'as', 'by', 'with', 'from', 'that', 'this', 'it', 'its',
        'are', 'was', 'be', 'been', 'being', 'have', 'has', 'had',
        'through', 'about', 'between', 'than', 'more', 'not', 'but',
    }
    words = [w for w in norm.split() if w not in stopwords]
    return Counter(words)


def cosine_similarity_bow(a: Counter, b: Counter) -> float:
    """Cosine similarity between two bag-of-words Counters."""
    if not a or not b:
        return 0.0
    common = set(a.keys()) & set(b.keys())
    dot = sum(a[k] * b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def detect_cycle_exact(
    observations: list[str],
    min_period: int = 3,
    max_period: int = 50,
    min_confirmations: int = 3,
) -> tuple[int | None, int | None]:
    """Detect cycle using exact normalized string matching.

    Returns (lock_in_obs, period) or (None, None) if no cycle found.
    lock_in_obs is 1-indexed (observation number).
    """
    n = len(observations)
    norms = [normalize_observation(obs) for obs in observations]

    for period in range(min_period, min(max_period + 1, n // min_confirmations)):
        # For each possible start position
        for start in range(n - period * min_confirmations):
            # Check if observations repeat with this period starting here
            confirmed = 0
            for k in range(1, (n - start) // period):
                if norms[start + k * period] == norms[start]:
                    confirmed += 1
                else:
                    break

            if confirmed >= min_confirmations:
                # Verify the full cycle: all positions in the period repeat
                all_match = True
                for offset in range(period):
                    base = norms[start + offset]
                    for k in range(1, min_confirmations + 1):
                        idx = start + offset + k * period
                        if idx >= n or norms[idx] != base:
                            all_match = False
                            break
                    if not all_match:
                        break

                if all_match:
                    return start + 1, period  # 1-indexed

    return None, None


def detect_cycle_similarity(
    observations: list[str],
    min_period: int = 3,
    max_period: int = 50,
    threshold: float = 0.85,
    min_confirmations: int = 3,
) -> tuple[int | None, int | None]:
    """Detect cycle using bag-of-words cosine similarity.

    Returns (lock_in_obs, period) or (None, None) if no cycle found.
    lock_in_obs is 1-indexed.
    """
    n = len(observations)
    bows = [observation_to_bow(obs) for obs in observations]

    for period in range(min_period, min(max_period + 1, n // min_confirmations)):
        for start in range(n - period * min_confirmations):
            # Check all positions in the candidate period
            all_match = True
            for offset in range(period):
                base_bow = bows[start + offset]
                confirmed = 0
                for k in range(1, min_confirmations + 1):
                    idx = start + offset + k * period
                    if idx >= n:
                        all_match = False
                        break
                    sim = cosine_similarity_bow(base_bow, bows[idx])
                    if sim >= threshold:
                        confirmed += 1
                    else:
                        all_match = False
                        break
                if not all_match:
                    break

            if all_match:
                return start + 1, period  # 1-indexed

    return None, None


def extract_cycle_vocabulary(
    observations: list[str],
    lock_in_obs: int,
    period: int,
) -> list[str]:
    """Extract the vocabulary of the limit cycle.

    Returns the sequence of normalized observations forming one period
    of the cycle, starting from lock_in_obs (1-indexed).
    """
    start = lock_in_obs - 1  # Convert to 0-indexed
    return [normalize_observation(observations[start + i])
            for i in range(min(period, len(observations) - start))]


def assign_states(
    observations: list[str],
    cycle_vocab: list[str] | None = None,
    similarity_threshold: float = 0.80,
) -> list[str]:
    """Assign each observation to a discrete state for Markov analysis.

    If cycle_vocab is provided, observations matching a cycle element
    are labeled by their cycle position. Others get a unique label.

    Falls back to normalized exact text as the state label if no cycle
    vocabulary is available.
    """
    if cycle_vocab is None:
        return [normalize_observation(obs) for obs in observations]

    cycle_bows = [observation_to_bow(cv) for cv in cycle_vocab]
    states = []
    exploration_counter = 0

    for obs in observations:
        norm = normalize_observation(obs)
        obs_bow = observation_to_bow(obs)

        # Try to match to cycle vocabulary
        best_sim = 0.0
        best_idx = -1
        for idx, (cv, cv_bow) in enumerate(zip(cycle_vocab, cycle_bows)):
            # Check exact match first (fast path)
            if norm == cv:
                best_idx = idx
                best_sim = 1.0
                break
            sim = cosine_similarity_bow(obs_bow, cv_bow)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        if best_sim >= similarity_threshold:
            states.append(f"cycle_{best_idx}")
        else:
            states.append(f"explore_{exploration_counter}")
            exploration_counter += 1

    return states


def analyze_run(
    text: str,
    min_period: int = 3,
    max_period: int = 50,
    exact_confirmations: int = 3,
    similarity_threshold: float = 0.85,
    similarity_confirmations: int = 3,
) -> CycleResult:
    """Full cycle analysis for a single Pull Methodology run.

    Tries exact matching first (faster, more precise). Falls back to
    similarity-based detection if exact matching fails.
    """
    observations = parse_observations(text)
    n = len(observations)

    if n == 0:
        return CycleResult(n_observations=0)

    # Try exact detection first
    lock_in, period = detect_cycle_exact(
        observations, min_period, max_period, exact_confirmations
    )

    # Fall back to similarity detection
    if lock_in is None:
        lock_in, period = detect_cycle_similarity(
            observations, min_period, max_period,
            similarity_threshold, similarity_confirmations
        )

    # Compute unique observations
    norms = [normalize_observation(obs) for obs in observations]
    unique_norms = set(norms)
    n_unique = len(unique_norms)

    result = CycleResult(
        n_observations=n,
        n_unique=n_unique,
        unique_ratio=n_unique / n if n > 0 else 0.0,
        observations=observations,
    )

    if lock_in is not None and period is not None:
        result.has_cycle = True
        result.lock_in_obs = lock_in
        result.cycle_period = period

        cycle_vocab = extract_cycle_vocabulary(observations, lock_in, period)
        result.cycle_vocabulary = cycle_vocab

        # Exploration vocabulary: unique observations before lock-in
        # that don't appear in the cycle
        cycle_set = set(cycle_vocab)
        exploration = []
        for i in range(lock_in - 1):  # 0-indexed, up to lock_in exclusive
            norm = norms[i]
            if norm not in cycle_set and norm not in exploration:
                exploration.append(norm)
        result.exploration_vocabulary = exploration

        # State sequence for Markov analysis
        result.state_sequence = assign_states(observations, cycle_vocab)
    else:
        result.state_sequence = assign_states(observations)

    return result


def build_transition_matrix(
    state_sequence: list[str],
    max_obs: int | None = None,
) -> tuple[list[str], list[list[float]]]:
    """Build a Markov transition matrix from a state sequence.

    Args:
        state_sequence: List of state labels.
        max_obs: If set, only use the first max_obs observations.

    Returns:
        (state_names, transition_matrix) where transition_matrix[i][j]
        is P(state j | state i).
    """
    seq = state_sequence[:max_obs] if max_obs else state_sequence

    # Collect unique states in order of first appearance
    seen = {}
    state_names = []
    for s in seq:
        if s not in seen:
            seen[s] = len(state_names)
            state_names.append(s)

    n_states = len(state_names)
    counts = [[0] * n_states for _ in range(n_states)]

    for i in range(len(seq) - 1):
        src = seen[seq[i]]
        dst = seen[seq[i + 1]]
        counts[src][dst] += 1

    # Normalize rows to probabilities
    matrix = []
    for row in counts:
        total = sum(row)
        if total > 0:
            matrix.append([c / total for c in row])
        else:
            matrix.append([0.0] * n_states)

    return state_names, matrix
