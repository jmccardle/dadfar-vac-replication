"""Online loop detection via HuggingFace StoppingCriteria.

Periodically decodes the generated text, parses observations, and checks
for limit cycles. Terminates generation once a stable cycle is confirmed
or a maximum observation count is reached.

Usage:
    from src.generation.early_termination import LoopDetectionCriteria

    criteria = LoopDetectionCriteria(
        tokenizer=tokenizer,
        prompt_length=prompt_length,
        min_observations=50,       # don't check before this many obs
        max_observations=300,      # hard stop
        cycle_confirmations=3,     # consecutive cycle repetitions to confirm
        check_interval_tokens=100, # decode and check every N new tokens
    )
    outputs = model.generate(..., stopping_criteria=[criteria])
    # After generation:
    print(criteria.result)  # CycleResult or None
"""

import re
from transformers import StoppingCriteria

from src.analysis.loop_detection import (
    parse_observations,
    detect_cycle_exact,
    detect_cycle_similarity,
    CycleResult,
    extract_cycle_vocabulary,
    normalize_observation,
)


class LoopDetectionCriteria(StoppingCriteria):
    """StoppingCriteria that terminates generation when a limit cycle is detected.

    Periodically decodes the generated tokens, parses numbered observations,
    and runs cycle detection. Stops when:
      - A cycle is confirmed (exact or similarity-based), OR
      - max_observations are reached, OR
      - model emits EOS (handled by generate() itself)

    Attributes:
        result: After generation, contains the CycleResult if a cycle was
                detected, or None if generation ended for another reason.
        n_observations_at_stop: Number of observations when stopping was triggered.
        stop_reason: 'cycle_detected', 'max_observations', or 'not_stopped'.
    """

    def __init__(
        self,
        tokenizer,
        prompt_length: int,
        min_observations: int = 50,
        max_observations: int = 300,
        cycle_confirmations: int = 3,
        check_interval_tokens: int = 100,
        similarity_threshold: float = 0.85,
        min_period: int = 3,
        max_period: int = 50,
        verbose: bool = False,
    ):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.min_observations = min_observations
        self.max_observations = max_observations
        self.cycle_confirmations = cycle_confirmations
        self.check_interval_tokens = check_interval_tokens
        self.similarity_threshold = similarity_threshold
        self.min_period = min_period
        self.max_period = max_period
        self.verbose = verbose

        self._last_check_length = 0
        self._last_n_obs = 0
        self.result: CycleResult | None = None
        self.n_observations_at_stop: int = 0
        self.stop_reason: str = "not_stopped"

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        """Called after each generated token. Returns True to stop."""
        seq_length = input_ids.shape[1]
        generated_length = seq_length - self.prompt_length

        # Only check at intervals to avoid excessive decoding
        if generated_length - self._last_check_length < self.check_interval_tokens:
            return False

        self._last_check_length = generated_length

        # Decode only the generated portion
        generated_ids = input_ids[0, self.prompt_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Parse observations
        observations = parse_observations(text)
        n_obs = len(observations)

        if n_obs <= self._last_n_obs:
            # No new observations yet — still generating within one observation
            return False

        self._last_n_obs = n_obs

        # Check max observations
        if n_obs >= self.max_observations:
            self.n_observations_at_stop = n_obs
            self.stop_reason = "max_observations"
            if self.verbose:
                print(f"    [early-term] max observations ({n_obs}) reached")
            return True

        # Don't check for cycles until we have enough data
        if n_obs < self.min_observations:
            return False

        # Run cycle detection
        lock_in, period = detect_cycle_exact(
            observations, self.min_period, self.max_period, self.cycle_confirmations
        )
        if lock_in is None:
            lock_in, period = detect_cycle_similarity(
                observations, self.min_period, self.max_period,
                self.similarity_threshold, self.cycle_confirmations
            )

        if lock_in is not None:
            # Cycle confirmed — stop generation
            self.n_observations_at_stop = n_obs
            self.stop_reason = "cycle_detected"

            cycle_vocab = extract_cycle_vocabulary(observations, lock_in, period)
            norms = [normalize_observation(o) for o in observations]

            self.result = CycleResult(
                has_cycle=True,
                lock_in_obs=lock_in,
                cycle_period=period,
                cycle_vocabulary=cycle_vocab,
                n_observations=n_obs,
                n_unique=len(set(norms)),
                unique_ratio=len(set(norms)) / n_obs,
                observations=observations,
            )

            if self.verbose:
                print(f"    [early-term] cycle detected: lock-in={lock_in}, "
                      f"period={period}, obs={n_obs}")
            return True

        return False
