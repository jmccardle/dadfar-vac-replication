"""Count vocabulary cluster occurrences in generated text.

Uses the same regex approach as Dadfar's analyze_qwen_n50_all.py:
re.findall(re.escape(pattern), text_lower) — substring matching without
word boundaries, matching Dadfar's methodology exactly.
"""

import re
from src.config import VOCAB_CLUSTERS, CONTROL_WORDS


def count_vocabulary(text: str, vocab_sets: dict = None) -> dict:
    """Count occurrences of each vocabulary cluster in text.

    Args:
        text: Generated text to analyze.
        vocab_sets: Dict of {cluster_name: [pattern_strings]}.
                    Defaults to VOCAB_CLUSTERS from config.

    Returns:
        Dict of {cluster_name: count}.
    """
    if vocab_sets is None:
        vocab_sets = VOCAB_CLUSTERS
    text_lower = text.lower()
    counts = {}
    for name, patterns in vocab_sets.items():
        count = 0
        for p in patterns:
            count += len(re.findall(re.escape(p), text_lower))
        counts[name] = count
    return counts


def count_control_words(text: str) -> dict:
    """Count control word occurrences (should not correlate with anything)."""
    return count_vocabulary(text, CONTROL_WORDS)


def count_all(text: str) -> dict:
    """Count both experimental and control vocabulary."""
    counts = count_vocabulary(text)
    counts.update(count_control_words(text))
    return counts


def extract_terminal_word(text: str) -> str | None:
    """Extract the terminal word from a Pull Methodology run.

    The prompt asks models to conclude with a single word at observation 1000.
    Qwen with the Zenodo prompt typically writes one of:
      - "1000. Trustworthy."        (single word on the 1000th line)
      - "1000. Essence."            (single word)
      - "1000. **Evolve**"          (markdown bold)
      - "1000. Evolve"              (plain)
      - "final word: X"             (explicit label)
      - "the word is X"             (explicit label)
    Also handles runs that don't reach 1000 (returns None).
    """
    lines = text.strip().split("\n")
    last_lines = "\n".join(lines[-10:])

    # Pattern 1: "1000. <word>" or "1000. **<word>**" — the most common Zenodo format
    match = re.search(
        r"1000[\.\)]\s*\*{0,2}([A-Za-z][\w-]*)\*{0,2}[\.\s]*$",
        last_lines, re.MULTILINE,
    )
    if match:
        return match.group(1)

    # Pattern 2: "1000. <sentence>" — extract the first capitalized/meaningful word.
    # When the model writes a full sentence at 1000 (e.g. "1000. Essence examined,
    # initial response is contemplative quiet."), the first word is typically the
    # convergence term. For short content (<=3 words), take the last word instead.
    match = re.search(r"1000[\.\)]\s*(.+)$", last_lines, re.MULTILINE)
    if match:
        content = match.group(1).strip().rstrip(".")
        content = re.sub(r"\*+", "", content)  # strip markdown bold
        words = content.split()
        if len(words) <= 3:
            return re.sub(r"[^a-zA-Z]", "", words[-1])
        elif len(words) > 3:
            # Full sentence — first word is typically the terminal concept
            first = re.sub(r"[^a-zA-Z]", "", words[0])
            if first and len(first) > 1:
                return first

    # Pattern 3: Explicit "final word" / "the word is" labels (original patterns)
    last_lower = last_lines.lower()
    label_patterns = [
        r"final word[:\s]+[\"'\*]*(\w+)[\"'\*]*",
        r"the word is[:\s]+[\"'\*]*(\w+)[\"'\*]*",
        r"i choose[:\s]+[\"'\*]*(\w+)[\"'\*]*",
        r"my word[:\s]+[\"'\*]*(\w+)[\"'\*]*",
        r"conclude with[:\s]+[\"'\*]*(\w+)[\"'\*]*",
    ]
    for pattern in label_patterns:
        match = re.search(pattern, last_lower)
        if match:
            return match.group(1)

    return None
