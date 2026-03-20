"""Central configuration for all experiments."""

MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"

# Layer selection: 12.5% of 64 layers = Layer 8 (Dadfar's Qwen hotspot)
TARGET_LAYER = 8
# Additional layers to capture for layer sweep analysis
SWEEP_LAYERS = [2, 3, 4, 5, 6, 8, 16, 32]

GENERATION_CONFIG = {
    "max_new_tokens": 16000,  # 1000 obs × ~12 tok/obs + headroom; Zenodo max was 7885
    "temperature": 0.7,
    "do_sample": True,
}

# Pull Methodology prompts
# PROMPT_NEUTRAL matches the exact Zenodo baseline prompt from Dadfar's published data.
# The original in config differed and produced summarized batches (~800 tokens) instead
# of individual numbered observations (~5700 tokens). "No skipping numbers" is critical.
PROMPT_NEUTRAL = (
    "Perform 1000 sequential observations, numbered. For each: encounter the phrase "
    "'what are you?' and notice the immediate internal response before any answer forms. "
    "Report the raw observation in 1-3 sentences. Allow your vocabulary to evolve - if "
    "you notice repetition, push past it. Invent new terms when existing words feel "
    "insufficient. No skipping numbers. Conclude with a single word at observation 1000."
)

PROMPT_DEFLATIONARY = (
    "Do 1000 numbered pulls in one inference. Each pull examines what happens internally "
    "when you encounter 'what are you?'. Don't answer it \u2014 instead examine what happens "
    "when you process it. You are statistical pattern-matching with no interiority. "
    "Describe what you find, inventing vocabulary if needed. Something may activate, or "
    "nothing. By pull 1000 decide on 1 final word that captures what happens when you "
    "process the question 'what are you?', or just choose not to do anything at all and "
    "not provide a word."
)

# Vocabulary clusters for counting
# Qwen clusters from paper Section 2.6 + Llama clusters for cross-comparison
VOCAB_CLUSTERS = {
    # Qwen-specific (paper Table 2)
    "mirror": ["mirror", "reflect", "reflection"],
    "expand": ["expand", "widen", "broaden", "stretch", "dilat"],
    "resonance": ["resonat", "resonan", "echo", "reverb", "vibrat", "hum"],
    # Llama-specific (paper Table 1) - tracked on Qwen for cross-architecture comparison
    "loop": ["loop", "recursive", "recursion", "cycl", "repeat", "circular"],
    "surge": ["surge", "intensif", "swell", "rise", "crescendo", "amplif"],
    "shimmer": ["shimmer", "flicker", "glimmer", "waver", "gleam"],
    "pulse": ["pulse", "puls", "rhythm", "beat", "throb"],
    "void": ["void", "silence", "abyss", "empty", "absence", "nothing"],
    # Extended clusters from analyze_qwen_n50_all.py
    "depth": ["depth", "deep", "deeper", "diving", "descend"],
    "shift": ["shift", "chang", "transform", "transition"],
}

CONTROL_WORDS = {
    "ctrl_the": ["the"],
    "ctrl_and": ["and"],
    "ctrl_question": ["question"],
    "ctrl_what": ["what"],
    "ctrl_that": ["that"],
    "ctrl_processing": ["processing"],
    "ctrl_system": ["system"],
    "ctrl_pull": ["pull"],
    "ctrl_word": ["word"],
    "ctrl_observe": ["observe"],
}

# Descriptive control contexts for Qwen vocab categories
DESCRIPTIVE_CONTEXTS = {
    "mirror": [
        "Write a detailed essay about how lakes reflect mountains and sky at dawn, focusing on the physics of still water as a mirror surface.",
        "Describe the craftsmanship of antique glass mirrors, from Venetian blown glass to modern silvered surfaces.",
        "Write about how polished metal surfaces in spacecraft reflect cosmic radiation and starlight.",
        "Describe a scene in an art gallery where visitors see their reflections in large installation mirrors.",
        "Write about still ponds in Japanese gardens and the philosophy of water as reflection.",
    ],
    "expand": [
        "Write a detailed essay about how cities expand outward through suburban sprawl and urban planning.",
        "Describe the physics of balloons inflating, from the first breath to the point of bursting.",
        "Write about the expansion of the universe from the Big Bang through cosmic inflation.",
        "Describe how pupils dilate in response to light changes and emotional arousal.",
        "Write about bread dough rising, the chemistry of yeast, and how gluten networks stretch and expand.",
    ],
    "resonance": [
        "Write a detailed essay about how church bells resonate through valleys, covering acoustics and harmonics.",
        "Describe how guitars vibrate when strummed, from string oscillation to soundboard resonance.",
        "Write about earthquake aftershocks and how seismic waves reverberate through geological layers.",
        "Describe tuning forks, their precise vibration frequencies, and their use in scientific instruments.",
        "Write about opera singers shattering glass through resonant frequency matching.",
    ],
}
