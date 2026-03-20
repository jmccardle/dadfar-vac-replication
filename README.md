# [Re] Vocabulary-Activation Correspondence in Self-Referential LLM Processing

**Author:** John P. McCardle, FFwF Robotics, LLC (john@ffwf.net)
**License:** CC BY 4.0
**Dataset DOI:** [10.5281/zenodo.19139301](https://doi.org/10.5281/zenodo.19139301)
**Original paper:** Dadfar (2026), [arXiv:2602.11358](https://arxiv.org/abs/2602.11358)

This repository contains the data, code, and analysis scripts for a
replication of Dadfar (2026) "Vocabulary-Activation Correspondence in
Self-Referential LLM Processing."

**Result: The replication fails.** VAC as reported does not survive
length correction or cross-model validation. The underlying autoregressive
dynamics (limit cycles, vocabulary narrowing) are real and reproducible,
but they are properties of extended generation, not of self-referential
processing specifically.

## Quick Start (Analysis Only — No GPU)

```bash
pip install -r requirements-analysis.txt
python3 scripts/revision_r05_fdr_correction.py     # BH-FDR: 4/22 survive
python3 scripts/revision_r07_survival_analysis.py   # Kaplan-Meier: p=0.005
python3 scripts/revision_r10_bimodality_test.py     # Hartigan dip: p<0.0001
python3 scripts/phase_b_spectral_vac_analysis.py    # Spectral confound: α=1.46
```

## Repository Structure

```
├── scripts/               # Analysis and generation scripts
│   ├── revision_r01_*.py  # TOST equivalence test
│   ├── revision_r02_*.py  # F-statistic permutation tests
│   ├── revision_r03_*.py  # VAC permutation test
│   ├── revision_r04_*.py  # Hedges' g correction
│   ├── revision_r05_*.py  # BH-FDR correction
│   ├── revision_r06_*.py  # Log-transformed partial correlations
│   ├── revision_r07_*.py  # Kaplan-Meier survival analysis
│   ├── revision_r08_*.py  # Lock-in resistance validation
│   ├── revision_r09_*.py  # Temperature pairwise tests
│   ├── revision_r10_*.py  # Hartigan's dip test for bimodality
│   ├── revision_r11_*.py  # Cosine direction permutation test
│   ├── phase_b_*.py       # Spectral confound analysis
│   ├── phase_e1_*.py      # Cross-prompt activation analysis
│   ├── phase_e2_*.py      # Controlled-length VAC analysis
│   └── phase_d_*.py       # Generation: control runs, layer sweep
│
├── src/                   # Source code
│   ├── generation/        # Pull Methodology runner, activation hooks
│   ├── metrics/           # Activation metric computation, vocabulary counting
│   ├── analysis/          # Correspondence analysis, loop detection
│   ├── figures/           # Figure generation utilities
│   └── config.py          # Model config, prompts, vocabulary clusters
│
├── notebooks/             # Jupyter notebooks (executed, with output)
│   ├── 01_colab_reproduce.ipynb    # Self-contained Colab reproduction
│   ├── 01_reproduce_experiments.ipynb
│   └── 02_generate_figures.ipynb
│
├── outputs/
│   ├── runs/              # Pre-computed results (JSON) + raw text
│   │   ├── extended_baseline/     # 50 Qwen runs, 28K token cap
│   │   ├── baseline/              # 50 Qwen runs, 8K token cap (Dadfar's config)
│   │   ├── phase_d_controls/      # 125 runs across 8 conditions
│   │   ├── phase_d3_temperature/  # Temperature ablation (T=0.3,0.7,1.0)
│   │   └── phase_d_layer_sweep/   # 8-layer × 8-condition sweep
│   └── analysis/          # Pre-computed statistics from revision scripts
│
├── zenodo/data/           # Dadfar's published Zenodo reference data
│
├── rescience/             # ReScience C paper (LaTeX)
│   └── content.tex
│
├── requirements-analysis.txt    # pip deps for analysis (no GPU)
└── requirements-generation.txt  # pip deps for generation (GPU required)
```

## Data

All run data is included as JSON results files containing pre-computed
activation metrics, vocabulary counts, cycle detection results, and
compliance data. Raw model-generated text is included as .txt files.

**Raw activation tensors are NOT included** (270 GB). They are available
on Zenodo with a single-run sample for pipeline verification.

### Key Data Files

| File | Description |
|------|-------------|
| `outputs/runs/extended_baseline/extended_baseline_results.json` | 50 Qwen runs (28K cap): metrics, vocab, terminal words |
| `outputs/runs/phase_d_controls/phase_d_results.json` | 125 runs (8 conditions): metrics, vocab, cycle detection |
| `outputs/runs/phase_d_layer_sweep/layer_sweep_results.json` | Layer sweep: per-layer activation metrics |
| `outputs/runs/phase_d3_temperature/temperature_ablation_results.json` | Temperature ablation results |
| `zenodo/data/qwen_baseline_n50.json` | Dadfar's published Qwen data |
| `zenodo/data/llama_baseline_n50.json` | Dadfar's published Llama 70B data |

## Key Results

| Analysis | Script | Finding |
|----------|--------|---------|
| Bimodality | `revision_r10` | Hartigan's dip = 0.350, p < 0.0001 |
| Spectral confound | `phase_b` | α = 1.46, R² = 0.928 |
| BH-FDR correction | `revision_r05` | Only 4/22 partial correlations survive |
| Survival analysis | `revision_r07` | Log-rank χ² = 20.2, p = 0.005 |
| Lock-in validation | `revision_r08` | r = −0.433 with TTR (construct validity) |
| VAC universality | `revision_r03` | Nonsense: 31 sig. pairs vs. baseline's 25 (p = 0.40) |
| Direction test | `revision_r11` | Δcos = 0.08, p = 0.285 (not special) |
| Temperature | `revision_r09` | KW p = 0.83 (temperature-invariant) |
| Effect sizes | `revision_r04` | d = 8.02 → g = 6.42 (Hedges' correction) |

## Hardware Used

- **Qwen 2.5-32B**: NVIDIA RTX 4090 (24 GB), 4-bit NF4
- **Llama 3.1 70B**: NVIDIA RTX 6000 Ada (48 GB) via RunPod, 4-bit NF4
- **Other models**: NVIDIA RTX 4090

## Citation

```bibtex
@article{mccardle2026revac,
  author  = {McCardle, John P.},
  title   = {[Re] Vocabulary-Activation Correspondence in Self-Referential LLM Processing},
  year    = {2026},
  note    = {ReScience C submission}
}
```
