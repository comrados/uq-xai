# UQ-XAI

This repository accompanies the paper "When Not to Explain: Uncertainty as a Gate for XAI Reliability".

Citation (placeholder):

```bibtex
@article{placeholder,
  title={Paper Title Placeholder},
  author={Author, A. and Author, B.},
  journal={Venue Placeholder},
  year={20XX}
}
```

## Description

This repository provides a framework for studying how epistemic uncertainty relates to XAI reliability in tabular and image settings. It includes model wrappers for uncertainty decomposition, perturbation and adversarial data generation, and explainer pipelines to measure stability/faithfulness and test epistemic gating or reliability signaling. Scripts reproduce the experiments and figures from the paper.

## Installation

Python version used: 3.11.13

```bash
git clone https://github.com/comrados/uq-xai
cd uq-xai

pip install -r requirements.txt
```

## Project Structure

```
.
├── config/                # Global settings and experiment configuration
├── data/                  # Dataset loading, splitting, caching, perturbations
├── evaluation/            # Metrics for performance, uncertainty, explanations
├── experiments/           # Tabular experiments and analysis scripts
├── experiments_images/    # Image experiments and analysis scripts
├── explainers/            # SHAP, LIME, Integrated Gradients, SmoothGrad
├── models/                # Base model implementations and registry
├── results/               # Generated outputs (tables, plots, pickles)
├── tables/                # LaTeX tables and artifacts for the paper
├── uncertainty/           # UQ wrappers and uncertainty decomposition
├── visualization/         # Plotting and table-generation utilities
├── LICENSE                # BSD-3-Clause license text
├── README.md              # This file
├── requirements.txt       # Main dependencies
└── exact_requirements.txt # Exact dependency snapshot
```

## License

BSD-3-Clause
