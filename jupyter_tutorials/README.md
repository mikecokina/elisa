# ELISA tutorials

The numbered notebooks introduce system construction, orbital and surface geometry, synthetic observations, datasets and fitting workflows. Each notebook contains its own scientific context and any limitations of its examples.

## Setup

Use Python 3.11 or later with this repository's version of ELISA and its dependencies installed, plus Jupyter and an IPython kernel in the same environment. Select that kernel when opening the notebooks.

Run notebooks with `jupyter_tutorials/` as the working directory so relative paths to bundled observations and results in `demo_data/` resolve correctly. Configure atmosphere and limb-darkening tables through ELISA settings before running examples that use those assets.

The notebooks use inline plotting. Run cells in order from a fresh kernel when recomputing an example. Generated RV fitting results and chains are written to `notebook_outputs/`. Demo 12 writes newly computed LC results and chains to `notebook_outputs/demo12/`; its default workflow loads the saved least-squares result and the 12.0, 12.1 and 12.2 MCMC chains from `demo_data/assets/` (produced by `fit_lc_12_0.py`, `fit_lc_12_1.py` and `fit_lc_12_2.py`) and computes all diagnostics from them, while least-squares and MCMC recomputation are enabled separately.
