# ELISA tutorials

The numbered notebooks introduce system construction, orbital and surface geometry, synthetic observations, datasets and fitting workflows. Each notebook contains its own scientific context and any limitations of its examples.

## Setup

Use Python 3.11 or later with this repository's version of ELISA and its dependencies installed, plus Jupyter and an IPython kernel in the same environment. Select that kernel when opening the notebooks.

Run notebooks with `jupyter_tutorials/` as the working directory so relative paths to bundled observations and results in `demo_data/` resolve correctly. Configure atmosphere and limb-darkening tables through ELISA settings before running examples that use those assets.

The notebooks use inline plotting. Run cells in order from a fresh kernel when recomputing an example. Generated RV fitting results and chains are written to `notebook_outputs/`.
