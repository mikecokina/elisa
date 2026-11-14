"""Stage 12.1 of demo 12: restart the MCMC from the 12.0 walkers with wider ``ln_f`` bounds.

The 12.0 chain pushed the noise-scale nuisance parameter ``ln_f`` against its upper bound of 2. This stage
widens the bounds to ``[-10, 6]``, converts the final 12.0 walker positions from the old normalization to the
new one, redraws ``ln_f`` uniformly in ``LN_F_INITIAL_RANGE`` and samples 1000 production steps after a
300-step warm-up. The model, data and informative priors are unchanged.

Inputs: ``lc_mcmc_12_0_params.json`` and the cached ``lc_mcmc_12_0_chain.json.gz``.
Outputs: ``lc_mcmc_12_1_params.json`` and the cached ``lc_mcmc_12_1_chain.json.gz`` (Git-ignored cache).

A new chain has a different checksum than the published one; update the manifest in the notebook if you
want it to replace the published asset. Running this script takes hours; the notebook never calls it.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from time import perf_counter
from typing import Any

# Set before importing NumPy/ELISA, including in spawned MCMC workers.
NUMERICAL_THREADS = 1
for thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[thread_variable] = str(NUMERICAL_THREADS)

import astropy.units as u  # noqa: E402
import numpy as np  # noqa: E402
from demo12_assets import (  # noqa: E402
    ASSETS_DIR,
    CHAIN_CACHE_DIR,
    DATA_FILE,
    chain_archive,
    compress_chain,
    read_chain,
    report,
)

from elisa import settings  # noqa: E402
from elisa.analytics import LCBinaryAnalyticsTask, LCData  # noqa: E402
from elisa.analytics.params.parameters import BinaryInitialParameters  # noqa: E402

INPUT_RESULT = ASSETS_DIR / "lc_mcmc_12_0_params.json"
INPUT_CHAIN = chain_archive("12_0")
OUTPUT_RESULT = ASSETS_DIR / "lc_mcmc_12_1_params.json"
OUTPUT_CHAIN = CHAIN_CACHE_DIR / "lc_mcmc_12_1_chain.json"

PERIOD = 2.47028
EPOCH = 54953.900507000006
NSTEPS = 1000
BURN_IN = 300  # Re-equilibrate after changing the noise-scale prior.
MCMC_PROCESSES = settings.NUMBER_OF_MCMC_PROCESSES
DISCRETIZATION = 10.0
INTERP_THRESHOLD = 100
SAMPLING = "adaptive"
LIMB_DARKENING_LAW = "cosine"
SEED = 121
PROGRESS = True
INITIAL_SIGMA_SCALE = 1.0  # Keep the physical starting ensemble; optional cloud expansion.
LN_F_MIN = -10.0
LN_F_MAX = 6.0  # Exploratory bound: verify that the new posterior does not hit it.
LN_F_INITIAL_RANGE = (2.0, 4.0)  # Spread the noise scale away from the old upper boundary.


def prepare_restart() -> tuple[dict[str, Any], np.ndarray]:
    """Build the 12.1 parameter dictionary and the initial walker state from the 12.0 outputs.

    :returns: Parameter dictionary for the fit and normalized initial walker positions (new bounds).
    :rtype: tuple[dict[str, Any], numpy.ndarray]
    """
    result = json.loads(INPUT_RESULT.read_text())
    saved_chain = read_chain(INPUT_CHAIN)
    labels = saved_chain["fitable_parameters"]
    chain = np.asarray(saved_chain["flat_chain"]).reshape(saved_chain["chain_shape"])

    mcmc_initial = deepcopy(result)
    mcmc_initial.pop("r_squared", None)
    for section in mcmc_initial.values():
        for parameter in section.values():
            parameter.pop("confidence_interval", None)
            parameter.pop("estimate", None)
    # The original chain records Gaussian prior means and widths. In ELISA,
    # `value` is also the Gaussian prior mean, so retain it for those parameters.
    # Walker positions below supply the actual starting points near the new fit.
    for label in labels:
        component, name = label.split("@")
        prior = saved_chain["fitable"][label]
        parameter = mcmc_initial[component][name]
        parameter["sigma"] = prior["sigma"]
        if prior["sigma"] is not None:
            parameter["value"] = prior["value"]

    mcmc_initial["nuisance"]["ln_f"]["min"] = LN_F_MIN
    mcmc_initial["nuisance"]["ln_f"]["max"] = LN_F_MAX

    parameters = BinaryInitialParameters(**deepcopy(mcmc_initial)).get_fitable()
    column_order = [labels.index(label) for label in parameters]
    initial_state = chain[-1][:, column_order].copy()

    if INITIAL_SIGMA_SCALE != 1.0:
        # Stretch joint samples around the saved posterior medians. This retains
        # their correlations; it does not create new Gaussian parameter priors.
        bounds = np.array([saved_chain["normalization"][label] for label in parameters])
        centre = np.array(
            [result[component][name]["value"] for component, name in (label.split("@") for label in parameters)],
        )
        centre = (centre - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
        samples = chain.reshape(-1, chain.shape[-1])[:, column_order]
        candidates = centre + INITIAL_SIGMA_SCALE * (samples - centre)
        candidates = candidates[np.all((candidates > 0) & (candidates < 1), axis=1)]
        rows = np.random.default_rng(SEED).choice(
            len(candidates),
            size=initial_state.shape[0],
            replace=False,
        )
        initial_state = candidates[rows]

    # The saved states use the OLD normalization. Convert via physical values
    # before supplying them to a fit with changed bounds.
    old_bounds = np.array([saved_chain["normalization"][label] for label in parameters])
    new_bounds = np.array([(parameter.min, parameter.max) for parameter in parameters.values()])
    physical_state = old_bounds[:, 0] + initial_state * (old_bounds[:, 1] - old_bounds[:, 0])
    noise_column = list(parameters).index("nuisance@ln_f")
    physical_state[:, noise_column] = np.random.default_rng(SEED).uniform(
        *LN_F_INITIAL_RANGE,
        size=initial_state.shape[0],
    )
    initial_state = (physical_state - new_bounds[:, 0]) / (new_bounds[:, 1] - new_bounds[:, 0])

    return mcmc_initial, initial_state


def main() -> None:
    """Restart the MCMC from the 12.0 walkers and store the summary and the compressed chain."""
    settings.configure(
        LOG_CONFIG="fit",
        LIMB_DARKENING_LAW=LIMB_DARKENING_LAW,
        NUMBER_OF_PROCESSES=1,
        NUMBER_OF_MCMC_PROCESSES=MCMC_PROCESSES,
    )
    kepler_data = LCData.load_from_file(
        filename=str(DATA_FILE),
        x_unit=u.d,
        y_unit=u.dimensionless_unscaled,
        data_columns=(0, 1, 2),
    )
    kepler_data.convert_to_phases(
        PERIOD,
        EPOCH,
        centre=0.0,
    )
    mcmc_initial, initial_state = prepare_restart()
    np.random.seed(SEED)  # noqa: NPY002 - emcee draws from the global NumPy state
    task = LCBinaryAnalyticsTask(
        data={"Kepler": kepler_data},
        method="mcmc",
        expected_morphology="detached",
    )
    started = perf_counter()
    task.fit(
        x0=mcmc_initial,
        initial_state=initial_state,
        nsteps=NSTEPS,
        nwalkers=initial_state.shape[0],
        burn_in=BURN_IN,
        discretization=DISCRETIZATION,
        interp_treshold=INTERP_THRESHOLD,
        samples=SAMPLING,
        fit_id=str(OUTPUT_CHAIN),
        save=True,
        progress=PROGRESS,
    )
    task.save_result(str(OUTPUT_RESULT))
    sampler = task.fit_cls.fit_method_instance.last_sampler
    archive = compress_chain(OUTPUT_CHAIN)
    report(f"Mean production acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
    report(f"ln_f = {task.get_result()['nuisance']['ln_f']['value']:.5f} (bounds {LN_F_MIN}, {LN_F_MAX})")
    report(f"MCMC refit completed in {perf_counter() - started:.1f} seconds.")
    report(f"New {NSTEPS}-step segment: {archive}")
    report(f"New parameter summaries: {OUTPUT_RESULT}")


if __name__ == "__main__":
    main()


"""
2026-11-07 00:58:03,608 - 65907 - elisa.analytics.binary_fit.lc_fit - INFO: Fitting and processing of results finished successfully.

BINARY SYSTEM
Parameter                                          value            -1 sigma            +1 sigma                unit    status                                            
------------------------------------------------------------------------------------------------------------------------------
Mass ratio (q=M_2/M_1):                           1.0508              0.0091              0.0286                        Variable                                          
Semi major axis (a):                             11.4845              0.0046              0.0218              solRad    11.2 / sin(radians(system@inclination))           
Inclination (i):                                   77.22                0.47                 0.1                 deg    Variable                                          
Eccentricity (e):                               0.030826             0.00022             5.1e-05                        Variable                                          
Argument of periastron (omega):                   168.95                0.36                2.11                 deg    Variable                                          
Orbital period (P):                              2.47028                   -                   -                   d    Fixed                                             
Additional light (l_3):                            0.049               0.027                0.01                        Variable                                          
Phase shift:                                 -0.00017605                   -                   -                        Fixed                                             
------------------------------------------------------------------------------------------------------------------------------
PRIMARY COMPONENT
Parameter                                          value            -1 sigma            +1 sigma                unit    status                                            
------------------------------------------------------------------------------------------------------------------------------
Mass (M_1):                                        1.624                   -                   -             solMass    Derived                                           
Surface potential (Omega_1):                       6.684               0.257               0.048                        Variable                                          
Critical potential at L_1:                         3.962                   -                   -                   -    Derived                                           
Synchronicity (F_1):                             1.06412             0.00048             0.00011                        (1 + system@eccentricity)**2 / (1 - system@eccentricity**2)**(3.0/2.0)
Polar gravity (log g):                              4.03                   -                   -            log(cgs)    Derived                                           
Equivalent radius (R_equiv):                     0.17932                   -                   -                 SMA    Derived                                           

Periastron radii
Polar radius:                                    0.17801                   -                   -                 SMA    Derived                                           
Backward radius:                                 0.18083                   -                   -                 SMA    Derived                                           
Side radius:                                     0.17919                   -                   -                 SMA    Derived                                           
Forward radius:                                  0.18133                   -                   -                 SMA    Derived                                           

Atmospheric parameters
Effective temperature (T_eff1):                   5801.0                21.0               247.0                   K    Variable                                          
Bolometric luminosity (L_bol):                      4.33                   -                   -               L_Sol    Derived                                           
Gravity darkening factor (G_1):                      1.0                   -                   -                        Fixed                                             
Albedo (A_1):                                        1.0                   -                   -                        Fixed                                             
Metallicity (log10(X_Fe/X_H)):                       0.0                   -                   -                        Fixed                                             
------------------------------------------------------------------------------------------------------------------------------
SECONDARY COMPONENT
Parameter                                          value            -1 sigma            +1 sigma                unit    status                                            
------------------------------------------------------------------------------------------------------------------------------
Mass (M_2):                                        1.707                   -                   -             solMass    Derived                                           
Surface potential (Omega_2):                       5.349               0.073               0.054                        Variable                                          
Critical potential at L_1:                        3.9631                   -                   -                   -    Derived                                           
Synchronicity (F_2):                             1.06412             0.00048             0.00011                        (1 + system@eccentricity)**2 / (1 - system@eccentricity**2)**(3.0/2.0)
Polar gravity (log g):                             3.794                   -                   -            log(cgs)    Derived                                           
Equivalent radius (R_equiv):                     0.24464                   -                   -                 SMA    Derived                                           

Periastron radii
Polar radius:                                    0.24032                   -                   -                 SMA    Derived                                           
Backward radius:                                 0.24926                   -                   -                 SMA    Derived                                           
Side radius:                                     0.24414                   -                   -                 SMA    Derived                                           
Forward radius:                                  0.25172                   -                   -                 SMA    Derived                                           

Atmospheric parameters
Effective temperature (T_eff2):                   5762.0                19.0               241.0                   K    Variable                                          
Bolometric luminosity (L_bol):                      7.84                   -                   -               L_Sol    Derived                                           
Gravity darkening factor (G_2):                      1.0                   -                   -                        Fixed                                             
Albedo (A_2):                                        1.0                   -                   -                        Fixed                                             
Metallicity (log10(X_Fe/X_H)):                       0.0                   -                   -                        Fixed                                             
------------------------------------------------------------------------------------------------------------------------------
Fit R^2:                                        0.999246                   -                   -                None    Derived                                           
------------------------------------------------------------------------------------------------------------------------------
Mean production acceptance fraction: 0.102
ln_f = 3.50777 (bounds -10.0, 6.0)
MCMC refit completed in 6793.8 seconds.
New 1000-step segment: /home/mike/Work/projects/elisa/jupyter_tutorials/demo_data/assets/lc_mcmc_12_1_chain.json
New parameter summaries: /home/mike/Work/projects/elisa/jupyter_tutorials/demo_data/assets/lc_mcmc_12_1_params.json

Process finished with exit code 0
"""  # noqa: E501, W291
