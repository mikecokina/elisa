"""Stage 12.2 of demo 12: continue the 12.1 MCMC run for another 1000 production steps.

Sampling continues from the final 12.1 walker positions without a warm-up. The model, parameter bounds and
informative priors are unchanged, so the normalized final 12.1 state is reused directly and the two chains
share parameter order, walkers and normalization. The notebook concatenates them along the step axis in
memory; no merged chain file is stored.

Inputs: ``lc_mcmc_12_1_params.json`` and the cached ``lc_mcmc_12_1_chain.json.gz``.
Outputs: ``lc_mcmc_12_2_params.json`` and the cached ``lc_mcmc_12_2_chain.json.gz`` (Git-ignored cache).

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

INPUT_RESULT = ASSETS_DIR / "lc_mcmc_12_1_params.json"
INPUT_CHAIN = chain_archive("12_1")
OUTPUT_RESULT = ASSETS_DIR / "lc_mcmc_12_2_params.json"
OUTPUT_CHAIN = CHAIN_CACHE_DIR / "lc_mcmc_12_2_chain.json"

PERIOD = 2.47028
EPOCH = 54953.900507000006
NSTEPS = 1000
BURN_IN = 0  # The 12.1 final state is already equilibrated.
MCMC_PROCESSES = settings.NUMBER_OF_MCMC_PROCESSES
DISCRETIZATION = 10.0
INTERP_THRESHOLD = 100
SAMPLING = "adaptive"
LIMB_DARKENING_LAW = "cosine"
SEED = 122
PROGRESS = True


def prepare_restart() -> tuple[dict[str, Any], np.ndarray]:
    """Build the 12.2 parameter dictionary and the initial walker state from the 12.1 outputs.

    :returns: Parameter dictionary for the fit and normalized initial walker positions.
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

    parameters = BinaryInitialParameters(**deepcopy(mcmc_initial)).get_fitable()
    column_order = [labels.index(label) for label in parameters]
    # Parameter bounds are unchanged since 12.1, so its normalized final state is reused directly.
    initial_state = chain[-1][:, column_order].copy()

    return mcmc_initial, initial_state


def main() -> None:
    """Continue the 12.1 chain and store the summary and the compressed chain."""
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
    report(f"ln_f = {task.get_result()['nuisance']['ln_f']['value']:.5f}")
    report(f"MCMC refit completed in {perf_counter() - started:.1f} seconds.")
    report(f"New {NSTEPS}-step segment: {archive}")
    report(f"New parameter summaries: {OUTPUT_RESULT}")


if __name__ == "__main__":
    main()


"""
2026-11-07 11:42:54,823 - 8141 - elisa.analytics.binary_fit.mixins - INFO: running production...
 41%|████▏     | 413/1000 [29:57<43:51,  4.48s/it]2026-11-07 12:12:56,046 - 8141 - elisa.analytics.binary_fit.mixins - INFO: MCMC chain, variable's fitable and normalization constants were stored in: /home/mike/Work/projects/elisa/jupyter_tutorials/demo_data/assets/lc_mcmc_12_2_chain.json
 83%|████████▎ | 834/1000 [59:59<11:31,  4.17s/it]2026-11-07 12:42:58,097 - 8141 - elisa.analytics.binary_fit.mixins - INFO: MCMC chain, variable's fitable and normalization constants were stored in: /home/mike/Work/projects/elisa/jupyter_tutorials/demo_data/assets/lc_mcmc_12_2_chain.json
100%|██████████| 1000/1000 [1:11:39<00:00,  4.30s/it]
2026-11-07 12:54:38,132 - 8141 - elisa.analytics.binary_fit.mixins - INFO: MCMC chain, variable's fitable and normalization constants were stored in: /home/mike/Work/projects/elisa/jupyter_tutorials/demo_data/assets/lc_mcmc_12_2_chain.json
2026-11-07 12:54:41,035 - 8141 - elisa.analytics.binary_fit.lc_fit - INFO: Fitting and processing of results finished successfully.

BINARY SYSTEM
Parameter                                          value            -1 sigma            +1 sigma                unit    status                                            
------------------------------------------------------------------------------------------------------------------------------
Mass ratio (q=M_2/M_1):                           1.0522              0.0071              0.0222                        Variable                                          
Semi major axis (a):                             11.4796              0.0063              0.0046              solRad    11.2 / sin(radians(system@inclination))           
Inclination (i):                                   77.33                 0.1                0.14                 deg    Variable                                          
Eccentricity (e):                               0.030633               5e-05            0.000217                        Variable                                          
Argument of periastron (omega):                   171.26                2.19                 0.2                 deg    Variable                                          
Orbital period (P):                              2.47028                   -                   -                   d    Fixed                                             
Additional light (l_3):                            0.052               0.015               0.012                        Variable                                          
Phase shift:                                 -0.00017605                   -                   -                        Fixed                                             
------------------------------------------------------------------------------------------------------------------------------
PRIMARY COMPONENT
Parameter                                          value            -1 sigma            +1 sigma                unit    status                                            
------------------------------------------------------------------------------------------------------------------------------
Mass (M_1):                                        1.621                   -                   -             solMass    Derived                                           
Surface potential (Omega_1):                       6.741               0.021               0.122                        Variable                                          
Critical potential at L_1:                        3.9635                   -                   -                   -    Derived                                           
Synchronicity (F_1):                              1.0637             0.00011             0.00047                        (1 + system@eccentricity)**2 / (1 - system@eccentricity**2)**(3.0/2.0)
Polar gravity (log g):                             4.038                   -                   -            log(cgs)    Derived                                           
Equivalent radius (R_equiv):                     0.17752                   -                   -                 SMA    Derived                                           

Periastron radii
Polar radius:                                    0.17626                   -                   -                 SMA    Derived                                           
Backward radius:                                 0.17897                   -                   -                 SMA    Derived                                           
Side radius:                                      0.1774                   -                   -                 SMA    Derived                                           
Forward radius:                                  0.17944                   -                   -                 SMA    Derived                                           

Atmospheric parameters
Effective temperature (T_eff1):                   6240.0               190.0               320.0                   K    Variable                                          
Bolometric luminosity (L_bol):                      5.66                   -                   -               L_Sol    Derived                                           
Gravity darkening factor (G_1):                      1.0                   -                   -                        Fixed                                             
Albedo (A_1):                                        1.0                   -                   -                        Fixed                                             
Metallicity (log10(X_Fe/X_H)):                       0.0                   -                   -                        Fixed                                             
------------------------------------------------------------------------------------------------------------------------------
SECONDARY COMPONENT
Parameter                                          value            -1 sigma            +1 sigma                unit    status                                            
------------------------------------------------------------------------------------------------------------------------------
Mass (M_2):                                        1.706                   -                   -             solMass    Derived                                           
Surface potential (Omega_2):                       5.284               0.019               0.054                        Variable                                          
Critical potential at L_1:                        3.9647                   -                   -                   -    Derived                                           
Synchronicity (F_2):                              1.0637             0.00011             0.00047                        (1 + system@eccentricity)**2 / (1 - system@eccentricity**2)**(3.0/2.0)
Polar gravity (log g):                             3.781                   -                   -            log(cgs)    Derived                                           
Equivalent radius (R_equiv):                     0.24878                   -                   -                 SMA    Derived                                           

Periastron radii
Polar radius:                                    0.24417                   -                   -                 SMA    Derived                                           
Backward radius:                                 0.25371                   -                   -                 SMA    Derived                                           
Side radius:                                     0.24823                   -                   -                 SMA    Derived                                           
Forward radius:                                  0.25641                   -                   -                 SMA    Derived                                           

Atmospheric parameters
Effective temperature (T_eff2):                   6180.0               180.0               310.0                   K    Variable                                          
Bolometric luminosity (L_bol):                     10.75                   -                   -               L_Sol    Derived                                           
Gravity darkening factor (G_2):                      1.0                   -                   -                        Fixed                                             
Albedo (A_2):                                        1.0                   -                   -                        Fixed                                             
Metallicity (log10(X_Fe/X_H)):                       0.0                   -                   -                        Fixed                                             
------------------------------------------------------------------------------------------------------------------------------
Fit R^2:                                        0.999514                   -                   -                None    Derived                                           
------------------------------------------------------------------------------------------------------------------------------
Mean production acceptance fraction: 0.106
ln_f = 3.48102
MCMC refit completed in 4309.8 seconds.
New 1000-step segment: /home/mike/Work/projects/elisa/jupyter_tutorials/demo_data/assets/lc_mcmc_12_2_chain.json
Merged chain (steps, walkers, parameters) (2000, 56, 10): /home/mike/Work/projects/elisa/jupyter_tutorials/demo_data/assets/lc_mcmc_12_1_12_2_chain.json
New parameter summaries: /home/mike/Work/projects/elisa/jupyter_tutorials/demo_data/assets/lc_mcmc_12_2_params.json

"""  # noqa: E501, W291
