"""Stage A and stage 12.0 of demo 12: least-squares fit and the initial MCMC run.

The least-squares result (``lc_least_squares.json``) is either loaded or recomputed. The MCMC is initialized
from it in a small ball of normalized coordinates, with Gaussian priors on the mass ratio and the primary
effective temperature and the noise-scale nuisance parameter ``ln_f`` bounded to ``[-10, 2]``.

Outputs:

* ``demo_data/assets/lc_mcmc_12_0_params.json`` - parameter summary (tracked in the ELISA repository),
* ``demo_data/demo12/chains/lc_mcmc_12_0_chain.json.gz`` - compressed chain (Git-ignored cache).

A new chain has a different checksum than the published one; update the manifest in the notebook if you
want it to replace the published asset. Running this script takes hours; the notebook never calls it.
"""

from __future__ import annotations

from copy import deepcopy
from time import perf_counter

import astropy.units as u
import numpy as np
from demo12_assets import ASSETS_DIR, CHAIN_CACHE_DIR, DATA_FILE, compress_chain, report

from elisa import settings
from elisa.analytics import LCBinaryAnalyticsTask, LCData
from elisa.analytics.params.parameters import BinaryInitialParameters

LOAD_LEAST_SQUARES_RESULT = True  # True: load saved LS; False: fit and save LS. Always run MCMC.
LEAST_SQUARES_FILE = ASSETS_DIR / "lc_least_squares.json"
OUTPUT_RESULT = ASSETS_DIR / "lc_mcmc_12_0_params.json"
OUTPUT_CHAIN = CHAIN_CACHE_DIR / "lc_mcmc_12_0_chain.json"
PERIOD = 2.47028
EPOCH = 54953.900507000006

NSTEPS = 2000
BURN_IN = 500
PROCESSES = settings.NUMBER_OF_PROCESSES
MCMC_PROCESSES = settings.NUMBER_OF_MCMC_PROCESSES
NWALKERS = max(32, 2 * MCMC_PROCESSES)
MAX_NFEV = 200
DISCRETIZATION = 10.0
INTERP_THRESHOLD = 100
SAMPLING = "adaptive"
LIMB_DARKENING_LAW = "cosine"
SEED = 12
INITIAL_SPREAD = 0.01
PROGRESS = True

LC_INITIAL = {
    "system": {
        "inclination": {
            "value": 76.0,
            "fixed": False,
            "min": 65,
            "max": 90,
            "unit": u.deg,
        },
        "eccentricity": {
            "value": 0.02,
            "fixed": False,
            "min": 0.0,
            "max": 0.1,
        },
        "mass_ratio": {
            "value": 1.08,
            "fixed": True,
        },
        "argument_of_periastron": {
            "value": 233,
            "fixed": False,
            "min": 0,
            "max": 360,
            "unit": u.deg,
        },
        "semi_major_axis": {
            "constraint": "11.2 / sin(radians(system@inclination))",
            "unit": u.solRad,
        },
        "period": {
            "value": PERIOD,
            "fixed": True,
            "unit": u.d,
        },
        "additional_light": {
            "value": 0.0,
            "fixed": False,
            "min": 0.0,
            "max": 0.5,
        },
        "phase_shift": {
            "value": 0,
            "fixed": False,
            "min": -0.02,
            "max": 0.02,
        },
    },
    "primary": {
        "t_eff": {
            "value": 7022.0,
            "fixed": True,
            "unit": u.K,
        },
        "surface_potential": {
            "value": 6.1,
            "fixed": False,
            "min": 5.0,
            "max": 10.0,
        },
        "albedo": {
            "value": 1.0,
            "fixed": True,
        },
        "gravity_darkening": {
            "value": 1.0,
            "fixed": True,
        },
        "synchronicity": {
            "constraint": "(1 + system@eccentricity)**2 / (1 - system@eccentricity**2)**(3.0/2.0)",
        },
        "metallicity": {
            "value": 0.0,
            "fixed": True,
            "unit": None,
        },
    },
    "secondary": {
        "t_eff": {
            "value": 6804,
            "fixed": False,
            "min": 3500.0,
            "max": 10000.0,
            "unit": u.K,
        },
        "surface_potential": {
            "value": 6.2,
            "fixed": False,
            "min": 4.0,
            "max": 10.0,
        },
        "albedo": {
            "value": 1.0,
            "fixed": True,
        },
        "gravity_darkening": {
            "value": 1.0,
            "fixed": True,
        },
        "synchronicity": {
            "constraint": "(1 + system@eccentricity)**2 / (1 - system@eccentricity**2)**(3.0/2.0)",
        },
        "metallicity": {
            "value": 0.0,
            "fixed": True,
            "unit": None,
        },
    },
}


def main() -> None:
    """Run the least-squares stage and the initial MCMC, then store results and the compressed chain."""
    settings.configure(
        LOG_CONFIG="fit",
        LIMB_DARKENING_LAW=LIMB_DARKENING_LAW,
        NUMBER_OF_PROCESSES=PROCESSES,
        NUMBER_OF_MCMC_PROCESSES=1,
    )
    CHAIN_CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
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

    least_squares_task = LCBinaryAnalyticsTask(
        data={"Kepler": kepler_data},
        method="least_squares",
        expected_morphology="detached",
    )
    if LOAD_LEAST_SQUARES_RESULT:
        least_squares_result = least_squares_task.load_result(str(LEAST_SQUARES_FILE))
        report(f"Loaded least-squares result from {LEAST_SQUARES_FILE}.")
    else:
        started = perf_counter()
        least_squares_result = least_squares_task.fit(
            x0=deepcopy(LC_INITIAL),
            discretization=DISCRETIZATION,
            interp_treshold=INTERP_THRESHOLD,
            samples=SAMPLING,
            max_nfev=MAX_NFEV,
        )
        least_squares_task.save_result(str(LEAST_SQUARES_FILE))
        report(f"Least squares completed in {perf_counter() - started:.1f} seconds.")

    mcmc_initial = deepcopy(least_squares_result)
    mcmc_initial.pop("r_squared", None)
    mcmc_initial["system"]["phase_shift"]["fixed"] = True
    mcmc_initial["system"]["mass_ratio"] = {
        "value": 1.08,
        "fixed": False,
        "sigma": 0.04,
        "min": 0.001,
        "max": 2.0,
    }
    mcmc_initial["primary"]["t_eff"] = {
        "value": 7022.0,
        "fixed": False,
        "sigma": 380.0,
        "min": 3500.0,
        "max": 10000.0,
        "unit": u.K,
    }
    mcmc_initial["nuisance"] = {
        "ln_f": {
            "value": -5.0,
            "fixed": False,
            "min": -10.0,
            "max": 2.0,
        },
    }

    # Initialize walkers near the fitted solution in normalized coordinates.
    parameters = BinaryInitialParameters(**deepcopy(mcmc_initial)).get_fitable()
    centre = np.array([(p.value - p.min) / (p.max - p.min) for p in parameters.values()])
    initial_state = np.random.default_rng(SEED).uniform(
        np.maximum(centre - INITIAL_SPREAD, 1e-8),
        np.minimum(centre + INITIAL_SPREAD, 1 - 1e-8),
        size=(NWALKERS, len(parameters)),
    )
    np.random.seed(SEED)  # noqa: NPY002 - emcee draws from the global NumPy state
    settings.configure(
        NUMBER_OF_PROCESSES=1,
        NUMBER_OF_MCMC_PROCESSES=MCMC_PROCESSES,
    )
    mcmc_task = LCBinaryAnalyticsTask(
        data={"Kepler": kepler_data},
        method="mcmc",
        expected_morphology="detached",
    )
    started = perf_counter()
    mcmc_task.fit(
        x0=mcmc_initial,
        nsteps=NSTEPS,
        nwalkers=NWALKERS,
        burn_in=BURN_IN,
        initial_state=initial_state,
        discretization=DISCRETIZATION,
        interp_treshold=INTERP_THRESHOLD,
        samples=SAMPLING,
        fit_id=str(OUTPUT_CHAIN),
        save=True,
        progress=PROGRESS,
    )
    mcmc_task.save_result(str(OUTPUT_RESULT))
    archive = compress_chain(OUTPUT_CHAIN)
    report(f"MCMC completed in {perf_counter() - started:.1f} seconds.")
    report(f"Compressed chain: {archive}")


if __name__ == "__main__":
    main()
