"""Fit the RT LMi light curves from demo 13.

Set ``RUN_LEAST_SQUARES`` and ``RUN_MCMC`` independently below. The unspotted
least-squares fit starts from the existing demo 13 result because the notebook
does not provide its original starting values, and overwrites that result when
finished. MCMC adds one primary spot to the saved least-squares
result and fits both the system and the spot.

Fit results are written to ``demo_data/assets/`` and the compressed MCMC chain
to ``demo_data/demo13/chains/``. MCMC can use the existing least-squares
result directly or a newly fitted result from this script.
Both stages use the same surface discretization and interpolation threshold.
``NSTEPS`` counts production steps after ``BURN_IN``; the exported chain stores
production samples only, with axes (step, walker, parameter). The default run
uses the bundled LSQ result and performs only MCMC. Run this script explicitly
to reproduce the expensive fit; the notebook loads the published archive.
"""

from __future__ import annotations

import gzip
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import astropy.units as u

from elisa import settings
from elisa.analytics import LCBinaryAnalyticsTask, LCData

RUN_LEAST_SQUARES = False
RUN_MCMC = True

DATA_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = DATA_DIR / "assets"
CHAIN_CACHE_DIR = Path(__file__).resolve().parent / "chains"
LEAST_SQUARES_RESULT = ASSETS_DIR / "RT_LMi_lc_lstsqr.json"
MCMC_RESULT = ASSETS_DIR / "RT_LMi_spot_fit.json"
MCMC_CHAIN = CHAIN_CACHE_DIR / "RT_LMi_spot_fit.json"

REFERENCE_MAGNITUDE = 11.35
DISCRETIZATION = 8
INTERP_THRESHOLD = 150
NSTEPS = 2000
NWALKERS = 100
BURN_IN = 1500
PROGRESS = True

PRIMARY_SPOTS = [
    {
        "label": "spot1",
        "longitude": {"value": 230, "fixed": False, "min": 180, "max": 270, "unit": u.degree},
        "latitude": {"value": 45, "fixed": False, "min": 0, "max": 90, "unit": u.degree},
        # Broaden the priors beyond the previous posterior's active radius boundary.
        "angular_radius": {"value": 50, "fixed": False, "min": 5, "max": 80, "unit": u.degree},
        "temperature_factor": {"value": 0.98, "fixed": False, "min": 0.85, "max": 1.0},
    },
]


def compress_chain(chain_file: Path) -> Path:
    """Compress the chain without a timestamp and remove the uncompressed file.

    :param chain_file: Path to the JSON chain.
    :type chain_file: Path
    :returns: Path to the compressed chain.
    :rtype: Path
    """
    archive = chain_file.with_name(f"{chain_file.name}.gz")
    with (
        chain_file.open("rb") as source,
        archive.open("wb") as target,
        gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as compressed,
    ):
        while block := source.read(1 << 20):
            compressed.write(block)
    chain_file.unlink()
    return archive


def main() -> None:
    """Fit unspotted RT LMi with LSQ, then jointly sample binary and spot parameters.

    Each stage is independently enabled by its RUN_* flag. LSQ restarts from
    the bundled unspotted result and saves back to that path. MCMC reloads that
    result (including when LSQ just ran), preserves its fixed parameters and
    constraints, then adds a primary spot and updates the potential bounds.
    Parameter summaries go to assets/; only the MCMC chain goes to the ignored
    chains/ cache, where deterministic gzip replaces the plain JSON export.
    """
    if not RUN_LEAST_SQUARES and not RUN_MCMC:
        sys.stdout.write("Both fit stages are disabled.\n")
        return

    if RUN_MCMC and not RUN_LEAST_SQUARES and not LEAST_SQUARES_RESULT.is_file():
        msg = f"Least-squares result not found: {LEAST_SQUARES_RESULT}."
        raise FileNotFoundError(msg)

    settings.configure(LOG_CONFIG="fit")
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        f"Generic.Bessell.{band}": LCData.load_from_file(
            filename=str(DATA_DIR / "lc_data" / f"RTLMi_{band}.dat"),
            x_unit=u.d,
            y_unit=u.mag,
            reference_magnitude=REFERENCE_MAGNITUDE,
        )
        for band in ("B", "V", "R", "I")
    }

    if RUN_LEAST_SQUARES:
        least_squares_task = LCBinaryAnalyticsTask(
            data=data,
            method="least_squares",
            expected_morphology="over-contact",
        )
        initial = least_squares_task.load_result(str(LEAST_SQUARES_RESULT))
        started = perf_counter()
        least_squares_task.fit(
            x0=initial,
            discretization=DISCRETIZATION,
            interp_treshold=INTERP_THRESHOLD,
        )
        least_squares_task.save_result(str(LEAST_SQUARES_RESULT))
        sys.stdout.write(f"Least squares completed in {perf_counter() - started:.1f} seconds: {LEAST_SQUARES_RESULT}\n")

    if RUN_MCMC:
        CHAIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        mcmc_task = LCBinaryAnalyticsTask(data=data, method="mcmc", expected_morphology="over-contact")
        initial = deepcopy(mcmc_task.load_result(str(LEAST_SQUARES_RESULT)))
        initial["primary"]["spots"] = deepcopy(PRIMARY_SPOTS)
        # Roche critical potentials bound this fixed-q, circular, synchronous system.
        potential = initial["primary"]["surface_potential"]
        potential.update(min=2.38, max=2.60)
        if not potential["min"] < potential["value"] < potential["max"]:
            msg = f"LSQ primary surface potential {potential['value']} must lie strictly within (2.38, 2.60)."
            raise ValueError(msg)
        started = perf_counter()
        mcmc_task.fit(
            x0=initial,
            nsteps=NSTEPS,
            nwalkers=NWALKERS,
            burn_in=BURN_IN,
            discretization=DISCRETIZATION,
            interp_treshold=INTERP_THRESHOLD,
            progress=PROGRESS,
            fit_id=str(MCMC_CHAIN),
            save=True,
        )
        mcmc_task.save_result(str(MCMC_RESULT))
        archive = compress_chain(MCMC_CHAIN)
        sys.stdout.write(f"MCMC completed in {perf_counter() - started:.1f} seconds: {MCMC_RESULT}\n")
        sys.stdout.write(f"Compressed chain: {archive}\n")


if __name__ == "__main__":
    main()
