"""Benchmark LC prediction overhead separately from real physical synthesis.

Run with the desired source tree on PYTHONPATH and real ELISA support tables
configured. Synthesis is executed and timed first; timed prediction calls then
replay those actual raw curves, excluding the engine's cost. This isolates
validation, phase conversion, mapping/interpolation and normalization. Older
implementations without prepared mappings are supported for before/after runs.

Example::

    PYTHONPATH=src python scripts/benchmark/lc_prediction_overhead.py --number 1000 --repeat 9
"""
import argparse
import inspect
import json
import logging
import statistics
import timeit
from functools import partial
from time import perf_counter
from typing import Any

import numpy as np

from elisa import settings
from elisa.analytics.binary_fit import lc_prediction
from elisa.analytics.models import lc
from elisa.binary_system.system import BinarySystem
from elisa.observer.observer import Observer


def _replay(*_args: Any, curves: dict, **_kwargs: Any) -> dict:
    """Return actual precomputed synthesis arrays, excluding synthesis from timing."""
    return curves


def benchmark(number: int, repeat: int) -> list[dict]:
    """Return median warmed timings in microseconds for direct/interpolated cases."""
    settings.configure(NUMBER_OF_PROCESSES=1, ATM_ATLAS="bb", LIMB_DARKENING_LAW="linear",
                       USE_SINGLE_LD_COEFFICIENTS=True, MESH_GENERATOR="auto")
    physical = {
        "system@inclination": 85., "system@eccentricity": 0.,
        "system@argument_of_periastron": 0., "system@period": 3.,
        "system@mass_ratio": .7, "system@semi_major_axis": 10.,
    }
    for component, temperature in [("primary", 6500.), ("secondary", 5000.)]:
        physical.update({f"{component}@{key}": value for key, value in {
            "t_eff": temperature, "surface_potential": 6., "gravity_darkening": .32,
            "albedo": .5, "metallicity": 0.}.items()})
    bands = ["Generic.Bessell.V", "Generic.Bessell.B"]
    prepared = "_validated_reducers" in inspect.signature(lc_prediction.predict_light_curves).parameters
    validate = getattr(lc_prediction, "validate_observations", None)
    if validate is None:
        validate = lc_prediction._validate_observations  # noqa: SLF001 - pre-cleanup source compatibility
    records = []
    for name, size, repetitions, interpolated in [
        ("small_direct", 8, 1, False), ("large_direct", 302, 20, False),
        ("large_interpolated", 302, 20, True),
    ]:
        xs = np.linspace(0., .99, size)
        maps = {band: np.tile(np.arange(size)[::(-1 if i else 1)], repetitions)
                for i, band in enumerate(bands)}
        x_data = {band: xs[index] for band, index in maps.items()}
        y_data = {band: np.ones(len(index)) for band, index in maps.items()}
        errors = {band: np.full(len(index), .01) for band, index in maps.items()}
        support = np.linspace(-.01, 1.01, 302) if interpolated else None
        observer = Observer(passband=bands, system=None)
        observer.system_cls = BinarySystem
        synthesis_support = xs if support is None else support
        durations = []
        for _ in range(3):
            start = perf_counter()
            raw = lc.synthetic_binary(synthesis_support, 10., observer, **physical)
            durations.append(perf_counter()-start)
        kwargs = {"fit_xs": support, "x_data": x_data, "y_err": errors,
                  "synthesis": partial(_replay, curves=raw)}
        reducers = validate(xs, y_data, maps, x_data, errors)
        if prepared:
            kwargs["_validated_reducers"] = reducers
        predict = partial(lc_prediction.predict_light_curves,
                          xs, y_data, maps, 10., observer, 10000, physical, **kwargs)
        predict()
        timings = timeit.repeat(predict, number=number, repeat=repeat)
        static = timeit.repeat(partial(validate, xs, y_data, maps, x_data, errors),
                               number=number, repeat=repeat)
        records.append({"case": name, "prepared": prepared, "observations": sum(map(len, y_data.values())),
                            "prediction_us": statistics.median(timings)*1e6/number,
                            "validation_us": statistics.median(static)*1e6/number,
                            "physical_synthesis_us": statistics.median(durations)*1e6,
                            "number": number, "repeat": repeat})
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--number", type=int, default=1000)
    parser.add_argument("--repeat", type=int, default=9)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    print(json.dumps(benchmark(args.number, args.repeat), indent=2))  # noqa: T201
