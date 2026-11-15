"""Capture deterministic scientific arrays for comparison with a Git baseline.

Run in each checkout with that checkout on PYTHONPATH:
    python /path/to/tests/synthesis_baseline.py output.npz
Compare the two captures with numpy.testing.assert_array_equal for every key.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

# Initialize the same constants and settings as ELISA's scientific regression tests.
import tests  # noqa: F401 - initialize constants before importing ELISA

# isort: split
from elisa import Observer, settings
from elisa.binary_system.system import BinarySystem
from tests import utils
from tests.tests_analytics.fitting_helpers import BINARY_PARAMETERS, V

if TYPE_CHECKING:
    import pathlib


def capture(destination: str | pathlib.Path) -> None:
    settings.configure(
        LIMB_DARKENING_LAW="linear",
        NUMBER_OF_PROCESSES=1,
        USE_SINGLE_LD_COEFFICIENTS=False,
        REFLECTION_EFFECT=True,
        REFLECTION_EFFECT_ITERATIONS=2,
        MESH_GENERATOR="auto",
        SUPPRESS_WARNINGS=True,
    )
    output = {}
    for kind in ["plain", "spot", "pulsation", "over-contact"]:
        cfg = deepcopy(BINARY_PARAMETERS)
        for component in ["primary", "secondary"]:
            cfg[component].update(discretization_factor=10.0, synchronicity=1.0, atmosphere="ck04")
        if kind == "spot":
            cfg["primary"]["spots"] = [
                {"longitude": 90.0, "latitude": 60.0, "angular_radius": 25.0, "temperature_factor": 0.9},
            ]
        elif kind == "pulsation":
            cfg["primary"]["pulsations"] = [
                {"l": 1, "m": 1, "amplitude": 50.0, "frequency": 16.0, "mode_axis_theta": 30.0},
            ]
        elif kind == "over-contact":
            cfg["primary"]["surface_potential"] = cfg["secondary"]["surface_potential"] = 3.0
        binary = BinarySystem.from_json(cfg, _verify=False)
        container = utils.prepare_orbital_position_container(binary)
        container.build(components_distance=1.0)
        for component in ["primary", "secondary"]:
            star = getattr(container, component)
            for field in [
                "points",
                "faces",
                "areas",
                "temperatures",
                "log_g",
                "normals",
                "potential_gradient_magnitudes",
            ]:
                output[f"{kind}_{component}_{field}"] = np.asarray(getattr(star, field))
        observer = Observer(passband=[V], system=binary)
        _, curve = observer.lc(phases=np.array([0.0, 0.1, 0.25, 0.5, 0.7]))
        output[f"{kind}_ck04_curve"] = curve[V]
        for component in ["primary", "secondary"]:
            getattr(binary, component).atmosphere = "bb"
        _, curve = observer.lc(phases=np.array([0.0, 0.1, 0.25, 0.5, 0.7]))
        output[f"{kind}_bb_curve"] = curve[V]
    np.savez(destination, **output)
    print(f"Saved {len(output)} deterministic arrays to {destination}")


if __name__ == "__main__":
    capture(sys.argv[1])
