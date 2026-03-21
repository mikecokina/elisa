from __future__ import annotations

"""Unit tests for SingleSystem.from_json using standard and radius parameter dicts.

These tests check that the constructor accepts the two supported JSON shapes and
produces a usable :class:`elisa.single_system.system.SingleSystem` instance.
"""

from typing import TYPE_CHECKING, Any

import numpy as np

from elisa import units as u
from elisa.single_system.system import SingleSystem

if TYPE_CHECKING:
    # Import TypedDicts only for type-checking to avoid runtime import issues.
    from elisa.single_system.system import SingleRadiusParams, SingleStandardParams


def _numeric_value(x: Any) -> float:
    """Return a numeric float value for quantities or plain numbers.

    This helper converts astropy Quantity-like objects or plain numbers to a
    float for comparison in tests.

    :param x: Value to convert.
    :returns: Float representation of the input value.
    :rtype: float
    """
    # noinspection PyBroadException
    try:
        return float(x)
    except Exception:
        return x  # let the assertion fail if not numeric


def test_single_system_from_standard_params() -> None:
    """Construct SingleSystem from the "standard" parameter mapping.

    The mapping follows the example provided in the issue and must produce a
    valid SingleSystem instance with expected numeric star properties.
    """
    standard_params: SingleStandardParams = {
        "system": {
            "inclination": 90.0,
            "rotation_period": 10.1,
            "gamma": 10000,
            "reference_time": 0.5,
            "phase_shift": 0.0,
        },
        "star": {
            "mass": 1.0,
            "t_eff": 5772.0,
            "gravity_darkening": 0.32,
            "discretization_factor": 5,
            "metallicity": 0.0,
            "polar_log_g": "4.43775 dex(cm.s-2)",
        },
    }

    single = SingleSystem.from_json(standard_params)
    assert isinstance(single, SingleSystem)
    assert hasattr(single, "star")

    t_eff = _numeric_value(single.star.t_eff)
    mass = _numeric_value(single.star.mass)

    expected_mass = (1.0 * u.solMass).to(u.DefaultStarUnits.mass).value

    assert np.isclose(t_eff, 5772.0)
    assert np.isclose(mass, expected_mass)


def test_single_system_from_radius_params() -> None:
    """Construct SingleSystem from the "radius" parameter mapping.

    The radius-style JSON uses an ``equivalent_radius`` entry (string or
    quantity). The function should accept the input and return a valid
    SingleSystem instance.
    """
    radius_params: SingleRadiusParams = {
        "system": {
            "inclination": 90.0,
            "rotation_period": 10.1,
            "gamma": 10000,
            "reference_time": 0.5,
            "phase_shift": 0.0,
        },
        "star": {
            "mass": 1.0,
            "t_eff": 5772.0,
            "gravity_darkening": 0.32,
            "discretization_factor": 5,
            "metallicity": 0.0,
            "equivalent_radius": "1 solRad",
        },
    }

    single = SingleSystem.from_json(radius_params)
    assert isinstance(single, SingleSystem)
    assert hasattr(single, "star")

    # mass and t_eff should still be preserved after transform
    t_eff = _numeric_value(single.star.t_eff)
    mass = _numeric_value(single.star.mass)

    expected_mass = (1.0 * u.solMass).to(u.DefaultStarUnits.mass).value

    assert np.isclose(t_eff, 5772.0)
    assert np.isclose(mass, expected_mass)
