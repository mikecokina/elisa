"""Community binary inputs are validated before component masses are derived."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import units as u
from elisa.base.error import PhysicalStateError
from elisa.binary_system.system import BinarySystem
from elisa.binary_system.utils import transform_json_community_to_std
from tests.tests_analytics.fitting_helpers import BINARY_PARAMETERS


@pytest.fixture
def community_binary_config() -> dict[str, Any]:
    config = deepcopy(BINARY_PARAMETERS)
    config["primary"]["synchronicity"] = 1.0
    config["secondary"]["synchronicity"] = 1.0
    return config


@pytest.mark.parametrize(
    "mass_ratio",
    [
        pytest.param(-1.0, id="singular-negative-ratio"),
        pytest.param(0.0, id="zero-ratio"),
        pytest.param(-0.5, id="negative-ratio"),
    ],
)
def test_nonpositive_ratio_raises_physical_error_before_mass_arithmetic(
    community_binary_config: dict[str, Any], mass_ratio: float,
) -> None:
    community_binary_config["system"]["mass_ratio"] = mass_ratio

    # q = -1 must be rejected before the mass formula divides by 1 + q.
    with np.errstate(all="raise"), pytest.raises(PhysicalStateError, match="mass_ratio"):
        BinarySystem.from_json(community_binary_config, _verify=False)


@pytest.mark.parametrize(
    "mass_ratio",
    [
        pytest.param(np.nan, id="nan-ratio"),
        pytest.param(np.inf, id="positive-infinite-ratio"),
        pytest.param(-np.inf, id="negative-infinite-ratio"),
    ],
)
def test_nonfinite_ratio_raises_native_numerical_error(
    community_binary_config: dict[str, Any], mass_ratio: float,
) -> None:
    community_binary_config["system"]["mass_ratio"] = mass_ratio

    with np.errstate(all="raise"), pytest.raises(FloatingPointError, match="mass_ratio") as caught:
        BinarySystem.from_json(community_binary_config, _verify=False)

    assert type(caught.value) is FloatingPointError


def test_positive_ratio_conversion_preserves_masses_and_input_units(
    community_binary_config: dict[str, Any],
) -> None:
    quantity_config = deepcopy(community_binary_config)
    quantity_config["system"]["period"] = 3 * 86400 * u.s
    quantity_config["system"]["semi_major_axis"] = 10 * u.solRad

    numeric_result = transform_json_community_to_std(community_binary_config)
    quantity_result = transform_json_community_to_std(quantity_config)

    for component in ("primary", "secondary"):
        expected_mass = numeric_result[component]["mass"]
        assert expected_mass > 0
        assert_allclose(quantity_result[component]["mass"], expected_mass, rtol=2e-15)

    primary_mass = numeric_result["primary"]["mass"]
    secondary_mass = numeric_result["secondary"]["mass"]
    assert_allclose(secondary_mass / primary_mass, 0.7)
    assert quantity_result["system"]["period"] == 3 * 86400 * u.s

    binary = BinarySystem.from_json(quantity_result, _verify=False)

    assert binary.period == 3.0
    assert_allclose(binary.semi_major_axis, (10 * u.solRad).to_value(u.m), rtol=2e-15)
