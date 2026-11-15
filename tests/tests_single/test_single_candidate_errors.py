"""Single model initialization exposes radius failures as invalid model parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

from elisa import SingleSystem, Star
from elisa.analytics.binary_fit.failure_policy import SYNTHESIS_FAILURES
from elisa.base.error import ElisaMaxIterationError, ElisaNumericError, ElisaParametersError
from elisa.single_system import radius

if TYPE_CHECKING:
    import elisa.base.error
    import elisa.single_system.system


def make_single(rotation_period: float = 25.38) -> elisa.single_system.system.SingleSystem:
    star = Star(mass=1.0, t_eff=5772.0, polar_log_g=4.43775, gravity_darkening=0.32)
    return SingleSystem(star=star, inclination=90.0, rotation_period=rotation_period)


@pytest.mark.parametrize("solver_name", ["calculate_polar_radius", "calculate_equatorial_radius"])
@pytest.mark.parametrize("category", [ElisaNumericError, ElisaMaxIterationError, TypeError, ValueError, RuntimeError])
def test_constructor_classifies_radius_failure_as_parameters_and_recovers(
    solver_name: str,
    category: type[
        RuntimeError
        | TypeError
        | ValueError
        | elisa.base.error.ElisaMaxIterationError
        | elisa.base.error.ElisaNumericError
    ],
) -> None:
    baseline = make_single().star.equatorial_radius
    cause = ArithmeticError("original cause")
    failure = category("radius solver failure")
    failure.__cause__ = cause
    with (
        patch.object(radius, solver_name, autospec=True, side_effect=failure),
        pytest.raises(ElisaParametersError, match="supplied system parameters") as caught,
    ):
        make_single()
    assert caught.value.__cause__ is failure
    assert caught.value.__cause__.__cause__ is cause
    assert isinstance(caught.value, SYNTHESIS_FAILURES)
    assert make_single().star.equatorial_radius == baseline


@pytest.mark.parametrize("category", [ElisaNumericError, ElisaMaxIterationError, TypeError])
def test_reinitialization_classifies_radius_failure_as_parameters_and_recovers(
    category: type[TypeError | elisa.base.error.ElisaMaxIterationError | elisa.base.error.ElisaNumericError],
) -> None:
    system = make_single()
    expected = system.star.equatorial_radius
    failure = category("radius solver failure during init")
    with (
        patch.object(radius, "calculate_equatorial_radius", autospec=True, side_effect=failure),
        pytest.raises(ElisaParametersError, match="supplied system parameters") as caught,
    ):
        system.init()
    assert caught.value.__cause__ is failure
    assert isinstance(caught.value, SYNTHESIS_FAILURES)
    system.init()
    assert system.star.equatorial_radius == expected


def test_rotational_instability_is_a_model_parameter_rejection() -> None:
    make_single()
    with pytest.raises(ElisaParametersError, match="break-up") as caught:
        make_single(rotation_period=0.1)
    assert isinstance(caught.value, SYNTHESIS_FAILURES)
    assert np.isfinite(make_single().star.equatorial_radius)


def test_unverified_json_without_a_model_parameterization_uses_model_category() -> None:
    data = {
        "system": {"inclination": 90.0, "rotation_period": 25.38},
        "star": {"mass": 1.0, "t_eff": 5772.0, "gravity_darkening": 0.32},
    }
    with pytest.raises(ElisaParametersError, match="JSON is invalid") as caught:
        SingleSystem.from_json(data, _verify=False)
    assert isinstance(caught.value, SYNTHESIS_FAILURES)
    data["star"]["polar_log_g"] = 4.43775
    assert np.isfinite(SingleSystem.from_json(data, _verify=False).star.equatorial_radius)
