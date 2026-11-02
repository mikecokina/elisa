"""Completeness checks respect current component values and reinitialization."""

from copy import deepcopy

import numpy as np
import pytest

from elisa import SingleSystem, Star
from elisa.base.error import ElisaInitialParametersError, ElisaParametersError, ElisaUnexpectedError
from elisa.base.system import System
from elisa.binary_system.system import BinarySystem
from tests.test_reinitialization import make_binary, make_single


def test_missing_binary_properties_are_initial_configuration_errors():
    star = Star(mass=1.0, t_eff=6000.0)
    with pytest.raises(ElisaInitialParametersError, match="surface_potential.*synchronicity"):
        System.object_params_validity_check({"primary": star}, BinarySystem.STAR_MANDATORY_KWARGS)


def test_missing_single_gravity_is_an_initial_configuration_error():
    with pytest.raises(ElisaInitialParametersError, match="polar_log_g"):
        SingleSystem(Star(mass=1.0, t_eff=6000.0), inclination=80.0, rotation_period=25.0)


def test_current_properties_complete_an_originally_partial_star_without_resetting_it():
    star = Star(mass=1.0, t_eff=6000.0)
    star.surface_potential = 6.0
    star.synchronicity = 1.0
    original_kwargs = deepcopy(star.kwargs)
    System.object_params_validity_check({"primary": star}, BinarySystem.STAR_MANDATORY_KWARGS)
    assert star.kwargs == original_kwargs
    star.init()
    assert star.surface_potential == 6.0
    assert star.synchronicity == 1.0


@pytest.mark.parametrize("invalid", [float("nan"), np.inf, -np.inf, -1.0, "bad", []])
def test_invalid_current_required_properties_are_candidate_errors(invalid):
    star = Star(mass=1.0, t_eff=6000.0)
    star.mass = invalid
    with pytest.raises(ElisaParametersError):
        System.object_params_validity_check({"star": star}, ("mass", "t_eff"))


def test_new_nan_is_not_mistaken_for_an_omitted_property():
    star = Star(mass=1.0, t_eff=6000.0)
    star.surface_potential = float("nan")
    with pytest.raises(ElisaParametersError):
        System.object_params_validity_check({"primary": star}, ("surface_potential",))


def test_missing_internal_attribute_is_a_contract_failure():
    star = Star(mass=1.0, t_eff=6000.0)
    del star.mass
    with pytest.raises(ElisaUnexpectedError, match="internal attribute 'mass'"):
        System.object_params_validity_check({"star": star}, ("mass",))


@pytest.mark.parametrize("owner", ["binary", "single"])
def test_invalid_required_values_fail_before_reinitialization_and_can_be_corrected(owner):
    model = make_binary() if owner == "binary" else make_single()
    star = model.secondary if owner == "binary" else model.star
    mass = star.mass
    original_axis = model.semi_major_axis if owner == "binary" else model.angular_velocity
    if owner == "binary":
        primary_kwargs = deepcopy(model.primary.kwargs)
    star.mass = np.inf
    with pytest.raises(ElisaParametersError):
        model.init()
    assert (model.semi_major_axis if owner == "binary" else model.angular_velocity) == original_axis
    if owner == "binary":
        assert model.primary.kwargs == primary_kwargs
    star.mass = mass
    model.init()
    model.init()
    assert star.mass == mass
