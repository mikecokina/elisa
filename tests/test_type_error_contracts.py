"""Type contract failures abort model evaluation instead of rejecting candidates."""

from types import SimpleNamespace

import numpy as np
import pytest

from elisa.analytics.binary_fit.failure_policy import SYNTHESIS_FAILURES
from elisa.base.error import ElisaUnexpectedError
from elisa.base.surface.coverage import surface_area_coverage
from elisa.base.surface.gravity import eval_args_for_magnitude_gradient
from elisa.base.surface.mesh import correct_component_mesh, symmetry_point_reduction
from elisa.base.system import System
from elisa.binary_system.orbit.orbit import Orbit
from elisa.binary_system.system import BinarySystem
from elisa.binary_system.utils import transform_json_community_to_std


@pytest.mark.parametrize("operation", [
    lambda: surface_area_coverage(1.5, np.array([], dtype=int), np.array([])),
    lambda: surface_area_coverage(-1, np.array([], dtype=int), np.array([])),
    lambda: eval_args_for_magnitude_gradient(object()),
    lambda: correct_component_mesh(object(), 0.0, np.array([])),
    lambda: symmetry_point_reduction(np.array([]), 1.5),
    lambda: symmetry_point_reduction(np.array([]), -1),
    lambda: System.object_params_validity_check({"primary": object()}, []),
])
def test_surface_and_component_type_contracts_are_fatal(operation):
    with pytest.raises(ElisaUnexpectedError) as caught:
        operation()
    assert not isinstance(caught.value, SYNTHESIS_FAILURES)


@pytest.mark.parametrize("section", ["primary", "secondary", "system"])
def test_community_conversion_requires_mapping_sections(section):
    data = {"primary": {}, "secondary": {}, "system": {}}
    data[section] = []
    with pytest.raises(ElisaUnexpectedError, match=section):
        transform_json_community_to_std(data)


def test_orbit_inclination_contract_preserves_original_type_error():
    orbit = Orbit(period=1.0, eccentricity=0.1, argument_of_periastron=0.0, inclination=1.0)
    orbit.inclination = object()
    with pytest.raises(ElisaUnexpectedError) as caught:
        orbit.get_conjuction()
    assert isinstance(caught.value.__cause__, TypeError)


def test_libration_potential_requires_numeric_solver_result(monkeypatch):
    monkeypatch.setattr(BinarySystem, "lagrangian_points_static", lambda *args: SimpleNamespace())
    with pytest.raises(ElisaUnexpectedError, match="unsupported radius type"):
        BinarySystem.libration_potentials_static(1.0, 1.0)


def test_binary_component_names_are_an_internal_contract():
    invalid_system = SimpleNamespace(components={"unknown": object()})
    with pytest.raises(ElisaUnexpectedError, match="unexpected component 'unknown'"):
        BinarySystem.setup_periastron_critical_potential(invalid_system)


def test_curve_router_requires_known_curve_type():
    from elisa.binary_system.curves.c_router import resolve_curve_method

    with pytest.raises(ElisaUnexpectedError, match="Invalid curve selector"):
        resolve_curve_method(SimpleNamespace(), "unknown")
