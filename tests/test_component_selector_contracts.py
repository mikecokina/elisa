"""Invalid fixed selectors abort evaluation at public and internal boundaries."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from elisa.analytics.binary_fit.failure_policy import SYNTHESIS_FAILURES
from elisa import settings
from elisa.base.error import ElisaOperationInputError, ElisaSettingsError, ElisaUnexpectedError
from elisa.binary_system.radius import calculate_radius
from elisa.binary_system.system import BinarySystem
from elisa.binary_system.utils import component_to_list
from elisa.conf.enums import BinaryComponent, RVMethod
from elisa.graphic import graphics
from elisa.observer.observer import Observer


@pytest.mark.parametrize("operation", [
    lambda: calculate_radius(1.0, 1.0, 5.0, "unknown", 1.0, 0.0, 0.0),
    lambda: BinarySystem.critical_potential_static("unknown", 1.0, 1.0, 1.0),
])
def test_unknown_physical_component_is_a_fatal_contract(operation):
    with pytest.raises(ElisaUnexpectedError, match="[Uu]nexpected component") as caught:
        operation()
    assert not isinstance(caught.value, SYNTHESIS_FAILURES)


@pytest.mark.parametrize("renderer", [graphics.binary_surface, graphics.binary_wireframe])
def test_renderer_rejects_unknown_component_selector(renderer):
    try:
        with pytest.raises(ElisaOperationInputError, match="plot component selector"):
            renderer(components_to_plot="unknown", elevation=0.0, inclination=90.0, azimuth=0.0, surface_colors=None)
    finally:
        plt.close("all")


@pytest.mark.parametrize("component", list(BinaryComponent))
def test_component_enum_preserves_string_selectors_and_physics(component):
    assert component_to_list(component) == [component.value]
    assert type(component_to_list(component)[0]) is str
    expected_radius = calculate_radius(1.0, 1.0, 5.0, component.value, 1.0, 0.0, 0.0)
    actual_radius = calculate_radius(1.0, 1.0, 5.0, component, 1.0, 0.0, 0.0)
    assert actual_radius == expected_radius
    expected_potential = BinarySystem.critical_potential_static(component.value, 1.0, 1.0, 1.0)
    actual_potential = BinarySystem.critical_potential_static(component, 1.0, 1.0, 1.0)
    np.testing.assert_equal(actual_potential, expected_potential)


@pytest.mark.parametrize("method", [RVMethod.KINEMATIC, "kinematic"])
def test_binary_rv_method_enum_preserves_string_dispatch(monkeypatch, method):
    sentinel = object()
    monkeypatch.setattr(
        "elisa.binary_system.system.rv.kinematic_radial_velocity",
        lambda system, **kwargs: sentinel,
    )

    assert BinarySystem.compute_rv(object(), method=method) is sentinel


def test_unknown_binary_rv_method_is_a_fatal_contract():
    with pytest.raises(ElisaUnexpectedError, match="Unexpected RV computation method") as caught:
        BinarySystem.compute_rv(object(), method="unknown")
    assert not isinstance(caught.value, SYNTHESIS_FAILURES)


def test_public_component_selection_is_an_operation_input_error():
    with pytest.raises(ElisaOperationInputError, match="Unsupported component selector"):
        component_to_list("unknown")


def test_observer_rejects_unknown_rv_method_before_synthesis():
    with pytest.raises(ElisaOperationInputError, match="Unsupported RV computation method"):
        Observer.rv(object(), method="unknown")


def test_settings_reject_unknown_rv_method_without_changing_configuration():
    original = settings.RV_METHOD
    with pytest.raises(ElisaSettingsError, match="RV_METHOD"):
        settings.configure(RV_METHOD="unknown")
    assert settings.RV_METHOD == original
