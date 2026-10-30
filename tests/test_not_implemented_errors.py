"""Unimplemented cold operations use an ELISA marker and remain builtin-compatible."""

from unittest.mock import Mock

import numpy as np
import pytest

from elisa import units as u
from elisa.analytics.dataset.base import RVData
from elisa.base.container import PositionContainer
from elisa.base.error import ElisaNotImplementedError, raise_elisa_not_implemented_error
from elisa.base.orbit.orbit import AbstractOrbit
from elisa.binary_system.orbit.container import OrbitalSupplements
from elisa.single_system.lc import compute_general_lightcurve


def test_not_implemented_marker_preserves_builtin_inheritance():
    assert ElisaNotImplementedError.__bases__ == (NotImplementedError,)
    with pytest.raises(NotImplementedError) as caught:
        raise_elisa_not_implemented_error("missing operation")
    assert type(caught.value) is ElisaNotImplementedError
    assert str(caught.value) == "missing operation"


def test_not_implemented_helper_retains_the_explicit_cause():
    cause = RuntimeError("backend has no implementation")
    with pytest.raises(ElisaNotImplementedError) as caught:
        raise_elisa_not_implemented_error("missing operation", cause)
    assert caught.value.__cause__ is cause


@pytest.mark.parametrize("name", sorted(PositionContainer.__abstractmethods__))
def test_position_container_stubs_keep_the_abstract_contract_and_report_the_method(name):
    method = getattr(PositionContainer, name)
    assert method.__isabstractmethod__
    # Invoke the stub explicitly: production containers override these methods.
    with pytest.raises(ElisaNotImplementedError, match=f"PositionContainer.{name}"):
        method(None)


def test_abstract_orbit_stub_uses_the_marker():
    assert AbstractOrbit.orbital_motion.__isabstractmethod__
    with pytest.raises(ElisaNotImplementedError, match="AbstractOrbit.orbital_motion"):
        AbstractOrbit.orbital_motion(None, 0.0)


def test_orbital_supplement_conversion_reports_its_missing_implementation():
    with pytest.raises(ElisaNotImplementedError, match="to_orbital_position"):
        OrbitalSupplements().to_orbital_position()


def test_generic_single_lightcurve_stub_preserves_its_message():
    with pytest.raises(ElisaNotImplementedError, match="use concrete generators"):
        compute_general_lightcurve(Mock())


def test_unimplemented_smoothing_method_preserves_the_dataset():
    data = RVData(
        x_data=np.array([0.0, 0.5]), y_data=np.array([1.0, 2.0]),
        x_unit=u.dimensionless_unscaled, y_unit=u.m / u.s,
    )
    with pytest.raises(ElisaNotImplementedError, match="Method unavailable is not implemented"):
        data.smooth(method="unavailable")
    np.testing.assert_array_equal(data.x_data, [0.0, 0.5])
    np.testing.assert_array_equal(data.y_data, [1.0, 2.0])
