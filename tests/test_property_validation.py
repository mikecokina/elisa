"""Physical domains, numerical failures and units at model property boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from astropy.units import UnitConversionError
from numpy.testing import assert_allclose, assert_array_equal

from elisa import units as u
from elisa.analytics.params.transform import BinaryInitialProperties, StarInitialProperties
from elisa.base.error import PhysicalStateError
from elisa.base.star import Star
from elisa.base.transform import BodyProperties, StarProperties, SystemProperties
from elisa.binary_system.curves.community import RadialVelocitySystem
from elisa.binary_system.orbit.transform import OrbitProperties
from elisa.binary_system.system import BinarySystem
from elisa.binary_system.transform import BinarySystemProperties, RadialVelocityObserverProperties

if TYPE_CHECKING:
    from collections.abc import Callable


# Units are the existing bare-number input units; None marks numeric-only methods.
# The values describe individual-property domains, not full-system admissibility.
PROPERTY_CASES = [
    (SystemProperties.period, u.d, [0.1, 1.0], [-1.0, 0.0]),
    (BinarySystemProperties.period, u.d, [1.0], [-1.0, 0.0]),
    (RadialVelocityObserverProperties.period, u.d, [1.0], [-1.0, 0.0]),
    (OrbitProperties.period, u.d, [1.0], [-1.0, 0.0]),
    (SystemProperties.inclination, u.deg, [0.0, 90.0, 180.0], [-1.0, 181.0]),
    (OrbitProperties.inclination, u.rad, [0.0, 1.0, np.pi], [-0.1, np.pi + 0.1]),
    (SystemProperties.gamma, u.m / u.s, [-1.0, 0.0, 1.0], []),
    (SystemProperties.additional_light, u.dimensionless_unscaled, [0.0, 0.5, 1.0], [-0.1, 1.1]),
    (SystemProperties.semi_major_axis, u.solRad, [1.0], [-1.0, 0.0]),
    (SystemProperties.distance, u.pc, [1.0], [-1.0, 0.0]),
    (BodyProperties.synchronicity, u.dimensionless_unscaled, [0.1, 1.0], [-1.0, 0.0]),
    (BodyProperties.albedo, u.dimensionless_unscaled, [0.0, 0.5, 1.0], [-0.1, 1.1]),
    (BodyProperties.discretization_factor, u.deg, [0.0, 45.0, 90.0], [-1.0, 91.0]),
    (BodyProperties.t_eff, u.K, [1.0, 6000.0], [-1.0, 0.0]),
    (StarProperties.t_eff, u.K, [6000.0], [-1.0, 0.0]),
    (StarProperties.equivalent_radius, u.solRad, [1.0], [-1.0, 0.0]),
    (StarProperties.mass, u.solMass, [1.0], [-1.0, 0.0]),
    (StarProperties.surface_potential, u.dimensionless_unscaled, [0.0, 1.0], [-1.0]),
    (StarProperties.metallicity, None, [-1.0, 0.0, 1.0], []),
    (StarProperties.polar_log_g, u.dex(u.cm / u.s**2), [-1.0, 0.0, 1.0], []),
    (StarProperties.gravity_darkening, u.dimensionless_unscaled, [0.0, 0.5, 1.0], [-0.1, 1.1]),
    (BinarySystemProperties.eccentricity, None, [0.0, np.nextafter(1.0, 0.0)], [-0.1, 1.0]),
    (RadialVelocityObserverProperties.eccentricity, None, [0.0, 0.5], [-0.1, 1.0]),
    (OrbitProperties.eccentricity, None, [0.0, np.nextafter(1.0, 0.0)], [-0.1, 1.0]),
    (BinarySystemProperties.argument_of_periastron, u.deg, [-720.0, 0.0, 360.0, 720.0], []),
    (OrbitProperties.argument_of_periastron, u.rad, [-1.0, 0.0, 1.0], []),
    (BinarySystemProperties.phase_shift, None, [-1.0, 0.0, 1.0], []),
    (BinarySystemProperties.primary_minimum_time, u.d, [-1.0, 0.0, 1.0], []),
    (BinarySystemProperties.t0, u.d, [-1.0, 0.0, 1.0], []),
    (RadialVelocityObserverProperties.mass_ratio, u.dimensionless_unscaled, [0.1, 1.0, 2.0], [-1.0, 0.0]),
    (RadialVelocityObserverProperties.asini, u.solRad, [0.0, 1.0], [-1.0]),
    (BinaryInitialProperties.semi_major_axis, u.solRad, [1.0], [-1.0, 0.0]),
    (BinaryInitialProperties.mass, u.solMass, [1.0], [-1.0, 0.0]),
    (StarInitialProperties.mass, u.solMass, [1.0], [-1.0, 0.0]),
]


@pytest.mark.parametrize(("transform", "input_unit", "valid", "invalid"), PROPERTY_CASES)
def test_physical_domains_preserve_units_and_boundaries(
    transform: Callable, input_unit: Any, valid: list[float], invalid: list[float],
) -> None:
    for value in valid:
        actual = transform(value)
        assert np.isfinite(actual)
        if input_unit is not None:
            assert_allclose(transform(value * input_unit), actual, rtol=2e-15, atol=0)
    for value in invalid:
        with pytest.raises(PhysicalStateError):
            transform(value)
        if input_unit is not None:
            with pytest.raises(PhysicalStateError):
                transform(value * input_unit)


@pytest.mark.parametrize(("transform", "input_unit", "valid", "invalid"), PROPERTY_CASES)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_properties_are_numerical_failures(
    transform: Callable, input_unit: Any, valid: list[float], invalid: list[float], value: float,
) -> None:
    for supplied in [value] if input_unit is None else [value, value * input_unit]:
        with np.errstate(divide="ignore", invalid="ignore"), pytest.raises(FloatingPointError) as caught:
            transform(supplied)
        assert type(caught.value) is FloatingPointError


def test_conversion_conventions_and_incompatible_units_are_unchanged() -> None:
    hours: Any = "24 h"
    invalid_type: Any = {"period": 1.0}
    assert SystemProperties.period(86400 * u.s) == 1.0
    assert OrbitProperties.period(hours) == 1.0
    assert RadialVelocityObserverProperties.period("1 d") == 1.0
    assert SystemProperties.gamma(-1 * u.km / u.s) == -1000.0
    assert_allclose(StarProperties.mass(1.0), u.solMass.to(u.kg))
    assert_allclose(SystemProperties.semi_major_axis(1.0), u.solRad.to(u.m))
    assert BinaryInitialProperties.mass(1.0) == 1.0
    assert BinaryInitialProperties.semi_major_axis(1.0) == 1.0
    assert_allclose(SystemProperties.inclination(90.0), np.pi / 2)
    assert OrbitProperties.inclination(1.0) == 1.0
    for transform in (SystemProperties.period, OrbitProperties.period):
        with pytest.raises(UnitConversionError):
            transform(1.0 * u.m)
        with pytest.raises(TypeError):
            transform(invalid_type)


@pytest.mark.parametrize("period", [-1.0, 0.0, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("quantity", [False, True])
def test_rv_and_binary_constructors_reject_invalid_period(period: float, quantity: bool) -> None:
    supplied = period * u.d if quantity else period
    error_type = PhysicalStateError if np.isfinite(period) else FloatingPointError
    with pytest.raises(error_type, match="period"):
        RadialVelocitySystem(
            eccentricity=0.0, argument_of_periastron=0.0, period=supplied,
            mass_ratio=0.7, asini=1.0, gamma=0.0,
        )
    primary, secondary = (
        Star(mass=mass, t_eff=6000.0, surface_potential=6.0, synchronicity=1.0)
        for mass in (1.0, 0.7)
    )
    with pytest.raises(error_type, match="period"):
        BinarySystem(
            primary=primary, secondary=secondary, inclination=90.0, period=supplied,
            eccentricity=0.0, argument_of_periastron=0.0,
        )


def test_valid_binary_and_zero_projected_rv_axis() -> None:
    primary, secondary = (
        Star(mass=mass, t_eff=6000.0, surface_potential=6.0, synchronicity=1.0)
        for mass in (1.0, 0.7)
    )
    binary = BinarySystem(
        primary=primary, secondary=secondary, inclination=0.0, period=86400 * u.s,
        eccentricity=0.0, argument_of_periastron=0.0,
    )
    assert binary.period == 1.0
    assert binary.morphology == "detached"
    rv = RadialVelocitySystem(
        eccentricity=0.0, argument_of_periastron=0.0, period=1.0,
        mass_ratio=0.7, asini=0.0, gamma=-100.0,
    )
    curves = rv.compute_rv(phases=np.array([0.0, 0.25]), position_method=rv.get_positions_method())
    for curve in curves.values():
        assert_array_equal(curve, [-100.0, -100.0])
