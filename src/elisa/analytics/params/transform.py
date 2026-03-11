"""Convert input parameters to default internal units."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import elisa.units as u
from elisa.analytics.params import conf
from elisa.base.transform import (
    WHEN_FLOAT64,
    SpotProperties,
    StarProperties,
    TransformProperties,
)
from elisa.binary_system.transform import (
    BinarySystemProperties,
    RadialVelocityObserverProperties,
)
from elisa.pulse.transform import PulsationModeProperties

if TYPE_CHECKING:
    from elisa.types import AstropyQuantity as Quantity

    from elisa.types import Float


def angular(value: Float | Quantity) -> Float:
    """Transform all angular units to ELISa's default internal angular unit.

    :param value: Numeric value to transform, can be a float or astropy Quantity.
    :type value: float | Quantity
    :returns: Transformed angular value in default internal angular unit.
    :rtype: float
    :raises TypeError: If input is not a numeric type or astropy Quantity.
    """
    if isinstance(value, u.Quantity):
        value = np.float64(value.to(conf.DEFAULT_FLOAT_ANGULAR_UNIT))
    elif isinstance(value, WHEN_FLOAT64):
        value = np.float64(value)
    else:
        error_msg = (
            "Input of variable is not (numpy.)int or (numpy.)float "
            "nor astropy.unit.quantity.Quantity instance."
        )
        raise TypeError(error_msg)
    return value


class BinaryInitialProperties(TransformProperties):
    """Handler for conversion of binary system parameters units."""

    @staticmethod
    def semi_major_axis(value: Float | Quantity) -> float:
        """Transform semi-major axis value to internal units.

        :param value: Semi-major axis value to transform.
        :type value: float | Quantity
        :returns: Transformed semi-major axis value in solar radii.
        :rtype: float
        :raises TypeError: If input is not a numeric type or astropy Quantity.
        :raises ValueError: If value is negative.
        """
        if isinstance(value, u.Quantity):
            value = np.float64(value.to(u.solRad))
        elif isinstance(value, WHEN_FLOAT64):
            value = np.float64(value)
        else:
            error_msg = (
                "Input of variable `semi_major_axis` is not (numpy.)int or (numpy.)float "
                "nor astropy.unit.quantity.Quantity instance."
            )
            raise TypeError(error_msg)
        if value < 0:
            error_msg = "Value of `semi_major_axis` cannot be negative."
            raise ValueError(error_msg)
        return value

    @staticmethod
    def mass(value: Float | Quantity) -> Float:
        """Transform mass value to internal units.

        :param value: Mass value to transform.
        :type value: float | Quantity
        :returns: Transformed mass value in default mass unit.
        :rtype: float
        :raises TypeError: If input is not a numeric type or astropy Quantity.
        :raises ValueError: If value is not positive.
        """
        if isinstance(value, u.Quantity):
            value = np.float64(value.to(conf.DEFAULT_FLOAT_MASS_UNIT))
        elif isinstance(value, WHEN_FLOAT64):
            value = np.float64(value)
        else:
            error_msg = (
                "User input is not (numpy.)int or (numpy.)float "
                "nor astropy.unit.quantity.Quantity instance."
            )
            raise TypeError(error_msg)
        if value <= 0:
            error_msg = "Invalid mass, use value > 0!"
            raise ValueError(error_msg)
        return value

    # ...existing code...
    eccentricity = BinarySystemProperties.eccentricity
    argument_of_periastron = angular
    inclination = angular
    gamma = BinarySystemProperties.gamma
    period = BinarySystemProperties.period
    mass_ratio = RadialVelocityObserverProperties.mass_ratio
    asini = RadialVelocityObserverProperties.asini
    additional_light = BinarySystemProperties.additional_light
    primary_minimum_time = BinarySystemProperties.primary_minimum_time


class StarInitialProperties(StarProperties):
    """Handler for conversion of star component parameters units."""

    mass = BinaryInitialProperties.mass


class SpotInitialProperties(SpotProperties):
    """Handler for conversion of spot parameters units."""

    latitude = angular
    longitude = angular
    angular_radius = angular


class NuisanceInitialProperties(TransformProperties):
    """Handler for unit conversion of nuisance fit parameters."""

    @staticmethod
    def ln_f(value: Float) -> Float:
        """Identity transformation for ln_f nuisance parameter.

        :param value: Nuisance parameter value.
        :type value: float
        :returns: Unchanged parameter value.
        :rtype: float
        """
        return value


class PulsationModeInitialProperties(PulsationModeProperties):
    """Handler for unit conversion of pulsation mode parameters."""

    mode_axis_theta = angular
    mode_axis_phi = angular
