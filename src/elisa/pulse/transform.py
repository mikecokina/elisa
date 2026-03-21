"""Pulsation mode parameter transformation and validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from elisa import const as c
from elisa import units as u
from elisa.base.transform import (
    WHEN_FLOAT64,
    SystemProperties,
    deg_transform,
    quantity_transform,
)
from elisa.base.types import FLOAT, INT

if TYPE_CHECKING:
    from elisa.types import AstropyQuantity as Quantity


class PulsationModeProperties(SystemProperties):
    """Validator and transformer for pulsation mode parameters."""

    @staticmethod
    def l(value: float) -> int:  # noqa: E743
        """Validate and return the angular degree of the pulsation mode.

        The angular degree must be a non-negative integer representing the
        spherical harmonic degree of the pulsation mode.

        :param value: Angular degree value (integer or integer-convertible float).
        :type value: int | float
        :returns: Validated angular degree as integer.
        :rtype: int
        :raises TypeError: If value is not an integer or integer-convertible number.
        """
        if int(value) - value == 0:
            value = int(value)
        if not isinstance(value, (int, INT)):
            error_msg = "Angular degree `l` is not (numpy.)int"
            raise TypeError(error_msg)
        return value

    @staticmethod
    def m(value: float) -> int:
        """Validate and return the azimuthal order of the pulsation mode.

        The azimuthal order can be positive, negative, or zero, and must be
        an integer with absolute value not exceeding the degree `l`.

        :param value: Azimuthal order value (integer or integer-convertible float).
        :type value: int | float
        :returns: Validated azimuthal order as integer.
        :rtype: int
        :raises TypeError: If value is not an integer or integer-convertible number.
        """
        if int(value) - value == 0:
            value = int(value)
        if not isinstance(value, (int, INT)):
            error_msg = "Angular degree `m` is not (numpy.)int"
            raise TypeError(error_msg)
        return value

    @staticmethod
    def amplitude(value: float | Quantity | str) -> float:
        """Validate and return the radial velocity amplitude of the pulsation mode.

        Converts input to the default velocity unit and validates that the
        amplitude is non-negative.

        :param value: Radial velocity amplitude (float, int, or astropy Quantity).
        :type value: float | int | u.Quantity | str
        :returns: Validated amplitude in default internal velocity unit.
        :rtype: float
        :raises TypeError: If value is not a numeric type or Quantity.
        :raises ValueError: If amplitude is negative.
        """
        if isinstance(value, (u.Quantity, str)):
            retval = u.Quantity(value) if isinstance(value, str) else value
            retval = np.float64(retval.to(u.VELOCITY_UNIT))
        elif isinstance(value, (int, INT, float, FLOAT)):
            retval = FLOAT(value)
        else:
            error_msg = (
                "Value of `amplitude` is not (numpy.)int or (numpy.)float "
                "nor astropy.unit.quantity.Quantity instance."
            )
            raise TypeError(error_msg)
        if retval < 0:
            error_msg = "Temperature amplitude of mode has to be non-negative number."
            raise ValueError(error_msg)

        return retval

    @staticmethod
    def frequency(value: float | Quantity | str) -> float:
        """Validate and return the frequency of the pulsation mode.

        Converts input to the default frequency unit and validates that the
        frequency is non-negative.

        :param value: Frequency of pulsation (float, int, or astropy Quantity).
        :type value: float | int | Quantity | str
        :returns: Validated frequency in default internal frequency unit.
        :rtype: float
        :raises TypeError: If value is not a numeric type or Quantity.
        :raises ValueError: If frequency is negative.
        """
        if isinstance(value, (u.Quantity, str)):
            retval = u.Quantity(value) if isinstance(value, str) else value
            retval = np.float64(retval.to(u.FREQUENCY_UNIT))
        elif isinstance(value, (int, INT, float, FLOAT)):
            retval = (
                (FLOAT(value) * u.DefaultPulsationsInputUnits["frequency"])
                .to(u.FREQUENCY_UNIT)
                .value
            )
        else:
            error_msg = (
                "Value of `frequency` is not (numpy.)int or (numpy.)float "
                "nor astropy.unit.quantity.Quantity instance."
            )
            raise TypeError(error_msg)
        if retval < 0:
            error_msg = "Frequency of the mode has to be non-negative number."
            raise ValueError(error_msg)

        return retval

    @staticmethod
    def start_phase(value: float | Quantity | str) -> float:
        """Validate and return the phase shift of the pulsation mode.

        Transforms the start phase to the default angular unit.

        :param value: Start phase value (float, int, or astropy Quantity).
        :type value: float | int | u.Quantity | str
        :returns: Validated start phase in default internal angular unit.
        :rtype: float
        """
        return quantity_transform(
            value,
            u.DefaultPulsationsUnits.start_phase,
            WHEN_FLOAT64,
            u.DefaultPulsationsInputUnits.start_phase,
        )

    @staticmethod
    def mode_axis_theta(value: float | Quantity | str) -> float:
        """Validate and return the latitudinal coordinate of the pulsation mode axis.

        The latitudinal coordinate (colatitude) must be in the range [0, pi).
        If no unit is supplied, degrees are assumed.

        :param value: Latitudinal angle (float, int, or astropy Quantity in degrees).
        :type value: float | int | u.Quantity | str
        :returns: Validated latitudinal angle in radians, in range [0, pi).
        :rtype: float
        :raises ValueError: If value is outside the valid range [0, pi).
        """
        retval = deg_transform(
            value,
            u.DefaultPulsationsUnits.mode_axis_theta,
            WHEN_FLOAT64,
            u.DefaultPulsationsInputUnits.mode_axis_theta,
        )
        if not 0 <= retval < c.PI:
            error_msg = (
                f"Value of `mode_axis_theta`: {retval} is outside bounds (0, pi)."
            )
            raise ValueError(error_msg)

        return retval

    @staticmethod
    def mode_axis_phi(value: float | Quantity | str) -> float:
        """Validate and return the azimuthal coordinate of the pulsation mode axis.

        The azimuthal coordinate (longitude) can be any angle.
        If no unit is supplied, degrees are assumed.

        :param value: Azimuthal angle (float, int, or astropy Quantity in degrees).
        :type value: float | int | u.Quantity | str
        :returns: Validated azimuthal angle in radians.
        :rtype: float
        """
        return deg_transform(
            value,
            u.DefaultPulsationsUnits.mode_axis_phi,
            WHEN_FLOAT64,
            u.DefaultPulsationsInputUnits.mode_axis_phi,
        )

    @staticmethod
    def temperature_perturbation_phase_shift(
        value: float | Quantity | str,
    ) -> float:
        """Validate and return the phase shift between geometric and temperature perturbations.

        Defines the phase shift between surface geometry perturbations and
        temperature perturbations of the pulsation mode.

        :param value: Phase shift value (float, int, or astropy Quantity).
        :type value: float | int | u.Quantity | str
        :returns: Phase shift in default internal angular unit.
        :rtype: float
        """
        return deg_transform(
            value,
            u.DefaultPulsationsUnits.temperature_perturbation_phase_shift,
            WHEN_FLOAT64,
            u.DefaultPulsationsInputUnits.temperature_perturbation_phase_shift,
        )

    @staticmethod
    def horizontal_to_radial_amplitude_ratio(value: float) -> float:
        """Validate and return the ratio of horizontal to radial displacement amplitudes.

        Defines the amplitude ratio between the horizontal (non-radial) component
        and the radial component of the pulsation displacement. This ratio is
        dimensionless.

        :param value: Amplitude ratio (float or int).
        :type value: float | int
        :returns: Validated amplitude ratio.
        :rtype: float
        :raises TypeError: If value is not a numeric type.
        """
        if not isinstance(value, (int, INT, float, FLOAT)):
            error_msg = "Parameter is not is not (numpy.)int or (numpy.)float"
            raise TypeError(error_msg)
        return value

    @staticmethod
    def tidally_locked(value: bool) -> bool:  # noqa: FBT001
        """Validate and return whether the pulsation mode is tidally locked.

        If True, the mode axis is fixed with respect to the tidal axis and does
        not drift with the stellar surface. If False, the mode axis drifts with
        stellar rotation.

        :param value: Boolean flag indicating if mode is tidally locked.
        :type value: bool
        :returns: Validated boolean value.
        :rtype: bool
        :raises TypeError: If value is not a boolean.
        """
        if not isinstance(value, bool):
            error_msg = "Parameter `tidally_locked` can contain only boolean"
            raise TypeError(error_msg)
        return value


