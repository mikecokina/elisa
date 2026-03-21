"""Validation transformers for analytics task data properties."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from elisa import settings
from elisa.base.transform import TransformProperties

if TYPE_CHECKING:
    from elisa.analytics.dataset.base import LCData, RVData


class RVBinaryAnalyticsTask(TransformProperties):
    """Validator for radial velocity binary analytics task data.

    Evaluates whether the observational data in a radial velocity binary
    analytics task are supplied in valid form, ensuring proper data types
    and valid counterpart designations.
    """

    @staticmethod
    def data(value: Any) -> dict[str, RVData]:
        """Validate and return radial velocity dataset dictionary.

        Verifies that the input is a dictionary with valid binary counterpart
        keys and RVData value instances.

        :param value: Input data to validate
        :type value: Any
        :return: Validated radial velocity data dictionary
        :rtype: dict[str, RVData]
        :raises TypeError: If value is not a dict or contains non-RVData values
        :raises ValueError: If dict keys are not valid binary counterpart designations
        """
        if not isinstance(value, dict):
            error_msg: str = "`radial_velocities` are not of type `dict`"
            raise TypeError(error_msg)

        for key, val in value.items():
            if key not in settings.BINARY_COUNTERPARTS:
                error_msg = (
                    f"{key} is invalid designation for radial velocity dataset. "
                    f"Please choose from {list(settings.BINARY_COUNTERPARTS.keys())}"
                )
                raise ValueError(error_msg)

            # Import here to avoid circular imports
            from elisa.analytics.dataset.base import RVData  # noqa: PLC0415

            if not isinstance(val, RVData):
                error_msg = f"{val} is not an instance of RVData class."
                raise TypeError(error_msg)

        return value


class LCBinaryAnalyticsProperties(TransformProperties):
    """Validator for light curve binary analytics task data.

    Evaluates whether the observational data in a light curve binary
    analytics task are supplied in valid form, ensuring proper data types
    and valid passband designations.
    """

    @staticmethod
    def data(value: Any) -> dict[str, LCData]:
        """Validate and return light curve dataset dictionary.

        Verifies that the input is a dictionary with valid passband keys
        and LCData value instances.

        :param value: Input data to validate
        :type value: Any
        :return: Validated light curve data dictionary
        :rtype: dict[str, LCData]
        :raises TypeError: If value is not a dict or contains non-LCData values
        :raises ValueError: If dict keys are not valid passband designations
        """
        if not isinstance(value, dict):
            error_msg: str = "`light_curves` are not of type `dict`"
            raise TypeError(error_msg)

        for key, val in value.items():
            if key not in settings.PASSBANDS:
                available_passbands: str = "\n".join(
                    f"  - {pb}" for pb in settings.PASSBANDS
                )
                error_msg = (
                    f"{key} is invalid passband. Please choose from available "
                    f"passbands:\n{available_passbands}"
                )
                raise ValueError(error_msg)

            # Import here to avoid circular imports
            from elisa.analytics.dataset.base import LCData  # noqa: PLC0415

            if not isinstance(val, LCData):
                error_msg = f"{val} is not an instance of LCData class."
                raise TypeError(error_msg)

        return value

