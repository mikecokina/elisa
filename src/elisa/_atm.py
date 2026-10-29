"""Isolated, currently unused atmosphere validation helpers.

Kept only so they are not forgotten; nothing in production code or tests depends on this module.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from elisa import utils
from elisa.atm import (
    atm_file_prefix_to_quantity_list,
    get_list_of_all_atm_tables,
    parse_domain_quantities_from_atm_table_filename,
    validated_atlas,
)
from elisa.base.error import ElisaError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.types import Float


def build_atm_validation_hypertable(atlas: str) -> dict:
    """Build a validation table for atmosphere model parameter bounds.

    Creates a nested dictionary mapping temperature → gravity values for
    each temperature in the atlas. Used to validate that log_g values are
    valid for their corresponding temperatures.

    :param atlas: Atmosphere atlas name (e.g., 'ck04', 'k93').
    :type atlas: str
    :returns: Dictionary with temperatures as keys and dicts of gravity and
        metallicity allowable values.
    :rtype: dict
    """
    atlas = validated_atlas(atlas)
    all_files = get_list_of_all_atm_tables(atlas)
    filenames = (Path(f).name for f in all_files)
    quantities = sorted(
        [parse_domain_quantities_from_atm_table_filename(f) for f in filenames],
        key=lambda x: x[0],
    )
    temp_qroups = itertools.groupby(quantities, key=lambda x: x[0])
    return {
        str(int(temp_qroup[0])): {
            "gravity": sorted(set(np.array(list(temp_qroup[1])).T[1])),
            "metallicity": atm_file_prefix_to_quantity_list("metallicity", atlas),
        }
        for temp_qroup in temp_qroups
    }


def is_out_of_bound(
    in_arr: NDArray | list,
    values: Float | Iterable,
    tolerance: Float,
) -> list[bool]:
    """Check if values are outside the bounds of an array with tolerance.

    :param in_arr: Reference array defining valid range.
    :type in_arr: NDArray | list
    :param values: Value(s) to check.
    :type values: Float | Iterable
    :param tolerance: Tolerance for out-of-bounds check.
    :type tolerance: Float
    :returns: List of booleans; True where value is out of bounds.
    :rtype: list[bool]
    """
    values = [values] if not isinstance(values, Iterable) else values
    top, bottom = max(in_arr) + tolerance, min(in_arr) - tolerance
    return [not bottom <= val <= top for val in values]


# pay attention to those methods below
# in the future for different atm model might happen that function won't be valid anymore


def validate_temperature(
    temperature: NDArray,
    atlas: str,
    *,
    _raise: bool = True,
) -> bool:
    """Validate that temperatures are within atlas bounds.

    :param temperature: Temperature values to validate in K.
    :type temperature: NDArray
    :param atlas: Atmosphere atlas name (e.g., 'ck04', 'k93').
    :type atlas: str
    :param _raise: If True, raise TemperatureError on invalid values;
        otherwise return False.
    :type _raise: bool
    :returns: True if all temperatures are valid, False otherwise.
    :rtype: bool
    :raises TemperatureError: If temperatures are out of bounds and _raise=True.
    """
    atlas = validated_atlas(atlas)
    allowed = sorted(atm_file_prefix_to_quantity_list("temperature", atlas))
    invalid = any(bool(allowed[-1] < t or t < allowed[0]) for t in temperature)
    if invalid:
        if _raise:
            msg = "Any temperature in system atmosphere is out of bound; it is usually caused by invalid physical model"
            raise ElisaError(msg)
        return False
    return True


def validate_metallicity(
    metallicity: Float | Iterable,
    atlas: str,
    *,
    _raise: bool = True,
) -> bool:
    """Validate that metallicity is within atlas bounds.

    :param metallicity: Metallicity value(s) [Fe/H] to validate.
    :type metallicity: Float | Iterable
    :param atlas: Atmosphere atlas name (e.g., 'ck04', 'k93').
    :type atlas: str
    :param _raise: If True, raise MetallicityError on invalid values;
        otherwise return False.
    :type _raise: bool
    :returns: True if metallicity is valid, False otherwise.
    :rtype: bool
    :raises MetallicityError: If metallicity is out of bounds and _raise=True.
    """
    out_of_bound_tol = 0.1  # Tolerance for out-of-bounds values
    atlas = validated_atlas(atlas)
    allowed = sorted(atm_file_prefix_to_quantity_list("metallicity", atlas))
    out_of_bound = is_out_of_bound(allowed, metallicity, out_of_bound_tol)
    if any(out_of_bound):
        if _raise:
            min_allowed = min(allowed) - out_of_bound_tol
            max_allowed = max(allowed) + out_of_bound_tol
            msg = (
                f"Any metallicity in system atmosphere is out of bound, "
                f"allowed values are in range <{min_allowed}, {max_allowed}>; "
                f"it is usually caused by invalid physical model"
            )
            raise ElisaError(msg)
        return False
    return True


def validate_logg_temperature_constraint(
    temperature: NDArray,
    log_g: NDArray,
    atlas: str,
    *,
    _raise: bool = True,
) -> bool:
    """Validate that log_g values are consistent with temperature in atlas.

    Checks that each log_g value is valid for its corresponding temperature
    according to the atmosphere atlas bounds.

    :param temperature: Temperature values in K.
    :type temperature: NDArray
    :param log_g: Surface gravity values in log10(cm/s²).
    :type log_g: NDArray
    :param atlas: Atmosphere atlas name (e.g., 'ck04', 'k93').
    :type atlas: str
    :param _raise: If True, raise GravityError on invalid values;
        otherwise return False.
    :type _raise: bool
    :returns: True if all log_g values are valid, False otherwise.
    :rtype: bool
    :raises GravityError: If log_g is out of bounds and _raise=True.
    """
    validation_hypertable = build_atm_validation_hypertable(atlas)
    allowed = sorted(atm_file_prefix_to_quantity_list("temperature", atlas))

    invalid = [
        is_out_of_bound(
            validation_hypertable[str(int(utils.find_nearest_value(allowed, t)[0]))]["gravity"],
            [g],
            0.1,
        )[0]
        for t, g in zip(temperature, log_g, strict=True)
    ]
    if np.any(invalid):
        if _raise:
            msg = (
                "Any gravity (log_g) in system atmosphere is out of bound; "
                "it is usually caused by invalid physical model"
            )
            raise ElisaError(msg)
        return False
    return True


def validate_atm(
    temperature: NDArray,
    log_g: NDArray,
    metallicity: Float | Iterable,
    atlas: str,
    *,
    _raise: bool = True,
) -> bool:
    """Validate atmosphere parameters against atlas bounds.

    Runs validate_temperature, validate_metallicity, and
    validate_logg_temperature_constraint in sequence.

    :param temperature: Temperature values in K.
    :type temperature: NDArray
    :param log_g: Surface gravity in log10(cm/s²).
    :type log_g: NDArray
    :param metallicity: Metallicity value(s) [Fe/H].
    :type metallicity: Float | Iterable
    :param atlas: Atmosphere atlas name (e.g., 'ck04', 'k93').
    :type atlas: str
    :param _raise: If True, raise error on invalid values;
        otherwise return False.
    :type _raise: bool
    :returns: True if all atmosphere parameters are valid, False otherwise.
    :rtype: bool
    :raises TemperatureError, MetallicityError, GravityError: If
        respective parameters are out of bounds and _raise=True.
    """
    try:
        metallicity = [metallicity] * len(temperature) if not isinstance(metallicity, Iterable) else metallicity
        validate_temperature(temperature, atlas)
        validate_metallicity(metallicity, atlas)
        validate_logg_temperature_constraint(temperature, log_g, atlas)
    except (ElisaError, ValueError):
        if not _raise:
            return False
        raise
    return True
