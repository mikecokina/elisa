"""Utility functions for data file reading, conversion, and smoothing operations."""

from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from elisa import settings, utils
from elisa import units as u
from elisa.base.error import ElisaAnalyticsDataValidationError
from elisa.base.types import FLOAT, INT

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from elisa.types import Float


def convert_data(
    data: NDArray[Float],
    unit: u.Unit,
    to_unit: u.Unit,
) -> NDArray[Float]:
    """Convert data array to desired unit or leave dimensionless unchanged.

    Performs unit conversion on the input data array. If the input unit is
    dimensionless, the data is returned unchanged. Otherwise, the data is
    converted to the target unit.

    :param data: Input data array to be converted.
    :type data: NDArray[Float]
    :param unit: Unit of the input data array.
    :type unit: u.Unit
    :param to_unit: Target unit for conversion.
    :type to_unit: u.Unit
    :returns: Converted data array in target unit, or unchanged if dimensionless.
    :rtype: NDArray[Float]
    """
    if unit == u.dimensionless_unscaled:
        return data
    return (data * unit).to(to_unit).value


def convert_flux(
    data: NDArray[Float],
    unit: u.Unit,
    *,
    zero_point: Float | None = None,
) -> NDArray[Float]:
    """Convert flux data from magnitudes to normalized flux.

    If the input flux is in magnitudes (mag or mmag), it is converted to
    normalized flux. Dimensionless flux is returned unchanged.

    :param data: Input flux array.
    :type data: NDArray[Float]
    :param unit: Unit of the input flux (dimensionless, mag, or mmag).
    :type unit: u.Unit
    :param zero_point: Reference magnitude for magnitude-to-flux conversion.
        Required if unit is magnitude-based.
    :type zero_point: Float | None
    :returns: Converted flux array (dimensionless or normalized).
    :rtype: NDArray[Float]
    :raises ElisaAnalyticsDataValidationError: If unit is magnitude-based but zero_point is not provided.
    """
    if unit in [u.mag, u.mmag] and zero_point is None:
        error_msg = (
            "You supplied your data in magnitudes. Please also specify "
            "a zero point using keyword argument `reference_magnitude`."
        )
        raise ElisaAnalyticsDataValidationError(error_msg)

    if unit == u.mag:
        data = utils.magnitude_to_flux(data, zero_point)
    elif unit == u.mmag:
        data = utils.magnitude_to_flux(data / 1000.0, zero_point)

    return data


def convert_flux_error(
    error: NDArray | Sequence[Float],
    unit: u.Unit,
    *,
    zero_point: Float | None = None,
    flux: NDArray | Sequence[Float] | None = None,
) -> NDArray[np.float64]:
    """Propagate magnitude uncertainties using the Gaussian-flux approximation.

    Uses sigma_F = ln(10)/2.5 * F * sigma_m at each observation. This
    first-order approximation assumes small Gaussian magnitude uncertainties.

    :param error: Uncertainties in the same unit as the observations.
    :param unit: Dimensionless, mag or mmag uncertainty unit.
    :param zero_point: Reference magnitude, required for magnitude units.
    :param flux: Pointwise converted flux using the same reference magnitude.
        Required for magnitude units; a zero point alone cannot set flux errors.
    :returns: Absolute flux uncertainties, with mmag errors divided by 1000.
    :raises ElisaAnalyticsDataValidationError: If magnitude conversion metadata or shapes are invalid.
    """
    errors = np.asarray(error, dtype=float)
    if unit not in (u.mag, u.mmag):
        return errors
    fluxes = _magnitude_error_flux(errors, zero_point, flux)
    sigma_mag = errors / 1000.0 if unit == u.mmag else errors
    return (np.log(10.0) / 2.5) * fluxes * sigma_mag



def _magnitude_error_flux(
    errors: NDArray[np.float64], zero_point: Float | None, flux: NDArray | Sequence[Float] | None,
) -> NDArray[np.float64]:
    """Validate magnitude-error metadata and its pointwise flux correspondence.

    :param errors: Magnitude or mmag uncertainty vector.
    :param zero_point: Reference magnitude used to convert the observations.
    :param flux: Converted pointwise flux vector.
    :returns: Converted flux with the same shape as the errors.
    :raises ElisaAnalyticsDataValidationError: If the reference, flux or array correspondence is invalid.
    """
    if zero_point is None or not np.isfinite(zero_point):
        msg = "Magnitude uncertainty conversion requires a finite zero point"
        raise ElisaAnalyticsDataValidationError(msg)
    if flux is None:
        msg = "Magnitude uncertainty conversion requires pointwise converted flux"
        raise ElisaAnalyticsDataValidationError(msg)
    fluxes = np.asarray(flux, dtype=float)
    if errors.ndim != 1 or errors.shape != fluxes.shape:
        msg = "Magnitude errors and converted flux must be matching one-dimensional arrays"
        raise ElisaAnalyticsDataValidationError(msg)
    return fluxes


def convert_unit(unit: u.Unit, to_unit: u.Unit) -> u.Unit:
    """Replace unit with desired unit or leave dimensionless unchanged.

    Returns the input unit if it is dimensionless, otherwise returns the
    target unit for conversion.

    :param unit: Old unit to be potentially replaced.
    :type unit: u.Unit
    :param to_unit: Target unit for conversion.
    :type to_unit: u.Unit
    :returns: New unit (original if dimensionless, target otherwise).
    :rtype: u.Unit
    """
    if unit == u.dimensionless_unscaled:
        return unit
    return to_unit


def read_data_file(
    filename: str,
    data_columns: tuple[int, ...],
    *,
    delimiter: str = settings.DELIM_WHITESPACE,
) -> NDArray[Float]:
    """Load finite numeric observation columns, rejecting malformed data.

    Reads a data file and extracts specified columns. Lines starting with
    ``#`` are treated as comments and skipped. Malformed or nonfinite selected values raise an error.

    :param filename: Path to the data file to read.
    :type filename: str
    :param data_columns: Tuple of column indices to extract
        (time, observable, error).
    :type data_columns: tuple[int, ...]
    :param delimiter: Regex pattern defining column separator.
    :type delimiter: str
    :returns: Numeric array with N rows and len(data_columns) selected observation columns.
    :rtype: NDArray[Float]
    :raises ElisaAnalyticsDataValidationError: If the file content or selected columns are invalid.
    :raises OSError: If the file cannot be opened.
    """
    data = _read_observation_table(filename, delimiter)
    columns = _select_observation_columns(data, data_columns, filename)
    return _numeric_observation_values(columns, filename, data_columns)


def _read_observation_table(filename: str, delimiter: str) -> pd.DataFrame:
    """Parse an observation table while preserving file I/O errors."""
    try:
        return pd.read_csv(
            filename,
            header=None,
            comment="#",
            delimiter=delimiter,
            engine="python",
            on_bad_lines="error",
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        msg = f"Cannot parse observation file `{filename}`: {exc}"
        raise ElisaAnalyticsDataValidationError(msg) from exc


def _select_observation_columns(
    data: pd.DataFrame, data_columns: tuple[int, ...], filename: str,
) -> pd.DataFrame:
    """Select requested observation columns and report invalid selections."""
    try:
        return data[list(data_columns)]
    except (KeyError, TypeError, ValueError) as exc:
        msg = f"Cannot select observation columns {data_columns!r} in file `{filename}`: {exc}"
        raise ElisaAnalyticsDataValidationError(msg) from exc


def _numeric_observation_values(
    columns: pd.DataFrame, filename: str, data_columns: tuple[int, ...],
) -> NDArray[Float]:
    """Convert selected observations to finite float values."""
    try:
        numeric = columns.apply(lambda column: pd.to_numeric(column, errors="raise"))
        values = numeric.to_numpy(dtype=FLOAT)
    except (TypeError, ValueError, ArithmeticError) as exc:
        msg = f"Invalid numeric observations in file `{filename}`, columns {data_columns!r}: {exc}"
        raise ElisaAnalyticsDataValidationError(msg) from exc
    if not np.all(np.isfinite(values)):
        msg = f"Observation file `{filename}` contains nonfinite values in columns {data_columns!r}"
        raise ElisaAnalyticsDataValidationError(msg)
    return values


def downselect_data(data: NDArray[Float], downselect_ratio: float | None) -> NDArray[Float]:
    """Uniformly subsample a validated nonempty observation table.

    :param data: Validated observation rows.
    :param downselect_ratio: Finite real selection ratio in (0, 1], or None to keep all rows.
    :returns: Rows selected with the existing int(1/downselect_ratio) stride.
    :raises ElisaAnalyticsDataValidationError: If the ratio has an invalid type or range.
    """
    if downselect_ratio is None:
        return data
    if (
        isinstance(downselect_ratio, (bool, np.bool_))
        or not isinstance(downselect_ratio, Real)
        or not 0 < downselect_ratio <= 1
    ):
        msg = "`downselect_ratio` must be a finite real number in (0, 1]"
        raise ElisaAnalyticsDataValidationError(msg)
    # A stride at least as large as the table selects only the first row.
    # Handle it before taking the reciprocal, which could overflow for tiny ratios.
    if downselect_ratio <= 1.0 / len(data):
        return data[:1]
    return data[::int(1.0 / downselect_ratio)]


def central_moving_average(
    dt_set: Any,
    *,
    n_bins: int = 100,
    radius: int = 2,
    cyclic_boundaries: bool = True,
) -> None:
    """Smooth data using central moving average binning.

    Performs central moving average smoothing on a dataset by binning the
    phase curve into ``n_bins`` bins. For each bin, the average value is
    calculated from points within ``radius`` number of bins. The dataset is
    modified in-place.

    Use this function only on phased data (phase 0-1).

    :param dt_set: Dataset object to be smoothed (modified in-place).
        Must have ``x_data``, ``y_data``, and ``y_err`` attributes.
    :type dt_set: LCData | RVData
    :param n_bins: Number of bins into which the phase curve is divided.
    :type n_bins: int
    :param radius: Number of bins (in each direction) used for averaging.
    :type radius: int
    :param cyclic_boundaries: If True, last bin is neighbor of first bin
        (for periodic data). If False, boundaries are handled differently.
    :type cyclic_boundaries: bool
    :returns: None (modifies dt_set in-place).
    :rtype: None
    """
    bin_centres, bin_masks = _moving_average_bins(dt_set.x_data, n_bins, radius, cyclic_boundaries=cyclic_boundaries)
    bin_averages, bin_errors = _average_binned_observations(dt_set.y_data, dt_set.y_err, bin_masks)

    dt_set.x_data = bin_centres
    dt_set.y_data = bin_averages
    dt_set.y_err = bin_errors


def _moving_average_bins(
    coordinates: NDArray, n_bins: int, radius: int, *, cyclic_boundaries: bool,
) -> tuple[NDArray[np.float64], list[NDArray[np.bool_]]]:
    """Prepare the centers and observation masks for nonempty moving-average bins.

    :param coordinates: Observation phases used to define bin boundaries.
    :param n_bins: Number of bins across the observation interval.
    :param radius: Number of neighboring bins on either side.
    :param cyclic_boundaries: Whether neighboring bins wrap around the interval.
    :returns: Nonempty bin centers and corresponding observation masks.
    """
    bin_boundaries = np.linspace(
        coordinates.min(),
        coordinates.max(),
        num=n_bins + 1,
        endpoint=True,
    )
    bin_centres = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])
    bin_idxs = np.digitize(coordinates, bin_boundaries[1:], right=True)

    # Create bin masks based on boundaries
    if cyclic_boundaries:
        bins = [(np.arange(ii - radius, ii + radius + 1, 1, dtype=INT) % n_bins) for ii in range(n_bins)]
    else:
        bins = [np.arange(0, ii + radius + 1, 1, dtype=INT) for ii in range(radius)]
        bins += [np.arange(ii - radius, ii + radius + 1, 1, dtype=INT) for ii in range(radius, n_bins - radius)]
        bins += [np.arange(ii - radius, n_bins, 1, dtype=INT) for ii in range(n_bins - radius, n_bins)]

    # Calculate bin masks and identify non-empty bins
    bin_masks = [np.isin(bin_idxs, bins[ii]) for ii in range(n_bins)]
    bin_masks_count = np.sum(bin_masks, axis=1)

    non_empty_bins = bin_masks_count > 0
    selected_masks = [mask for mask, nonempty in zip(bin_masks, non_empty_bins, strict=True) if nonempty]
    return bin_centres[non_empty_bins], selected_masks


def _average_binned_observations(
    values: NDArray, errors: NDArray | None, bin_masks: list[NDArray[np.bool_]],
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Average observations with inverse-variance weights when errors are supplied.

    :param values: Observed values before smoothing.
    :param errors: Supplied measurement errors, or None to preserve their absence.
    :param bin_masks: Observation masks for each nonempty bin.
    :returns: Bin averages and mean supplied errors, preserving omitted errors.
    """
    if errors is None:
        return np.array([np.average(values[mask]) for mask in bin_masks]), None
    averages = []
    mean_errors = []
    for mask in bin_masks:
        bin_errors = errors[mask]
        weights = 1 / bin_errors**2
        averages.append(np.average(values[mask], weights=weights))
        mean_errors.append(np.mean(bin_errors))
    return np.asarray(averages), np.asarray(mean_errors)
