"""Utility functions for data file reading, conversion, and smoothing operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from packaging import version

from elisa import settings, utils
from elisa import units as u
from elisa.base.types import FLOAT, INT

if TYPE_CHECKING:
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
    :raises ValueError: If unit is magnitude-based but zero_point is not provided.
    """
    if unit in [u.mag, u.mmag] and zero_point is None:
        error_msg = (
            "You supplied your data in magnitudes. Please also specify "
            "a zero point using keyword argument `reference_magnitude`."
        )
        raise ValueError(error_msg)

    if unit == u.mag:
        data = utils.magnitude_to_flux(data, zero_point)
    elif unit == u.mmag:
        data = utils.magnitude_to_flux(data / 1000.0, zero_point)

    return data


def convert_flux_error(
    error: NDArray[Float],
    unit: u.Unit,
    *,
    zero_point: Float | None = None,
) -> NDArray[Float]:
    """Convert flux error array from magnitudes to normalized flux.

    If error data is in magnitudes (mag or mmag), it is converted to
    normalized flux error. Dimensionless error is returned unchanged.

    :param error: Input flux error array.
    :type error: NDArray[Float]
    :param unit: Unit of the input error (dimensionless, mag, or mmag).
    :type unit: u.Unit
    :param zero_point: Reference magnitude for magnitude-to-flux conversion.
        Required if unit is magnitude-based.
    :type zero_point: Float | None
    :returns: Converted error array (dimensionless or normalized flux error).
    :rtype: NDArray[Float]
    :raises ValueError: If unit is magnitude-based but zero_point is not provided.
    """
    if unit in [u.mag, u.mmag] and zero_point is None:
        error_msg = (
            "You supplied your data in magnitudes. Please also specify "
            "a zero point using keyword argument `reference_magnitude`."
        )
        raise ValueError(error_msg)

    if unit == u.mag:
        error = utils.magnitude_error_to_flux_error(error)
    elif unit == u.mmag:
        error = utils.magnitude_error_to_flux_error(error / 1000.0)

    return error


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
    """Load observation data from file, handling missing values gracefully.

    Reads a data file and extracts specified columns. Lines starting with
    ``#`` are treated as comments and skipped. Bad lines are skipped during
    reading, and any remaining NaN values are removed.

    :param filename: Path to the data file to read.
    :type filename: str
    :param data_columns: Tuple of column indices to extract
        (time, observable, error).
    :type data_columns: tuple[int, ...]
    :param delimiter: Regex pattern defining column separator.
    :type delimiter: str
    :returns: Loaded data as (N x 3) array with columns for time, observable,
        and error.
    :rtype: NDArray[Float]
    """
    reader_kwargs: dict[str, bool | str] = {"on_bad_lines": "skip"}
    if version.parse(pd.__version__) < version.parse("1.3.0"):
        reader_kwargs = {"error_bad_lines": False}

    data = pd.read_csv(
        filename,
        header=None,
        comment="#",
        delimiter=delimiter,
        engine="python",
        **reader_kwargs,
    )[list(data_columns)]
    data = data.apply(lambda s: pd.to_numeric(s, errors="coerce")).dropna()
    return data.to_numpy(dtype=FLOAT)


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
    bin_boundaries = np.linspace(
        dt_set.x_data.min(),
        dt_set.x_data.max(),
        num=n_bins + 1,
        endpoint=True,
    )
    bin_centres = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])
    bin_idxs = np.digitize(dt_set.x_data, bin_boundaries[1:], right=True)

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
    iterator = np.arange(n_bins)[non_empty_bins]

    # Calculate binned averages and errors
    if dt_set.y_err is not None:
        bin_averages = np.array(
            [
                np.average(
                    dt_set.y_data[bin_masks[ii]],
                    weights=1 / dt_set.y_err[bin_masks[ii]] ** 2,
                )
                for ii in iterator
            ],
        )
        bin_errors = np.array([np.mean(dt_set.y_err[bin_masks[ii]]) for ii in iterator])
    else:
        bin_averages = np.array([np.average(dt_set.y_data[bin_masks[ii]]) for ii in iterator])
        bin_errors = np.array([np.std(dt_set.y_data[bin_masks[ii]]) for ii in iterator])

    # Update dataset with binned results
    dt_set.x_data = bin_centres[non_empty_bins]
    dt_set.y_data = bin_averages
    dt_set.y_err = bin_errors
