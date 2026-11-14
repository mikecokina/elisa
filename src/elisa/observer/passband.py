from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import interpolate

from elisa import settings
from elisa.base.error import raise_elisa_passband_error, raise_elisa_unexpected_error
from elisa.observer.plot import PassbandPlot

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.types import Float


_MIN_PASSBAND_SAMPLES = 2
_NANOMETERS_TO_ANGSTROMS = 10.0


def _validated_passband_interval(interval: object) -> tuple[float, float]:
    """Validate finite, ordered wavelength bounds of a fixed passband."""
    try:
        bounds = np.asarray(interval, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise_elisa_passband_error(f"Invalid passband wavelength interval: {interval!r}", exc)

    has_two_bounds = bounds.shape == (2,)
    bounds_are_finite = np.all(np.isfinite(bounds))
    if not (has_two_bounds and bounds_are_finite):
        raise_elisa_passband_error(f"Passband wavelength interval must contain two finite bounds: {interval!r}")

    left, right = bounds
    left_is_nonnegative = left >= 0
    bounds_are_increasing = left < right
    if not (left_is_nonnegative and bounds_are_increasing):
        raise_elisa_passband_error(f"Passband wavelength bounds must satisfy 0 <= left < right: {interval!r}")

    return float(left), float(right)


def _validated_passband_table(df: pd.DataFrame, passband: str) -> tuple[NDArray, NDArray]:
    """Validate fixed response samples before constructing an interpolator."""
    if not isinstance(df, pd.DataFrame):
        raise_elisa_passband_error(f"Passband {passband!r} requires a DataFrame response table")

    try:
        raw_wavelengths = df[settings.PASSBAND_DATAFRAME_WAVE].to_numpy()
        raw_throughput = df[settings.PASSBAND_DATAFRAME_THROUGHPUT].to_numpy()
        wavelengths_are_complex = np.iscomplexobj(raw_wavelengths)
        throughput_is_complex = np.iscomplexobj(raw_throughput)
        if wavelengths_are_complex or throughput_is_complex:
            raise_elisa_passband_error(f"Passband {passband!r} samples must be real numbers")

        wavelengths = np.asarray(raw_wavelengths, dtype=float)
        throughput = np.asarray(raw_throughput, dtype=float)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise_elisa_passband_error(f"Invalid wavelength/throughput columns for passband {passband!r}: {exc}", exc)

    wavelengths_are_vector = wavelengths.ndim == 1
    throughput_is_vector = throughput.ndim == 1
    has_enough_samples = wavelengths.size >= _MIN_PASSBAND_SAMPLES
    sample_shapes_match = wavelengths.shape == throughput.shape
    valid_sample_shape = (
        wavelengths_are_vector
        and throughput_is_vector
        and has_enough_samples
        and sample_shapes_match
    )
    if not valid_sample_shape:
        msg = f"Passband {passband!r} requires at least two matching wavelength/throughput samples"
        raise_elisa_passband_error(msg)

    wavelengths_are_finite = np.all(np.isfinite(wavelengths))
    throughput_is_finite = np.all(np.isfinite(throughput))
    if not (wavelengths_are_finite and throughput_is_finite):
        raise_elisa_passband_error(f"Passband {passband!r} wavelength/throughput samples must be finite")

    first_wavelength_is_nonnegative = wavelengths[0] >= 0
    wavelengths_are_increasing = np.all(wavelengths[1:] > wavelengths[:-1])
    if not (first_wavelength_is_nonnegative and wavelengths_are_increasing):
        raise_elisa_passband_error(f"Passband {passband!r} wavelengths must be nonnegative and strictly increasing")

    has_nonzero_throughput = np.any(throughput != 0)
    if not has_nonzero_throughput:
        raise_elisa_passband_error(f"Passband {passband!r} throughput must not be identically zero")

    return wavelengths, throughput


def _read_passband_csv(file_path: Path) -> pd.DataFrame:
    """Read response samples, preserving source errors as passband errors."""
    try:
        # noinspection PyArgumentList
        table = pd.read_csv(file_path)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        msg = f"Cannot read passband table {file_path}: {exc}"
        raise_elisa_passband_error(msg, exc)
    return table


def _wavelengths_to_angstroms(wavelengths: NDArray, source: Path) -> NDArray:
    """Convert validated nanometer samples without allowing numerical overflow."""
    try:
        with np.errstate(over="raise", invalid="raise"):
            converted = wavelengths * _NANOMETERS_TO_ANGSTROMS
    except FloatingPointError as exc:
        msg = f"Invalid wavelength scale in passband table {source}"
        raise_elisa_passband_error(msg, exc)
    return converted


def init_bolometric_passband() -> tuple[PassbandContainer, Float, Float]:
    """Initialize bolometric passband and its wavelength boundaries.

    Creates a bolometric passband container with constant throughput of 1.0 across
    the full wavelength range. Returns the passband container along with its left
    and right bandwidth boundaries.

    :returns: Tuple containing the bolometric passband container, right bandwidth
        (maximum float), and left bandwidth (0.0).
    :rtype: tuple[PassbandContainer, float, float]
    """
    df = pd.DataFrame(
        {
            settings.PASSBAND_DATAFRAME_THROUGHPUT: [1.0, 1.0],
            settings.PASSBAND_DATAFRAME_WAVE: [50.0, 2000000.0],
        },
    )
    right_bandwidth = sys.float_info.max
    left_bandwidth = 0.0
    bol_passband = PassbandContainer(table=df, passband="bolometric")

    return bol_passband, right_bandwidth, left_bandwidth


def init_rv_passband(
    wavelength_interval: tuple | None = None,
    wave_column: str | None = None,
    throughput_column: str | None = None,
) -> tuple[PassbandContainer, Float, Float]:
    """Initialize the passband used to calculate radial velocities.

    Creates a passband container with constant throughput of 1.0 within
    the specified radial velocity wavelength interval. If an argument
    is not provided, its value is obtained from the current settings.

    :param wavelength_interval: Wavelength interval, or the current
        RV_LAMBDA_INTERVAL setting if None.
    :param wave_column: Wavelength column name, or the current
        PASSBAND_DATAFRAME_WAVE setting if None.
    :param throughput_column: Throughput column name, or the current
        PASSBAND_DATAFRAME_THROUGHPUT setting if None.
    :returns: Tuple containing the radial velocity passband container,
        right bandwidth (interval[1]), and left bandwidth (interval[0]).
    :rtype: tuple[PassbandContainer, Float, Float]
    :raises ElisaPassbandError: If the wavelength interval is invalid.
    """
    interval = _validated_passband_interval(
        settings.RV_LAMBDA_INTERVAL if wavelength_interval is None else wavelength_interval,
    )
    wave_column = settings.PASSBAND_DATAFRAME_WAVE if wave_column is None else wave_column
    throughput_column = (
        settings.PASSBAND_DATAFRAME_THROUGHPUT
        if throughput_column is None
        else throughput_column
    )

    df = pd.DataFrame(
        {
            throughput_column: [1.0, 1.0],
            wave_column: interval,
        },
    )

    right_bandwidth = interval[1]
    left_bandwidth = interval[0]
    psmbnd = PassbandContainer(table=df, passband="rv_band")

    return psmbnd, right_bandwidth, left_bandwidth


def bolometric(x: Float | list | NDArray) -> Float | list | NDArray:
    r"""Bolometric passband interpolation function.

    Returns constant throughput of 1.0 for all wavelengths. This function implements
    a passband that is transparent across all wavelengths, effectively acting as
    :math:`\lambda(x) = 1.0`.

    :param x: Wavelength value(s). Can be a scalar float or array-like (list/NDArray).
    :type x: float | list | NDArray
    :returns: Constant value(s) of 1.0 in the same shape as input. Returns float for
        scalar input, list for list input, or NDArray for array input.
    :rtype: float | list | NDArray
    """
    if isinstance(x, (float, int)):
        return 1.0
    if isinstance(x, list):
        return [1.0] * len(x)
    if isinstance(x, np.ndarray):
        return np.ones(len(x))
    # This interpolator is an internal synthesis callback. Its caller must supply
    # supported wavelength values; parameter drift cannot change their Python type.
    # A different type is a programming contract failure, not a bad fixed passband
    # or a candidate model that fitting should penalize.
    msg = f"Bolometric passband interpolation received unsupported wavelength type: {type(x)}"
    return raise_elisa_unexpected_error(msg)


class PassbandContainer:
    """Data container for storing passband response curves.

    Stores wavelength-dependent passband throughput data and provides interpolation
    utilities for wavelength-dependent calculations. Fully initialized PassbandContainer
    instances contain the following attributes:

    - left_bandwidth: Left wavelength boundary (minimum wavelength) of the passband.
    - right_bandwidth: Right wavelength boundary (maximum wavelength) of the passband.
    - table: pandas.DataFrame containing wavelength and throughput columns defining the
      passband response curve.
    - passband: Name identifier of the passband.
    - akima: Interpolation function for throughput values across the passband wavelength range.
    """

    def __init__(self, table: pd.DataFrame, passband: str) -> None:
        """Initialize a PassbandContainer instance.

        Sets up a PassbandContainer object to store the relationship between wavelengths
        and throughput values for a given passband. The table is processed to compute
        bandwidth boundaries and set up an interpolation function.

        :param table: DataFrame containing wavelength and throughput columns.
        :type table: pd.DataFrame
        :param passband: Name identifier of the passband.
        :type passband: str
        """
        if not isinstance(passband, str) or not passband:
            raise_elisa_passband_error(f"Invalid passband identifier: {passband!r}")
        self.left_bandwidth = np.nan
        self.right_bandwidth = np.nan
        self.akima = None
        self._table = pd.DataFrame({})
        self.wave_unit = "angstrom"
        self.passband = passband
        # in case this np.pi will stay here, there will be rendundant multiplication in intensity integration
        self.wave_to_si_mult = 1e-10
        self.plot = PassbandPlot(self)

        self.table = table

    @property
    def table(self) -> pd.DataFrame:
        """Get the passband table.

        Returns the pandas DataFrame containing wavelength and throughput data
        for this passband.

        :returns: DataFrame containing wavelength and throughput columns.
        :rtype: pd.DataFrame
        """
        return self._table

    @table.setter
    def table(self, df: pd.DataFrame) -> None:
        """Set the passband table and precompute interpolation function.

        Precomputes left and right bandwidth boundaries for the given table and
        sets up an interpolation function. For 'bolometric' and 'rv_band' passbands,
        uses a simple constant function returning 1.0. For other passbands, uses
        scipy's Akima1DInterpolator for smooth interpolation across wavelengths.

        :param df: DataFrame containing wavelength and throughput columns.
        :type df: pd.DataFrame
        :raises ElisaPassbandError: If response columns, samples or wavelength bounds are invalid.
        """
        wave, throughput = _validated_passband_table(df, self.passband)
        if self.passband.lower() in ("bolometric", "rv_band"):
            akima = bolometric
        else:
            try:
                akima = interpolate.Akima1DInterpolator(wave, throughput)
            except (ValueError, TypeError, FloatingPointError, OverflowError) as exc:
                raise_elisa_passband_error(f"Cannot interpolate passband {self.passband!r}: {exc}", exc)
        self._table = df
        self.akima = akima
        self.left_bandwidth = wave[0]
        self.right_bandwidth = wave[-1]

    @staticmethod
    def get_passband_df(passband: str) -> pd.DataFrame:
        """Read passband response curve data from CSV file.

        Loads passband throughput curve from a CSV file based on the passband name.
        The wavelength values are converted from nanometers to angstroms (multiplied by 10).

        :param passband: Name of the passband to load.
        :type passband: str
        :returns: DataFrame containing wavelength and throughput columns for the passband.
        :rtype: pd.DataFrame
        :raises ElisaPassbandError: If the name, source table or response samples are invalid.
        """
        passband_is_string = isinstance(passband, str)
        passband_is_supported = passband_is_string and passband in settings.PASSBANDS
        if not passband_is_supported:
            msg = f"Invalid or unsupported passband function: {passband}"
            raise_elisa_passband_error(msg)

        file_path = Path(settings.PASSBAND_TABLES) / f"{passband}.csv"
        table = _read_passband_csv(file_path)
        wavelengths, _ = _validated_passband_table(table, passband)
        table[settings.PASSBAND_DATAFRAME_WAVE] = _wavelengths_to_angstroms(wavelengths, file_path)
        return table

    def get_bandwidth(self) -> tuple[Float, Float]:
        """Get the passband wavelength boundaries.

        Returns the left (minimum) and right (maximum) wavelength boundaries
        of the passband response curve.

        :returns: Tuple containing (left_bandwidth, right_bandwidth).
        :rtype: tuple[float, float]
        """
        return self.left_bandwidth, self.right_bandwidth

    @classmethod
    def from_name(cls, passband: str) -> PassbandContainer:
        """Create a PassbandContainer from a passband name.

        Loads the passband response curve from the built-in database and creates
        a new PassbandContainer instance with the loaded data.

        :param passband: Name of the passband to load.
        :type passband: str
        :returns: Initialized PassbandContainer with loaded passband data.
        :rtype: PassbandContainer
        """
        df = cls.get_passband_df(passband)
        return cls(table=df, passband=passband)
