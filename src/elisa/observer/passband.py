from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import interpolate

from elisa import settings
from elisa.observer.plot import PassbandPlot

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.types import Float


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


def init_rv_passband() -> tuple[PassbandContainer, Float, Float]:
    """Initialize passband used to calculate radial velocities.

    Creates a passband container with constant throughput of 1.0 within the
    radial velocity wavelength interval. Returns the passband container along
    with its left and right bandwidth boundaries.

    :returns: Tuple containing the radial velocity passband container, right
        bandwidth (RV_LAMBDA_INTERVAL[1]), and left bandwidth (RV_LAMBDA_INTERVAL[0]).
    :rtype: tuple[PassbandContainer, float, float]
    """
    df = pd.DataFrame(
        {
            settings.PASSBAND_DATAFRAME_THROUGHPUT: [1.0, 1.0],
            settings.PASSBAND_DATAFRAME_WAVE: settings.RV_LAMBDA_INTERVAL,
        },
    )
    right_bandwidth = settings.RV_LAMBDA_INTERVAL[1]
    left_bandwidth = settings.RV_LAMBDA_INTERVAL[0]
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
        return np.array([1.0] * len(x))
    msg = f"Unexpected type for x: {type(x)}"
    raise TypeError(msg)


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
        """
        self._table = df
        self.akima = (
            bolometric
            if self.passband.lower() in ["bolometric", "rv_band"]
            else interpolate.Akima1DInterpolator(
                df[settings.PASSBAND_DATAFRAME_WAVE],
                df[settings.PASSBAND_DATAFRAME_THROUGHPUT],
            )
        )
        self.left_bandwidth = df[settings.PASSBAND_DATAFRAME_WAVE].min()
        self.right_bandwidth = df[settings.PASSBAND_DATAFRAME_WAVE].max()

    @staticmethod
    def get_passband_df(passband: str) -> pd.DataFrame:
        """Read passband response curve data from CSV file.

        Loads passband throughput curve from a CSV file based on the passband name.
        The wavelength values are converted from nanometers to angstroms (multiplied by 10).

        :param passband: Name of the passband to load.
        :type passband: str
        :returns: DataFrame containing wavelength and throughput columns for the passband.
        :rtype: pd.DataFrame
        :raises ValueError: If the passband name is not valid or unsupported.
        """
        if passband not in settings.PASSBANDS:
            msg = f"Invalid or unsupported passband function: {passband}"
            raise ValueError(msg)
        file_path = Path(settings.PASSBAND_TABLES) / f"{passband}.csv"
        # noinspection PyArgumentList
        df = pd.read_csv(file_path)
        df[settings.PASSBAND_DATAFRAME_WAVE] *= 10.0
        return df

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
