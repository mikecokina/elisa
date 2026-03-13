"""Pulsation mode data container and initialization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from elisa import const as c
from elisa import settings, units, utils
from elisa import umpy as up
from elisa.logger import getLogger
from elisa.pulse.transform import PulsationModeProperties

if TYPE_CHECKING:
    from elisa.units import _DefaultPulsationsInputUnits, _DefaultPulsationsUnits

logger = getLogger("pulse.mode")


class PulsationMode:
    """Container for pulsation mode parameters and computed properties.

    Stores all parameters defining a single pulsation mode including spherical
    harmonic degree, azimuthal order, amplitude, frequency, and optional
    orientation and displacement parameters. Automatically computes derived
    properties such as angular frequency and spherical harmonic normalization
    constant.

    Attributes
    ----------
    l : int
        Spherical harmonic degree (non-negative integer).
    m : int
        Azimuthal order (integer with ``|m| <= l``).
    amplitude : float
        Radial velocity amplitude in m/s.
    frequency : float
        Pulsation frequency in s^-1.
    start_phase : float
        Initial phase offset in radians. Defaults to 0.
    mode_axis_theta : float
        Latitude of mode axis in radians. Defaults to 0.
    mode_axis_phi : float
        Longitude of mode axis in radians. Defaults to 0.
    temperature_perturbation_phase_shift : float
        Phase shift between geometry and temperature perturbations.
    horizontal_to_radial_amplitude_ratio : float | None
        Ratio of horizontal to radial displacement amplitudes.
    temperature_amplitude_factor : float | None
        Temperature perturbation amplitude factor.
    tidally_locked : bool
        Whether mode axis is tidally locked. Defaults to False.
    radial_amplitude : float | None
        Computed radial displacement amplitude (set externally).
    horizontal_amplitude : float | None
        Computed horizontal displacement amplitude (set externally).
    angular_frequency : float
        Angular frequency (2π * frequency).
    renorm_const : float
        Spherical harmonics renormalization constant.
    points : np.ndarray | None
        Surface points in tilted spherical coordinates.
    point_harmonics : np.ndarray | None
        Spherical harmonics values at surface points.
    point_harmonics_derivatives : np.ndarray | None
        Derivatives of harmonics at surface points.
    complex_displacement : np.ndarray | None
        Complex displacement vector at surface points.
    tilt_phi : float | None
        Azimuthal tilt angle correction.
    tilt_theta : float | None
        Latitudinal tilt angle correction.

    """

    MANDATORY_KWARGS: tuple[str, ...] = ("l", "m", "amplitude", "frequency")

    OPTIONAL_KWARGS: tuple[str, ...] = (
        "start_phase",
        "mode_axis_theta",
        "mode_axis_phi",
        "temperature_perturbation_phase_shift",
        "horizontal_to_radial_amplitude_ratio",
        "temperature_amplitude_factor",
        "tidally_locked",
    )
    ALL_KWARGS: tuple[str, ...] = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a PulsationMode instance.

        Validates input parameters, transforms them to internal units, and
        initializes the pulsation mode with all required and optional properties.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments including mandatory (l, m, amplitude, frequency)
            and optional parameters for configuring the pulsation mode.

        Raises
        ------
        ValueError
            If mandatory parameters are missing or invalid values provided.
        TypeError
            If invalid keyword arguments are supplied.

        Examples
        --------
        Create a pulsation mode with mandatory parameters::

            mode = PulsationMode(
                l=2,
                m=1,
                amplitude=1.0 * u.m / u.s,
                frequency=1.0 / u.d
            )

        Create a mode with optional parameters::

            mode = PulsationMode(
                l=2,
                m=1,
                amplitude=1.0 * u.m / u.s,
                frequency=1.0 / u.d,
                start_phase=0.5,
                tidally_locked=True
            )

        """
        utils.invalid_kwarg_checker(
            kwargs=kwargs,
            kwarglist=PulsationMode.ALL_KWARGS,
            instance=self.__class__,
        )
        utils.check_missing_kwargs(
            PulsationMode.MANDATORY_KWARGS,
            kwargs,
            instance_of=PulsationMode,
        )
        kwargs = self.transform_input(**kwargs)

        # Get logger
        logger.info("initialising object %s", self.__class__.__name__)
        logger.debug(
            "setting property components of class instance %s",
            self.__class__.__name__,
        )

        # Initialize spherical harmonic parameters
        self.l: int = -1
        self.m: int = -1
        self.amplitude: float = up.NaN
        self.frequency: float = up.NaN

        # Initialize phase and orientation parameters
        self.start_phase: float = 0
        self.mode_axis_theta: float = 0
        self.mode_axis_phi: float = 0
        self.temperature_perturbation_phase_shift: float = settings.DEFAULT_TEMPERATURE_PERTURBATION_PHASE_SHIFT

        # Initialize amplitude and temperature parameters
        self.horizontal_to_radial_amplitude_ratio: float | None = None
        self.temperature_amplitude_factor: float | None = None
        self.tidally_locked: bool = False

        # Initialize computed amplitude attributes
        self.radial_amplitude: float | None = None
        self.horizontal_amplitude: float | None = None

        # Initialize surface-related auxiliary variables
        self.points: np.ndarray | None = None
        self.point_harmonics: np.ndarray | None = None
        self.point_harmonics_derivatives: np.ndarray | None = None
        self.complex_displacement: np.ndarray | None = None
        self.tilt_phi: float | None = None
        self.tilt_theta: float | None = None

        # Initialize properties from input kwargs
        self.init_properties(**kwargs)

        # Compute derived properties
        self.angular_frequency: float = c.FULL_ARC * self.frequency

        # Spherical harmonics renormalization constant to rms = 1
        self.renorm_const: float = 2 * c.PI**0.5

        # Validate mode parameters
        self.validate_mode()

    @property
    def default_input_units(self) -> _DefaultPulsationsInputUnits:
        """Return default units for initialization parameters without explicit units.

        Returns
        -------
        elisa.units.DefaultPulsationsInputUnits
            Default pulsation input units container.

        """
        return units.DefaultPulsationsInputUnits

    @property
    def default_internal_units(self) -> _DefaultPulsationsUnits:
        """Return set of internal units for pulsation parameters.

        Returns
        -------
        elisa.units.DefaultPulsationsUnits
            Default pulsation internal units' container.

        """
        return units.DefaultPulsationsUnits

    @staticmethod
    def transform_input(**kwargs) -> dict:
        """Transform and validate input keyword arguments.

        Converts input parameters to appropriate internal units and validates
        their values using PulsationModeProperties validators.

        Parameters
        ----------
        kwargs : dict
            Raw input keyword arguments.

        Returns
        -------
        dict
            Transformed and validated keyword arguments.

        """
        return PulsationModeProperties.transform_input(**kwargs)

    def validate_mode(self) -> None:
        """Validate that azimuthal order constraint ``|m| <= l`` is satisfied.

        Raises
        ------
        ValueError
            If ``|m| > l``, which is physically invalid for spherical harmonics.

        """
        if np.abs(self.m) > self.l:
            error_msg = (
                f"Absolute value of azimuthal order m: {self.m} cannot be higher than degree of the mode l: {self.l}."
            )
            raise ValueError(error_msg)

    def init_properties(self, **kwargs) -> None:
        """Initialize pulsation mode properties from input keyword arguments.

        Sets attributes on the instance for each key-value pair in kwargs
        after validation and transformation.

        Parameters
        ----------
        kwargs : dict
            Validated and transformed keyword arguments.

        """
        logger.debug("initialising properties of PulsationMode, values: %s", kwargs)
        for kwarg, value in kwargs.items():
            setattr(self, kwarg, value)
