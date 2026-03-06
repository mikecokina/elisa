from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

from elisa import settings
from elisa import umpy as up
from elisa import units as u
from elisa.binary_system.curves.community import RadialVelocitySystem
from elisa.logger import getLogger
from elisa.observer import utils as outils
from elisa.observer.passband import PassbandContainer, init_bolometric_passband
from elisa.observer.plot import Plot
from elisa.photometric_standards.standards_handlers import load_standard
from elisa.utils import is_empty, jd_to_phase

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from elisa import BinarySystem, SingleSystem
    from elisa.types import Float
    from elisa.units import Unit

logger = getLogger("observer.observer")


class Observables:
    """Wrapper class for accessing observable calculation methods from Observer.

    Provides convenient access to light curve and radial velocity calculations
    through wrapper methods that delegate to the Observer instance.
    """

    def __init__(self, observer: Observer) -> None:
        """Initialize the Observables wrapper.

        :param observer: Parent Observer instance.
        :type observer: Observer
        """
        self.observer = observer

    def lc(
            self,
            from_phase: Float | None = None,
            to_phase: Float | None = None,
            phase_step: Float | None = None,
            phases: ArrayLike | None = None,
            from_time: Float | None = None,
            to_time: Float | None = None,
            time_step: Float | None = None,
            times: ArrayLike | None = None,
            flux_unit: Unit | None = None,
            *,
            normalize: bool = False,
    ) -> tuple[NDArray, dict[str, NDArray]]:
        """Calculate light curves by delegating to the Observer instance.

        See :meth:`Observer.lc` for parameter documentation.

        :returns: Tuple containing phases and flux curves.
        :rtype: tuple[NDArray, dict[str, NDArray]]
        """
        return self.observer.lc(
            from_phase,
            to_phase,
            phase_step,
            phases,
            normalize=normalize,
            from_time=from_time,
            to_time=to_time,
            time_step=time_step,
            times=times,
            flux_unit=flux_unit,
        )

    def rv(
            self,
            from_phase: Float | None = None,
            to_phase: Float | None = None,
            phase_step: Float | None = None,
            phases: ArrayLike | None = None,
            *,
            normalize: bool = False,
            method: str | None = None,
            from_time: Float | None = None,
            to_time: Float | None = None,
            time_step: Float | None = None,
            times: ArrayLike | None = None,
    ) -> tuple[NDArray, dict[str, NDArray]]:
        """Calculate radial velocity curves by delegating to the Observer instance.

        See :meth:`Observer.rv` for parameter documentation.

        :returns: Tuple containing phases and radial velocity curves.
        :rtype: tuple[NDArray, dict[str, NDArray]]
        """
        return self.observer.rv(
            from_phase,
            to_phase,
            phase_step,
            phases,
            normalize=normalize,
            method=method,
            from_time=from_time,
            to_time=to_time,
            time_step=time_step,
            times=times,
        )


class Observer:
    """Observer class for synthetic photometric and spectroscopic observations.

    Responsible for calculating synthetic observations based on an input system model.
    Supports light curves and radial velocity curves calculation in any provided passband.

    List of available passbands is accessible via:

    .. code-block:: python

        from elisa import settings
        settings.PASSBANDS

    This class supports calculation of light curves (lc) and radial velocity curves (rv)
    using :meth:`lc` and :meth:`rv` methods.

    After initialization, the following attributes are available:

    - left_bandwidth: Minimum wavelength encompassing all desired passbands.
    - right_bandwidth: Maximum wavelength encompassing all desired passbands.
    - passband: Dictionary mapping passband names to PassbandContainer instances.
    - phases: Calculated observation phases.
    - fluxes: Calculated flux values for each passband.
    - radial_velocities: Calculated radial velocity curves for system components.
    """

    def __init__(
            self,
            passband: str | list[str] | None = None,
            system: BinarySystem | SingleSystem | None = None,
    ) -> None:
        r"""Initialize an Observer instance.

        :param passband: Passband name(s) for light curve observations. Can be a single
            passband name or list of passband names. For valid names see
            :data:`elisa.settings.PASSBANDS`. Defaults to None (empty list).
        :type passband: str | list[str] | None
        :param system: System instance (BinarySystem or SingleSystem) to observe.
        :type system: object | None
        """
        if passband is None:
            passband = []
        logger.info("initialising Observer instance")
        # specifying what kind of system is observed
        self._system: BinarySystem | SingleSystem = system
        self._system_cls = type(self._system)

        self.left_bandwidth = sys.float_info.max
        self.right_bandwidth = 0.0
        self.passband: dict[str, PassbandContainer] = {}
        self.init_passband(passband)

        self._observables = ["lc", "rv"]
        self.phases: NDArray | None = None
        self.times: NDArray | None = None
        self.fluxes: dict[str, NDArray] | None = None
        self.magnitudes: dict[str, NDArray] | None = None
        self._flux_unit = u.W / u.m ** 2
        self.radial_velocities: dict[str, NDArray] = {}
        self.rv_unit: object | None = None
        self.zero_points: dict[str, object] = {"system": None}

        self.plot = Plot(self)
        self.observe = Observables(self)

    @property
    def system_cls(self) -> type:
        """Get the system class type.

        :returns: Class type of the system.
        :rtype: type
        """
        return self._system_cls

    @system_cls.setter
    def system_cls(self, value: type) -> None:
        """Set the system class type.

        :param value: Class type to set.
        :type value: type
        """
        self._system_cls = value

    @property
    def flux_unit(self) -> object:
        """Get the flux unit.

        :returns: Current flux unit.
        :rtype: object
        """
        return self._flux_unit

    @flux_unit.setter
    def flux_unit(self, value: object) -> None:
        """Set the flux unit.

        :param value: Flux unit to set (astropy Unit or compatible).
        :type value: object
        """
        self._flux_unit = u.Unit(value)

    def init_passband(self, passband: str | list[str]) -> None:
        """Initialize passbands for the Observer instance.

        Fills the `self.passband` dictionary with PassbandContainer instances for each
        requested passband. Automatically updates global left and right bandwidth boundaries
        to encompass all requested passbands. For multiple passbands defined on different
        wavelength intervals, the global limits represent the full encompassing range.

        Example: Passbands with intervals [350, 650] and [450, 750] will result in
        global boundaries [350, 750].

        :param passband: Single passband name or list of passband names.
        :type passband: str | list[str]
        """
        passband = [passband] if isinstance(passband, str) else passband
        for band in passband:
            if band == "bolometric":
                psbnd, right_bandwidth, left_bandwidth = init_bolometric_passband()
            else:
                psbnd = PassbandContainer.from_name(passband=band)
                left_bandwidth, right_bandwidth = psbnd.get_bandwidth()

            self.setup_bandwidth(left_bandwidth=left_bandwidth, right_bandwidth=right_bandwidth)
            self.passband[band] = psbnd

    def setup_bandwidth(self, left_bandwidth: Float, right_bandwidth: Float) -> None:
        """Update global wavelength boundaries based on new passband limits.

        Compares supplied left and right bandwidth values with currently set boundaries.
        If the supplied values extend beyond the current boundaries, they replace the
        current global wavelength limits.

        :param left_bandwidth: Left (minimum) wavelength boundary of the passband.
        :type left_bandwidth: float
        :param right_bandwidth: Right (maximum) wavelength boundary of the passband.
        :type right_bandwidth: float
        """
        self.left_bandwidth = min(self.left_bandwidth, left_bandwidth)
        self.right_bandwidth = max(self.right_bandwidth, right_bandwidth)

    def lc(
            self,
            from_phase: Float | None = None,
            to_phase: Float | None = None,
            phase_step: Float | None = None,
            phases: ArrayLike | None = None,
            *,
            normalize: bool = False,
            from_time: Float | None = None,
            to_time: Float | None = None,
            time_step: Float | None = None,
            times: ArrayLike | None = None,
            flux_unit: Unit | None = None,
    ) -> tuple[NDArray, dict[str, NDArray]]:
        """Calculate synthetic light curves.

        Computes a light curve based on input parameters and the System supplied during
        initialization of the Observer instance. Returns light curves for each passband
        defined in the Observer instance. Times of observations can be supplied in either
        time or phase domain.

        If normalize is True, fluxes are normalized to maximum=1 (dimensionless). Otherwise,
        flux values are adjusted for distance and converted to the specified flux_unit if
        provided.

        :param from_phase: Starting phase of the observation. Defaults to None.
        :type from_phase: float | None
        :param to_phase: End phase of the observation. Defaults to None.
        :type to_phase: float | None
        :param phase_step: Phase increment of the observations. Defaults to None.
        :type phase_step: float | None
        :param phases: Array of phases at which to perform observations. If provided,
            from_phase, to_phase, and phase_step are ignored. Defaults to None.
        :type phases: ArrayLike | None
        :param normalize: If True, output is normalized to maximum=1. Defaults to False.
        :type normalize: bool
        :param from_time: Starting time of the observation. Defaults to None.
        :type from_time: float | None
        :param to_time: End time of the observation. Defaults to None.
        :type to_time: float | None
        :param time_step: Time increment of the observations. Defaults to None.
        :type time_step: float | None
        :param times: Array of times at which to perform observations. If provided,
            from_time, to_time, and time_step are ignored. Defaults to None.
        :type times: ArrayLike | None
        :param flux_unit: Unit of flux (astropy Unit). If None, uses observer's flux_unit.
            Defaults to None.
        :type flux_unit: object | None
        :returns: Tuple containing phases and flux curves for each passband.
        :rtype: tuple[NDArray, dict[str, NDArray]]
        :raises ValueError: If flux_unit is specified with normalize=True (conflicting parameters).
        """
        if normalize:
            if flux_unit in [None, u.dimensionless_unscaled]:
                self.flux_unit = u.dimensionless_unscaled
            else:
                msg = (
                    "You can either produce normalized light curve or specify `flux_unit` other "
                    "than dimensionless unscaled. Change input parameters."
                )
                raise ValueError(msg)
        else:
            self.flux_unit = u.Unit(flux_unit) if flux_unit is not None else self.flux_unit

        phases = self.manage_time_series(
            from_phase,
            to_phase,
            phase_step,
            phases,
            from_time,
            to_time,
            time_step,
            times,
        )

        # reduce phases to only unique ones from interval (0, 1) in general case without pulsations
        base_phases, base_phases_to_origin = self.phase_interval_reduce(phases)

        logger.info("observation is running")
        # calculates lines of sight for corresponding phases
        position_method = self._system.get_positions_method()

        lc_kwargs = {
            "passband": self.passband,
            "left_bandwidth": self.left_bandwidth,
            "right_bandwidth": self.right_bandwidth,
            "phases": base_phases,
            "position_method": position_method,
        }

        curves = self._system.compute_lightcurve(**lc_kwargs)

        # remap unique phases back to original phase interval
        for items in curves:
            curves[items] = np.array(curves[items])[base_phases_to_origin]

            # adding additional light
            correction = (
                    np.mean(curves[items])
                    * self._system.additional_light
                    / (1.0 - self._system.additional_light)
            )
            curves[items] += correction

        self.phases = phases + self._system.phase_shift
        if normalize or self.flux_unit == u.dimensionless_unscaled:
            self.fluxes, _ = outils.normalize_light_curve(
                y_data=curves,
                kind="maximum",
                top_fraction_to_average=0.0,
            )
        else:
            curves = outils.adjust_flux_for_distance(curves, self._system.distance)
            if self.flux_unit in [None, u.W / u.m ** 2]:
                self.fluxes = curves
            elif self.flux_unit == u.mag:
                if self.zero_points["system"] != settings.MAGNITUDE_SYSTEM.lower():
                    self.zero_points = load_standard(settings.MAGNITUDE_SYSTEM)
                self.fluxes = outils.convert_to_magnitudes(curves, self.zero_points)
                self.magnitudes = self.fluxes
            else:
                msg = f"Unknown value for `Observer.flux_unit`: {self.flux_unit}"
                raise ValueError(msg)

        logger.info("observation finished")
        return self.phases, self.fluxes

    def rv(
            self,
            from_phase: Float | None = None,
            to_phase: Float | None = None,
            phase_step: Float | None = None,
            phases: ArrayLike | None = None,
            *,
            normalize: bool = False,
            method: str | None = None,
            from_time: Float | None = None,
            to_time: Float | None = None,
            time_step: Float | None = None,
            times: ArrayLike | None = None,
    ) -> tuple[NDArray, dict[str, NDArray]]:
        """Calculate synthetic radial velocity curves.

        Computes a radial velocity curve based on input parameters and the System supplied
        during initialization of the Observer instance. Times of observations can be supplied
        in either time or phase domain.

        The method can compute radial velocities using different algorithms specified by
        the `method` parameter.

        :param from_phase: Starting phase of the observation. Defaults to None.
        :type from_phase: float | None
        :param to_phase: End phase of the observation. Defaults to None.
        :type to_phase: float | None
        :param phase_step: Phase increment of the observations. Defaults to None.
        :type phase_step: float | None
        :param phases: Array of phases at which to perform observations. If provided,
            from_phase, to_phase, and phase_step are ignored. Defaults to None.
        :type phases: ArrayLike | None
        :param normalize: If True, output is normalized to maximum=1 (dimensionless).
            Defaults to False.
        :type normalize: bool
        :param method: Method for calculation of radial velocities. Can be 'kinematic' or
            'radiometric'. If None, uses the default from settings. Defaults to None.
        :type method: str | None
        :param from_time: Starting time of the observation. Defaults to None.
        :type from_time: float | None
        :param to_time: End time of the observation. Defaults to None.
        :type to_time: float | None
        :param time_step: Time increment of the observations. Defaults to None.
        :type time_step: float | None
        :param times: Array of times at which to perform observations. If provided,
            from_time, to_time, and time_step are ignored. Defaults to None.
        :type times: ArrayLike | None
        :returns: Tuple containing phases and radial velocity curves for each system component.
        :rtype: tuple[NDArray, dict[str, NDArray]]
        """
        method = settings.RV_METHOD if method is None else method

        phases = self.manage_time_series(
            from_phase,
            to_phase,
            phase_step,
            phases,
            from_time,
            to_time,
            time_step,
            times,
        )

        # reduce phases to only unique ones from interval (0, 1) in general case without pulsations
        base_phases, base_phases_to_origin = self.phase_interval_reduce(phases)

        self.radial_velocities = self._system.compute_rv(
            phases=base_phases,
            position_method=self._system.get_positions_method(),
            method=method,
        )

        # remap unique phases back to original phase interval
        for items in self.radial_velocities:
            self.radial_velocities[items] = np.array(self.radial_velocities[items])[
                base_phases_to_origin
            ]

        self.phases = phases + self._system.phase_shift
        self.rv_unit = u.m / u.s
        if normalize:
            self.rv_unit = u.dimensionless_unscaled
            _max = np.max([np.max(item) for item in self.radial_velocities.values()])
            self.radial_velocities = {
                key: value / _max for key, value in self.radial_velocities.items()
            }

        return self.phases, self.radial_velocities

    def phase_interval_reduce(
        self,
        phases: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Reduce phase interval to base interval (0, 1) for LC without pulsations.

        This method reduces the original phase interval to the base interval (0, 1) in
        the case of light curves without pulsations. It optimizes calculations by finding
        unique phases and returning a mapping to reconstruct the original phase array.

        The reduction is skipped if the system has pulsations or is asynchronous with spots,
        as these require calculation for each phase.

        :param phases: Array of phases to reduce.
        :type phases: NDArray
        :returns: Tuple containing base phases (unique phases in interval (0, 1)) and
            reverse indices (mask applicable to base_phases which reconstructs original phases).
        :rtype: tuple[NDArray, NDArray]
        """
        from elisa.binary_system.system import BinarySystem  # noqa: PLC0415
        from elisa.single_system.system import SingleSystem  # noqa: PLC0415

        if self._system_cls == BinarySystem or str(self._system_cls) == str(BinarySystem):
            # function shouldn't search for base phases if system has pulsations or is asynchronous with spots
            has_pulsation_test = (
                self._system.primary.has_pulsations() | self._system.secondary.has_pulsations()
            )

            test1 = (self._system.primary.synchronicity != 1.0) & self._system.primary.has_spots()
            test2 = (self._system.secondary.synchronicity != 1.0) & self._system.secondary.has_spots()
            asynchronous_spotty_test = test1 | test2

            if has_pulsation_test | asynchronous_spotty_test:
                return phases, up.arange(phases.shape[0])
            base_interval = np.round(phases % 1, 9)
            unique, inverse = np.unique(base_interval, return_inverse=True)
            return unique, inverse  # type: ignore[assignment]

        if self._system_cls == SingleSystem or str(self._system_cls) == str(SingleSystem):
            has_pulsation_test = self._system.star.has_pulsations()
            has_spot_test = self._system.star.has_spots()

            # the most complex case, has to be solved for each phase
            if has_pulsation_test:
                return phases, up.arange(phases.shape[0])
            # in case of just spots on surface, unique (0.1) phases are only needed
            if has_spot_test and not has_pulsation_test:
                base_interval = np.round(phases % 1, 9)
                unique, inverse = np.unique(base_interval, return_inverse=True)
                return unique, inverse  # type: ignore[assignment]
            # in case of clear surface no pulsations and spots, only single observation is needed
            return np.zeros(1), np.zeros(phases.shape[0], dtype=int)

        if self._system_cls == RadialVelocitySystem or str(self._system_cls) == str(
            RadialVelocitySystem,
        ):
            return phases, up.arange(phases.shape[0], dtype=int)

        msg = "Not implemented."
        raise NotImplementedError(msg)

    def manage_time_series(
            self,
            from_phase: Float | None = None,
            to_phase: Float | None = None,
            phase_step: Float | None = None,
            phases: ArrayLike | None = None,
            from_time: Float | None = None,
            to_time: Float | None = None,
            time_step: Float | None = None,
            times: ArrayLike | None = None,
    ) -> NDArray:
        """Convert input time series parameters into photometric phases.

        Converts time or phase domain parameters into an array of phases for observation.
        Either phase-domain parameters (from_phase, to_phase, phase_step, or phases array)
        or time-domain parameters (from_time, to_time, time_step, or times array) must be
        provided, but not both.

        :param from_phase: Starting phase of the observation. Defaults to None.
        :type from_phase: float | None
        :param to_phase: End phase of the observation. Defaults to None.
        :type to_phase: float | None
        :param phase_step: Phase increment of the observations. Defaults to None.
        :type phase_step: float | None
        :param phases: Array of phases at which to perform observations. If provided,
            from_phase, to_phase, and phase_step are ignored. Defaults to None.
        :type phases: ArrayLike | None
        :param from_time: Starting time of the observation. Defaults to None.
        :type from_time: float | None
        :param to_time: End time of the observation. Defaults to None.
        :type to_time: float | None
        :param time_step: Time increment of the observations. Defaults to None.
        :type time_step: float | None
        :param times: Array of times at which to perform observations. If provided,
            from_time, to_time, and time_step are ignored. Defaults to None.
        :type times: ArrayLike | None
        :returns: Array of phases for observations.
        :rtype: NDArray
        :raises ValueError: If time series parameters are invalid or inconsistently provided.
        """
        phases_supplied = not (
                phases is None
                and (from_phase is None or to_phase is None or phase_step is None)
        )
        times_supplied = not (
                times is None and (from_time is None or to_time is None or time_step is None)
        )

        message = (
            "Please pick arguments either from phase-domain: `from_phase`, `to_phase`, "
            "`phase_step` or `phases` or from time-domain parameters: `from_time`, `to_time`, "
            "`time_step` or `times`."
        )
        if not (phases_supplied or times_supplied):
            msg = f"Missing arguments.\n{message}"
            raise ValueError(msg)
        if phases_supplied and times_supplied:
            msg = f"You specified time series in phase and time domain at once.\n{message}"
            raise ValueError(msg)

        if times_supplied:
            if is_empty(times):
                times = up.arange(from_time, to_time, time_step)

            phases = jd_to_phase(times, self._system.period, self._system.t0, centre=0.5)
        elif is_empty(phases):
            phases = up.arange(from_phase, to_phase, phase_step)

        return np.array(phases) - self._system.phase_shift
