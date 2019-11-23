import numpy as np

from elisa.binary_system.curves import rv
from elisa.binary_system.orbit import orbit
from elisa.binary_system.transform import RadialVelocityObserverProperties
from elisa.logger import getLogger
from elisa.observer import plot
from elisa.utils import is_empty

from elisa import (
    umpy as up,
    units,
    const
)

logger = getLogger('binary_system.curves.community')


class Observables(object):
    def __init__(self, observer):
        self.observer = observer

    def rv(self, from_phase=None, to_phase=None, phase_step=None, phases=None, normalize=False):
        return self.observer.radial_velocity(from_phase, to_phase, phase_step, phases, normalize)


class RadialVelocityObserver(object):
    """
    Standalone class to compute binary components center of mass radial velocity via astro-community parameters.

    Community parameters::

        ``asini`` -- semi_major_axis * sin(inclination)
        ``eccentricit``
        ``argument_of_periastron``
        ``period``
        ``true_anomaly``
        ``gamma``
        ``mass_ratio``

    Input (initialization) parameters

    :param eccentricity: Union[(numpy.)int, (numpy.)float];
    :param argument_of_periastron: Union[(numpy.)float, (numpy.)int, astropy.units.quantity.Quantity];
    :param period: Union[(numpy.)float, (numpy.)int, astropy.units.quantity.Quantity]; Orbital period of binary
                   star system. If unit is not specified, default period unit is assumed (days).
    Output parameters (computed on init):
    """

    inclination = const.HALF_PI
    phase_shift = 0.0

    def __init__(self, eccentricity, argument_of_periastron, period, mass_ratio, asini, gamma):
        self.plot = Plot(self)

        kwargs = self.transform_input(**dict(
            eccentricity=eccentricity,
            argument_of_periastron=argument_of_periastron,
            period=period,
            mass_ratio=mass_ratio,
            asini=asini,
            gamma=gamma
        ))

        self.eccentricity = np.nan
        self.argument_of_periastron = np.nan
        self.period = np.nan
        self.mass_ratio = np.nan
        self.asini = np.nan
        self.gamma = np.nan
        self.orbit = None
        self.rv_unit = units.dimensionless_unscaled

        self.observe = Observables(self)

        self.init_properties(**kwargs)
        self.init_orbit()

    def init_properties(self, **kwargs):
        """
        Setup properties from input.

        :param kwargs: Dict; all supplied input properties
        """
        for key, val in kwargs.items():
            setattr(self, key, val)

    @staticmethod
    def transform_input(**kwargs):
        """
        Transform and validate input kwargs.

        :param kwargs: Dict;
        :return: Dict;
        """
        return RadialVelocityObserverProperties.transform_input(**kwargs)

    def init_orbit(self):
        """
        Orbit class fro binary like system.
        """
        logger.debug(f"re/initializing orbit in class instance {self.__class__.__name__}")
        orbit_kwargs = {key: getattr(self, key) for key in orbit.Orbit.ALL_KWARGS}
        self.orbit = orbit.Orbit(**orbit_kwargs)

    def radial_velocity(self, from_phase=None, to_phase=None, phase_step=None, phases=None, normalize=False):
        """
        Method for simulation of observation radial velocity curves in community manner (on params `asini` and `q`).

        :param normalize: bool;
        :param from_phase: float;
        :param to_phase: float;
        :param phase_step: float;
        :param phases: Iterable float;
        :return: Tuple[numpy.array, numpy.array, numpy.array]; phases, primary rv, secondary rv
        """
        if phases is None and (from_phase is None or to_phase is None or phase_step is None):
            raise ValueError("Missing arguments. Specify phases.")

        if is_empty(phases):
            phases = up.arange(start=from_phase, stop=to_phase, step=phase_step)
        phases = np.array(phases)
        position_method = self.orbit.orbital_motion
        orbital_motion = position_method(phase=phases)
        r1, r2 = self.distance_to_center_of_mass(self.mass_ratio, orbital_motion)

        sma_primary = rv.orbital_semi_major_axes(r1[-1], self.orbit.eccentricity, orbital_motion[:, 2][-1])
        sma_secondary = rv.orbital_semi_major_axes(r2[-1], self.orbit.eccentricity, orbital_motion[:, 2][-1])

        period = np.float64((self.period * units.PERIOD_UNIT).to(units.s))
        asini = np.float64((self.asini * units.solRad).to(units.m))

        sma_primary *= asini
        sma_secondary *= asini

        primary_rv = self._radial_velocity(sma_primary, self.eccentricity, self.argument_of_periastron,
                                           period, orbital_motion[:, 2]) * -1.0

        secondary_rv = self._radial_velocity(sma_secondary, self.eccentricity, self.argument_of_periastron,
                                             period, orbital_motion[:, 2])

        primary_rv += self.gamma
        secondary_rv += self.gamma

        self.rv_unit = units.m / units.s
        if normalize:
            self.rv_unit = units.dimensionless_unscaled
            _max = np.max([primary_rv, secondary_rv])
            primary_rv /= _max
            secondary_rv /= _max

        return phases, primary_rv, secondary_rv

    @staticmethod
    def distance_to_center_of_mass(q, positions):
        distance = positions[:, 0]
        com_from_primary = (q * distance) / (1.0 + q)
        return com_from_primary, distance - com_from_primary

    @staticmethod
    def _radial_velocity(asini, eccentricity, argument_of_periastron, period, true_anomaly):
        """
        Compute radial velocity for given paramters.

        :param asini: float
        :param eccentricity: float
        :param argument_of_periastron: float
        :param true_anomaly: float or numpy.array
        :param period: float
        :return: float or numpy.array
        """
        a = 2.0 * up.pi * asini
        b = period * up.sqrt(1.0 - up.power(eccentricity, 2))
        c = up.cos(true_anomaly + argument_of_periastron) + (eccentricity * up.cos(argument_of_periastron))
        return a * c / b


class Plot(object):
    def __init__(self, observer):
        self.observer = observer

    rv_curve = plot.Plot.rv_curve
