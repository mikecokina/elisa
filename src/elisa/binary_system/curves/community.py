import numpy as np

from .. orbit import orbit
from .. transform import RadialVelocityObserverProperties
from ... logger import getLogger
from ... import (
    umpy as up,
    units as u,
    const
)

logger = getLogger('binary_system.curves.community')


class RadialVelocitySystem(object):
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
    """

    inclination = const.HALF_PI
    phase_shift = 0.0

    def __init__(self, eccentricity, argument_of_periastron, period, mass_ratio, asini, gamma):
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
        self.rv_unit = u.dimensionless_unscaled

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

        :param kwargs: Dict; model parameters
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

    def radial_velocity(self, **kwargs):
        """
        Method for producing synthetic radial velocity curves in community format (on params `asini` and `q`).

        :param kwargs: Dict;
        :**kwargs options**:
            * :phases: numpy.array; photometric phases used to calculate synthetic radial velocities
            * :position_method: callable; function producing an array of elisa.const.Position tuples that describe
                                          position and orientation of system components at given photometric phase

        """
        phases = kwargs.pop("phases")
        position_method = kwargs.pop("position_method")
        orbital_motion = position_method(phase=phases)

        sma_primary, sma_secondary = self.distance_to_center_of_mass(self.mass_ratio, 1.0)

        period = np.float64((self.period * u.PERIOD_UNIT).to(u.s))
        asini = np.float64((self.asini * u.solRad).to(u.m))

        sma_primary *= asini
        sma_secondary *= asini

        primary_rv = self._radial_velocity(sma_primary, self.eccentricity, self.argument_of_periastron,
                                           period, orbital_motion[:, 2]) * -1.0

        secondary_rv = self._radial_velocity(sma_secondary, self.eccentricity, self.argument_of_periastron,
                                             period, orbital_motion[:, 2])

        rv_dict = {'primary':  primary_rv + self.gamma, 'secondary': secondary_rv + self.gamma}

        return rv_dict

    @staticmethod
    def distance_to_center_of_mass(q, distance):
        """
        Returns distance from component's centre of mass to the barycentre.

        :param q: float; mass ratio
        :param distance: float; components distance
        :return: Tuple; primary, secondary com distance from barycentre
        """
        com_from_primary = (q * distance) / (1.0 + q)
        return com_from_primary, distance - com_from_primary

    @staticmethod
    def _radial_velocity(asini, eccentricity, argument_of_periastron, period, true_anomaly):
        """
        Compute radial velocity for given paramters.

        :param asini: float;
        :param eccentricity: float;
        :param argument_of_periastron: float;
        :param true_anomaly: Union[float, numpy.array];
        :param period: float
        :return: Union[float, numpy.array];
        """
        a = 2.0 * up.pi * asini
        b = period * up.sqrt(1.0 - up.power(eccentricity, 2))
        c = up.cos(true_anomaly + argument_of_periastron) + (eccentricity * up.cos(argument_of_periastron))
        return - a * c / b

    def get_positions_method(self):
        return self.orbit.orbital_motion

    def compute_rv(self, **kwargs):
        """
        Method for producing synthetic radial velocity curves in community format (on params `asini` and `q`).

        :param kwargs: Dict;
        :**kwargs options**:
            * :phases: numpy.array; photometric phases used to calculate synthetic radial velocities
            * :position_method: callable; function producing an array of elisa.const.Position tuples that describe
                                          position and orientation of system components at given photometric phase

        """
        return self.radial_velocity(**kwargs)

    @staticmethod
    def prepare_json(data):
        """
        Filtering out parameters necessary to initialize RadialVelocitySystem

        :param data: Dict; possibly containing items not valid for initialization of RadialVelocitySystem instance
        :return: Dict; set of argument necessary to initialize RadialVelocitySystem instance
        """
        return dict(
            eccentricity=data['eccentricity'],
            argument_of_periastron=data['argument_of_periastron'],
            period=data['period'],
            mass_ratio=data['mass_ratio'],
            asini=data['asini'],
            gamma=data['gamma']
        )
