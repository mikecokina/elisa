import numpy as np

from ... import (
    utils,
    const,
    umpy as up
)
from ... logger import getLogger
from ... binary_system.orbit.transform import OrbitProperties
from ... base.orbit.orbit import AbstractOrbit

logger = getLogger('binary_system.orbit.orbit')


def angular_velocity(period, eccentricity, distance):
    """
    Compute angular velocity for given components distance.
    This can be derived from facts that::


        w = dp/dt

        P * 1/2 * dp/dt = pi * a * b

        e = sqrt(1 - (b/a)^2)

    where a, b are respectively semi major and semi minor axis, P is period and e is eccentricity.


    :param period: float;
    :param eccentricity: float;
    :param distance: float;
    :return: float;
    """
    return ((2.0 * up.pi) / (period * 86400.0 * (distance ** 2))) * up.sqrt(
        (1.0 - eccentricity) * (1.0 + eccentricity))  # $\rad.sec^{-1}$


def primary_orbital_speed(m1, m2, a_red, components_distance):
    """
    Returns orbital speed of primary component with respect to the system centre of mass.

    :param m1: float; primary mass
    :param m2: float; secondary mass
    :param a_red: float; semi major axis of the primary component with respect to the system centre of mass
    :param components_distance: float;
    :return: float;
    """
    m = m1 + m2
    return m2 * np.sqrt((const.G / m) * ((2 / components_distance) - (m2 / (a_red * m))))


def velocity_vector_angle(eccentricity, true_anomaly):
    """
    Returns sine and cosine of angle between velocity vector and join vector.

    :param eccentricity: float;
    :param true_anomaly: float;
    :return: Tuple;
    """
    den = np.sqrt(1 + eccentricity**2 + 2 * eccentricity * np.cos(true_anomaly))
    sin = (1 + eccentricity * np.cos(true_anomaly)) / den
    cos = - (eccentricity * np.sin(true_anomaly)) / den
    return sin, cos


def create_orb_vel_vectors(system, components_distance):
    """
    Returns orbital velocity vectors for both components in reference frame of centre of mass.

    :param system: elisa.binary_system.container;
    :param components_distance: float;
    :return: float;
    """
    a_red = system.semi_major_axis * system.mass_ratio / (1 + system.mass_ratio)

    speed = primary_orbital_speed(system.primary.mass, system.secondary.mass, a_red,
                                  system.semi_major_axis * components_distance)

    sin, cos = velocity_vector_angle(system.eccentricity, system.position.true_anomaly)

    velocity = {'primary': np.array([cos * speed, -sin * speed, 0])}
    velocity['secondary'] = - velocity['primary'] / system.mass_ratio

    return velocity


def distance_to_center_of_mass(primary_mass, secondary_mass, distance):
    """
    Return distance from primary and from secondary component to center of mass.

    :param primary_mass: float
    :param secondary_mass: float
    :param distance: Union[float, numpy.array]
    :return: Tuple[Union[float, numpy.array]];
    """
    mass = primary_mass + secondary_mass
    com_from_primary = (distance * secondary_mass) / mass
    return com_from_primary, distance - com_from_primary


def orbital_semi_major_axes(r, eccentricity, true_anomaly):
    """
    Return orbital semi major axis from component distance, eccentricity and true anomaly.

    :param r: float or numpy.array; distance from center of mass to object
    :param eccentricity: float or numpy.array; orbital eccentricity
    :param true_anomaly: float or numpy.array; true anomaly of orbital motion
    :return: Union[float, numpy.array]
    """
    return r * (1.0 + eccentricity * up.cos(true_anomaly)) / (1.0 - up.power(eccentricity, 2))


def component_distance_from_mean_anomaly(eccentricity, true_anomaly):
    """
    Return component distance in SMA from semi-major axis, eccentricity and true anomaly.

    :param r: float or numpy.array; distance from center of mass to object
    :param eccentricity: float or numpy.array; orbital eccentricity
    :param true_anomaly: float or numpy.array; true anomaly of orbital motion
    :return: Union[float, numpy.array]
    """
    return (1.0 - up.power(eccentricity, 2)) / (1.0 + eccentricity * up.cos(true_anomaly))


def get_approx_ecl_angular_width(forward_radius1, forward_radius2, components_distance, inclination):
    """
    Returns angular width of the eclipse assuming spherical components.

    :param forward_radius1: float;
    :param forward_radius2: float;
    :param components_distance: float; in SMA
    :param inclination: float; in radians
    :return: tuple; angular half-width of the eclipse and the inner plateau
    """
    # tilt of the orbital plane and z-axis in the observer reference frame
    tilt = np.abs(const.HALF_PI - inclination)
    # maximum apparent distance between components where eclipse is possible
    r_outer = forward_radius1 + forward_radius2
    r_inner = np.abs(forward_radius1 - forward_radius2)
    # closest aparent distances of component centres
    r_close = components_distance * np.sin(tilt)

    # checking if eclipses occur

    nu_outer = 0.0 if r_close >= r_outer else \
        np.arcsin(np.sqrt(
            np.power(r_outer/components_distance, 2) - np.power(np.sin(tilt), 2)
        ))
    nu_inner = 0.0 if r_close >= r_inner else \
        np.arcsin(np.sqrt(
            np.power(r_inner / components_distance, 2) - np.power(np.sin(tilt), 2)
        ))

    return nu_outer, nu_inner


class Orbit(AbstractOrbit):
    """
    Object representing orbit of a binary system. Accessible as an attribute of an BinarySystem object

    Input parameters:

    :param period: Union[(numpy.)float, (numpy.)int, astropy.units.quantity.Quantity]; Orbital period of binary
                   star system. If unit is not specified, default period unit is assumed (days).
    :param inclination: ; Union[float, astropy.units.Quantity]; If unitless values is supplied, default unit
                          suppose to be radians.
    :param eccentricity: Union[(numpy.)int, (numpy.)float];
    :param argument_of_periastron: Union[(numpy.)float, (numpy.)int, astropy.units.quantity.Quantity]; If unit is
                                   not supplied, value in radians is assumed.
    :param phase_shift: float;

    Output parameters:

    :periastron_distance: float; Distance in periastron (in unit of semi major axis (a = a1 + a2) of relative motion)
    :periastron_phase: float; true (not photometric but computed from mean anomaly) phase of periastron.
    """

    MANDATORY_KWARGS = ['period', 'inclination', 'eccentricity', 'argument_of_periastron']
    OPTIONAL_KWARGS = ['phase_shift']
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, **kwargs):
        utils.invalid_kwarg_checker(kwargs, Orbit.ALL_KWARGS, Orbit)
        utils.check_missing_kwargs(self.__class__.MANDATORY_KWARGS, kwargs, instance_of=self.__class__)
        kwargs = OrbitProperties.transform_input(**kwargs)

        super(Orbit, self).__init__(**kwargs)

        # default valeus of properties
        self.period = np.nan
        self.eccentricity = np.nan
        self.argument_of_periastron = np.nan
        self.inclination = np.nan

        self.periastron_distance = np.nan
        self.periastron_phase = np.nan
        self.semi_major_axis = np.nan
        self.phase_shift = 0.0

        # values of properties
        logger.debug(f"setting properties of orbit")
        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])

        self.periastron_distance = self.compute_periastron_distance()
        self.conjunctions = self.get_conjuction()
        self.periastron_phase = - self.conjunctions["primary_eclipse"]["true_phase"] % 1

    @classmethod
    def phase_to_mean_anomaly(cls, phase):
        """
        returns mean anomaly of points on orbit as a function of phase

        :param phase: Union[numpy.array, float];
        :return: Union[numpy.array, float];
        """
        return const.FULL_ARC * phase

    @classmethod
    def mean_anomaly_to_phase(cls, mean_anomaly):
        """
        returns phase of points on orbit as a function of mean anomaly

        :param mean_anomaly: Union[numpy.array, float];
        :return:  Union[numpy.array, float];
        """
        return mean_anomaly / const.FULL_ARC

    def mean_anomaly_fn(self, eccentric_anomaly: float, *args) -> float:
        """
        Definition of Kepler equation for scipy _solver in Orbit.eccentric_anomaly.

        :param eccentric_anomaly: float;
        :param args: Tuple; (mean_anomaly, )
        :return: float;
        """
        mean_anomaly, = args
        return eccentric_anomaly - self.eccentricity * up.sin(eccentric_anomaly) - mean_anomaly

    def mean_anomaly_to_eccentric_anomaly(self, mean_anomaly: float) -> float:
        """
        Solves Kepler equation for eccentric anomaly via mean anomaly.

        :param mean_anomaly: float;
        :return: float;
        """
        import scipy.optimize
        try:
            solution = scipy.optimize.newton(self.mean_anomaly_fn, 1.0, args=(mean_anomaly, ), tol=1e-10)
            if not up.isnan(solution):
                if solution < 0:
                    solution += const.FULL_ARC
                return solution
            else:
                return False
        except Exception as e:
            logger.debug(f"solver scipy.optimize.newton in function Orbit.eccentric_anomaly did not provide "
                         f"solution.\n Reason: {e}")
            return False

    def eccentric_anomaly_to_mean_anomaly(self, eccentric_anomaly):
        """
        Returns mean anomaly as a function of eccentric anomaly calculated using Kepler equation.

        :param eccentric_anomaly: numpy.array;
        :return: numpy.array
        """
        return (eccentric_anomaly - self.eccentricity * up.sin(eccentric_anomaly)) % const.FULL_ARC

    def eccentric_anomaly_to_true_anomaly(self, eccentric_anomaly):
        """
        Returns true anomaly as a function of eccentric anomaly and eccentricity.

        :param eccentric_anomaly: Union[numpy.array, float]
        :return: Union[numpy.array, float]
        """
        true_anomaly = 2.0 * up.arctan(
            up.sqrt((1.0 + self.eccentricity) / (1.0 - self.eccentricity)) * up.tan(eccentric_anomaly / 2.0))
        true_anomaly[true_anomaly < 0] += const.FULL_ARC
        return true_anomaly

    def true_anomaly_to_eccentric_anomaly(self, true_anomaly):
        """
        Returns eccentric anomaly as a function of true anomaly and eccentricity.

        :param true_anomaly: Union[numpy.array, float];
        :return: Union[numpy.array, float];
        """
        eccentric_anomaly = \
            2.0 * up.arctan(up.sqrt((1.0 - self.eccentricity) / (1.0 + self.eccentricity)) * up.tan(true_anomaly / 2.0))
        eccentric_anomaly[eccentric_anomaly < 0] += const.FULL_ARC
        return eccentric_anomaly

    def relative_radius(self, true_anomaly):
        """
        Calculates the length of radius vector of elipse where a=1.

        :param true_anomaly: Union[numpy.array, float];
        :return: Union[numpy.array, float];
        """
        return (1.0 - self.eccentricity ** 2) / (1.0 + self.eccentricity * up.cos(true_anomaly))

    def true_anomaly_to_azimuth(self, true_anomaly):
        """
        Convert true anomaly angle to azimuth angle measured from y axis in 2D plane.

        ::

            azimuth 0 alligns with y axis
                     | 0
                pi/2 |
                -----------
                     |
                     |

        :param true_anomaly: Union[numpy.array, float];
        :return: Union[numpy.array, float];
        """
        return (true_anomaly + self.argument_of_periastron) % const.FULL_ARC

    def azimuth_to_true_anomaly(self, azimuth):
        """
        Calculates the azimuth form given true anomaly

        :param azimuth: Union[numpy.array, float]
        :return: Union[numpy.array, float]
        """
        return (azimuth - self.argument_of_periastron) % const.FULL_ARC

    def orbital_motion(self, phase):
        """
        Function takes photometric phase of the binary system as input and calculates positions of the secondary
        component in the frame of reference of primary component.

        :param phase: Union[numpy.array, float]; photometric phases
        :return: numpy.array; matrix consisting of column stacked vectors distance, azimut angle, true anomaly and phase

        ::

            numpy.array((r1, az1, nu1, phs1),
                        (r2, az2, nu2, phs2),
                         ...       
                        (rN, azN, nuN, phsN))
        """
        # ability to accept scalar as input
        if isinstance(phase, (int, np.int, float, np.float)):
            phase = np.array([np.float(phase)])
        # photometric phase to phase measured from periastron
        true_phase = self.true_phase(phase=phase, phase_shift=self.conjunctions['primary_eclipse']['true_phase'])

        mean_anomaly = self.phase_to_mean_anomaly(phase=true_phase)
        eccentric_anomaly = np.array([self.mean_anomaly_to_eccentric_anomaly(mean_anomaly=xx)
                                      for xx in mean_anomaly])
        true_anomaly = self.eccentric_anomaly_to_true_anomaly(eccentric_anomaly=eccentric_anomaly)
        distance = self.relative_radius(true_anomaly=true_anomaly)
        azimut_angle = self.true_anomaly_to_azimuth(true_anomaly=true_anomaly)

        return np.column_stack((distance, azimut_angle, true_anomaly, phase))

    def orbital_motion_from_azimuths(self, azimuth):
        """
        Function takes azimuths of the binary system (angle between ascending node (y) as input and calculates
        positions of the secondary component in the frame of reference of primary component.

        :param azimuth: Union[numpy.array, float];
        :return: numpy.array; matrix consisting of column stacked vectors distance,
                              azimut angle, true anomaly and phase

        ::

               numpy.array((r1, az1, ni1, phs1),
                           (r2, az2, ni2, phs2),
                            ...
                           (rN, azN, niN, phsN))

        """
        true_anomaly = self.azimuth_to_true_anomaly(azimuth)
        distance = self.relative_radius(true_anomaly=true_anomaly)
        eccentric_anomaly = self.true_anomaly_to_eccentric_anomaly(true_anomaly)
        mean_anomaly = self.eccentric_anomaly_to_mean_anomaly(eccentric_anomaly)
        true_phase = self.mean_anomaly_to_phase(mean_anomaly)
        phase = self.phase(true_phase, phase_shift=self.conjunctions['primary_eclipse']['true_phase'])
        return np.column_stack((distance, azimuth, true_anomaly, phase))

    def get_conjuction(self):
        """
        Compute and return photometric phase of conjunction (eclipses).
        We assume that primary component is placed in center of coo system and observation unit vector is [1, 0, 0]

        return dictionary is in shape::

            {
                type_of_eclipse <`primary_eclipse` or `secondary_eclipse`>: {
                    'true_phase': float,
                    'true_anomaly': float,
                    'mean_anomaly': float,
                    'eccentric_anomaly'float:
                },
                ...
            }

        :return: Dict;
        """
        # determining order of eclipses
        conjuction_arc_list = []
        try:
            if 0 <= self.inclination <= const.PI / 2.0:
                conjuction_arc_list = [const.PI / 2.0, 3.0 * const.PI / 2.0]
            elif const.PI / 2.0 < self.inclination <= const.PI:
                conjuction_arc_list = [3.0 * const.PI / 2.0, const.PI / 2.0]
        except Exception as e:
            raise TypeError(f'Invalid type of {self.__class__.__name__}.inclination - {str(e)}.')

        conjunction_quantities = dict()
        for alpha, idx in list(zip(conjuction_arc_list, ['primary_eclipse', 'secondary_eclipse'])):
            # true anomaly of conjunction (measured from periastron counter-clokwise)
            true_anomaly_of_conjuction = self.azimuth_to_true_anomaly(alpha)  # \nu_{con}

            # eccentric anomaly of conjunction (measured from apse line)
            eccentric_anomaly_of_conjunction = (2.0 * up.arctan(
                up.sqrt((1.0 - self.eccentricity) / (1.0 + self.eccentricity)) *
                up.tan(true_anomaly_of_conjuction / 2.0))) % const.FULL_ARC

            # mean anomaly of conjunction (measured from apse line)
            mean_anomaly_of_conjunction = (eccentric_anomaly_of_conjunction -
                                           self.eccentricity *
                                           up.sin(eccentric_anomaly_of_conjunction)) % const.FULL_ARC

            # true phase of conjunction (measured from apse line)
            true_phase_of_conjunction = (mean_anomaly_of_conjunction / const.FULL_ARC) % 1.0

            conjunction_quantities[idx] = dict()
            conjunction_quantities[idx]["true_anomaly"] = true_anomaly_of_conjuction
            conjunction_quantities[idx]["eccentric_anomaly"] = eccentric_anomaly_of_conjunction
            conjunction_quantities[idx]["mean_anomaly"] = mean_anomaly_of_conjunction
            conjunction_quantities[idx]["true_phase"] = true_phase_of_conjunction

        return conjunction_quantities

    def compute_periastron_distance(self):
        """
        Calculates relative periastron distance in SMA units.

        :return: float;
        """
        periastron_distance = self.relative_radius(true_anomaly=np.array([0])[0])
        logger.debug(f"setting property periastron_distance "
                     f"of class instance {self.__class__.__name__} to {periastron_distance}")
        return periastron_distance
