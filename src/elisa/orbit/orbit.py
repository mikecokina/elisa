import numpy as np

from elisa import logger, utils, const, umpy as up
from elisa.orbit.transform import OrbitProperties


def angular_velocity(period, eccentricity, distance):
    """
    Compute angular velocity for given components distance.
    This can be derived from facts that::


        w = dp/dt

        P * 1/2 * dp/dt = pi * a * b

        e = sqrt(1 - (b/a)^2)

    where a, b are respectively semi major and semi minor axis, P is period and e is eccentricity.


    :param period:
    :param eccentricity:
    :param distance: float
    :return: float
    """
    return ((2.0 * up.pi) / (period * 86400.0 * (distance ** 2))) * up.sqrt(
        (1.0 - eccentricity) * (1.0 + eccentricity))  # $\rad.sec^{-1}$


class Orbit(object):

    MANDATORY_KWARGS = ['period', 'inclination', 'eccentricity', 'argument_of_periastron']
    OPTIONAL_KWARGS = ['phase_shift']
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, suppress_logger=False, **kwargs):
        utils.invalid_kwarg_checker(kwargs, Orbit.ALL_KWARGS, Orbit)
        utils.check_missing_kwargs(self.__class__.MANDATORY_KWARGS, kwargs, instance_of=self.__class__)
        kwargs = OrbitProperties.transform_input(**kwargs)
        self._logger = logger.getLogger(name=self.__class__.__name__, suppress=suppress_logger)

        # default valeus of properties
        self.period = np.nan
        self.eccentricity = np.nan
        self.argument_of_periastron = np.nan
        self.inclination = np.nan

        self.periastron_distance = np.nan
        self.periastron_phase = np.nan
        self.semimajor_axis = np.nan
        self.phase_shift = 0.0

        # values of properties
        self._logger.debug(f"setting properties of orbit")
        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])

        self.periastron_distance = self.compute_periastron_distance()
        self.periastron_phase = - self.get_conjuction()["primary_eclipse"]["true_phase"] % 1

    @classmethod
    def true_phase(cls, phase, phase_shift):
        """
        Returns shifted phase of the orbit by the amount phase_shift.

        :param phase: ndarray
        :param phase_shift: float
        :return: ndarray
        """
        return phase + phase_shift

    @staticmethod
    def phase(true_phase, phase_shift):
        """
        reverts the phase shift introduced in function true phase

        :param true_phase: numpy.array
        :param phase_shift: numpy.float
        :return:
        """
        return true_phase - phase_shift

    @classmethod
    def phase_to_mean_anomaly(cls, phase):
        """
        returns mean anomaly of points on orbit as a function of phase

        :param phase: ndarray
        :return: ndarray
        """
        return const.FULL_ARC * phase

    @classmethod
    def mean_anomaly_to_phase(cls, mean_anomaly):
        """
        returns phase of points on orbit as a function of mean anomaly

        :param mean_anomaly: numpy.array
        :return:
        """
        return mean_anomaly / const.FULL_ARC

    def mean_anomaly_fn(self, eccentric_anomaly: float, *args) -> float:
        """
        Definition of Kepler equation for scipy _solver in Orbit.eccentric_anomaly.

        :param eccentric_anomaly: float
        :param args: Tuple; (mean_anomaly, )
        :return: float
        """
        mean_anomaly, = args
        return eccentric_anomaly - self.eccentricity * up.sin(eccentric_anomaly) - mean_anomaly

    def mean_anomaly_to_eccentric_anomaly(self, mean_anomaly: float) -> float:
        """
        Solves Kepler equation for eccentric anomaly via mean anomaly.

        :param mean_anomaly: float
        :return: float
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
            self._logger.debug(f"Solver scipy.optimize.newton in function Orbit.eccentric_anomaly did not provide "
                               f"solution.\n Reason: {e}")
            return False

    def eccentric_anomaly_to_mean_anomaly(self, eccentric_anomaly):
        """
        returns mean anomaly as a function of eccentric anomaly calculated using Kepler equation

        :param eccentric_anomaly: numpy.array
        :return:
        """
        return (eccentric_anomaly - self.eccentricity * up.sin(eccentric_anomaly)) % const.FULL_ARC

    def eccentric_anomaly_to_true_anomaly(self, eccentric_anomaly):
        """
        Returns true anomaly as a function of eccentric anomaly and eccentricity.

        :param eccentric_anomaly: ndarray
        :return: ndarray
        """
        true_anomaly = 2.0 * up.arctan(
            up.sqrt((1.0 + self.eccentricity) / (1.0 - self.eccentricity)) * up.tan(eccentric_anomaly / 2.0))
        true_anomaly[true_anomaly < 0] += const.FULL_ARC
        return true_anomaly

    def true_anomaly_to_eccentric_anomaly(self, true_anomaly):
        """
        returns eccentric anomaly as a function of true anomaly and eccentricity

        :param true_anomaly: numpy.array
        :return:
        """
        eccentric_anomaly = \
            2.0 * up.arctan(up.sqrt((1.0 - self.eccentricity) / (1.0 + self.eccentricity)) * up.tan(true_anomaly / 2.0))
        eccentric_anomaly[eccentric_anomaly < 0] += const.FULL_ARC
        return eccentric_anomaly

    def relative_radius(self, true_anomaly):
        """
        calculates the length of radius vector of elipse where a=1

        :param true_anomaly: numpy.array
        :return: numpy.array
        """
        return (1.0 - self.eccentricity ** 2) / (1.0 + self.eccentricity * up.cos(true_anomaly))

    def true_anomaly_to_azimuth(self, true_anomaly):
        """
        Convert true anomaly angle to azimuth angle measured from -y axis in 2D plane.

        ::

        azimuth 0 alligns with -y axis
                 |
                 |      pi/2
            -----------
                 |
                 | 0

        :param true_anomaly: ndarray or float
        :return: ndarray or float
        """
        return (true_anomaly + self.argument_of_periastron) % const.FULL_ARC

    def azimuth_to_true_anomaly(self, azimuth):
        """
        Calculates the azimuth form given true anomaly

        :param azimuth: numpy.array
        :return:
        """
        return (azimuth - self.argument_of_periastron) % const.FULL_ARC

    def orbital_motion(self, phase):
        """
        Function takes photometric phase of the binary system as input and calculates positions of the secondary
        component in the frame of reference of primary component.

        :param phase: ndarray or float
        :return: ndarray; matrix consisting of column stacked vectors distance, azimut angle, true anomaly and phase

        ::

            ndarray((r1, az1, ni1, phs1),
                    (r2, az2, ni2, phs2),
                     ...
                    (rN, azN, niN, phsN))
        """
        # ability to accept scalar as input
        if isinstance(phase, (int, np.int, float, np.float)):
            phase = np.array([np.float(phase)])
        # photometric phase to phase measured from periastron
        true_phase = self.true_phase(phase=phase, phase_shift=self.get_conjuction()['primary_eclipse']['true_phase'])
        true_phase = self.phase(true_phase=true_phase, phase_shift=self.phase_shift)

        mean_anomaly = self.phase_to_mean_anomaly(phase=true_phase)
        eccentric_anomaly = np.array([self.mean_anomaly_to_eccentric_anomaly(mean_anomaly=xx)
                                      for xx in mean_anomaly])
        true_anomaly = self.eccentric_anomaly_to_true_anomaly(eccentric_anomaly=eccentric_anomaly)
        distance = self.relative_radius(true_anomaly=true_anomaly)
        azimut_angle = self.true_anomaly_to_azimuth(true_anomaly=true_anomaly)

        return np.column_stack((distance, azimut_angle, true_anomaly, phase))

    def orbital_motion_from_azimuths(self, azimuth):
        """
        function takes azimuths of the binary system (angle between ascending node (-y) as input and calculates
        positions of the secondary component in the frame of reference of primary component

        :param azimuth: numpy.array or numpy.float
        :return: numpy.array: matrix consisting of column stacked vectors distance, azimut angle, true anomaly and phase
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
        phase = self.phase(true_phase, phase_shift=self.get_conjuction()['primary_eclipse']['true_phase'])
        return np.column_stack((distance, azimuth, true_anomaly, phase))

    def get_conjuction(self):
        """
        Compute and return photometric phase of conjunction (eclipses).
        We assume that primary component is placed in center of coo system and observation unit vector is [-1, 0, 0]

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

        :return: Dict
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
            true_anomaly_of_conjuction = (alpha - self.argument_of_periastron) % const.FULL_ARC  # \nu_{con}

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

        :return: float
        """
        periastron_distance = self.relative_radius(true_anomaly=np.array([0])[0])
        self._logger.debug(f"setting property periastron_distance "
                           f"of class instance {self.__class__.__name__} to {periastron_distance}")
        return periastron_distance
