import numpy as np

from astropy import units as u
from elisa.engine import utils, logger, units
from elisa.engine import const


class Orbit(object):

    MANDATORY_KWARGS = ['period', 'inclination', 'eccentricity', 'argument_of_periastron']
    OPTIONAL_KWARGS = []
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, suppress_logger=False, **kwargs):
        utils.invalid_kwarg_checker(kwargs, Orbit.ALL_KWARGS, Orbit)
        utils.check_missing_kwargs(self.__class__.MANDATORY_KWARGS, kwargs, instance_of=self.__class__)

        self._logger = logger.getLogger(name=self.__class__.__name__, suppress=suppress_logger)

        # default valeus of properties
        self._period: np.float64 = np.nan
        self._inclination: np.float64 = np.nan
        self._eccentricity: np.float64 = np.nan
        self._argument_of_periastron: np.float64 = np.nan
        self._periastron_distance: np.float64 = np.nan
        self._perastron_phase: np.float64 = np.nan
        self._semimajor_axis: np.float64 = np.nan

        # values of properties
        for kwarg in kwargs:
            self._logger.debug(f"setting property {kwarg} "
                               f"of class instance {self.__class__.__name__} to {kwargs[kwarg]}")
            setattr(self, kwarg, kwargs[kwarg])

        self._periastron_distance = self.compute_periastron_distance()
        self._perastron_phase = - self.get_conjuction()["primary_eclipse"]["true_phase"] % 1

    @property
    def semimajor_axis(self):
        """
        Returns semimajor axis in SI units.
        
        :return: float
        """
        return self._semimajor_axis

    @semimajor_axis.setter
    def semimajor_axis(self, semimajor_axis):
        """
        Semimajor axis setter.
        
        :param semimajor_axis: float
        :return:
        """
        self._semimajor_axis = semimajor_axis
    
    @property
    def periastron_phase(self):
        """
        Photometric phase of periastron.
        
        :return: float
        """
        return self._perastron_phase

    @property
    def period(self):
        """
        Returns orbital period of the binary system in default period unit.

        :return: numpy.float
        """
        return self._period

    @period.setter
    def period(self, period):
        """
        Setter for orbital period of binary system orbit.

        :param period: float
        :return:
        """
        if isinstance(period, u.quantity.Quantity):
            self._period = np.float64(period.to(units.PERIOD_UNIT))
        elif isinstance(period, (int, np.int, float, np.float)):
            self._period = np.float64(period)
        else:
            raise TypeError('Input of variable `period` is not (numpy.)int or (numpy.)float '
                            'nor astropy.unit.quantity.Quantity instance.')
        self._logger.debug(f"setting property period "
                           f"of class instance {self.__class__.__name__} to {self._period}")

    @property
    def inclination(self):
        """
        Returns inclination of binary system orbit.

        :return: float
        """
        return self._inclination

    @inclination.setter
    def inclination(self, inclination):
        """
        Setter for inclination of binary system orbit.

        :param inclination: float
        :return:
        """
        if isinstance(inclination, u.quantity.Quantity):
            self._inclination = np.float64(inclination.to(units.ARC_UNIT))
        elif isinstance(inclination, (int, np.int, float, np.float)):
            self._inclination = np.float64(inclination)
        else:
            raise TypeError('Input of variable `inclination` is not (numpy.)int or (numpy.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

    @property
    def eccentricity(self):
        """
        Returns eccentricity of binary system orbit.

        :return: float
        """
        return self._eccentricity

    @eccentricity.setter
    def eccentricity(self, eccentricity):
        """
        Setter for eccentricity of binary system orbit.

        :param eccentricity: float
        :return:
        """
        self._eccentricity = eccentricity

    @property
    def argument_of_periastron(self):
        """
        Returns argument of periastron of binary system orbit.

        :return: float
        """
        return self._argument_of_periastron

    @argument_of_periastron.setter
    def argument_of_periastron(self, argument_of_periastron):
        """
        Setter for argument of periastron. 
        If unit is not supplied, value in degrees is assumed.

        :param argument_of_periastron: (numpy.)int, (numpy.)float, astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(argument_of_periastron, u.quantity.Quantity):
            self._argument_of_periastron = np.float64(argument_of_periastron.to(units.ARC_UNIT))
        elif isinstance(argument_of_periastron, (int, np.int, float, np.float)):
            self._argument_of_periastron = np.float64((argument_of_periastron * u.deg).to(units.ARC_UNIT))
        else:
            raise TypeError('Input of variable `argument_of_periastron` is not (numpy.)int or (numpy.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

    @classmethod
    def true_phase(cls, phase, phase_shift):
        """
        Returns shifted phase of the orbit by the amount phase_shift.

        :param phase: ndarray
        :param phase_shift: float
        :return: ndarray
        """
        return phase + phase_shift

    @classmethod
    def mean_anomaly(cls, phase):
        """
        Returns mean anomaly of points on orbit described by phase.

        :param phase: ndarray
        :return: ndarray
        """
        return const.FULL_ARC * phase

    def mean_anomaly_fn(self, eccentric_anomaly, *args):
        """
        Definition of Kepler equation for scipy _solver in Orbit.eccentric_anomaly.

        :param eccentric_anomaly: float
        :param args: Tuple; (mean_anomaly, )
        :return: float
        """
        mean_anomaly, = args
        return (eccentric_anomaly - self.eccentricity * np.sin(eccentric_anomaly)) - mean_anomaly

    def eccentric_anomaly(self, mean_anomaly):
        """
        Solves Kepler equation for eccentric anomaly via mean anomaly.

        :param mean_anomaly: float
        :return: float
        """
        import scipy.optimize
        try:
            solution = scipy.optimize.newton(self.mean_anomaly_fn, 1.0, args=(mean_anomaly, ), tol=1e-10)
            if not np.isnan(solution):
                if solution < 0:
                    solution += const.FULL_ARC
                return solution
            else:
                return False
        except Exception as e:
            self._logger.debug(f"Solver scipy.optimize.newton in function Orbit.eccentric_anomaly did not provide "
                               f"solution.\n Reason: {e}")
            return False

    def true_anomaly(self, eccentric_anomaly):
        """
        Returns true anomaly as a function of eccentric anomaly and eccentricity.

        :param eccentric_anomaly: ndarray
        :return: ndarray
        """
        true_anomaly = 2.0 * np.arctan(
            np.sqrt((1.0 + self.eccentricity) / (1.0 - self.eccentricity)) * np.tan(eccentric_anomaly / 2.0))
        true_anomaly[true_anomaly < 0] += const.FULL_ARC
        return true_anomaly

    def relative_radius(self, true_anomaly):
        """
        Calculates the length of radius vector of elipse where a = 1.

        :param true_anomaly: ndarray
        :return: ndarray
        """
        return (1.0 - np.power(self.eccentricity, 2)) / (1.0 + self.eccentricity * np.cos(true_anomaly))

    def true_anomaly_to_azimuth(self, true_anomaly):
        """
        Convert true anomaly angle to azimuth angle measured from -y axis in 2D plane.

        ::
                 |
                 |      pi/2
            -----------
                 |
                 | 0

        :param true_anomaly: ndarray or float
        :return: ndarray or float
        """
        azimut = true_anomaly + self.argument_of_periastron
        azimut %= const.FULL_ARC
        return azimut

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

        mean_anomaly = self.mean_anomaly(phase=true_phase)
        eccentric_anomaly = np.array([self.eccentric_anomaly(mean_anomaly=xx)
                                      for xx in mean_anomaly])
        true_anomaly = self.true_anomaly(eccentric_anomaly=eccentric_anomaly)
        distance = self.relative_radius(true_anomaly=true_anomaly)
        azimut_angle = self.true_anomaly_to_azimuth(true_anomaly=true_anomaly)

        return np.column_stack((distance, azimut_angle, true_anomaly, phase))

    def get_conjuction(self):
        """
        Compute and return photometric phase of conjunction (eclipses).
        We assume that primary component is situated in center of coo system and observation unit vector is [-1, 0, 0]

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
            raise TypeError(f'Invalid type of {self.__class__.__name__}.inclination.')

        conjunction_quantities = dict()
        for alpha, idx in list(zip(conjuction_arc_list, ['primary_eclipse', 'secondary_eclipse'])):
            # true anomaly of conjunction (measured from periastron counter-clokwise)
            true_anomaly_of_conjuction = (alpha - self.argument_of_periastron) % const.FULL_ARC  # \nu_{con}

            # eccentric anomaly of conjunction (measured from apse line)
            eccentric_anomaly_of_conjunction = (2.0 * np.arctan(
                np.sqrt((1.0 - self.eccentricity) / (1.0 + self.eccentricity)) *
                np.tan(true_anomaly_of_conjuction / 2.0))) % const.FULL_ARC

            # mean anomaly of conjunction (measured from apse line)
            mean_anomaly_of_conjunction = (eccentric_anomaly_of_conjunction -
                                           self.eccentricity *
                                           np.sin(eccentric_anomaly_of_conjunction)) % const.FULL_ARC

            # true phase of conjunction (measured from apse line)
            true_phase_of_conjunction = (mean_anomaly_of_conjunction / const.FULL_ARC) % 1.0

            conjunction_quantities[idx] = dict()
            conjunction_quantities[idx]["true_anomaly"] = true_anomaly_of_conjuction
            conjunction_quantities[idx]["eccentric_anomaly"] = eccentric_anomaly_of_conjunction
            conjunction_quantities[idx]["mean_anomaly"] = mean_anomaly_of_conjunction
            conjunction_quantities[idx]["true_phase"] = true_phase_of_conjunction

        return conjunction_quantities

    @property
    def periastron_distance(self):
        """
        Return periastron distance.

        :return: float
        """
        return self._periastron_distance

    def compute_periastron_distance(self):
        """
        Calculates relative periastron distance in SMA units.

        :return: float
        """
        periastron_distance = self.relative_radius(true_anomaly=np.array([0])[0])
        self._logger.debug(f"setting property periastron_distance "
                           f"of class instance {self.__class__.__name__} to {periastron_distance}")
        return periastron_distance
