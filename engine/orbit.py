import logging
import numpy as np
from engine import const as c
from astropy import units as u
import engine.units as U
from engine import utils

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : [%(levelname)s] : %(name)s : %(message)s')


class Orbit(object):

    KWARGS = ['period', 'inclination', 'eccentricity', 'argument_of_periastron']
    OPTIONAL_KWARGS = []
    ALL_KWARGS = KWARGS + OPTIONAL_KWARGS

    def __init__(self, **kwargs):
        utils.invalid_kwarg_checker(kwargs, Orbit.ALL_KWARGS, Orbit)
        self._logger = logging.getLogger(Orbit.__name__)

        # default valeus of properties
        self._period = None
        self._inclination = None
        self._eccentricity = None
        self._argument_of_periastron = None
        self._periastron_distance = None
        self._perastron_phase = None

        # values of properties
        for kwarg in Orbit.KWARGS:
            if kwarg in kwargs:
                if kwargs[kwarg] is not None:
                    self._logger.debug("Setting property {} "
                                       "of class instance {} to {}".format(kwarg, Orbit.__name__, kwargs[kwarg]))
                    setattr(self, kwarg, kwargs[kwarg])

        self._periastron_distance = self.compute_periastron_distance()
        self._perastron_phase = - self.get_conjuction()["primary_eclipse"]["true_phase"] % 1

    @property
    def periastron_phase(self):
        """
        photometric phase of periastron
        :return:
        """
        return self._perastron_phase

    @property
    def period(self):
        """
        returns orbital period of the binary system in default period unit

        :return: numpy.float
        """
        return self._period

    @period.setter
    def period(self, period):
        """
        setter for orbital period of binary system orbit

        :param period: np.float
        :return:
        """
        self._period = period

    @property
    def inclination(self):
        """
        returns inclination of binary system orbit

        :return: numpy.float
        """
        return self._inclination

    @inclination.setter
    def inclination(self, inclination):
        """
        setter for inclination of binary system orbit

        :param inclination: numpy.float
        :return:
        """
        if isinstance(inclination, u.quantity.Quantity):
            self._inclination = np.float64(inclination.to(U.ARC_UNIT))
        elif isinstance(inclination, (int, np.int, float, np.float)):
            self._inclination = np.float64(inclination)
        else:
            raise TypeError('Input of variable `inclination` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

        if not 0 <= self.inclination <= c.PI:
            raise ValueError('Eccentricity value of {} is out of bounds (0, pi).'.format(self.inclination))

    @property
    def eccentricity(self):
        """
        returns eccentricity of binary system orbit

        :return: numpy.float
        """
        return self._eccentricity

    @eccentricity.setter
    def eccentricity(self, eccentricity):
        """
        setter for eccentricity of binary system orbit

        :param eccentricity: numpy.float
        :return:
        """
        self._eccentricity = eccentricity

    @property
    def argument_of_periastron(self):
        """
        returns argument of periastron of binary system orbit

        :return: numpy.float
        """
        return self._argument_of_periastron

    @argument_of_periastron.setter
    def argument_of_periastron(self, argument_of_periastron):
        """
        setter for argument of periastron of binary system orbit

        :param argument_of_periastron: numpy.float
        :return:
        """
        if isinstance(argument_of_periastron, u.quantity.Quantity):
            self._argument_of_periastron = np.float64(argument_of_periastron.to(U.ARC_UNIT))
        elif isinstance(argument_of_periastron, (int, np.int, float, np.float)):
            self._argument_of_periastron = np.float64(argument_of_periastron)
        else:
            raise TypeError('Input of variable `argument_of_periastron` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')
        if not 0 <= self._argument_of_periastron <= c.FULL_ARC:
            self._argument_of_periastron %= c.FULL_ARC

    @classmethod
    def true_phase(cls, phase=None, phase_shift=None):
        """
        returns shifted phase of the orbit by the amount phase_shift

        :param phase: numpy.array
        :param phase_shift: numpy.float
        :return:
        """
        return phase + phase_shift

    @classmethod
    def mean_anomaly(cls, phase=None):
        """
        returns mean anomaly of points on orbit described by phase

        :param phase: numpy.array
        :return: numpy.array
        """
        return c.FULL_ARC * phase

    def mean_anomaly_fn(self, eccentric_anomaly=None, *args):
        """
        definition of Kepler equation for scipy _solver in Orbit.eccentric_anomaly

        :param eccentric_anomaly: numpy.float
        :param args: mean_anomaly: numpy.float
        :return: numpy.float
        """
        mean_anomaly, = args
        return (eccentric_anomaly - self.eccentricity * np.sin(eccentric_anomaly)) - mean_anomaly

    def eccentric_anomaly(self, mean_anomaly=None):
        """
        solves Kepler equation for eccentric anomaly via mean anomaly

        :param mean_anomaly: numpy.float
        :return: numpy.float
        """
        import scipy.optimize
        try:
            solution = scipy.optimize.newton(self.mean_anomaly_fn, 1.0, args=(mean_anomaly,),
                                             tol=1e-10)
            if not np.isnan(solution):
                if solution < 0:
                    solution += c.FULL_ARC
                return solution
            else:
                return False
        except Exception as e:
            self._logger.debug("Solver scipy.optimize.newton in function Orbit.eccentric_anomaly did not provide "
                               "solution.\n Reason: {}".format(e))
            return False

    def true_anomaly(self, eccentric_anomaly=None):
        """
        returns true anomaly as a function of eccentric anomaly and eccentricity

        :param eccentric_anomaly: numpy.array
        :return: numpy.array
        """
        true_anomaly = 2.0 * np.arctan(
            np.sqrt((1.0 + self.eccentricity) / (1.0 - self.eccentricity)) * np.tan(eccentric_anomaly / 2.0))
        true_anomaly[true_anomaly < 0] += c.FULL_ARC
        return true_anomaly

    def relative_radius(self, true_anomaly=None):
        """
        calculates the length of radius vector of elipse where a=1

        :param true_anomaly: numpy.array
        :return: numpy.array
        """
        return (1.0 - self.eccentricity ** 2) / (1.0 + self.eccentricity * np.cos(true_anomaly))

    def true_anomaly_to_azimuth(self, true_anomaly=None):
        azimut = true_anomaly + self.argument_of_periastron
        azimut %= c.FULL_ARC
        return azimut

    def orbital_motion(self, phase=None):
        """
        function takes photometric phase of the binary system as input and calculates positions of the secondary
        component in the frame of reference of primary component

        :param phase: np.array or np.float
        :return: np.array: matrix consisting of column stacked vectors distance, azimut angle, true anomaly and phase
                           np.array((r1, az1, ni1, phs1),
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
        compute and return photometric phase of conjunction (eclipses)

        we assume that primary component is situated in center of coo system and observation unit
        vector is [-1, 0, 0]

        return dictionary is in shape {type_of_eclipse: {'true_phase': ,
                                                         'true_anomaly': ,
                                                         'mean_anomaly': ,
                                                         eccentric_anomaly: }, ...}

        :return: dict(dict)
        """
        # determining order of eclipses
        conjuction_arc_list = []
        try:
            if 0 <= self.inclination <= c.PI / 2.0:
                conjuction_arc_list = [c.PI / 2.0, 3.0 * c.PI / 2.0]
            elif c.PI / 2.0 < self.inclination <= c.PI:
                conjuction_arc_list = [3.0 * c.PI / 2.0, c.PI / 2.0]
        except:
            raise TypeError('Invalid type of {0}.inclination.'.format(Orbit.__name__))

        conjunction_quantities = {}
        for alpha, idx in list(zip(conjuction_arc_list, ['primary_eclipse', 'secondary_eclipse'])):
            # true anomaly of conjunction (measured from periastron counter-clokwise)
            true_anomaly_of_conjuction = (alpha - self.argument_of_periastron) % c.FULL_ARC  # \nu_{con}

            # eccentric anomaly of conjunction (measured from apse line)
            eccentric_anomaly_of_conjunction = (2.0 * np.arctan(
                np.sqrt((1.0 - self.eccentricity) / (1.0 + self.eccentricity)) *
                np.tan(true_anomaly_of_conjuction / 2.0))) % c.FULL_ARC

            # mean anomaly of conjunction (measured from apse line)
            mean_anomaly_of_conjunction = (eccentric_anomaly_of_conjunction -
                                           self.eccentricity * np.sin(eccentric_anomaly_of_conjunction)) % c.FULL_ARC

            # true phase of conjunction (measured from apse line)
            true_phase_of_conjunction = (mean_anomaly_of_conjunction / c.FULL_ARC) % 1.0

            conjunction_quantities[idx] = {}
            conjunction_quantities[idx]["true_anomaly"] = true_anomaly_of_conjuction
            conjunction_quantities[idx]["eccentric_anomaly"] = eccentric_anomaly_of_conjunction
            conjunction_quantities[idx]["mean_anomaly"] = mean_anomaly_of_conjunction
            conjunction_quantities[idx]["true_phase"] = true_phase_of_conjunction

        return conjunction_quantities

    @property
    def periastron_distance(self):
        """
        return periastron distance

        :return:
        """
        return self._periastron_distance

    def compute_periastron_distance(self):
        """
        calculates relative periastron distance in SMA units

        :return: float
        """
        periastron_distance = self.relative_radius(true_anomaly=np.array([0])[0])
        self._logger.debug("Setting property {} "
                           "of class instance {} to {}".format('periastron_distance', Orbit.__name__,
                                                               periastron_distance))
        return periastron_distance
