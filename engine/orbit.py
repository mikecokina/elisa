import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : [%(levelname)s] : %(name)s : %(message)s')


class Orbit(object):

    KWARGS = ['period', 'inclination', 'eccentricity', 'periastron', 'phase_shift']

    def __init__(self, **kwargs):
        self.is_property(kwargs)
        self._logger = logging.getLogger(Orbit.__name__)

        # default valeus of properties
        self._period = None
        self._inclination = None
        self._eccentricity = None
        self._periastron = None
        self._phase_shift = None

        # values of properties
        for kwarg in Orbit.KWARGS:
            if kwarg in kwargs:
                if kwargs[kwarg] is not None:
                    self._logger.debug("Setting property {} "
                                       "of class instance {} to {}".format(kwarg, Orbit.__name__, kwargs[kwarg]))
                    setattr(self, kwarg, kwargs[kwarg])

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
        self._logger.debug("Setting property period "
                           "of class instance {} to {}".format(Orbit.__name__, self._period))

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
        self._inclination = inclination
        self._logger.debug("Setting property inclination "
                           "of class instance {} to {}".format(Orbit.__name__, self._inclination))

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
        self._logger.debug("Setting property eccentricity "
                           "of class instance {} to {}".format(Orbit.__name__, self._eccentricity))

    @property
    def periastron(self):
        """
        returns argument of periastron of binary system orbit

        :return: numpy.float
        """
        return self._periastron

    @periastron.setter
    def periastron(self, periastron):
        """
        setter for argument of periastron of binary system orbit

        :param periastron: numpy.float
        :return:
        """
        self._periastron = periastron
        self._logger.debug("Setting property periastron "
                           "of class instance {} to {}".format(Orbit.__name__, self._periastron))

    @property
    def phase_shift(self):
        """
        returns phase shift of the primary eclipse minimum with respect to ephemeris
        true_phase is used during calculations, where: true_phase = phase + phase_shift

        :return: numpy.float
        """
        return self._phase_shift

    @phase_shift.setter
    def phase_shift(self, phase_shift):
        """
        setter for phase shift of the primary eclipse minimum with respect to ephemeris
        this will cause usage of true_phase during calculations, true_phase = phase + phase_shift

        :param phase_shift: numpy.float
        :return:
        """
        self._phase_shift = phase_shift
        self._logger.debug("Setting property phase_shift "
                           "of class instance {} to {}".format(Orbit.__name__, self._phase_shift))

    @classmethod
    def is_property(cls, kwargs):
        """
        method for checking if keyword arguments are valid properties of this class

        :param kwargs: dict
        :return:
        """
        is_not = ['`{}`'.format(k) for k in kwargs if k not in cls.KWARGS]
        if is_not:
            raise AttributeError('Arguments {} are not valid {} properties.'.format(', '.join(is_not), cls.__name__))

    @classmethod
    def true_phase(cls, phase, phase_shift):
        """
        returns shifted phase of the orbit by the amount phase_shift

        :param phase: numpy.array
        :param phase_shift: numpy.float
        :return:
        """
        return phase + phase_shift

    @classmethod
    def mean_anomaly(cls, phase):
        """
        returns mean anomaly of points on orbit described by phase

        :param phase: numpy.array
        :return: numpy.array
        """
        return 2.0 * np.pi * phase

    @classmethod
    def mean_anomaly_fn(cls, eccentric_anomaly, *args):
        """
        definition of Kepler eq+uation for scipy solver

        :param eccentric_anomaly: numpy.arrray
        :param args: mean_anomally: numpy_array
                    eccentricity: numpy.float
        :return: numpy.array
        """
        mean_anomaly, eccentricity = args
        return (eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly)) - mean_anomaly

    # @classmethod
    def orbital_motion(cls, phase):
        true_phase = cls.true_phase(phase, cls.phase_shift) # ked je pred funkciu @classmethod cls.phase_shift vracia objekt a nie hodnotu
        mean_anomaly = cls.mean_anomaly(phase)
        # return position
