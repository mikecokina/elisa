from engine.system import System
from engine.orbit import Orbit
from astropy import units as u
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : [%(levelname)s] : %(name)s : %(message)s')


class BinarySystem(System):

    KWARGS = ['gamma', 'inclination', 'period', 'eccentricity', 'periastron']

    def __init__(self, name=None, **kwargs):
        self.is_property(kwargs)
        super(BinarySystem, self).__init__(name=name, **kwargs)

        # get logger
        self._logger = logging.getLogger(BinarySystem.__name__)

        # default values of properties
        self._inclination = None
        self._period = None
        self._eccentricity = None
        self._periastron = None
        self._orbit = None

        # orbit initialisation
        self.init_orbit()

        # values of properties
        for kwarg in BinarySystem.KWARGS:
            if kwarg in kwargs:
                self._logger.debug("Setting property {} "
                                   "of class instance {} to {}".format(kwarg, BinarySystem.__name__, kwargs[kwarg]))
                setattr(self, kwarg, kwargs[kwarg])

    def init_orbit(self):
        self._logger.debug("Re/Initializing orbit in class instance {} ".format(BinarySystem.__name__))
        orbit_kwargs = {key: getattr(self, key) for key in Orbit.KWARGS}
        self._orbit = Orbit(**orbit_kwargs)

    @property
    def orbit(self):
        return self._orbit

    @property
    def period(self):
        """
        binarys system period

        :return: (np.)int, (np.)float, astropy.unit.quantity.Quantity
        """
        return self._period

    @period.setter
    def period(self, period):
        """
        set orbital period of bonary star system, if unit is not specified, default unit is assumed

        :param period: (np.)int, (np.)float, astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(period, u.quantity.Quantity):
            self._period = np.float64(period.to(self.get_period_unit()))
        elif isinstance(period, (int, np.int, float, np.float)):
            self._period = np.float64(period)
        else:
            raise TypeError('Input of variable `period` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')
        self.orbit._period = self._period
        self._logger.debug("Setting property period "
                           "of class instance {} and {} to {}".format(BinarySystem.__name__, Orbit.__name__, self._period))

    @property
    def inclination(self):
        """
        inclination of binary star system

        :return: (np.)int, (np.)float, astropy.unit.quantity.Quantity
        """
        return self._inclination

    @inclination.setter
    def inclination(self, inclination):
        """
        set orbit inclination of binary star system, if unit is not specified, default unit is assumed

        :param inclination: (np.)int, (np.)float, astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(inclination, u.quantity.Quantity):
            self._inclination = np.float64(inclination.to(self.get_arc_unit()))
        elif isinstance(inclination, (int, np.int, float, np.float)):
            self._inclination = np.float64(inclination)
        else:
            raise TypeError('Input of variable `inclination` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')
        self.orbit._inclination = self._inclination
        self._logger.debug("Setting property inclination "
                           "of class instance {} and {} to {}".format(BinarySystem.__name__, Orbit.__name__, self._inclination))

    @property
    def eccentricity(self):
        """
        eccentricity of orbit of binary star system

        :return: (np.)int, (np.)float
        """
        return self._eccentricity

    @eccentricity.setter
    def eccentricity(self, eccentricity):
        """
        set eccentricity

        :param eccentricity: (np.)int, (np.)float
        :return:
        """
        if eccentricity < 0 or eccentricity > 1 or not isinstance(eccentricity, (int, np.int, float, np.float)):
            raise TypeError('Input of variable `eccentricity` is not (np.)int or (np.)float or it is out of boundaries.')
        self._eccentricity = eccentricity
        self.orbit._eccentricity = self._eccentricity
        self._logger.debug("Setting property eccentricity "
                           "of class instance {} and {} to {}".format(BinarySystem.__name__, Orbit.__name__, self._eccentricity))

    @property
    def periastron(self):
        """
        argument of periastron

        :return: (np.)int, (np.)float, astropy.unit.quantity.Quantity
        """
        return self._periastron

    @periastron.setter
    def periastron(self, periastron):
        """
        set argumnt of periastron

        :param periastron: (np.)int, (np.)float, astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(periastron, u.quantity.Quantity):
            self._periastron = np.float64(periastron.to(self.get_arc_unit()))
        elif isinstance(periastron, (int, np.int, float, np.float)):
            self._periastron = np.float64(periastron)
        else:
            raise TypeError('Input of variable `periastron` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')
        self.orbit.periastron = self.periastron
        self._logger.debug("Setting property periastron "
                           "of class instance {} and {} to {}".format(BinarySystem.__name__, Orbit.__name__, self._periastron))

    def compute_lc(self):
        pass

    def get_info(self):
        pass

    @classmethod
    def is_property(cls, kwargs):
        is_not = ['`{}`'.format(k) for k in kwargs if k not in cls.KWARGS]
        if is_not:
            raise AttributeError('Arguments {} are not valid {} properties.'.format(', '.join(is_not), cls.__name__))
