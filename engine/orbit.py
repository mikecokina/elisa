import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : [%(levelname)s] : %(name)s : %(message)s')

class Orbit(object):

    KWARGS = ['period', 'inclination', 'eccentricity', 'periastron']

    def __init__(self, **kwargs):
        self.is_property(kwargs)
        self._logger = logging.getLogger(Orbit.__name__)

        self._period = None
        self._inclination = None
        self._eccentricity = None
        self._periastron = None

        # values of properties
        for kwarg in Orbit.KWARGS:
            if kwarg in kwargs:
                self._logger.debug("Setting property {} "
                                   "of class instance {} to {}".format(kwarg, Orbit.__name__, kwargs[kwarg]))
                setattr(self, kwarg, kwargs[kwarg])

    @property
    def period(self):
        return self._period

    @period.setter
    def period(self, period):
        self._period = period

    @property
    def inclination(self):
        return self._inclination

    @inclination.setter
    def inclination(self, inclination):
        self._inclination = inclination

    @property
    def eccentricity(self):
        return self._eccentricity

    @eccentricity.setter
    def eccentricity(self, eccentricity):
        self._eccentricity = eccentricity

    @property
    def periastron(self):
        return self._periastron

    @periastron.setter
    def periastron(self, periastron):
        self._periastron = periastron

    @classmethod
    def is_property(cls, kwargs):
        is_not = ['`{}`'.format(k) for k in kwargs if k not in cls.KWARGS]
        if is_not:
            raise AttributeError('Arguments {} are not valid {} properties.'.format(', '.join(is_not), cls.__name__))

