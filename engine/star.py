from engine.body import Body
from astropy import units as u
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : [%(levelname)s] : %(name)s : %(message)s')


class Star(Body):

    KWARGS = ['mass', 't_eff', 'vertices', 'faces', 'normals', 'temperatures', 'synchronicity', 'albedo',
              'polar_radius', 'surface_potential', 'backward_radius', 'gravity_darkening', 'polar_gravity_acceleration',
              'polar_log_g', 'equatorial_radius']

    def __init__(self, name=None, **kwargs):
        # get logger
        self._logger = logging.getLogger(Star.__name__)

        self.is_property(kwargs)
        super(Star, self).__init__(name=name, **kwargs)

        # default values of properties
        self._surface_potential = None
        self._backward_radius = None
        self._gravity_darkening = None
        self._synchronicity = None
        self._forward_radius = None
        self._side_radius = None
        self._polar_radius = None
        self._polar_gravity_acceleration = None
        self._polar_log_g = None
        self._equatorial_radius = None
        self._critical_surface_potential = None

        # values of properties
        for kwarg in Star.KWARGS:
            if kwarg in kwargs:
                self._logger.debug("Setting property {} "
                                   "of class instance {} to {}".format(kwarg, Star.__name__, kwargs[kwarg]))
                setattr(self, kwarg, kwargs[kwarg])

    @property
    def critical_surface_potential(self):
        return self._critical_surface_potential

    @critical_surface_potential.setter
    def critical_surface_potential(self, potential):
        self._critical_surface_potential = potential

    @property
    def surface_potential(self):
        """
        returns surface potential of Star
        usage: xy.Star

        :return: float64
        """
        return self._surface_potential

    @surface_potential.setter
    def surface_potential(self, potential):
        """
        setter for surface potential
        usage: xy.surface_potential = new_potential

        :param potential: float64
        """
        self._surface_potential = np.float64(potential)

    @property
    def backward_radius(self):
        """
        returns value of backward radius of an object in default unit
        usage: xy.backward_radius

        :return: float64
        """
        return self._backward_radius

    @property
    def forward_radius(self):
        """
        returns value of forward radius of an object in default unit returns None if it doesn't exist
        usage: xy.forward_radius

        :return: float64
        """
        return self._forward_radius

    @property
    def side_radius(self):
        """
        returns value of side radius of an object in default unit
        usage: xy.side_radius

        :return: float64
        """
        return self._side_radius

    @property
    def gravity_darkening(self):
        """
        returns gravity darkening
        usage: xy.gravity_darkening

        :return: float64
        """
        return self._backward_radius

    @gravity_darkening.setter
    def gravity_darkening(self, gravity_darkening):
        """
        setter for gravity darkening
        accepts values of gravity darkening in range (0, 1)

        :param gravity_darkening: float64
        """
        if 0 <= gravity_darkening <= 1:
            self._gravity_darkening = np.float64(gravity_darkening)
        else:
            raise ValueError('Parameter gravity darkening = {} is out of range (0, 1)'.format(gravity_darkening))

    @property
    def polar_radius(self):
        """
        returns polar radius in default units

        :return: float
        """
        return self._polar_radius

    @property
    def equatorial_radius(self):
        """
        returns equatorial radius in default units

        :return: float
        """
        return self._equatorial_radius

    @property
    def polar_gravity_acceleration(self):
        """
        returns polar gravity acceleration in default units

        :return: float
        """
        return self._polar_gravity_acceleration

    @property
    def polar_log_g(self):
        """
        returns logarythm of polar surface gravity in SI

        :return: float
        """
        return self._polar_log_g

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
