from engine.body import Body
from engine.spot import Spot
from astropy import units as u
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : [%(levelname)s] : %(name)s : %(message)s')


class Star(Body):

    KWARGS = ['mass', 't_eff', 'vertices', 'faces', 'normals', 'temperatures', 'synchronicity', 'albedo',
              'polar_radius', 'surface_potential', 'backward_radius', 'gravity_darkening', 'polar_gravity_acceleration',
              'polar_log_g', 'equatorial_radius', 'spots', 'discretization_factor']

    def __init__(self, name=None, **kwargs):
        self.is_property(kwargs)
        super(Star, self).__init__(name=name, **kwargs)

        # get logger
        self._logger = logging.getLogger(Star.__name__)

        # default values of properties
        self._surface_potential = None
        self._backward_radius = None
        self._gravity_darkening = None
        self._synchronicity = None
        self._forward_radius = None
        self._side_radius = None
        self._polar_gravity_acceleration = None
        self._polar_log_g = None
        self._equatorial_radius = None
        self._critical_surface_potential = None
        self._spots = None
        self._potential_gradients = None
        self._polar_potential_gradient = None

        # values of properties
        for kwarg in Star.KWARGS:
            if kwarg in kwargs:
                self._logger.debug("Setting property {} "
                                   "of class instance {} to {}".format(kwarg, Star.__name__, kwargs[kwarg]))
                setattr(self, kwarg, kwargs[kwarg])

    # def _add_spot(self, spot):
    #     self._spots = spot if isinstance(spot, Spot) and not self._spots else [self._spots]
    #
    def remove_spot(self, spot_index):
        del(self._spots[spot_index])

    @property
    def spots(self):
        return self._spots

    @spots.setter
    def spots(self, spots):
        # initialize spots dataframes
        if spots:
            self._spots = {idx: Spot(**spot_meta) for idx, spot_meta in enumerate(spots)}

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

        :return: float
        """
        return self._gravity_darkening

    @gravity_darkening.setter
    def gravity_darkening(self, gravity_darkening):
        """
        setter for gravity darkening
        accepts values of gravity darkening in range (0, 1)

        :param gravity_darkening: float
        """
        if 0 <= gravity_darkening <= 1:
            self._gravity_darkening = np.float64(gravity_darkening)
        else:
            raise ValueError('Parameter gravity darkening = {} is out of range (0, 1)'.format(gravity_darkening))

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

    @property
    def potential_gradients(self):
        """
        returns array of absolute values of potential gradients for each face of surface

        :return: numpy.array
        """
        return self._potential_gradients

    @potential_gradients.setter
    def potential_gradients(self, potential_gradients):
        """
        :param potential_gradients: np.array
        :return:
        """
        self._potential_gradients = potential_gradients

    @property
    def polar_potential_gradient(self):
        """
        returns array of absolute value of polar potential gradient

        :return: float
        """
        return self._polar_potential_gradient

    @polar_potential_gradient.setter
    def polar_potential_gradient(self, potential_gradient):
        """
        :param potential_gradient: float
        :return:
        """
        self._polar_potential_gradient = potential_gradient

    def calculate_polar_effective_temperature(self):
        """
        returns polar effective temperature

        :return: float
        """
        return self.t_eff * np.power(np.sum(self.areas) /
                                     np.sum(self.areas * np.power(self.potential_gradients /
                                                                  self.polar_potential_gradient,
                                                                  self.gravity_darkening)),
                                     0.25)

    def calculate_effective_temperatures(self, gradient_magnitudes=None):
        """
        calculates effective temperatures for given gradient magnitudes, if None given star surface t_effs are
        calculated

        :param gradient_magnitudes:
        :return:
        """
        gradient_magnitudes = self.potential_gradients if gradient_magnitudes is None else gradient_magnitudes
        t_eff_polar = self.calculate_polar_effective_temperature()
        t_eff_points = t_eff_polar * np.power(gradient_magnitudes / self.polar_potential_gradient,
                                              0.25 * self.gravity_darkening)
        return t_eff_points

    def is_property(self, kwargs):
        """
        method for checking if keyword arguments are valid properties of this class

        :param kwargs: dict
        :return:
        """
        is_not = ['`{}`'.format(k) for k in kwargs if k not in dir(self)]
        if is_not:
            raise AttributeError('Arguments {} are not valid {} properties.'.format(', '.join(is_not), Star.__name__))
