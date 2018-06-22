from engine.body import Body
from engine.spot import Spot
from engine.pulsations import PulsationMode
from engine import utils
from astropy import units as u
import numpy as np
import logging
from copy import copy
from scipy.special import sph_harm, lpmv
from scipy.optimize import brute, fmin

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : [%(levelname)s] : %(name)s : %(message)s')


class Star(Body):

    KWARGS = ['mass', 't_eff', 'vertices', 'faces', 'normals', 'temperatures', 'synchronicity', 'albedo',
              'polar_radius', 'surface_potential', 'backward_radius', 'gravity_darkening', 'polar_gravity_acceleration',
              'polar_log_g', 'equatorial_radius', 'spots', 'discretization_factor', 'pulsations']

    def __init__(self, name=None, **kwargs):
        self.is_property(kwargs)
        super(Star, self).__init__(name=name, **kwargs)

        # get logger
        self._logger = logging.getLogger(Star.__name__)

        # default values of properties
        self._surface_potential = None
        self._backward_radius = None
        self._polar_radius = None
        self._gravity_darkening = None
        self._synchronicity = None
        self._forward_radius = None
        self._side_radius = None
        self._polar_gravity_acceleration = None
        self._polar_log_g = None
        self._equatorial_radius = None
        self._critical_surface_potential = None
        self._spots = None
        self._potential_gradient_magnitudes = None
        self._polar_potential_gradient = None
        self._pulsations = None

        # values of properties
        for kwarg in Star.KWARGS:
            if kwarg in kwargs:
                self._logger.debug("Setting property {} "
                                   "of class instance {} to {}".format(kwarg, Star.__name__, kwargs[kwarg]))
                setattr(self, kwarg, kwargs[kwarg])

    @property
    def pulsations(self):
        return self._pulsations

    @pulsations.setter
    def pulsations(self, pulsations):
        if pulsations:
            self._pulsations = {idx: PulsationMode(**pulsation_meta) for idx, pulsation_meta in enumerate(pulsations)}

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
    def polar_radius(self):
        """
        returns value of polar radius of an object in default unit returns None if it doesn't exist
        usage: xy.polar_radius

        :return: float64
        """
        return self._polar_radius

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
    def potential_gradient_magnitudes(self):
        """
        returns array of absolute values of potential gradients for each face of surface

        :return: numpy.array
        """
        return self._potential_gradient_magnitudes

    @potential_gradient_magnitudes.setter
    def potential_gradient_magnitudes(self, potential_gradient_magnitudes):
        """
        :param potential_gradient_magnitudes: np.array
        :return:
        """
        self._potential_gradient_magnitudes = potential_gradient_magnitudes

    @property
    def polar_potential_gradient_magnitude(self):
        """
        returns array of absolute value of polar potential gradient

        :return: float
        """
        return self._polar_potential_gradient

    @polar_potential_gradient_magnitude.setter
    def polar_potential_gradient_magnitude(self, potential_gradient_magnitude):
        """
        :param potential_gradient: float
        :return:
        """
        self._polar_potential_gradient = potential_gradient_magnitude

    def calculate_polar_effective_temperature(self):
        """
        returns polar effective temperature

        :return: float
        """
        return self.t_eff * np.power(np.sum(self.areas) /
                                     np.sum(self.areas * np.power(self.potential_gradient_magnitudes /
                                                                  self.polar_potential_gradient_magnitude,
                                                                  self.gravity_darkening)),
                                     0.25)

    def calculate_effective_temperatures(self, gradient_magnitudes=None):
        """
        calculates effective temperatures for given gradient magnitudes, if None given star surface t_effs are
        calculated

        :param gradient_magnitudes:
        :return:
        """
        gradient_magnitudes = self.potential_gradient_magnitudes if gradient_magnitudes is None else gradient_magnitudes
        t_eff_polar = self.calculate_polar_effective_temperature()
        t_eff_points = t_eff_polar * np.power(gradient_magnitudes / self.polar_potential_gradient_magnitude,
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

    def add_pulsations(self, points=None, faces=None, temperatures=None):

        def alp(x, *args):
            """
            returns negative value from imaginary value of associated Legendre polynomial (ALP), used in minimizer to
            find global maximum of real part of spherical harmonics

            :param x: float - argument of function
            :param args: l - order of ALP
                         m - degree of ALP
            :return: float - negative of absolute value of ALP
            """
            l, m = args
            return -abs(lpmv(m, l, x))
            # return -abs(np.real(sph_harm(m, l, x[0], x[1])))

        def spherical_harmonics_renormalization_constant(l, m):
            old_settings = np.seterr(divide='ignore', invalid='ignore', over='ignore')
            Ns = int(np.power(5, np.ceil((l-m)/23))*((l-m)+1))
            output = brute(alp, ranges=((0.0, 1.0),), args=(l, m), Ns=Ns, finish=fmin, full_output=True)
            np.seterr(**old_settings)

            x = output[2][np.argmin(output[3])] if not 0 <= output[0] <= 1 else output[0]
            result = abs(np.real(sph_harm(m, l, 0, np.arccos(x))))
            return 1. / result

        if points is not None:
            self.points
            if faces is None or temperatures is None:
                raise ValueError('`points` argument is not None but `faces` or `temperature` is. Please supply the '
                                 'missing keyword arguments')
        else:
            points = copy(self.points)
            faces = copy(self.faces)
            temperatures = copy(self.temperatures)

        surface_centers = self.calculate_surface_centres(points, faces)
        centres = utils.cartesian_to_spherical(surface_centers)

        for pulsation_index, pulsation in self.pulsations.items():
            # generating of renormalised spherical harmonics (maximum value on sphere equuals to 1)
            constant = spherical_harmonics_renormalization_constant(pulsation.l, pulsation.m)
            spherical_harmonics = constant * np.real(sph_harm(pulsation.m, pulsation.l, centres[:, 1], centres[:, 2]))

            temperatures += pulsation.amplitude * spherical_harmonics

        return temperatures

    def renormalize_temperatures(self):
        # no need to calculate surfaces they had to be calculated already, otherwise there is nothing to renormalize
        total_surface = np.sum(self.areas)
        if self.spots:
            for spot_index, spot in self.spots.items():
                total_surface += np.sum(spot.areas)
        desired_flux_value = total_surface * self.t_eff

        current_flux = np.sum(self.areas * self.temperatures)
        if self.spots:
            for spot_index, spot in self.spots.items():
                current_flux += np.sum(spot.areas * spot.temperatures)

        coefficient = np.power(desired_flux_value / current_flux, 0.25)
        self.temperatures *= coefficient
        if self.spots:
            for spot_index, spot in self.spots.items():
                spot.temperatures *= coefficient