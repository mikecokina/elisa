from engine.body import Body
from engine.pulsations import PulsationMode
from engine import utils
from engine import const as c
from astropy import units as u
import numpy as np
import logging
from copy import copy
from scipy.special import sph_harm, lpmv
from scipy.optimize import brute, fmin

# temporary
from time import time


class Star(Body):

    # KWARGS = ['mass', 't_eff', 'vertices', 'faces', 'normals', 'temperatures', 'synchronicity', 'albedo',
    #           'polar_radius', 'surface_potential', 'backward_radius', 'gravity_darkening', 'polar_gravity_acceleration',
    #           'polar_log_g', 'equatorial_radius', 'spots', 'discretization_factor', 'pulsations']
    KWARGS = ['mass', 't_eff', 'gravity_darkening']

    OPTIONAL_KWARGS = ['surface_potential', 'polar_log_g', 'synchronicity', 'albedo', 'pulsations',
                       'discretization_factor', 'spots']
    ALL_KWARGS = KWARGS + OPTIONAL_KWARGS

    def __init__(self, name=None, **kwargs):
        utils.invalid_kwarg_checker(kwargs, Star.ALL_KWARGS, Star)
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
        self._potential_gradient_magnitudes = None
        self._polar_potential_gradient = None
        self._pulsations = None
        self._filling_factor = None
        self.kwargs = kwargs

        utils.check_missing_kwargs(Star.KWARGS, kwargs, instance_of=Star)

        # values of properties
        for kwarg in Star.ALL_KWARGS:
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
        :param potential_gradient_magnitude: float
        :return:
        """
        self._polar_potential_gradient = potential_gradient_magnitude

    @property
    def filling_factor(self):
        return self._filling_factor

    @filling_factor.setter
    def filling_factor(self, filling_factor):
        self._filling_factor = filling_factor

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
        if self.spots:  # temporary
            gradient_magnitudes = self.potential_gradient_magnitudes if gradient_magnitudes is None else \
                gradient_magnitudes
        else:
            gradient_magnitudes = self.potential_gradient_magnitudes[:self.base_symmetry_faces_number] if \
                gradient_magnitudes is None else gradient_magnitudes
        t_eff_polar = self.calculate_polar_effective_temperature()
        t_eff = t_eff_polar * np.power(gradient_magnitudes / self.polar_potential_gradient_magnitude,
                                       0.25 * self.gravity_darkening)

        return t_eff if self.spots else t_eff[self.face_symmetry_vector]

    def add_pulsations(self, points=None, faces=None, temperatures=None):
        """
        function returns temperature map with added temperature perturbations caused by pulsations

        :param points: np.array - if `None` star.points are used
        :param faces: np.array - if `None` star.faces are used
        :param temperatures: np.array - if `None` star.temperatures
        :return:
        """

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
            # generating spherical coordinate system in case of mode axis not identical with axis of rotation, new axis
            # is created by rotation of polar coordinates around z in positive direction by `mode_axis_phi' and than
            # rotating this coordinate system around y axis to tilt z axis by amount of `mode_axis_theta`
            if pulsation.mode_axis_phi == 0 and pulsation.mode_axis_theta == 0:
                phi, theta = centres[:, 1], centres[:, 2]
            else:
                phi_rot = (centres[:, 1] - pulsation.mode_axis_phi) % c.FULL_ARC  # rotation around Z
                # axis
                cos_phi = np.cos(phi_rot)
                sin_theta = np.sin(centres[:, 2])
                sin_axis_theta = np.sin(pulsation.mode_axis_theta)
                cos_theta = np.cos(centres[:, 2])
                # rotation around Y axis by `mode_axis_theta` angle
                cos_axis_theta = np.cos(pulsation.mode_axis_theta)
                theta = np.arccos(cos_phi * sin_theta * sin_axis_theta + cos_theta * cos_axis_theta)
                phi = np.arctan2(np.sin(phi_rot) * sin_theta, cos_phi * sin_theta * cos_axis_theta -
                                 cos_theta * sin_axis_theta)

            # generating of renormalised spherical harmonics (maximum value on sphere equuals to 1)
            constant = spherical_harmonics_renormalization_constant(pulsation.l, pulsation.m)
            spherical_harmonics = constant * np.real(sph_harm(pulsation.m, pulsation.l, phi, theta))

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
        self._logger.debug('Surface temperature map renormalized by a factor {0}'.format(coefficient))
        self.temperatures *= coefficient
        if self.spots:
            for spot_index, spot in self.spots.items():
                spot.temperatures *= coefficient

    @staticmethod
    def limb_darkening_factor(normal_vector=None, line_of_sight=None, coefficients=None, limb_darkening_law=None):
        """
        calculates limb darkening factor for given surface element given by radius vector and line of sight vector
        :param line_of_sight: numpy.array - vector (or vectors) of line of sight (normalized to 1 !!!)
        :param normal_vector: numpy.array - single or multiple normal vectors (normalized to 1 !!!)
        :param coefficients: np.float in case of linear law
                             np.array in other cases
        :param limb_darkening_law: str -  `linear` or `cosine`, `logarithmic`, `square_root`
        :return:  gravity darkening factor(s), the same type/shape as theta
        """
        if normal_vector is None:
            raise ValueError('Normal vector(s) was not supplied.')
        if line_of_sight is None:
            raise ValueError('Line of sight vector(s) was not supplied.')

        # if line_of_sight.ndim != 1 and normal_vector.ndim != line_of_sight.ndim:
        #     raise ValueError('`line_of_sight` should be either one vector or ther same amount of vectors as provided in'
        #                      ' radius vectors')

        if coefficients is None:
            raise ValueError('Limb darkening coefficients were not supplied.')
        elif limb_darkening_law is None:
            raise ValueError('Limb darkening rule was not supplied choose from: `linear` or `cosine`, `logarithmic`, '
                             '`square_root`.')
        elif limb_darkening_law in ['linear', 'cosine']:
            if not np.isscalar(coefficients):
                raise ValueError('Only one scalar limb darkening coefficient is required for linear cosine law. You '
                                 'used: {}'.format(coefficients))
        elif limb_darkening_law in ['logarithmic', 'square_root']:
            if not np.shape(coefficients) == (2,):
                raise ValueError('Invalid number of limb darkening coefficients. Expected 2, given: '
                                 '{}'.format(coefficients))

        cos_theta = np.sum(normal_vector * line_of_sight, axis=-1)
        if limb_darkening_law in ['linear', 'cosine']:
            return 1 - coefficients + coefficients * cos_theta
        elif limb_darkening_law == 'logarithmic':
            return 1 - coefficients[0] * (1 - cos_theta) - coefficients[1] * cos_theta * np.log(cos_theta)
        elif limb_darkening_law == 'square_root':
            return 1 - coefficients[0] * (1 - cos_theta) - coefficients[1] * (1 - np.sqrt(cos_theta))

    @staticmethod
    def calculate_bolometric_limb_darkening_factor(limb_darkening_law=None, coefficients=None):
        """
        Calculates limb darkening factor D(int) used when calculating flux from given intensity on surface.
        D(int) = integral over hemisphere (D(theta)cos(theta)

        :param limb_darkening_law: str -  `linear` or `cosine`, `logarithmic`, `square_root`
        :param coefficients: np.float in case of linear law
                             np.array in other cases
        :return: float - bolometric_limb_darkening_factor (scalar for the whole star)
        """
        if coefficients is None:
            raise ValueError('Limb darkening coefficients were not supplied.')
        elif limb_darkening_law is None:
            raise ValueError('Limb darkening rule was not supplied choose from: `linear` or `cosine`, `logarithmic`, '
                             '`square_root`.')
        elif limb_darkening_law in ['linear', 'cosine']:
            if not np.isscalar(coefficients):
                raise ValueError('Only one scalar limb darkening coefficient is required for linear cosine law. You '
                                 'used: {}'.format(coefficients))
        elif limb_darkening_law in ['logarithmic', 'square_root']:
            if not np.shape(coefficients) == (2,):
                raise ValueError('Invalid number of limb darkening coefficients. Expected 2, given: '
                                 '{}'.format(coefficients))

        if limb_darkening_law in ['linear', 'cosine']:
            return np.pi * (1 - coefficients / 3)
        elif limb_darkening_law == 'logarithmic':
            return np.pi * (1 - coefficients[0] / 3 + 2 * coefficients[1] / 9)
        elif limb_darkening_law == 'square_root':
            return np.pi * (1 - coefficients[0] / 3 - coefficients[1] / 5)

