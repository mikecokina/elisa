import numpy as np

from elisa.engine.base.body import Body
from elisa.engine.pulsations import PulsationMode
from elisa.engine import utils, logger
from elisa.engine import const as c
from copy import copy
from scipy.special import sph_harm, lpmv
from scipy.optimize import brute, fmin
from astropy import units as u
from elisa.engine import units as U


class Star(Body):
    KWARGS = ['mass', 't_eff', 'gravity_darkening']

    OPTIONAL_KWARGS = ['surface_potential', 'synchronicity', 'albedo', 'pulsations',
                       'discretization_factor', 'spots', 'metallicity', 'polar_log_g']
    ALL_KWARGS = KWARGS + OPTIONAL_KWARGS

    # this will be removed after full implementation of config system
    ATMOSPHERE_MODEL = 'black_body'

    def __init__(self, name=None, suppress_logger=False, **kwargs):
        utils.invalid_kwarg_checker(kwargs, Star.ALL_KWARGS, Star)
        super(Star, self).__init__(name=name, **kwargs)

        # get logger
        self._logger = logger.getLogger(self.__class__.__name__, suppress=suppress_logger)

        # default values of properties
        self._surface_potential = None
        self._backward_radius = None
        self._polar_radius = None
        self._gravity_darkening = None
        self._synchronicity = None
        self._forward_radius = None
        self._side_radius = None
        self._polar_log_g = None
        self._log_g = None
        self._equatorial_radius = None
        self._critical_surface_potential = None
        self._potential_gradient_magnitudes = None
        self._polar_potential_gradient_magnitude = None
        self._pulsations = None
        self._filling_factor = None
        self._metallicity = None
        self.kwargs = kwargs

        utils.check_missing_kwargs(Star.KWARGS, kwargs, instance_of=Star)

        # values of properties
        for kwarg in Star.ALL_KWARGS:
            if kwarg in kwargs:
                self._logger.debug(f"setting property {kwarg} of class instance "
                                   f"{self.__class__.__name__} to {kwargs[kwarg]}")
                setattr(self, kwarg, kwargs[kwarg])

    @property
    def polar_log_g(self):
        """
        returns logarithm of polar surface gravity in SI

        :return: float
        """
        return self._polar_log_g

    @polar_log_g.setter
    def polar_log_g(self, polar_log_g):
        """
        setter for polar surface gravity, if unit is not specified in astropy.units format, value in m/s^2 is assumed

        :param log_g:
        :return:
        """
        if isinstance(polar_log_g, u.quantity.Quantity):
            self._polar_log_g = np.float64(polar_log_g.to(U.LOG_ACCELERATION_UNIT))
        elif isinstance(polar_log_g, (int, np.int, float, np.float)):
            # self._polar_log_g = np.float64((log_g * u.dex(u.cm / u.s ** 2)).to(U.LOG_ACCELERATION_UNIT))
            self._polar_log_g = np.float64(polar_log_g)
        else:
            raise TypeError('Input of variable `polar_log_g` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')
        self._logger.debug("Setting property polar_log_g "
                           "of class instance {} to {}".format(Star.__name__, self._polar_log_g))

    @property
    def metallicity(self):
        """
        returns metallicity of the star, measured as log10(N_Fe/N_H)
        :return:
        """
        return self._metallicity

    @metallicity.setter
    def metallicity(self, metallicity):
        if isinstance(metallicity, (int, np.int, float, np.float)):
            self._metallicity = metallicity
        else:
            raise TypeError('Input of variable `metallicity` is not (np.)int or (np.)float '
                            'instance.')
        self._logger.debug("Setting property metalllicity "
                           "of class instance {} to {}".format(Star.__name__, self._metallicity))

    @property
    def log_g(self):
        """
        returns surface gravity array
        :return:
        """
        return self._log_g

    @log_g.setter
    def log_g(self, log_g):
        """
        setter for log g array
        :param log_g:
        :return:
        """
        self._log_g = log_g

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
        return self._polar_potential_gradient_magnitude

    @polar_potential_gradient_magnitude.setter
    def polar_potential_gradient_magnitude(self, potential_gradient_magnitude):
        """
        :param potential_gradient_magnitude: float
        :return:
        """
        self._polar_potential_gradient_magnitude = potential_gradient_magnitude

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
            if gradient_magnitudes is None:
                gradient_magnitudes = self.potential_gradient_magnitudes
        else:
            if gradient_magnitudes is None:
                gradient_magnitudes = self.potential_gradient_magnitudes[:self.base_symmetry_faces_number]

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

    def calculate_intensity(self, temperatures=None):
        """
        calculates overall radiant flux radiated from unit area with certain effective temperature using atmosphere
        model set in config file
        :param temperatures: array
        :return:
        """
        if temperatures is None:
            temperatures = self.temperatures

        if self.ATMOSPHERE_MODEL == 'black_body':
            const = c.S_BOLTZMAN / np.pi
            return const * np.power(temperatures, 4)
        # here will go other atmosphere models

    def calculate_spectral_radiance(self, temperatures=None, lambda_range=None, steps=None):
        """
        calculates spectral radiant flux radiated from unit area per solid angle with certain effective temperature
        using atmosphere model set in config file
        :param temperatures: array
        :return:
        """
        if temperatures is None:
            temperatures = self.temperatures

        if self.ATMOSPHERE_MODEL == 'black_body':
            k1 = 2 * c.PLANCK_CONST / c.C
            k2 = c.PLANCK_CONST / (c.BOLTZMAN_CONST * self.temperatures)
        # here insert other atmosphere models

    def calculate_normal_radiance(self, intensities=None, areas=None):
        """
        returns radiance from each face along its normal

        :param intensities: array
        :param areas: array
        :return:
        """
        if intensities is None:
            raise ValueError('Intensities for given faces were not supplied.')
        if areas is None:
            areas = self.areas
        if np.shape(intensities) != np.shape(areas):
            raise ValueError('Intensities and areas does not have the same shapes.')

        # in case of object from BinarySystem there is a factor major_semiaxis^2 missing this can be later renormalized
        #  using less computational power
        return intensities * areas

    def has_spots(self):
        return not len(self._spots) == 0



