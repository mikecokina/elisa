import numpy as np

from elisa.base.body import Body
from elisa.pulse.mode import PulsationMode
from elisa import utils, const as c, units as e_units

from copy import copy
from scipy.special import sph_harm, lpmv
from scipy.optimize import brute, fmin
from astropy import units as u
from elisa.utils import is_empty


class Star(Body):
    """
    :param log_g: numpy.array; Returns surface gravity array.
    :param filling_factor: float;

    ::
        Return filling factor of Star with another Star in binary system.
        This parameter makes sence only if Star is part of binary system.
        Filling factor is computed as (Omega_{inner} - Omega) / (Omega_{inner} - Omega_{outter}) where Omega_X denote
        potential value and `Omega` is potential of given Star. Inner and outter are critical inner and outter
        potentials for given binary star system.

    :param critical_surface_potential: float;
    ::

        Return critical surface potential (If used such potential in binary system,
        Star fills exactly Roche lobe in periastron).
        This parameter makes sence only if star is part of binary system.


    """
    MANDATORY_KWARGS = ['mass', 't_eff', 'gravity_darkening']
    OPTIONAL_KWARGS = ['surface_potential', 'synchronicity', 'albedo', 'pulsations',
                       'discretization_factor', 'spots', 'metallicity', 'polar_log_g']
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, name=None, suppress_logger=False, **kwargs):
        utils.invalid_kwarg_checker(kwargs, Star.ALL_KWARGS, Star)
        super(Star, self).__init__(name, self.__class__.__name__, suppress_logger, **kwargs)

        # default values of properties
        self.log_g = np.array([])
        self.filling_factor = np.nan
        self.critical_surface_potential = np.nan

        self._surface_potential = np.nan
        self._backward_radius = np.nan
        self._polar_radius = np.nan
        self._gravity_darkening = np.nan
        self._forward_radius = np.nan
        self._side_radius = np.nan
        self._polar_log_g = np.nan
        self._equatorial_radius = np.nan
        self._potential_gradient_magnitudes = np.nan
        self._polar_potential_gradient_magnitude = np.nan
        self._pulsations = dict()
        self._metallicity = np.nan

        self.init_parameters(**kwargs)

    def init_parameters(self, **kwargs):
        """
        Initialise instance paramters
        :param kwargs: Dict; initial parameters
        :return:
        """
        self._logger.debug(f"initialising properties of class instance {self.__class__.__name__}")
        for kwarg in Star.ALL_KWARGS:
            if kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])

    def has_pulsations(self):
        """
        Determine whether Star has defined pulsations.
        :return: bool
        """
        return len(self._pulsations) > 0

    @property
    def polar_log_g(self):
        """
        Returns logarithm of polar surface gravity in SI.

        :return: float
        """
        return self._polar_log_g

    @polar_log_g.setter
    def polar_log_g(self, polar_log_g):
        """
        Setter for polar surface gravity.
        If unit is not specified in astropy.units format, value in cgs unit is assumed (it means log(g) in cgs).

        :param polar_log_g: float, astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(polar_log_g, u.quantity.Quantity):
            self._polar_log_g = np.float64(polar_log_g.to(e_units.LOG_ACCELERATION_UNIT))
        elif isinstance(polar_log_g, (int, np.int, float, np.float)):
            self._polar_log_g = np.float64(polar_log_g)
        else:
            raise TypeError('Input of variable `polar_log_g` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

        self._logger.debug(f"setting property polar_log_g "
                           f"of class instance {self.__class__.__name__} to {self._polar_log_g}")

    @property
    def metallicity(self):
        """
        Returns metallicity of the star, measured as log10(N_Fe/N_H).
        :return: float
        """
        return self._metallicity

    @metallicity.setter
    def metallicity(self, metallicity):
        """
        Set metalicity. Float number is assumed to be in [M/H] (cgs) units.

        :param metallicity: float
        :return:
        """
        if isinstance(metallicity, (int, np.int, float, np.float)):
            self._metallicity = metallicity
        else:
            raise TypeError('Input of variable `metallicity` is not (np.)int or (np.)float '
                            'instance.')
        self._logger.debug(f"setting property metalllicity of class instance "
                           f"{self.__class__.__name__} to {self._metallicity}")

    @property
    def pulsations(self):
        """
        Return pulsation modes for given Star instance.

        :return: Dict:

        ::

        {index: PulsationMode}
        """
        return self._pulsations

    @pulsations.setter
    def pulsations(self, pulsations):
        """
        Set pulsation mode for given Star instance defined by dict.

        :param pulsations: Dict:

        ::

        [{"l": <int>, "m": <int>, "amplitude": <float>, "frequency": <float>}, ...]

        :return:
        """
        if pulsations:
            self._pulsations = {idx: PulsationMode(**pulsation_meta) for idx, pulsation_meta in enumerate(pulsations)}

    @property
    def surface_potential(self):
        """
        Returns surface potential of Star.

        :return: float
        """
        return self._surface_potential

    @surface_potential.setter
    def surface_potential(self, potential):
        """
        Setter for surface potential.

        :param potential: float
        """
        self._surface_potential = np.float64(potential)

    @property
    def backward_radius(self):
        """
        Returns value of backward radius of an object in default unit.

        :return: float
        """
        return self._backward_radius

    @property
    def forward_radius(self):
        """
        Returns value of forward radius of an object in default unit.
        Returns None if it doesn't exist (in case of W UMa binary systems)

        :return: float
        """
        return self._forward_radius

    @property
    def polar_radius(self):
        """
        Returns value of polar radius of an object in default unit.

        :return: float
        """
        return self._polar_radius

    @property
    def side_radius(self):
        """
        Returns value of side radius of an object in default unit.

        :return: float
        """
        return self._side_radius

    @property
    def gravity_darkening(self):
        """
        Returns value of gravity darkening.

        :return: float
        """
        return self._gravity_darkening

    @gravity_darkening.setter
    def gravity_darkening(self, gravity_darkening):
        """
        Setter for gravity darkening.
        Accepts values of gravity darkening in range (0, 1)

        :param gravity_darkening: float
        """
        if 0 <= gravity_darkening <= 1:
            self._gravity_darkening = np.float64(gravity_darkening)
        else:
            raise ValueError(f'Parameter gravity darkening = {gravity_darkening} is out of range (0, 1)')

    @property
    def equatorial_radius(self):
        """
        Returns equatorial radius in default units.

        :return: float
        """
        return self._equatorial_radius

    @property
    def potential_gradient_magnitudes(self):
        """
        Returns array of absolute values of potential gradients for each face of surface.

        :return: ndarray
        """
        return self._potential_gradient_magnitudes

    @potential_gradient_magnitudes.setter
    def potential_gradient_magnitudes(self, potential_gradient_magnitudes):
        """
        Set potential gradient magnitudes.

        :param potential_gradient_magnitudes: ndarray
        :return:
        """
        self._potential_gradient_magnitudes = potential_gradient_magnitudes

    @property
    def polar_potential_gradient_magnitude(self):
        """
        Returns value of magnitude of polar potential gradient.

        :return: float
        """
        return self._polar_potential_gradient_magnitude

    @polar_potential_gradient_magnitude.setter
    def polar_potential_gradient_magnitude(self, potential_gradient_magnitude):
        """
        Set magnituded of polar potential gradient.

        :param potential_gradient_magnitude: float
        :return:
        """
        self._polar_potential_gradient_magnitude = potential_gradient_magnitude

    def reset_spots_properties(self):
        """
        Reset computed spots properties
        """
        for _, spot_instance in self.spots.items():
            spot_instance.boundary = np.array([])
            spot_instance.boundary_center = np.array([])
            spot_instance.center = np.array([])

            spot_instance.points = np.array([])
            spot_instance.normals = np.array([])
            spot_instance.faces = np.array([])
            spot_instance.face_centres = np.array([])

            spot_instance.areas = np.array([])
            spot_instance.potential_gradient_magnitudes = np.array([])
            spot_instance.temperatures = np.array([])

            spot_instance._log_g = np.array([])

    def calculate_polar_effective_temperature(self):
        """
        Returns polar effective temperature.

        :return: float
        """
        return self.t_eff * np.power(np.sum(self.areas) /
                                     np.sum(self.areas * np.power(self.potential_gradient_magnitudes /
                                                                  self.polar_potential_gradient_magnitude,
                                                                  self.gravity_darkening)),
                                     0.25)

    def calculate_effective_temperatures(self, gradient_magnitudes=None):
        """
        Calculates effective temperatures for given gradient magnitudes.
        If None is given, star surface t_effs are calculated.

        :param gradient_magnitudes:
        :return:
        """
        if self.has_spots():  # temporary
            if is_empty(gradient_magnitudes):
                gradient_magnitudes = self.potential_gradient_magnitudes
        else:
            if is_empty(gradient_magnitudes):
                gradient_magnitudes = self.potential_gradient_magnitudes[:self.base_symmetry_faces_number]

        t_eff_polar = self.calculate_polar_effective_temperature()
        t_eff = t_eff_polar * np.power(gradient_magnitudes / self.polar_potential_gradient_magnitude,
                                       0.25 * self.gravity_darkening)

        return t_eff if self.spots else t_eff[self.face_symmetry_vector]

    def add_pulsations(self, points=None, faces=None, temperatures=None):
        """
        soon deprecated
        Function returns temperature map with added temperature perturbations caused by pulsations.

        :param time: float
        :param points: ndarray - if `None` Star.points are used
        :param faces: ndarray - if `None` Star.faces are used
        :param temperatures: ndarray - if `None` Star.temperatures
        :return: ndarray
        """

        def alp(x, *args):
            """
            Returns negative value from imaginary part of associated Legendre polynomial (ALP),
            used in minimizer to find global maximum of real part of spherical harmonics.

            :param x: float - argument of function
            :param args:

            ::

                l - angular degree of ALP
                m - degree of ALP

            :return: float; negative of absolute value of ALP
            """
            l, m = args
            return -abs(lpmv(m, l, x))
            # return -abs(np.real(sph_harm(m, l, x[0], x[1])))

        def spherical_harmonics_renormalization_constant(l: int, m: int):
            old_settings = np.seterr(divide='ignore', invalid='ignore', over='ignore')
            ns = int(np.power(5, np.ceil((l-m)/23))*((l-m)+1))
            output = brute(alp, ranges=((0.0, 1.0),), args=(l, m), Ns=ns, finish=fmin, full_output=True)
            np.seterr(**old_settings)

            x = output[2][np.argmin(output[3])] if not 0 <= output[0] <= 1 else output[0]
            result = abs(np.real(sph_harm(m, l, 0, np.arccos(x))))
            return 1.0 / result

        if not is_empty(points):
            if is_empty(faces) or is_empty(temperatures):
                raise ValueError('A `points` argument is not None but `faces` or `temperature` is.\n'
                                 'Please supply the missing keyword arguments')
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
        """
        In case of spot presence, renormalize temperatures to fit effective temperature again,
        since spots disrupt effective temperature of Star as entity.

        :return:
        """
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
        self._logger.debug(f'surface temperature map renormalized by a factor {coefficient}')
        self.temperatures *= coefficient
        if self.spots:
            for spot_index, spot in self.spots.items():
                spot.temperatures *= coefficient

    def properties_serializer(self):
        body_props = ['mass', 't_eff', 'points', 'faces', 'normals', 'temperatures', 'synchronicity', 'albedo',
                      'polar_radius', 'areas', 'discretization_factor', 'face_centres', 'spots',
                      'point_symmetry_vector', 'inverse_point_symmetry_matrix', 'base_symmetry_points_number',
                      'face_symmetry_vector', 'base_symmetry_faces_number', 'base_symmetry_points',
                      'base_symmetry_faces']
        star_props = ['surface_potential', 'backward_radius', 'polar_radius', 'gravity_darkening', 'synchronicity',
                      'forward_radius', 'side_radius', 'polar_log_g', 'equatorial_radius',
                      'critical_surface_potential', 'potential_gradient_magnitudes',
                      'polar_potential_gradient_magnitude', 'pulsations', 'filling_factor', 'metallicity', 'log_g']

        properties_list = body_props + star_props
        return StarProperties(**{prop: copy(getattr(self, prop)) for prop in properties_list})


class StarProperties(object):
    def __init__(self, **kwargs):
        self.properties = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        return self.properties

    def __getitem__(self, item):
        return getattr(self, item)

    def __str__(self):
        return str(self.to_dict())
