import numpy as np

from .. import utils
from ..base.body import Body
from ..base.container import StarPropertiesContainer
from ..base.transform import StarProperties
from ..pulse.mode import PulsationMode
from ..logger import getLogger

from copy import (
    copy,
    deepcopy
)

logger = getLogger('base.star')


class Star(Body):
    """
    Child class of elisa.base.body.Body representing Star.
    Class intherit parameters from elisa.base.body.Body and add following

    Input parameters:

    :param surface_potential: float;
    :param synchronicity: float;
    :param pulsations: List;
    :param metallicity: float;
    :param polar_log_g: float;
    :param gravity_darkening: float;

    Output parameters:

    :filling_factor: float;
    :critical_surface_potential: float;
    :pulsations: Dict[int, PulsationMode];
    :side_radius: float; Computed in periastron
    :forward_radius: float; Computed in periastron
    :backward_radius: float; Computed in periastron
    """

    MANDATORY_KWARGS = ['mass', 't_eff', 'gravity_darkening']
    OPTIONAL_KWARGS = ['surface_potential', 'synchronicity', 'albedo', 'pulsations',
                       'discretization_factor', 'spots', 'metallicity', 'polar_log_g']
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, name=None, **kwargs):
        utils.invalid_kwarg_checker(kwargs, Star.ALL_KWARGS, Star)
        super(Star, self).__init__(name, **kwargs)
        kwargs = self.transform_input(**kwargs)

        # default values of properties
        self.filling_factor = np.nan
        self.critical_surface_potential = np.nan
        self.surface_potential = np.nan
        self.metallicity = np.nan
        self.polar_log_g = np.nan
        self.gravity_darkening = np.nan
        self._pulsations = dict()

        self.side_radius = np.nan
        self.forward_radius = np.nan
        self.backward_radius = np.nan
        self.equivalent_radius = np.nan

        self.init_parameters(**kwargs)

    def transform_input(self, **kwargs):
        """
        Transform and validate input kwargs.

        :param kwargs: Dict;
        :return: Dict;
        """
        return StarProperties.transform_input(**kwargs)

    def init_parameters(self, **kwargs):
        """
        Initialise instance parameters

        :param kwargs: Dict; initial parameters
        :return:
        """
        logger.debug(f"initialising properties of class instance {self.__class__.__name__}")
        for kwarg in Star.ALL_KWARGS:
            if kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])

    def has_pulsations(self):
        """
        Determine whether Star has defined pulsations.

        :return: bool;
        """
        return len(self._pulsations) > 0

    @property
    def pulsations(self):
        """
        Return pulsation modes for given Star instance.

        :return: Dict;

        ::

            {index: PulsationMode}
        """
        return self._pulsations

    @pulsations.setter
    def pulsations(self, pulsations):
        """
        Set pulsation mode for given Star instance defined by dict.

        :param pulsations: Dict;

        ::

            [{"l": <int>, "m": <int>, "amplitude": <float>, "frequency": <float>}, ...]

        :return:
        """
        if pulsations:
            self._pulsations = {idx: PulsationMode(**pulsation_meta) for idx, pulsation_meta in enumerate(pulsations)}

    def properties_serializer(self):
        properties_list = ['mass', 't_eff', 'synchronicity', 'albedo', 'discretization_factor', 'polar_radius',
                           'equatorial_radius', 'gravity_darkening', 'surface_potential', 'pulsations',
                           'metallicity', 'polar_log_g', 'critical_surface_potential',
                           # todo: remove side_radius when figured out starting point for solver
                           'side_radius']
        props = {prop: copy(getattr(self, prop)) for prop in properties_list}
        props.update({
            "name": self.name,
            "spots": deepcopy(self.spots)
        })
        return props

    def to_properties_container(self):
        """
        Serialize instance of elisa.base.star.Star to elisa.base.container.StarPropertiesContainer.

        :return: elisa.base.container.StarPropertiesContainer
        """
        return StarPropertiesContainer(**self.properties_serializer())

    # todo: move it to star container
    # def add_pulsations(self, points=None, faces=None, temperatures=None):
    #     """
    #     soon deprecated
    #     Function returns temperature map with added temperature perturbations caused by pulsations.
    #     :param points: ndarray - if `None` Star.points are used
    #     :param faces: ndarray - if `None` Star.faces are used
    #     :param temperatures: ndarray - if `None` Star.temperatures
    #     :return: ndarray
    #     """
    #
    #     def alp(x, *args):
    #         """
    #         Returns negative value from imaginary part of associated Legendre polynomial (ALP),
    #         used in minimizer to find global maximum of real part of spherical harmonics.
    #         :param x: float - argument of function
    #         :param args:
    #
    #         ::
    #
    #             l - angular degree of ALP
    #             m - degree of ALP
    #
    #         :return: float; negative of absolute value of ALP
    #         """
    #         l, m = args
    #         return -abs(up.lpmv(m, l, x))
    #         # return -abs(np.real(up.sph_harm(m, l, x[0], x[1])))
    #
    #     def spherical_harmonics_renormalization_constant(l: int, m: int):
    #         old_settings = np.seterr(divide='ignore', invalid='ignore', over='ignore')
    #         ns = int(up.power(5, up.ceil((l-m)/23))*((l-m)+1))
    #         output = brute(alp, ranges=((0.0, 1.0),), args=(l, m), Ns=ns, finish=fmin, full_output=True)
    #         np.seterr(**old_settings)
    #
    #         x = output[2][np.argmin(output[3])] if not 0 <= output[0] <= 1 else output[0]
    #         result = abs(np.real(up.sph_harm(m, l, 0, up.arccos(x))))
    #         return 1.0 / result
    #
    #     if not is_empty(points):
    #         if is_empty(faces) or is_empty(temperatures):
    #             raise ValueError('A `points` argument is not None but `faces` or `temperature` is.\n'
    #                              'Please supply the missing keyword arguments')
    #     else:
    #         points = copy(self.points)
    #         faces = copy(self.faces)
    #         temperatures = copy(self.temperatures)
    #
    #     surface_centers = self.calculate_surface_centres(points, faces)
    #     centres = utils.cartesian_to_spherical(surface_centers)
    #
    #     for pulsation_index, pulsation in self.pulsations.items():
    #         # generating spherical coordinate system in case
    #         # of mode axis not identical with axis of rotation, new axis
    #         # is created by rotation of polar coordinates around z in positive direction by `mode_axis_phi' and than
    #         # rotating this coordinate system around y axis to tilt z axis by amount of `mode_axis_theta`
    #         if pulsation.mode_axis_phi == 0 and pulsation.mode_axis_theta == 0:
    #             phi, theta = centres[:, 1], centres[:, 2]
    #         else:
    #             phi_rot = (centres[:, 1] - pulsation.mode_axis_phi) % c.FULL_ARC  # rotation around Z
    #             # axis
    #             cos_phi = up.cos(phi_rot)
    #             sin_theta = up.sin(centres[:, 2])
    #             sin_axis_theta = up.sin(pulsation.mode_axis_theta)
    #             cos_theta = up.cos(centres[:, 2])
    #             # rotation around Y axis by `mode_axis_theta` angle
    #             cos_axis_theta = up.cos(pulsation.mode_axis_theta)
    #             theta = up.arccos(cos_phi * sin_theta * sin_axis_theta + cos_theta * cos_axis_theta)
    #             phi = up.arctan2(up.sin(phi_rot) * sin_theta, cos_phi * sin_theta * cos_axis_theta -
    #                              cos_theta * sin_axis_theta)
    #
    #         # generating of renormalised spherical harmonics (maximum value on sphere equuals to 1)
    #         constant = spherical_harmonics_renormalization_constant(pulsation.l, pulsation.m)
    #         spherical_harmonics = constant * np.real(up.sph_harm(pulsation.m, pulsation.l, phi, theta))
    #
    #         temperatures += pulsation.amplitude * spherical_harmonics
    #
    #     return temperatures
