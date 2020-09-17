import numpy as np

from .. import utils
from .. base.body import Body
from .. base.container import StarPropertiesContainer
from .. base.transform import StarProperties
from .. pulse.mode import PulsationMode
from .. logger import getLogger

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
