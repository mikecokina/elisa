import numpy as np

from .. import utils
from .. base.body import Body
from .. base.container import StarPropertiesContainer
from .. base.transform import StarProperties
from .. pulse.mode import PulsationMode
from .. logger import getLogger
from .. import units as u

from copy import (
    copy,
    deepcopy
)

logger = getLogger('base.star')


class Star(Body):
    """
    Child class of elisa.base.body.Body representing `Star`.
    
    Class can be imported directly:
    ::

        from elisa import Star
    
    Mandatory `Star` arguments if the instance is a component of the `SingleSystem`:

        :param mass: float; If mass is int, np.int, float, np.float, program assumes solar mass as it's unit.
                        If mass astropy.unit.quantity.Quantity instance, program converts it to default units.
        :param t_eff: float; Accepts value in any temperature unit. If your input is without unit,
                         function assumes that supplied value is in K.
        :param polar_log_g: float; log_10 of the polar surface gravity

    following mandatory arguments are also available:

        :param metallicity: float; log[M/H] default value is 0.0
        :param gravity_darkening: float; gravity darkening factor, if not supplied, it is interpolated from Claret 2003
                                         based on t_eff

    After initialization of the SingleSystem, following additional attributes of the Star instance are available:

        :critical_potential: float; potential of the star required to fill its Roche lobe
        :equivalent_radius: float; radius of a sphere with the same volume as a component (in SMA units)
        :polar_radius: float; radius of a star towards the pole of the star
        :equatorial_radius: float; radius of a star towards the pole of the star
    
    Mandatory `Star` arguments if the instance is a component of the `BinarySystem`:

        :param mass: float; If mass is int, np.int, float, np.float, program assumes solar mass as it's unit.
                            If mass astropy.unit.quantity.Quantity instance, program converts it to default units.
        :param t_eff: float; Accepts value in any temperature unit. If your input is without unit,
                             function assumes that supplied value is in K.
        :param surface_potential: float; generalized surface potential (Wilson 79)
        :param synchronicity: float; synchronicity F (omega_rot / omega_orb), equals 1 for synchronous rotation
        :param albedo: float; surface albedo, value from <0, 1> interval

    following mandatory arguments are also available:

        :param metallicity: float; log[M/H] default value is 0.0
        :param gravity_darkening: float; gravity darkening factor, if not supplied, it is interpolated from Claret 2003
                                         based on t_eff


    After initialization of the `BinarySystem`, following additional attributes of the `Star` instance are available:

        :critical_potential: float; potential of the star required to fill its Roche lobe
        :equivalent_radius: float; radius of a sphere with the same volume as a component (in SMA units)
        :filling_factor: float: calculated as (Omega_{inner} - Omega) / (Omega_{inner} - Omega_{outter})

                            :filling factor < 0: component does not fill its Roche lobe
                            :filling factor = 0: component fills preciselly its Roche lobe
                            :1 > filling factor > 0: component overflows its Roche lobe
                            :filling factor = 1: upper boundary of the filling factor, higher value would lead to 
                                                 the mass loss trough Lagrange point L2
                                                 
        Radii at periastron (in SMA units)
            :polar_radius: float; radius of a star towards the pole of the star
            :side_radius: float; radius of a star in the direction perpendicular to the pole and direction of a
                                 companion
            :backward_radius: float; radius of a star in the opposite direction as the binary companion
            :forward_radius: float; radius of a star towards the binary companion, returns numpy.nan if the system is
                                    over-contact

    Optional parameters of `Star` instances can be defined with the following arguments:
        :param spots: List[Dict[str, float]]; Spots definitions. Order in which the spots are defined will determine the
                                          layering of the spots (spot defined as first will lay bellow any subsequently
                                          defined overlapping spot). Example of spots definition:

        ::

            [
                 {"longitude": 90,
                  "latitude": 58,
                  "angular_radius": 15,
                  "temperature_factor": 0.9},
                 {"longitude": 85,
                  "latitude": 80,
                  "angular_radius": 30,
                  "temperature_factor": 1.05},
                 {"longitude": 45,
                  "latitude": 90,
                  "angular_radius": 30,
                  "temperature_factor": 0.95},
             ]

        :param pulsations: List[Dict[str, float]]; to be added soon
        :param atmosphere: str; atmosphere to use for given object instance, available atmosphere models:

            - `castelli`, `castelli-kurucz`, `ck` or `ck04`: atmosphere models in Castelli-Kurucz, 2004
            - `kurucz`, `k` or `k93`: atmosphere models in Kurucz, 1993

    After initialization, `Star` instance after initialization within the given `System` has its spot initialized in a
    `star_instance.spots` attribute as a list containing elisa.base.spot.Spot containers.

    """

    MANDATORY_KWARGS = ['mass', 't_eff']
    OPTIONAL_KWARGS = ['surface_potential', 'synchronicity', 'albedo', 'pulsations', 'atmosphere',
                       'spots', 'metallicity', 'polar_log_g', 'discretization_factor', 'gravity_darkening']
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, name=None, **kwargs):
        utils.invalid_kwarg_checker(kwargs, Star.ALL_KWARGS, Star)
        super(Star, self).__init__(name, **kwargs)
        kwargs = self.transform_input(**kwargs)

        # default values of properties
        self.filling_factor = np.nan
        self.critical_surface_potential = np.nan
        self.surface_potential = np.nan
        self.metallicity = 0.0
        self.polar_log_g = np.nan
        self.gravity_darkening = np.nan
        self._pulsations = list()

        self.side_radius = np.nan
        self.forward_radius = np.nan
        self.backward_radius = np.nan
        self.equivalent_radius = np.nan

        self.init_parameters(**kwargs)

    @property
    def default_input_units(self):
        """
        Returns set of default units of intialization parameters, in case, when provided without units.

        :return: elisa.units.DefaultStarInputUnits;
        """
        return u.DefaultStarInputUnits

    @property
    def default_internal_units(self):
        """
        Returns set of internal units of Star parameters.

        :return: elisa.units.DefaultStarUnits;
        """
        return u.DefaultStarUnits

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

    def kwargs_serializer(self):
        """
        Creating dictionary of keyword arguments in order to be able to
        reinitialize the class instance in init().

        :return: Dict;
        """
        default_units = {
            "mass": u.kg,
            "t_eff": u.K,
            "discretization_factor": u.rad
        }

        serialized_kwargs = dict()
        for kwarg in self.ALL_KWARGS:
            if kwarg in ["spots"]:
                # important: this relies on dict ordering
                value = [spot.kwargs_serializer() for spot in getattr(self, kwarg).values()]
            elif kwarg in default_units:
                value = getattr(self, kwarg)
                if not isinstance(value, u.Quantity):
                    value = value * default_units[kwarg]
            else:
                value = getattr(self, kwarg)

            serialized_kwargs[kwarg] = value
        return serialized_kwargs

    def init(self):
        self.__init__(**self.kwargs_serializer())

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
        if pulsations in [None, []]:
            self._pulsations = {}
        elif pulsations:
            self._pulsations = {idx: PulsationMode(**pulsation_meta) for idx, pulsation_meta in enumerate(pulsations)}

    def properties_serializer(self):
        """
        Prepares properties to be inherited from Star instance by StarContainer instance.

        :return: Dict;
        """
        properties_list = ['mass', 't_eff', 'synchronicity', 'albedo', 'discretization_factor', 'equivalent_radius',
                           'polar_radius', 'equatorial_radius', 'gravity_darkening', 'surface_potential', 'pulsations',
                           'metallicity', 'polar_log_g', 'critical_surface_potential', 'atmosphere',
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



