import numpy as np

from typing import Union, Dict
from copy import deepcopy
from scipy import optimize

from . import graphic
from . orbit import orbit
from . curves import lc, rv
from . surface import mesh
from . surface.temperature import interpolate_albedo
from . transform import BinarySystemProperties
from . curves import c_router
from . import (
    utils as bsutils,
    radius as bsradius,
    model
)
from . container import OrbitalPositionContainer

from .. base.error import MorphologyError
from .. base.container import SystemPropertiesContainer
from .. base.system import System
from .. base.star import Star
from .. base.curves import utils as rv_utils

from .. import settings
from .. logger import getLogger
from .. import (
    umpy as up,
    units as u,
    utils,
    const
)
from .. opt.fsolver import fsolve

logger = getLogger('binary_system.system')


class BinarySystem(System):
    """
    Class to store and calculate necessary properties of the binary system based on the user provided parameters.
    Child class of elisa.base.system.System.

    Class can be imported directly:
    ::

        >>> from elisa import BinarySystem

    After initialization, apart from the attributes already defined by the user with the arguments, user has access to
    the following attributes:

        :mass_ratio: float; secondary mass / primary mass
        :semi_major_axis: float; semi major axis of system in physical units
        :morphology: str; morphology of the system:

                      :`detached`: both components are not filling their respective Roche lobes,
                      :`semi-detached`: one of the components is filling its Roche lobe,
                      :`double-contact`: both components fill their Roche lobes,
                      :`over-contact`: components are physically connected with a ''neck'')

    `BinarySystem' requires instances of elisa.base.star.Star in `primary` and `secondary` argument with following
    mandatory arguments:

        :param mass: float; If mass is int, np.int, float, np.float, program assumes solar mass as it's unit.
                            If mass astropy.unit.quantity.Quantity instance, program converts it to default units.
        :param t_eff: float; Accepts value in any temperature unit. If your input is without unit,
                             function assumes that supplied value is in K.
        :param surface_potential: float; generalized surface potential (Wilson 79)
        :param synchronicity: float; synchronicity F (omega_rot / omega_orb), equals 1 for synchronous rotation

    following optional arguments are also available:

        :param metallicity: float; log[M/H] default value is 0.0
        :param gravity_darkening: float; gravity darkening factor, if not supplied, it is interpolated
                                         from Claret 2003 based on t_eff

        :param albedo: float; surface albedo, value from <0, 1> interval, if not supplied,
                              Claret 2001 will be used for interpolation


    Each component instance will after initialization contain following attributes:

        :critical_surface_potential: float; potential of the star required to fill its Roche lobe
        :equivalent_radius: float; radius of a sphere with the same volume as a component (in SMA units)
        :filling_factor: float: calculated as (Omega_{inner} - Omega) / (Omega_{inner} - Omega_{outter})

                            :filling factor < 0: component does not fill its Roche lobe
                            :filling factor = 0: component fills preciselly its Roche lobe
                            :1 > filling factor > 0: component overflows its Roche lobe
                            :filling factor = 1: upper boundary of the filling factor, higher value would lead to
                                                 the mass loss trough Lagrange point L2
        
        Radii at periastron (in SMA units)  
            :polar_radius: float; radius of a star towards the pole of the star
            :side_radius: float; radius of a star in the direction perpendicular to the pole
                                 and direction of a companion
            :backward_radius: float; radius of a star in the opposite direction as the binary companion
            :forward_radius: float; radius of a star towards the binary companion,
                                    returns numpy.nan if the system is over-contact

    The BinarySystem can be initialized either by using valid class arguments, e.g.:
    ::
    
        >>> from elisa import BinarySystem
        >>> from elisa import Star
        >>> from astropy import units as u
        
        >>> primary = Star(
        >>>     mass=2.15 * u.solMass,
        >>>     surface_potential=3.6,
        >>>     synchronicity=1.0,
        >>>      t_eff=10000 * u.K,
        >>>     gravity_darkening=1.0,
        >>>     discretization_factor=5,  # angular size (in degrees) of the surface elements
        >>>     albedo=0.6,
        >>>     metallicity=0.0,
        >>> )
        
        >>> secondary = Star(
        >>>     mass=0.45 * u.solMass,
        >>>     surface_potential=5.39,
        >>>     synchronicity=1.0,
        >>>     t_eff=8000 * u.K,
        >>>     gravity_darkening=1.0,
        >>>     albedo=0.6,
        >>>     metallicity=0,
        >>> )
        >>>
        >>> bs = BinarySystem(
        >>>     primary=primary,
        >>>     secondary=secondary,
        >>>     argument_of_periastron=58 * u.deg,
        >>>     gamma=-30.7 * u.km / u.s,
        >>>     period=2.5 * u.d,
        >>>     eccentricity=0.0,
        >>>     inclination=85 * u.deg,
        >>>     primary_minimum_time=2440000.00000 * u.d,
        >>>     phase_shift=0.0,
        >>> )
    
    or by using the BinarySystem.from_json(<dict>) function that accepts various parameter combination in form of
    dictionary such as:
    ::
    
        >>> data = {
        >>>     "system": {
        >>>         "inclination": 90.0,
        >>>         "period": 10.1,
        >>>         "argument_of_periastron": 90.0,
        >>>         "gamma": "0.0 m / s",  # you can define quantity using string representation of the astropy units
        >>>         "eccentricity": 0.3,
        >>>         "primary_minimum_time": 0.0,
        >>>         "phase_shift": 0.0
        >>>     },
        >>>     "primary": {
        >>>         "mass": 2.15,
        >>>         "surface_potential": 3.6,
        >>>         "synchronicity": 1.0,
        >>>         "t_eff": 10000.0,
        >>>         "gravity_darkening": 1.0,
        >>>         "discretization_factor": 5,
        >>>         "albedo": 1.0,
        >>>         "metallicity": 0.0,
        >>>         "atmosphere": "ck04"
        >>>     },
        >>>     "secondary": {
        >>>         "mass": 0.45,
        >>>         "surface_potential": 5.39,
        >>>         "synchronicity": 1.0,
        >>>         "t_eff": 8000.0,
        >>>         "gravity_darkening": 1.0,
        >>>         "albedo": 1.0,
        >>>         "metallicity": 0.0,
        >>>         "atmosphere": "black_body"
        >>>     }
        >>> }
        >>>
        >>> binary = BinarySystem.from_json(data)
    
    See documentation for `from_json` method for details.
                                    
    The orbit of the binary system can be modelled using function
    `calculate_orbital_motion(phases)`. E.g.:
    ::

        >>> binary.calculate_orbital_motion(np.linspace(0, 1))

    The class contains substantial plotting capability in the BinarySystem.plot module that contains following
    functions (further info in: see elisa.binary_system.graphics.plot):

        - orbit(args): plots an orbit of the binary system
        - equipotential(args): xy, yz, zx cross-sections of equipotential surface
        - mesh(args): 3D mesh (scatter) plot of the surface points
        - wireframe(args): wire frame model of the selected system components
        - surface(args): plot models of the binary components with various surface
                         colormaps (gravity_acceleration, temperature, radiance, ...)
          
    Plot function can be called as function of the plot module. E.g.:
    ::
    
        >>> binary.plot.surface(phase=0.1, colormap='temperature'))
         
    Similarly, an animation of the orbital motion can be produced using BinarySystem.animation module and its function 
    `orbital_motion(*args)`.


    List of valid input system arguments:

    :param primary: elisa.base.star.Star; instance of primary component
    :param secondary: elisa.base.star.Star; instance of secondary component
    :param inclination: Union[float, astropy.unit.quantity.Quantity]; Inclination of the system.
                        If unit is not supplied, value in degrees is assumed.
    :param period: Union[(numpy.)float, (numpy.)int, astropy.units.quantity.Quantity]; Orbital period of binary
                   star system. If unit is not specified, default period unit is assumed (days).
    :param eccentricity: Union[(numpy.)int, (numpy.)float]; from <0, 1> interval
    :param argument_of_periastron: Union[(numpy.)float, (numpy.)int, astropy.units.quantity.Quantity];
    :param gamma: Union[float, astropy.unit.quantity.Quantity]; Center of mass velocity.
                  Expected type is astropy.units.quantity.Quantity, numpy.float or numpy.int
                  otherwise TypeError will be raised. If unit is not specified, default velocity unit is assumed (m/s).
    :param phase_shift: float; Phase shift of the primary eclipse with respect to the ephemeris.
                               true_phase is used during calculations, where: true_phase = phase + phase_shift.;
    :param primary_minimum_time: Union[(numpy.)float, (numpy.)int, astropy.units.quantity.Quantity];
    :param additional_light: float; fraction of light that does not originate from the `BinarySystem`
    """

    MANDATORY_KWARGS = ['inclination', 'period', 'eccentricity', 'argument_of_periastron']
    OPTIONAL_KWARGS = ['gamma', 'phase_shift', 'additional_light', 'primary_minimum_time']
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    STAR_MANDATORY_KWARGS = ['mass', 't_eff', 'surface_potential', 'synchronicity']
    STAR_OPTIONAL_KWARGS = ['metallicity', 'gravity_darkening', 'albedo']
    STAR_ALL_KWARGS = STAR_MANDATORY_KWARGS + STAR_OPTIONAL_KWARGS

    def __init__(self, primary, secondary, name=None, **kwargs):
        # initial validity checks
        utils.invalid_kwarg_checker(kwargs, BinarySystem.ALL_KWARGS, self.__class__)
        utils.check_missing_kwargs(BinarySystem.MANDATORY_KWARGS, kwargs, instance_of=BinarySystem)
        self.object_params_validity_check(dict(primary=primary, secondary=secondary), self.STAR_MANDATORY_KWARGS)
        kwargs: Dict = self.transform_input(**kwargs)

        super(BinarySystem, self).__init__(name, **kwargs)

        logger.info(f"initialising object {self.__class__.__name__}")
        logger.debug(f"setting properties of components of class instance {self.__class__.__name__}")

        # graphic related properties
        self.plot = graphic.plot.Plot(self)
        self.animation = graphic.animation.Animation(self)

        # components
        self.primary: Star = primary
        self.secondary: Star = secondary
        self._components: Dict[str, Star] = dict(primary=self.primary, secondary=self.secondary)

        # default values of properties
        self.orbit: Union[None, orbit.Orbit] = None
        self.period: float = np.nan
        self.eccentricity: float = np.nan
        self.argument_of_periastron: float = np.nan
        self.primary_minimum_time: float = 0.0
        self.phase_shift: float = 0.0
        self.gamma: float = 0.0
        self.mass_ratio: float = self.secondary.mass / self.primary.mass

        # set attributes and test whether all parameters were initialized
        # we already ensured that all kwargs are valid and all mandatory kwargs are present so lets set class attributes
        self.init_properties(**kwargs)

        # calculation of dependent parameters
        logger.debug("computing semi-major axis")
        self.semi_major_axis: float = self.calculate_semi_major_axis()

        # orbit initialisation (initialise class Orbit from given BinarySystem parameters)
        self.init_orbit()

        # setup critical surface potentials in periastron
        logger.debug("setting up critical surface potentials of components in periastron")
        self.setup_periastron_critical_potential()

        logger.debug("setting up morphological classification of binary system")
        self.morphology: str = self.compute_morphology()

        self.setup_components_radii(components_distance=self.orbit.periastron_distance,
                                    calculate_equivalent_radius=True)
        self.setup_betas()
        self.setup_albedos()
        self.assign_pulsations_amplitudes(normalisation_constant=self.semi_major_axis)

        # adjust and setup discretization factor if necessary
        self.setup_discretisation_factor()

        # setting common reference to emphemeris
        self.t0: float = self.primary_minimum_time

    @property
    def default_input_units(self):
        """
        Returns set of default units of intialization parameters, in case, when provided without an units.

        :return: elisa.units.DefaultBinarySystemInputUnits;
        """
        return u.DefaultBinarySystemInputUnits

    @property
    def default_internal_units(self):
        """
        Returns set of internal units of system parameters.

        :return: elisa.units.DefaultBinarySystemUnits;
        """
        return u.DefaultBinarySystemUnits

    @classmethod
    def from_json(cls, data, _verify=True, _kind_of=None):
        """
        Create instance of BinarySystem from JSON in form such as::

            {
              "system": {
                "inclination": 90.0,
                "period": 10.1,
                "argument_of_periastron": "90.0 deg",  # string representation of astropy quntity is also valid
                "gamma": 0.0,
                "eccentricity": 0.3,
                "primary_minimum_time": 0.0,
                "phase_shift": 0.0
              },
              "primary": {
                "mass": 2.0,
                "surface_potential": 7.1,
                "synchronicity": 1.0,
                "t_eff": 6500.0,
                "gravity_darkening": 1.0,
                "discretization_factor": 5,
                "albedo": 1.0,
                "metallicity": 0.0,
                "atmosphere": "ck04"
              },
              "secondary": {
                "mass": 2.0,
                "surface_potential": 7.1,
                "synchronicity": 1.0,
                "t_eff": 6500.0,
                "gravity_darkening": 1.0,
                "discretization_factor": 5,
                "albedo": 1.0,
                "metallicity": 0.0,
                "atmosphere": "black_body"
              }
            }

            or

            {
              "system": {
                "inclination": 90.0,
                "period": 10.1,
                "argument_of_periastron": 90.0,
                "gamma": 0.0,
                "eccentricity": 0.3,
                "primary_minimum_time": 0.0,
                "phase_shift": 0.0,
                "semi_major_axis": 10.5,
                "mass_ratio": 0.5
              },
              "primary": {
                "surface_potential": 7.1,
                "synchronicity": 1.0,
                "t_eff": 6500.0,
                "gravity_darkening": 1.0,
                "discretization_factor": 5,
                "albedo": 1.0,
                "metallicity": 0.0,
                "atmosphere": "black_body"
              },
              "secondary": {
                "surface_potential": 7.1,
                "synchronicity": 1.0,
                "t_eff": 6500.0,
                "gravity_darkening": 1.0,
                "discretization_factor": 5,
                "albedo": 1.0,
                "metallicity": 0.0,
                "atmosphere": "black_body"
              }
            }

        Default units (when unit is not specified as string)::

             {
                "inclination": [degrees],
                "period": [days],
                "argument_of_periastron": [degrees],
                "gamma": [m/s],
                "eccentricity": [dimensionless],
                "primary_minimum_time": [d],
                "phase_shift": [dimensionless],
                "mass": [solMass],
                "surface_potential": [dimensionless],
                "synchronicity": [dimensionless],
                "t_eff": [K],
                "gravity_darkening": [dimensionless],
                "discretization_factor": [degrees],
                "albedo": [dimensionless],
                "metallicity": [dimensionless],

                "semi_major_axis": [solRad],
                "mass_ratio": [dimensionless]
            }

        :return: elisa.binary_system.system.BinarySystem;
        """
        data_cp = deepcopy(data)
        if _verify:
            bsutils.validate_binary_json(data_cp)

        kind_of = _kind_of or bsutils.resolve_json_kind(data_cp)
        if kind_of in ["community"]:
            data_cp = bsutils.transform_json_community_to_std(data_cp)

        primary, secondary = Star(**data_cp["primary"]), Star(**data_cp["secondary"])
        return cls(primary=primary, secondary=secondary, **data_cp["system"])

    @classmethod
    def from_fit_results(cls, results):
        """
        Building binary system from standard fit results format.

        :param results: Dict; {'component': {'param_name': {'value': value, fixed: ...}}}
        :return: elisa.binary_system.system.BinarySystem;
        """
        data = dict()
        for key, component in results.items():
            if key == 'r_squared':
                continue
            data[key] = dict()

            for param, content in component.items():
                if param in ['spots', 'pulsations']:
                    features = list()
                    for idx, feature in enumerate(content):
                        features.append(dict())
                        for f_param, f_content in feature.items():
                            if f_param == 'label':
                                continue
                            features[idx][f_param] = f_content['value']
                    data[key][param] = features
                else:
                    data[key][param] = content['value']

        return BinarySystem.from_json(data=data)

    def build_container(self, phase=None, time=None, build_pulsations=True):
        """
        Function returns `OrbitalPositionContainer` with fully built model binary system at
        user-defined photometric phase or time of observation.

        :param time: float; JD, Julian-Date
        :param phase: float; photometric phase
        :param build_pulsations: bool; whether build pulsation or not
        :return: elisa.binary_system.container.OrbitalPositionContainer;
        """
        if phase is not None and time is not None:
            raise ValueError('Please specify whether you want to build your container '
                             'EITHER at given photometric `phase` or at given `time`.')
        phase = phase if time is None else utils.jd_to_phase(time, period=self.period, t0=self.t0)

        position = self.calculate_orbital_motion(input_argument=phase, return_nparray=False, calculate_from='phase')[0]
        orbital_position_container = OrbitalPositionContainer.from_binary_system(self, position)
        orbital_position_container.build(build_pulsations=build_pulsations)

        logger.info(f'Orbital position container was successfully built at photometric phase {phase:.2f}.')
        return orbital_position_container

    def init(self):
        """
        Function to reinitialize BinarySystem class instance after changing parameter(s) of binary system.
        """
        # reinitialize components (in case when value in component instance was changed)
        for component in settings.BINARY_COUNTERPARTS:
            getattr(self, component).init()

        self.__init__(primary=self.primary, secondary=self.secondary, **self.kwargs_serializer())

    @property
    def components(self):
        """
        Return components object in Dict[str, elisa.base.star.Star].

        :return: Dict[str, elisa.base.star.Star];
        """
        return self._components

    def properties_serializer(self):
        """
        Return binary system properties in form of a JSON file.

        :return: Dict; {`primary`: {}, `secondary`: {}, `system`: {}}
        """
        props = BinarySystemProperties.transform_input(**self.kwargs_serializer())
        props.update({
            "semi_major_axis": self.semi_major_axis,
            "morphology": self.morphology,
            "mass_ratio": self.mass_ratio
        })
        return props

    def to_properties_container(self):
        return SystemPropertiesContainer(**self.properties_serializer())

    def init_orbit(self):
        """
        Orbit class in binary system.
        """
        logger.debug(f"re/initializing orbit in class instance {self.__class__.__name__} / {self.name}")
        orbit_kwargs = {key: getattr(self, key) for key in orbit.Orbit.ALL_KWARGS}
        self.orbit = orbit.Orbit(**orbit_kwargs)

    def is_eccentric(self):
        """
        Resolve whether system is eccentric.

        :return: bool;
        """
        return self.eccentricity > 0

    def is_synchronous(self):
        """
        Resolve whether system is synchronous (consider synchronous system
        if sychnronicity of both components is equal to 1).

        :return: bool;
        """
        return (self.primary.synchronicity == 1) & (self.secondary.synchronicity == 1)

    def calculate_semi_major_axis(self):
        """
        Calculates length of semi major axis using 3rd kepler default_law.

        :return: float;
        """
        period = np.float64((self.period * u.PERIOD_UNIT).to(u.s))
        return (const.G * (self.primary.mass + self.secondary.mass) * period ** 2 / (4 * const.PI ** 2)) ** (1.0 / 3)

    def compute_morphology(self):
        """
        Setup binary star class property `morphology`.
        Determines the morphology based on current system parameters
        and setup `morphology` parameter of `self` system instance.

        :return: str; morphology of the system (`detached`, `semi-detached`, `double-contact`, `over-contact`)
        """
        __PRECISSION__ = 1e-8
        __MORPHOLOGY__ = None
        if (self.primary.synchronicity == 1 and self.secondary.synchronicity == 1) and self.eccentricity == 0.0:
            lp = self.libration_potentials()

            self.primary.filling_factor = self.compute_filling_factor(self.primary.surface_potential, lp)
            self.secondary.filling_factor = self.compute_filling_factor(self.secondary.surface_potential, lp)

            if ((1 > self.secondary.filling_factor > 0) or (1 > self.primary.filling_factor > 0)) and \
                    (abs(self.primary.filling_factor - self.secondary.filling_factor) > __PRECISSION__):
                msg = "Detected over-contact binary system, but potentials of components are not the same."
                raise MorphologyError(msg)
            if self.primary.filling_factor > 1 or self.secondary.filling_factor > 1:
                raise MorphologyError("Non-Physical system: primary_filling_factor or "
                                      "secondary_filling_factor is greater then 1. "
                                      "Filling factor is obtained as following:"
                                      "(Omega_{inner} - Omega) / (Omega_{inner} - Omega_{outter})")

            if (abs(self.primary.filling_factor) < __PRECISSION__ and self.secondary.filling_factor < 0) or \
                    (self.primary.filling_factor < 0 and abs(self.secondary.filling_factor) < __PRECISSION__) or \
                    (abs(self.primary.filling_factor) < __PRECISSION__ and abs(self.secondary.filling_factor)
                     < __PRECISSION__):
                __MORPHOLOGY__ = "semi-detached"
            elif self.primary.filling_factor < 0 and self.secondary.filling_factor < 0:
                __MORPHOLOGY__ = "detached"
            elif 1 >= self.primary.filling_factor > 0:
                __MORPHOLOGY__ = "over-contact"
            elif self.primary.filling_factor > 1 or self.secondary.filling_factor > 1:
                raise MorphologyError("Non-Physical system: potential of components is to low.")

        else:
            self.primary.filling_factor, self.secondary.filling_factor = None, None
            if (abs(self.primary.surface_potential - self.primary.critical_surface_potential) < __PRECISSION__) and \
                    (abs(
                        self.secondary.surface_potential - self.secondary.critical_surface_potential) < __PRECISSION__):
                __MORPHOLOGY__ = "double-contact"

            elif (not (not (abs(
                    self.primary.surface_potential - self.primary.critical_surface_potential) < __PRECISSION__) or not (
                    self.secondary.surface_potential > self.secondary.critical_surface_potential))) or \
                    ((abs(
                        self.secondary.surface_potential - self.secondary.critical_surface_potential) < __PRECISSION__)
                     and (self.primary.surface_potential > self.primary.critical_surface_potential)):
                __MORPHOLOGY__ = "semi-detached"

            elif (self.primary.surface_potential > self.primary.critical_surface_potential) and (
                    self.secondary.surface_potential > self.secondary.critical_surface_potential):
                __MORPHOLOGY__ = "detached"

            else:
                raise MorphologyError("Non-Physical system. Change stellar parameters.")
        return __MORPHOLOGY__

    def setup_discretisation_factor(self):
        """
        Adjusting discretization factors of both components to have roughly similar sizes.
        If none of the components have their discretization factors set, smaller
        component is adjusted according to bigger. If secondary discretization factor
        was not set, it will be now with respect to primary component.
        """
        def _adjust_alpha(adj_component, ref_comp):
            return ref_comp.discretization_factor * \
                   (ref_comp.equivalent_radius / adj_component.equivalent_radius) * \
                   (ref_comp.t_eff / adj_component.t_eff) ** 2

        adj_comp, adj, ref = None, None, None
        # if both components are not specified, alpha of smaller component is adjusted to the bigger component
        if not self.primary.kwargs.get('discretization_factor') \
                and not self.secondary.kwargs.get('discretization_factor'):
            if self.secondary.equivalent_radius * self.secondary.t_eff ** 2 < \
                    self.primary.equivalent_radius * self.primary.t_eff ** 2:
                adj, ref = self.secondary, self.primary
                adj_comp = 'secondary'
            else:
                adj, ref = self.primary, self.secondary
                adj_comp = 'primary'
        # if only one alpha is supplied, the second is adjusted
        elif not self.secondary.kwargs.get('discretization_factor'):
            adj, ref = self.secondary, self.primary
            adj_comp = 'secondary'
        elif not self.primary.kwargs.get('discretization_factor'):
            adj, ref = self.primary, self.secondary
            adj_comp = 'primary'

        # adjust discretization factor for given component
        if adj_comp is not None:
            adj.discretization_factor = _adjust_alpha(adj, ref)

            if adj.discretization_factor > np.radians(settings.MAX_DISCRETIZATION_FACTOR):
                adj.discretization_factor = np.radians(settings.MAX_DISCRETIZATION_FACTOR)
            if adj.discretization_factor < np.radians(settings.MIN_DISCRETIZATION_FACTOR):
                adj.discretization_factor = np.radians(settings.MIN_DISCRETIZATION_FACTOR)

            logger.info(f"setting discretization factor of {adj_comp} component to "
                        f"{up.degrees(adj.discretization_factor):.2f} "
                        f"according to discretization factor of the companion.")

        # adjust discretization factor for spots
        for component in settings.BINARY_COUNTERPARTS:
            instance: Star = getattr(self, component)
            if instance.has_spots():
                for spot in instance.spots.values():
                    if not spot.kwargs.get('discretization_factor'):
                        spot.discretization_factor = instance.discretization_factor

    def transform_input(self, **kwargs):
        """
        Transform and validate input kwargs.

        :param kwargs: Dict;
        :return: Dict;
        """
        return BinarySystemProperties.transform_input(**kwargs)

    def setup_periastron_critical_potential(self):
        """
        Compute and set critical surface potential for both components.
        Critical surface potential is for component defined as potential when component fill its Roche lobe.
        """
        for component, instance in self.components.items():
            cp = self.critical_potential(component=component, components_distance=1.0 - self.eccentricity)
            setattr(instance, "critical_surface_potential", cp)

    def critical_potential(self, component, components_distance):
        """
        Return a critical potential for target component.

        :param component: str; define target component to compute critical potential; `primary` or `secondary`
        :param components_distance: numpy.float;
        :return: numpy.float;
        """
        synchronicity = self.primary.synchronicity if component == 'primary' else self.secondary.synchronicity
        return self.critical_potential_static(component, components_distance, self.mass_ratio, synchronicity)

    @staticmethod
    def critical_potential_static(component, components_distance, mass_ratio, synchronicity):
        """
        Static method for calculation of critical potential for EB system component.

        :param component: str; define target component to compute critical potential; `primary` or `secondary`
        :param components_distance: components_distance: numpy.float;
        :param mass_ratio: float;
        :param synchronicity: float;
        :return: numpy.float;
        """
        args1 = synchronicity, mass_ratio, components_distance
        args2 = args1 + (0.0, const.HALF_PI)
        solver_err = ValueError("Iteration process to solve critical potential seems "
                                "to lead nowhere (critical potential solver has failed).")

        if component == 'primary':
            solution = optimize.newton(model.primary_potential_derivative_x, x0=1e-6, args=args1, tol=1e-12)
            if not up.isnan(solution):
                precalc_args = model.pre_calculate_for_potential_value_primary(*args2)
                args = (mass_ratio,) + precalc_args
                return abs(model.potential_value_primary(solution, *args))
            else:
                raise solver_err

        elif component == 'secondary':
            solution = optimize.newton(model.secondary_potential_derivative_x, x0=1e-6, args=args1, tol=1e-12)
            if not up.isnan(solution):
                precalc_args = model.pre_calculate_for_potential_value_secondary(*args2)
                args = (mass_ratio,) + precalc_args
                return abs(model.potential_value_secondary(components_distance - solution, *args))
            else:
                raise solver_err

        else:
            raise ValueError("Parameter `component` has incorrect value. Use `primary` or `secondary`.")

    def libration_potentials(self):
        """
        Return potentials in L3, L1, L2 respectively.

        :return: List; [Omega(L3), Omega(L1), Omega(L2)];
        """
        return self.libration_potentials_static(self.orbit.periastron_distance, self.mass_ratio)

    @staticmethod
    def libration_potentials_static(periastron_distance, mass_ratio):
        """
        Static version of `libration_potentials` where potentials in L3, L1, L2 are calculated.

        :param periastron_distance: float;
        :param mass_ratio: float;
        :return: List;
        """
        def _potential(radius):
            theta, d = const.HALF_PI, periastron_distance
            if isinstance(radius, (float, int, np.float, np.int)):
                radius = [radius]
            elif not isinstance(radius, tuple([list, np.array])):
                raise ValueError("Incorrect value of variable `radius`.")

            p_values = []
            for r in radius:
                phi, r = (0.0, r) if r >= 0 else (const.PI, abs(r))

                block_a = 1.0 / r
                block_b = mass_ratio / (up.sqrt(up.power(d, 2) + up.power(r, 2) - (
                        2.0 * r * up.cos(phi) * up.sin(theta) * d)))
                block_c = (mass_ratio * r * up.cos(phi) * up.sin(theta)) / (up.power(d, 2))
                block_d = 0.5 * (1 + mass_ratio) * up.power(r, 2) * (
                        1 - up.power(up.cos(theta), 2))

                p_values.append(block_a + block_b - block_c + block_d)
            return p_values

        lagrangian_points = BinarySystem.lagrangian_points_static(periastron_distance, mass_ratio)
        return _potential(lagrangian_points)

    def lagrangian_points(self):
        """
        Compute Lagrangian points for current system parameters.

        :return: List; x-valeus of libration points [L3, L1, L2] respectively
        """
        return self.lagrangian_points_static(self.orbit.periastron_distance, self.mass_ratio)

    @staticmethod
    def lagrangian_points_static(periastron_distance, mass_ratio):
        """
        Static version of `lagrangian_points` that returns Lagrangian points for current system parameters.

        :param periastron_distance: float;
        :param mass_ratio: float;
        :return: List; x-valeus of libration points [L3, L1, L2] respectively
        """

        def _potential_dx(x, *args):
            """
            General potential derivatives in x when::

                primary.synchornicity = secondary.synchronicity = 1.0
                eccentricity = 0.0
            :param x: (numpy.)float
            :param args: Tuple; periastron distance of components
            :return: (numpy.)float
            """
            d, = args
            r_sqr, rw_sqr = x ** 2, (d - x) ** 2
            return - (x / r_sqr ** (3.0 / 2.0)) + ((mass_ratio * (d - x)) / rw_sqr ** (
                    3.0 / 2.0)) + (mass_ratio + 1) * x - mass_ratio / d ** 2

        xs = np.linspace(- periastron_distance * 3.0, periastron_distance * 3.0, 100)

        args_val = periastron_distance,
        round_to = 10
        points, lagrange = [], []

        for x_val in xs:
            try:
                # if there is no valid value (in case close to x=0.0, _potential_dx diverge)
                old_settings = np.seterr(divide='raise', invalid='raise')
                _potential_dx(round(x_val, round_to), *args_val)
                np.seterr(**old_settings)
            except Exception as e:
                logger.debug(f"invalid value passed to potential, exception: {str(e)}")
                continue

            try:
                solution, _, ier, _ = fsolve(_potential_dx, x_val, full_output=True, args=args_val, xtol=1e-12)
                if ier == 1:
                    if round(solution[0], 5) not in points:
                        try:
                            value_dx = abs(round(_potential_dx(solution[0], *args_val), 4))
                            use = True if value_dx == 0 else False
                        except Exception as e:
                            logger.debug(f"skipping solution for x: {x_val} due to exception: {str(e)}")
                            use = False

                        if use:
                            points.append(round(solution[0], 5))
                            lagrange.append(solution[0])
                            if len(lagrange) == 3:
                                break
            except Exception as e:
                logger.debug(f"solution for x: {x_val} lead to nowhere, exception: {str(e)}")
                continue

        return sorted(lagrange) if mass_ratio < 1.0 else sorted(lagrange, reverse=True)

    def compute_equipotential_boundary(self, components_distance, plane):
        """
        Compute a equipotential boundary of components (crossection of Hill plane).

        :param components_distance: (numpy.)float;
        :param plane: str; to compute crossection with plane `xy`, `yz` or `zx`
        :return: Tuple; (numpy.array, numpy.array) - (xs, ys) in Cartesian 2-D plane
        """

        components = ['primary', 'secondary']
        points_primary, points_secondary = [], []
        fn_map = {'primary': (model.potential_primary_fn, model.pre_calculate_for_potential_value_primary),
                  'secondary': (model.potential_secondary_fn, model.pre_calculate_for_potential_value_secondary)}

        angles = np.linspace(0, const.FULL_ARC, 500, endpoint=True)
        for component in components:
            component_instance = getattr(self, component)
            synchronicity = component_instance.synchronicity

            for angle in angles:
                if utils.is_plane(plane, 'xy'):
                    args, use = (synchronicity, self.mass_ratio, components_distance, angle, const.HALF_PI), False
                elif utils.is_plane(plane, 'yz'):
                    args, use = (synchronicity, self.mass_ratio, components_distance, const.HALF_PI, angle), False
                elif utils.is_plane(plane, 'zx'):
                    args, use = (synchronicity, self.mass_ratio, components_distance, 0.0, angle), False
                else:
                    raise ValueError('Invalid choice of crossection plane, use only: `xy`, `yz`, `zx`.')

                scipy_solver_init_value = np.array([components_distance / 10000.0])
                aux_args = (self.mass_ratio,) + fn_map[component][1](*args)
                args = (aux_args, component_instance.surface_potential)
                solution, _, ier, _ = fsolve(fn_map[component][0], scipy_solver_init_value,
                                             full_output=True, args=args, xtol=1e-12)

                # check for regular solution
                if ier == 1 and not up.isnan(solution[0]):
                    solution = solution[0]
                    if 30 >= solution >= 0:
                        use = True
                else:
                    continue

                if use:
                    if utils.is_plane(plane, 'yz'):
                        if component == 'primary':
                            points_primary.append([solution * up.sin(angle), solution * up.cos(angle)])
                        elif component == 'secondary':
                            points_secondary.append([solution * up.sin(angle), solution * up.cos(angle)])
                    elif utils.is_plane(plane, 'xz'):
                        if component == 'primary':
                            points_primary.append([solution * up.sin(angle), solution * up.cos(angle)])
                        elif component == 'secondary':
                            points_secondary.append([- (solution * up.sin(angle) - components_distance),
                                                     solution * up.cos(angle)])
                    else:
                        if component == 'primary':
                            points_primary.append([solution * up.cos(angle), solution * up.sin(angle)])
                        elif component == 'secondary':
                            points_secondary.append([- (solution * up.cos(angle) - components_distance),
                                                     solution * up.sin(angle)])

        return np.array(points_primary), np.array(points_secondary)

    def get_positions_method(self):
        """
        Return method to use for orbital motion computation.

        :return: callable;
        """
        return self.calculate_orbital_motion

    def calculate_orbital_motion(self, input_argument=None, return_nparray=False, calculate_from='phase'):
        """
        Calculate orbital motion for current system parameters and supplied phases or azimuths.

        :param calculate_from: str; 'phase' or 'azimuths' parameter based on which orbital motion should be calculated
        :param return_nparray: bool; if True positions in form of numpy arrays will be also returned
        :param input_argument: numpy.array;
        :return: Tuple[List[NamedTuple: elisa.const.Position], List[Integer]] or
                 List[NamedTuple: elisa.const.Position]
        """
        input_argument = np.array([input_argument]) if np.isscalar(input_argument) else input_argument
        orbital_motion = self.orbit.orbital_motion(phase=input_argument) if calculate_from == 'phase' \
            else self.orbit.orbital_motion_from_azimuths(azimuth=input_argument)
        idx = up.arange(np.shape(input_argument)[0], dtype=np.int)
        positions = np.hstack((idx[:, np.newaxis], orbital_motion))
        # return retval, positions if return_nparray else retval
        return positions if return_nparray else [const.Position(*p) for p in positions]

    def calculate_components_radii(self, components_distance):
        """
        Calculate component radii.
        Use methods to calculate polar, side, backward and if not W UMa also
        forward radius and assign to component instance.

        :param components_distance: float; distance of components in SMA unit
        :return: Dict[str, Dict[str, float]];
        """
        fns = [bsradius.calculate_polar_radius, bsradius.calculate_side_radius, bsradius.calculate_backward_radius]
        components = settings.BINARY_COUNTERPARTS

        if self.eccentricity == 0.0:
            corrected_potential = {component: getattr(self, component).surface_potential for component in components}
        else:
            corrected_potential = self.correct_potentials(distances=np.array([components_distance, ]))
            corrected_potential = {component: corrected_potential[component][0] for component in components}

        radii = dict(primary=dict(), secondary=dict())
        for component in components:
            instance: Star = getattr(self, component)

            kwargs = dict(synchronicity=instance.synchronicity,
                          mass_ratio=self.mass_ratio,
                          components_distance=components_distance,
                          surface_potential=corrected_potential[component],
                          component=component)

            for fn in fns:
                logger.debug(f'initialising {" ".join(str(fn.__name__).split("_")[1:])} '
                             f'for {component} component')

                param_name = f'{"_".join(str(fn.__name__).split("_")[1:])}'
                r = fn(**kwargs)
                radii[component][param_name] = r

            if self.morphology != 'over-contact':
                radii[component]['forward_radius'] = bsradius.calculate_forward_radius(**kwargs)

        return radii

    def setup_components_radii(self, components_distance, calculate_equivalent_radius=True):
        """
        Setup component radii.
        Use methods to calculate equivalent, polar, side, backward and if not W UMa also
        forward radius and assign to component instance.

        :param calculate_equivalent_radius: bool; some application do not require calculation of equivalent radius
        :param components_distance: float; distance of components in SMA unit
        """
        radii = self.calculate_components_radii(components_distance)

        for component, rs in radii.items():
            instance: Star = getattr(self, component)

            for key, value in rs.items():
                setattr(instance, key, value)

            if calculate_equivalent_radius:
                setattr(instance, 'equivalent_radius', self.calculate_equivalent_radius(component)[component])

    def setup_albedos(self):
        """
        Setup of default componet albedo.
        """
        for component, instance in self.components.items():
            instance.albedo = interpolate_albedo(instance.t_eff) if utils.is_empty(instance.albedo) else instance.albedo

    @staticmethod
    def compute_filling_factor(surface_potential, lagrangian_points):
        """
        Compute filling factor of given BinaryStar system.
        Filling factor is computed as::

            (Omega_{inner} - Omega) / (Omega_{inner} - Omega_{outter}),

        where Omega_X denote potential value and `Omega` is potential of given Star.
        Inner and outter are critical inner and outter potentials for given binary star system.

        :param surface_potential: float;
        :param lagrangian_points: List; lagrangian points in `order` (in order to ensure [L3, L1, L2])
        :return: float;
        """
        return (lagrangian_points[1] - surface_potential) / (lagrangian_points[1] - lagrangian_points[2])

    def correct_potentials(self, phases=None, component="all", iterations=2, distances=None):
        """
        Function calculates potential for each phase in phases in such way that conserves
        volume of the component.

        :param phases: numpy.array; if `distances` is not `None`, phases will be not used
        :param component: str; `primary`, `secondary` or `all` (=both)
        :param iterations: int;
        :param distances: numpy.array; if not `None`, corrected potentials will be calculated
                                       for given component distances
        :return: numpy.array;
        """
        if distances is None:
            if phases is None:
                raise ValueError('Either `phases` or components `distances` have to be supplied.')

            data = self.orbit.orbital_motion(phases)
            distances = data[:, 0]
        distances = np.array(distances)
        components = bsutils.component_to_list(component)

        potentials = dict()
        for component in components:
            star = getattr(self, component)
            new_potentials = star.surface_potential * np.ones(distances.shape)

            points_equator, points_meridian = \
                self.generate_equator_and_meridian_points(
                    components_distance=1.0,
                    component=component,
                    surface_potential=star.surface_potential
                )
            volume = utils.calculate_volume_ellipse_approx(points_equator, points_meridian)
            equiv_r_mean = utils.calculate_equiv_radius(volume)

            side_radii = np.empty(distances.shape)
            volume = np.empty(distances.shape)
            for _ in range(iterations):
                for idx, pot in enumerate(new_potentials):
                    radii_args = (star.synchronicity, self.mass_ratio, distances[idx], new_potentials[idx], component)
                    side_radii[idx] = bsradius.calculate_side_radius(*radii_args)

                    points_equator, points_meridian = \
                        self.generate_equator_and_meridian_points(
                            components_distance=distances[idx],
                            component=component,
                            surface_potential=new_potentials[idx]
                        )
                    volume[idx] = utils.calculate_volume_ellipse_approx(points_equator, points_meridian)

                equiv_r = utils.calculate_equiv_radius(volume)
                coeff = equiv_r_mean / equiv_r
                corrected_side_radii = coeff * side_radii

                new_potentials = [
                    bsutils.potential_from_radius(component, corrected_side_radii[idx], const.HALF_PI,
                                                  const.HALF_PI, distance, self.mass_ratio,
                                                  star.synchronicity) for idx, distance in enumerate(distances)
                ]

            potentials[component] = np.array(new_potentials)
        return potentials

    def calculate_equivalent_radius(self, component):
        """
        Function returns equivalent radius of the given component, i.e. radius of the 
        sphere with the same volume as given component.

        :param component: str; `primary`, `secondary` or `all` (=both)
        :return: Dict; {'primary': r_equiv, ...}
        """
        components = bsutils.component_to_list(component)
        r_equiv = dict()
        for component in components:
            star = getattr(self, component)
            points_equator, points_meridian = \
                self.generate_equator_and_meridian_points(
                    components_distance=1.0,
                    component=component,
                    surface_potential=star.surface_potential
                )

            volume = utils.calculate_volume_ellipse_approx(points_equator, points_meridian)
            r_equiv[component] = utils.calculate_equiv_radius(volume)
        return r_equiv

    def calculate_bolometric_luminosity(self, components):
        """
        Calculates bolometric luminosity of given component based on its effective
        temperature and equivalent radius using black body approximation.

        :param components: str; `primary`, `secondary` or `all` (=both)
        :return: Dict; {'primary': L_bol, ...}
        """
        components = bsutils.component_to_list(components)
        r_equiv = {component: getattr(self, component).equivalent_radius for component in components}

        luminosity = dict()
        for component in components:
            star = getattr(self, component)
            luminosity[component] = 4.0 * const.PI * np.power(r_equiv[component] * self.semi_major_axis, 2
                                                              ) * const.STEFAN_BOLTZMAN_CONST * np.power(star.t_eff, 4)

        return luminosity

    def generate_equator_and_meridian_points(self, components_distance, component, surface_potential):
        """
        Function calculates a two arrays of points contouring equator and meridian calculating for the same x-values.

        :param surface_potential: float;
        :param component: str; `primary` or `secondary`
        :param components_distance: float;
        :return: Tuple[numpy.array, numpy.array]; (points on equator, points on meridian)
        """

        forward_radius, x, a = None, None, None
        star = getattr(self, component)
        discretization_factor = star.discretization_factor

        rad_args = (star.synchronicity, self.mass_ratio, components_distance, surface_potential, component)
        backward_radius = bsradius.calculate_backward_radius(*rad_args)

        # generating equidistant angles
        if self.morphology == 'detached':
            num = int(const.PI // discretization_factor)
            theta = np.linspace(discretization_factor, const.PI - discretization_factor, num=num + 1,
                                endpoint=True)

            forward_radius = bsradius.calculate_forward_radius(*rad_args)

            # generating x coordinates for both meridian and equator
            a = 0.5 * (forward_radius + backward_radius)
            c = forward_radius - a
            x = a * up.cos(theta) + c

        elif self.morphology == 'over-contact':
            num = int(const.HALF_PI // discretization_factor)
            theta = np.linspace(const.HALF_PI + discretization_factor, const.PI - discretization_factor, num=num + 1,
                                endpoint=True)

            forward_radius = mesh.calculate_neck_position(self, return_polynomial=False) \
                if component == 'primary' else 1 - mesh.calculate_neck_position(self, return_polynomial=False)

            a = 0.5 * (forward_radius + backward_radius)
            c = forward_radius - a
            x_back = a * up.cos(theta) + c

            x_front = np.linspace(forward_radius, c, num=num + 1, endpoint=True)
            x = np.concatenate((x_front, x_back))

        elif self.morphology in ['semi-detached', 'double-contact']:
            num = int(const.HALF_PI // discretization_factor)
            theta = np.linspace(const.HALF_PI + discretization_factor, const.PI - discretization_factor, num=num,
                                endpoint=True)

            forward_radius = bsradius.calculate_forward_radius(*rad_args)

            a = 0.5 * (forward_radius + backward_radius)
            c = forward_radius - a

            x_front = np.linspace(forward_radius - 0.05 * a, c, num=num + 1, endpoint=True)
            x_back = a * up.cos(theta) + c

            x = np.concatenate((x_front, x_back))

        fn_cylindrical = getattr(model, f"potential_{component}_cylindrical_fn")
        precal_cylindrical = getattr(model, f"pre_calculate_for_potential_value_{component}_cylindrical")
        cylindrical_potential_derivative_fn = getattr(model, f"radial_{component}_potential_derivative_cylindrical")

        phi1, phi2 = const.HALF_PI * np.ones(x.shape), up.zeros(x.shape)
        phi, z = up.concatenate((phi1, phi2)), up.concatenate((x, x))

        args = (phi, z, components_distance, a / 2, precal_cylindrical, fn_cylindrical,
                cylindrical_potential_derivative_fn, surface_potential, self.mass_ratio, star.synchronicity)
        points = mesh.get_surface_points_cylindrical(*args)

        if self.morphology != 'over-contact':
            # add forward and backward points to meridians
            equator_points = np.vstack(([0, 0, forward_radius],
                                        points[:points.shape[0] // 2, :],
                                        [0, 0, -backward_radius]))
            meridian_points = np.vstack(([0, 0, forward_radius],
                                         points[points.shape[0] // 2:, :],
                                         [0, 0, -backward_radius]))

            return equator_points, meridian_points
        else:
            equator_points = np.vstack((points[:points.shape[0] // 2, :], [0, 0, -backward_radius]))
            meridian_points = np.vstack((points[points.shape[0] // 2:, :], [0, 0, -backward_radius]))

            return equator_points, meridian_points

    # light curves *****************************************************************************************************
    def compute_lightcurve(self, **kwargs):
        """
        This function decides which light curve generator function is used.
        Depending on the basic properties of the binary system.

        :param kwargs: Dict; arguments to be passed into light curve generator functions
        :**kwargs options**:
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** phases ** * - numpy.array
            * ** position_method ** * - method

        :return: Dict; {`passband`: numpy.array, }
        """
        curve_fn = c_router.resolve_curve_method(self, curve='lc')
        return curve_fn(**kwargs)

    def _compute_circular_synchronous_lightcurve(self, **kwargs):
        return lc.compute_circular_synchronous_lightcurve(self, **kwargs)

    def _compute_circular_spotty_asynchronous_lightcurve(self, **kwargs):
        return lc.compute_circular_spotty_asynchronous_lightcurve(self, **kwargs)

    def _compute_circular_pulsating_lightcurve(self, **kwargs):
        return lc.compute_circular_pulsating_lightcurve(self, **kwargs)

    def _compute_eccentric_spotty_lightcurve(self, **kwargs):
        return lc.compute_eccentric_spotty_lightcurve(self, **kwargs)

    def _compute_eccentric_lightcurve(self, **kwargs):
        return lc.compute_eccentric_lightcurve_no_spots(self, **kwargs)

    # radial velocity curves *******************************************************************************************
    def compute_rv(self, **kwargs):
        """
        This function decides which radial velocities generator function is
        used depending on the user-defined method.

        :param kwargs: Dict;
        :**kwargs options**:
            * :method: str; `kinematic` (motion of the centre of mass) or
                            `radiometric` (radiance weighted contribution of each visible element)
            * :position_method: callable; method for obtaining orbital positions
            * :phases: numpy.array; photometric phases

        :return: Dict; {`primary`: numpy.array, `secondary`: numpy.array}
        """
        if kwargs['method'] == 'kinematic':
            return rv.kinematic_radial_velocity(self, **kwargs)
        elif kwargs['method'] == 'radiometric':
            curve_fn = c_router.resolve_curve_method(self, curve='rv')
            kwargs = rv_utils.include_passband_data_to_kwargs(**kwargs)
            return curve_fn(**kwargs)
        else:
            raise ValueError(f"Unknown RV computing method `{kwargs['method']}`.\n"
                             f"List of available methods: [`kinematic`, `radiometric`].")

    def _compute_circular_synchronous_rv_curve(self, **kwargs):
        return rv.compute_circular_synchronous_rv_curve(self, **kwargs)

    def _compute_circular_spotty_asynchronous_rv_curve(self, **kwargs):
        return rv.compute_circular_spotty_asynchronous_rv_curve(self, **kwargs)

    def _compute_circular_pulsating_rv_curve(self, **kwargs):
        return rv.compute_circular_pulsating_rv_curve(self, **kwargs)

    def _compute_eccentric_spotty_rv_curve(self, **kwargs):
        return rv.compute_eccentric_spotty_rv_curve(self, **kwargs)

    def _compute_eccentric_rv_curve_no_spots(self, **kwargs):
        return rv.compute_eccentric_rv_curve_no_spots(self, **kwargs)
