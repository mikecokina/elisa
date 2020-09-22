import numpy as np
import scipy

from typing import Union
from copy import deepcopy
from scipy import optimize

from . import graphic
from . orbit import orbit
from . curves import lc, rv
from . surface import mesh
from . transform import BinarySystemProperties
from . curves import c_router
from . import (
    utils as bsutils,
    radius as bsradius,
    model
)

from .. base.error import MorphologyError
from .. base.container import SystemPropertiesContainer
from .. base.system import System
from .. base.star import Star

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
    Compute and initialise minimal necessary attributes to be used in light curves computation.
    Child class of elisa.base.system.System representing BinarySystem.
    Class intherit parameters from elisa.base.system.System and add following

    Input parameters:

    :param primary: elisa.base.star.Star; instance of primary component
    :param secondary: elisa.base.star.Star; instance of secondary component
    :param eccentricity: Union[(numpy.)int, (numpy.)float];
    :param argument_of_periastron: Union[(numpy.)float, (numpy.)int, astropy.units.quantity.Quantity];
    :param period: Union[(numpy.)float, (numpy.)int, astropy.units.quantity.Quantity]; Orbital period of binary
                   star system. If unit is not specified, default period unit is assumed (days).
    :param phase_shift: float; Phase shift of the primary eclipse minimum with respect to ephemeris
                               true_phase is used during calculations, where: true_phase = phase + phase_shift.;
    :param primary_minimum_time: Union[(numpy.)float, (numpy.)int, astropy.units.quantity.Quantity];

    Output parameters (computed on init):

    :mass_ratio: float; secondary mass / primary mass
    :orbit: elisa.orbit.orbit.Orbit; instance of orbit
    :plot: elisa.binary_system.graphic.plot.Plot;
    :animation: elisa.binary_system.graphic.animation.Animation;
    :semi_major_axis: float; semi major axis of system in physical units
    :morphology: str; morphology of current system
    """

    MANDATORY_KWARGS = ['inclination', 'period', 'eccentricity', 'argument_of_periastron']
    OPTIONAL_KWARGS = ['gamma', 'phase_shift', 'additional_light', 'primary_minimum_time']
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    COMPONENT_MANDATORY_KWARGS = ['mass', 't_eff', 'gravity_darkening', 'surface_potential', 'synchronicity',
                                  'albedo', 'metallicity']
    COMPONENT_OPTIONAL_KWARGS = []
    COMPONENT_ALL_KWARGS = COMPONENT_MANDATORY_KWARGS + COMPONENT_OPTIONAL_KWARGS

    def __init__(self, primary, secondary, name=None, **kwargs):
        # initial validity checks
        utils.invalid_kwarg_checker(kwargs, BinarySystem.ALL_KWARGS, self.__class__)
        utils.check_missing_kwargs(BinarySystem.MANDATORY_KWARGS, kwargs, instance_of=BinarySystem)
        self.object_params_validity_check(dict(primary=primary, secondary=secondary), self.COMPONENT_MANDATORY_KWARGS)
        kwargs = self.transform_input(**kwargs)

        super(BinarySystem, self).__init__(name, **kwargs)

        logger.info(f"initialising object {self.__class__.__name__}")
        logger.debug(f"setting properties of components of class instance {self.__class__.__name__}")

        self.plot = graphic.plot.Plot(self)
        self.animation = graphic.animation.Animation(self)

        self.primary = primary
        self.secondary = secondary
        self.mass_ratio = self.secondary.mass / self.primary.mass
        self._components = dict(primary=self.primary, secondary=self.secondary)

        # default values of properties
        self.orbit: Union[None, orbit.Orbit] = None
        self.period = np.nan
        self.eccentricity = np.nan
        self.argument_of_periastron = np.nan
        self.primary_minimum_time = 0.0
        self.phase_shift = 0.0
        self.gamma = 0.0

        # set attributes and test whether all parameters were initialized
        # we already ensured that all kwargs are valid and all mandatory kwargs are present so lets set class attributes
        self.init_properties(**kwargs)

        # calculation of dependent parameters
        logger.debug("computing semi-major axis")
        self.semi_major_axis = self.calculate_semi_major_axis()

        # orbit initialisation (initialise class Orbit from given BinarySystem parameters)
        self.init_orbit()

        # setup critical surface potentials in periastron
        logger.debug("setting up critical surface potentials of components in periastron")
        self.setup_periastron_critical_potential()

        logger.debug("setting up morphological classification of binary system")
        self.morphology = self.compute_morphology()

        self.setup_components_radii(components_distance=self.orbit.periastron_distance)
        self.assign_pulsations_amplitudes(normalisation_constant=self.semi_major_axis)

        # adjust and setup discretization factor if necessary
        self.setup_discretisation_factor()

    @classmethod
    def from_json(cls, data, _verify=True, _kind_of=None):
        """
        Create instance of BinarySystem from JSON in form like::

            {
              "system": {
                "inclination": 90.0,
                "period": 10.1,
                "argument_of_periastron": 90.0,
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
                "metallicity": 0.0
              },
              "secondary": {
                "mass": 2.0,
                "surface_potential": 7.1,
                "synchronicity": 1.0,
                "t_eff": 6500.0,
                "gravity_darkening": 1.0,
                "discretization_factor": 5,
                "albedo": 1.0,
                "metallicity": 0.0
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
                "metallicity": 0.0
              },
              "secondary": {
                "surface_potential": 7.1,
                "synchronicity": 1.0,
                "t_eff": 6500.0,
                "gravity_darkening": 1.0,
                "discretization_factor": 5,
                "albedo": 1.0,
                "metallicity": 0.0
              }
            }
            
        Currently, this approach require values in default units used in app.

        Default units::

             {
                "inclination": [degrees],
                "period": [days],
                "argument_of_periastron": [degrees],
                "gamma": [m/s],
                "eccentricity": [dimensionless],
                "primary_minimum_time": ,
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

        :return: elisa.binary_system.system.BinarySystem
        """
        data_cp = deepcopy(data)
        if _verify:
            bsutils.validate_binary_json(data_cp)

        kind_of = _kind_of or bsutils.resolve_json_kind(data_cp)
        if kind_of in ["community"]:
            data_cp = bsutils.transform_json_community_to_std(data_cp)

        primary, secondary = Star(**data_cp["primary"]), Star(**data_cp["secondary"])
        return cls(primary=primary, secondary=secondary, **data_cp["system"])

    def to_json(self):
        """
        Serialize BinarySystem instance to json.

        :return: Dict; json like
        """

        return {
            "system": {
                "inclination":  (self.inclination * u.rad).to(u.deg).value,
                "period": self.period,
                "argument_of_periastron": (self.argument_of_periastron * u.rad).to(u.deg).value,
                "gamma": self.gamma,
                "eccentricity": self.eccentricity,
                "primary_minimum_time": self.primary_minimum_time,
                "phase_shift": self.phase_shift
            },
            "primary": {
                "mass": (self.primary.mass * u.kg).to(u.solMass).value,
                "surface_potential": self.primary.surface_potential,
                "synchronicity": self.primary.synchronicity,
                "t_eff": self.primary.t_eff,
                "gravity_darkening": self.primary.gravity_darkening,
                "discretization_factor": (self.primary.discretization_factor * u.rad).to(u.deg).value,
                "albedo": self.primary.albedo,
                "metallicity": self.primary.metallicity
            },
            "secondary": {
                "mass": (self.secondary.mass * u.kg).to(u.solMass).value,
                "surface_potential": self.secondary.surface_potential,
                "synchronicity": self.secondary.synchronicity,
                "t_eff": self.secondary.t_eff,
                "gravity_darkening": self.primary.gravity_darkening,
                "discretization_factor": (self.secondary.discretization_factor * u.rad).to(u.deg).value,
                "albedo": self.secondary.albedo,
                "metallicity": self.secondary.metallicity
            },
        }

    def init(self):
        """
        Function to reinitialize BinarySystem class instance after changing parameter(s) of binary system.
        """
        self.__init__(primary=self.primary, secondary=self.secondary, **self.kwargs_serializer())

    @property
    def components(self):
        """
        Return components object in Dict.

        :return: Dict[str, elisa.base.Star];
        """
        return self._components

    def kwargs_serializer(self):
        """
        Creating dictionary of keyword arguments of BinarySystem class in order to be able to reinitialize the class
        instance in init().

        :return: Dict;
        """
        serialized_kwargs = dict()
        for kwarg in self.ALL_KWARGS:
            if kwarg in ['argument_of_periastron', 'inclination']:
                serialized_kwargs[kwarg] = getattr(self, kwarg) * u.ARC_UNIT
            else:
                serialized_kwargs[kwarg] = getattr(self, kwarg)
        return serialized_kwargs

    def properties_serializer(self):
        props = BinarySystemProperties.transform_input(**self.kwargs_serializer())
        props.update({
            "semi_major_axis": self.semi_major_axis,
            "morphology": self.morphology,
            "mass_ratio": self.mass_ratio
        })
        return props

    def to_properties_container(self):
        return SystemPropertiesContainer(**self.properties_serializer())

    def evaluate_stype(self):
        pass

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
        Calculates length semi major axis using 3rd kepler default_law.

        :return: float;
        """
        period = np.float64((self.period * u.PERIOD_UNIT).to(u.s))
        return (const.G * (self.primary.mass + self.secondary.mass) * period ** 2 / (4 * const.PI ** 2)) ** (1.0 / 3)

    def compute_morphology(self):
        """
        Setup binary star class property `morphology`.
        It find out morphology based on current system parameters
        and setup `morphology` parameter of `self` system instance.

        :return: str;
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
        If secondary discretization factor was not set, it will be now with respect to primary component.
        """
        if not self.secondary.kwargs.get('discretization_factor'):
            self.secondary.discretization_factor = (self.primary.discretization_factor * self.primary.polar_radius
                                                    / self.secondary.polar_radius * u.rad).value

            if self.secondary.discretization_factor > np.radians(settings.MAX_DISCRETIZATION_FACTOR):
                self.secondary.discretization_factor = np.radians(settings.MAX_DISCRETIZATION_FACTOR)
            if self.secondary.discretization_factor < np.radians(settings.MIN_DISCRETIZATION_FACTOR):
                self.secondary.discretization_factor = np.radians(settings.MIN_DISCRETIZATION_FACTOR)

            logger.info(f"setting discretization factor of secondary component to "
                        f"{up.degrees(self.secondary.discretization_factor):.2f} "
                        f"according to discretization factor of the primary component.")

        for component in settings.BINARY_COUNTERPARTS.keys():
            instance = getattr(self, component)
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
        for component in settings.BINARY_COUNTERPARTS:
            setattr(
                getattr(self, component), "critical_surface_potential",
                self.critical_potential(component=component, components_distance=1.0 - self.eccentricity)
            )

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
            solution = optimize.newton(model.primary_potential_derivative_x, 0.000001, args=args1, tol=1e-12)
            if not up.isnan(solution):
                precalc_args = model.pre_calculate_for_potential_value_primary(*args2)
                args = (mass_ratio,) + precalc_args
                return abs(model.potential_value_primary(solution, *args))
            else:
                raise solver_err

        elif component == 'secondary':
            solution = optimize.newton(model.secondary_potential_derivative_x, 0.000001, args=args1, tol=1e-12)
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
        :return: list
        """
        def potential(radius):
            theta, d = const.HALF_PI, periastron_distance
            if isinstance(radius, (float, int, np.float, np.int)):
                radius = [radius]
            elif not isinstance(radius, tuple([list, np.array])):
                raise ValueError("Incorrect value of variable `radius`")

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
        return potential(lagrangian_points)

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

        def potential_dx(x, *args):
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
                # if there is no valid value (in case close to x=0.0, potential_dx diverge)
                old_settings = np.seterr(divide='raise', invalid='raise')
                potential_dx(round(x_val, round_to), *args_val)
                np.seterr(**old_settings)
            except Exception as e:
                logger.debug(f"invalid value passed to potential, exception: {str(e)}")
                continue

            try:
                solution, _, ier, _ = fsolve(potential_dx, x_val, full_output=True, args=args_val, xtol=1e-12)
                if ier == 1:
                    if round(solution[0], 5) not in points:
                        try:
                            value_dx = abs(round(potential_dx(solution[0], *args_val), 4))
                            use = True if value_dx == 0 else False
                        except Exception as e:
                            logger.debug(f"skipping sollution for x: {x_val} due to exception: {str(e)}")
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
        :param plane: str; xy, yz, zx
        :return: Tuple; (numpy.array, numpy.array)
        """

        components = ['primary', 'secondary']
        points_primary, points_secondary = [], []
        fn_map = {'primary': (model.potential_primary_fn, model.pre_calculate_for_potential_value_primary),
                  'secondary': (model.potential_secondary_fn, model.pre_calculate_for_potential_value_secondary)}

        angles = np.linspace(-3 * const.HALF_PI, const.HALF_PI, 300, endpoint=True)
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
        if return_nparray:
            return positions
        else:
            return [const.Position(*p) for p in positions]

    def setup_components_radii(self, components_distance):
        """
        Setup component radii.
        Use methods to calculate polar, side, backward and if not W UMa also
        forward radius and assign to component instance.

        :param components_distance: float;
        """
        fns = [bsradius.calculate_polar_radius, bsradius.calculate_side_radius, bsradius.calculate_backward_radius]
        components = ['primary', 'secondary']

        if self.eccentricity == 0.0:
            corrected_potential = {component: getattr(self, component).surface_potential for component in components}
        else:
            corrected_potential = self.correct_potentials(distances=np.array([components_distance, ]))
            corrected_potential = {component: corrected_potential[component][0] for component in components}

        for component in components:
            component_instance = getattr(self, component)
            for fn in fns:
                logger.debug(f'initialising {" ".join(str(fn.__name__).split("_")[1:])} '
                             f'for {component} component')

                param = f'{"_".join(str(fn.__name__).split("_")[1:])}'
                kwargs = dict(synchronicity=component_instance.synchronicity,
                              mass_ratio=self.mass_ratio,
                              components_distance=components_distance,
                              surface_potential=corrected_potential[component],
                              component=component)
                r = fn(**kwargs)
                setattr(component_instance, param, r)
                if self.morphology != 'over-contact':
                    r = bsradius.calculate_forward_radius(**kwargs)
                    setattr(component_instance, 'forward_radius', r)

            # setting value of equivalent radius
            e_rad = self.calculate_equivalent_radius(components=component)[component]
            setattr(component_instance, 'equivalent_radius', e_rad)

    @staticmethod
    def compute_filling_factor(surface_potential, lagrangian_points):
        """
        Compute filling factor of given BinaryStar system.
        Filling factor is computed as::

            (Omega_{inner} - Omega) / (Omega_{inner} - Omega_{outter}),

        where Omega_X denote potential value and `Omega` is potential of given Star.
        Inner and outter are critical inner and outter potentials for given binary star system.

        :param surface_potential: float;
        :param lagrangian_points: List; lagrangian points in `order` (in order to ensure that L2)
        :return: float;
        """
        return (lagrangian_points[1] - surface_potential) / (lagrangian_points[1] - lagrangian_points[2])

    def correct_potentials(self, phases=None, component="all", iterations=2, distances=None):
        """
        Function calculates potential for each phase in phases in such way that conserves
        volume of the component.

        :param phases: numpy.array; if `distances` is not `None`, phases will be not used
        :param component: str; `primary`, `secondary` or None (=both)
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
        else:
            distances = np.array(distances)

        retval = {}
        components = bsutils.component_to_list(component)
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

                new_potentials = np.array(
                    [bsutils.potential_from_radius(component, corrected_side_radii[idx], const.HALF_PI,
                                                   const.HALF_PI, distance, self.mass_ratio,
                                                   star.synchronicity) for idx, distance in enumerate(distances)])

            retval[component] = new_potentials
        return retval

    def calculate_equivalent_radius(self, components):
        """
        Function returns equivalent radius of the given component, i.e. radius of the sphere with the same volume as
        given component.

        :param components: str; `primary`, `secondary` or None (=both)
        :return: dict; {'primary': r_equiv, ...}
        """
        components = bsutils.component_to_list(components)
        retval = dict()
        for component in components:
            star = getattr(self, component)
            points_equator, points_meridian = \
                self.generate_equator_and_meridian_points(
                    components_distance=1.0,
                    component=component,
                    surface_potential=star.surface_potential
                )

            volume = utils.calculate_volume_ellipse_approx(points_equator, points_meridian)
            retval[component] = utils.calculate_equiv_radius(volume)

        return retval

    def calculate_bolometric_luminosity(self, components):
        """
        Calculates bolometric luminosity of given component based on its effective temperature and equivalent radius
        using black body approximation.

        :param components: str; `primary`, `secondary` or None (=both)
        :return: dict; {'primary': L_bol, ...}
        """
        components = bsutils.component_to_list(components)
        r_equiv = {component: getattr(self, component).equivalent_radius for component in components}

        retval = dict()
        for component in components:
            star = getattr(self, component)
            retval[component] = 4 * const.PI * np.power(r_equiv[component] * self.semi_major_axis, 2) * \
                                const.S_BOLTZMAN * np.power(star.t_eff, 4)

        return retval

    def generate_equator_and_meridian_points(self, components_distance, component, surface_potential):
        """
        Function calculates a two arrays of points contouring equator and meridian calculating for the same x-values.

        :param surface_potential: float;
        :param component: string; `primary` or `secondary`
        :param components_distance: float;
        :return: Tuple; (points on equator, points on meridian)
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

        phi1 = const.HALF_PI * np.ones(x.shape)
        phi2 = up.zeros(x.shape)
        phi = up.concatenate((phi1, phi2))
        z = up.concatenate((x, x))

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
            equator_points = np.vstack((points[:points.shape[0] // 2, :],
                                        [0, 0, -backward_radius]))
            meridian_points = np.vstack((points[points.shape[0] // 2:, :],
                                         [0, 0, -backward_radius]))

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
            * ** atlas ** * - str
            * ** phases ** * - numpy.array
            * ** position_method ** * - method
        :return: Dict
        """
        fn_arr = (self._compute_circular_synchronous_lightcurve,
                  self._compute_circular_spotty_asynchronous_lightcurve,
                  self._compute_eccentric_spotty_lightcurve,
                  self._compute_eccentric_lightcurve)
        curve_fn = c_router.resolve_curve_method(self, fn_arr)

        return curve_fn(**kwargs)

    def _compute_circular_synchronous_lightcurve(self, **kwargs):
        return lc.compute_circular_synchronous_lightcurve(self, **kwargs)

    def _compute_circular_spotty_asynchronous_lightcurve(self, **kwargs):
        return lc.compute_circular_spotty_asynchronous_lightcurve(self, **kwargs)

    def _compute_eccentric_spotty_lightcurve(self, **kwargs):
        return lc.compute_eccentric_spotty_lightcurve(self, **kwargs)

    def _compute_eccentric_lightcurve(self, **kwargs):
        return lc.compute_eccentric_lightcurve_no_spots(self, **kwargs)

    # radial velocity curves *******************************************************************************************
    def compute_rv(self, **kwargs):
        if kwargs['method'] == 'point_mass':
            return rv.com_radial_velocity(self, **kwargs)
        if kwargs['method'] == 'radiometric':
            fn_arr = (self._compute_circular_synchronous_rv_curve,
                      self._compute_circular_spotty_asynchronous_rv_curve,
                      self._compute_eccentric_spotty_rv_curve,
                      self._compute_eccentric_rv_curve_no_spots)
            curve_fn = c_router.resolve_curve_method(self, fn_arr)

            kwargs = rv.include_passband_data_to_kwargs(**kwargs)
            return curve_fn(**kwargs)

    def _compute_circular_synchronous_rv_curve(self, **kwargs):
        return rv.compute_circular_synchronous_rv_curve(self, **kwargs)

    def _compute_circular_spotty_asynchronous_rv_curve(self, **kwargs):
        return rv.compute_circular_spotty_asynchronous_rv_curve(self, **kwargs)

    def _compute_eccentric_spotty_rv_curve(self, **kwargs):
        return rv.compute_eccentric_spotty_rv_curve(self, **kwargs)

    def _compute_eccentric_rv_curve_no_spots(self, **kwargs):
        return rv.compute_eccentric_rv_curve_no_spots(self, **kwargs)
