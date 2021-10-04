import numpy as np

from scipy import optimize
from copy import deepcopy

from . orbit import orbit
from . curves import lc, rv, c_router
from . transform import SingleSystemProperties
from . import (
    model,
    graphic,
    radius as sradius,
    utils as sys_utils
)
from . container import SinglePositionContainer
from .. logger import getLogger
from .. import const
from .. import (
    units,
    utils,
    const as c,
)
from .. base.system import System
from .. base.curves import utils as rv_utils
from .. opt.fsolver import fsolve
from .. base.star import Star

logger = getLogger('single_system.system')


class SingleSystem(System):
    """
    Class to store and calculate necessary properties of the single star system based on the user provided parameters.
    Child class of elisa.base.system.System.

    Class can be imported directly:
    ::

        from elisa import SingleSystem

    After initialization, apart from the attributes already defined by the user with the arguments, user has access to
    the following attributes:

        :angular_velocity: float; angular velocity of the stellar rotation

    `SingleSystem' requires instance of elisa.base.star.Star in `star` argument with the following
    mandatory arguments:

        :param mass: float; If mass is int, np.int, float, np.float, program assumes solar mass as it's unit.
                            If mass astropy.unit.quantity.Quantity instance, program converts it to default units.
        :param t_eff: float; Accepts value in any temperature unit. If your input is without unit,
                             function assumes that supplied value is in K.
        :param polar_log_g: float; log_10 of the polar surface gravity

    following mandatory arguments are also available:

        :param metallicity: float; log[M/H] default value is 0.0
        :param gravity_darkening: float; gravity darkening factor, if not supplied, it is interpolated from Claret 2003
                                         based on t_eff

    Each component instance will after initialization contain following attributes:

        :critical_surface_potential: float; potential of the star required to fill its Roche lobe
        :equivalent_radius: float; radius of a sphere with the same volume as a component
        :polar_radius: float; radius of a star towards the pole of the star
        :equatorial_radius: float; radius of a star towards the pole of the star

    The SingleSystem can be initialized either by using valid class arguments, e.g.:
    ::

        from astropy import units as u

        from elisa.single_system.system import SingleSystem
        from elisa.base.star import Star

        star = Star(
            mass=1.0*u.solMass,
            t_eff=5772*u.K,
            gravity_darkening=0.32,
            polar_log_g=4.43775*u.dex(u.cm/u.s**2),
            metallicity=0.0,
            discretization_factor=2
        )
        
        system = SingleSystem(
            star=star,
            gamma=0*u.km/u.s,
            inclination=90*u.deg,
            rotation_period=25.380*u.d,
            reference_time=0.0*u.d
        )

    or by using the SingleSystem.from_json(dict) function that accepts various parameter combination in form of
    dictionary such as:
    ::

        data = {
            "system": {
                "inclination": 90.0,
                "rotation_period": 10.1,
                "gamma": '10000 K',  # you can define quantity using a string representation of the astropy quantities
                "reference_time": 0.5,
                "phase_shift": 0.0
            },
            "star": {
                "mass": 1.0,
                "t_eff": 5772.0,
                "gravity_darkening": 0.32,
                "discretization_factor": 5,
                "metallicity": 0.0,
                "polar_log_g": "4.43775 dex(cm.s-2)"   # you can use logarithmic units using json/dict input
            }
        }

        single = SingleSystem.from_json(data)

    See documentation for `from_json` method for details.

    The rotation of the binary system can be modelled using function
    `calculate_lines_of_sight(phases)`. E.g.:
    ::

        single_instance.calculate_lines_of_sight(np.linspace(0, 1))

    The class contains substantial plotting capability in the SingleSystem.plot module that contains following
    functions (further info in: see elisa.single_system.graphics.plot):

        - equipotential(args): zx cross-sections of equipotential surface
        - mesh(args): 3D mesh (scatter) plot of the surface points
        - wireframe(args): wire frame model of the star
        - surface(args): plot model of the star with various surface colormaps (gravity_acceleration,
          temperature, radiance, ...)

    Plot function can be called as function of the plot module. E.g.:
    ::

        single_instance.plot.surface(phase=0.1, colormap='temperature)

    Similarly, an animation of the orbital motion can be produced using SingleSystem.animation module and its function
    `rotational_motion(*args)`.

    List of valid system arguments:

    :param star: elisa.base.star.Star; instance of the single star
    :param inclination: Union[float, astropy.unit.quantity.Quantity]; Inclination of the system.
                        If unit is not supplied, value in degrees is assumed.
    :param rotational_period: Union[(numpy.)float, (numpy.)int, astropy.units.quantity.Quantity]; Orbital period of
                              binary star system. If unit is not specified, default period unit is assumed (days).
    :param reference_time: Union[(numpy.)float, (numpy.)int, astropy.units.quantity.Quantity];
    :param phase_shift: float; Phase shift of the primary eclipse with respect to the ephemeris.
                               true_phase is used during calculations, where: true_phase = phase + phase_shift.
    :param additional_light: float; fraction of light that does not originate from the `BinarySystem`
    :param gamma: Union[float, astropy.unit.quantity.Quantity]; Center of mass velocity.
                  Expected type is astropy.units.quantity.Quantity, numpy.float or numpy.int
                  otherwise TypeError will be raised. If unit is not specified, default velocity unit is assumed (m/s).
    """

    MANDATORY_KWARGS = ['inclination', 'rotation_period']
    OPTIONAL_KWARGS = ['reference_time', 'phase_shift', 'additional_light', 'gamma']
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    STAR_MANDATORY_KWARGS = ['mass', 't_eff', 'polar_log_g']
    STAR_OPTIONAL_KWARGS = ['metallicity', 'gravity_darkening']
    STAR_ALL_KWARGS = STAR_MANDATORY_KWARGS + STAR_OPTIONAL_KWARGS

    def __init__(self, star, name=None, **kwargs):
        utils.invalid_kwarg_checker(kwargs, SingleSystem.ALL_KWARGS, self.__class__)
        utils.check_missing_kwargs(SingleSystem.MANDATORY_KWARGS, kwargs, instance_of=SingleSystem)
        self.object_params_validity_check(dict(star=star), self.STAR_MANDATORY_KWARGS)
        kwargs = self.transform_input(**kwargs)

        super(SingleSystem, self).__init__(name, **kwargs)

        logger.info(f"initialising object {self.__class__.__name__}")
        logger.debug(f"setting properties of a star in class instance {self.__class__.__name__}")

        self.plot = graphic.plot.Plot(self)
        self.animation = graphic.animation.Animation(self)

        self.star = star
        self._components = dict(star=self.star)

        # default values of properties
        self.orbit = None
        self.rotation_period = None
        self.reference_time = 0
        self.angular_velocity = None
        self.period = self.rotation_period
        self.phase_shift = 0.0

        # set attributes and test whether all parameters were initialized
        # we already ensured that all kwargs are valid and all mandatory kwargs are present so lets set class attributes
        self.init_properties(**kwargs)

        # calculation of dependent parameters
        self.angular_velocity = orbit.angular_velocity(self.rotation_period)
        self.star.surface_potential = model.surface_potential_from_polar_log_g(self.star.polar_log_g, self.star.mass)

        # orbit initialisation (initialise class Orbit from given BinarySystem parameters)
        self.init_orbit()

        self.setup_critical_potential()
        # checking if star is below break-up rotational velocity
        self.check_stability()

        # this is also check if star surface is closed
        self.setup_radii(calculate_equivalent_radius=True)
        self.setup_betas()
        self.assign_pulsations_amplitudes()
        self.setup_discretisation_factor()

        # setting common reference to emphemeris
        self.period = self.rotation_period
        self.t0 = self.reference_time

    @property
    def default_input_units(self):
        """
        Returns set of default units of intialization parameters, in case, when provided without units.

        :return: elisa.units.DefaultSingleSystemInputUnits;
        """
        return units.DefaultSingleSystemInputUnits

    @property
    def default_internal_units(self):
        """
        Returns set of internal units of system parameters.

        :return: elisa.units.DefaultSingleSystemUnits;
        """
        return units.DefaultSingleSystemUnits

    @classmethod
    def from_json(cls, data, _verify=True, _kind_of=None):
        """
        Create instance of BinarySystem from JSON in `standard` or `radius` format. Example of a `standard` format of
        parameters::

            {
                "system": {
                    "inclination": 90.0,
                    "rotation_period": 10.1,
                    "gamma": 10000,
                    "reference_time": 0.5,
                    "phase_shift": 0.0
                },
                "star": {
                    "mass": 1.0,
                    "t_eff": 5772.0,
                    "gravity_darkening": 0.32,
                    "discretization_factor": 5,
                    "metallicity": 0.0,
                    "polar_log_g": "4.43775 dex(cm.s-2)"   # you can also use logarithmic units using json/dict input
                }
            }

        Example of a `radius` format of parameters::

            {
                "system": {
                    "inclination": 90.0,
                    "rotation_period": 10.1,
                    "gamma": 10000,
                    "reference_time": 0.5,
                    "phase_shift": 0.0
                },
                "star": {
                    "mass": 1.0,
                    "t_eff": 5772.0,
                    "gravity_darkening": 0.32,
                    "discretization_factor": 5,
                    "metallicity": 0.0,
                    "equivalent_radius": "1.0 solRad"
                }
            }

        Default units::

             {
                "inclination": [degrees],
                "rotational_period": [days],
                "gamma": [m/s],
                "reference_time": [d],
                "phase_shift": [dimensionless],
                "mass": [solMass],
                "surface_potential": [dimensionless],
                "synchronicity": [dimensionless],
                "t_eff": [K],
                "gravity_darkening": [dimensionless],
                "discretization_factor": [degrees],
                "metallicity": [dimensionless],
                "semi_major_axis": [solRad],
                "mass_ratio": [dimensionless]
                "polar_log_g": [dex(m*s-2)]
                "equivalent_radius": [solRad]
            }

        :return: elisa.single_system.system.SingleSystem
        """
        data_cp = deepcopy(data)
        if _verify:
            sys_utils.validate_single_json(data_cp)

        kind_of = _kind_of or sys_utils.resolve_json_kind(data_cp)
        if kind_of in ["radius"]:
            data_cp = sys_utils.transform_json_radius_to_std(data_cp)

        star = Star(**data_cp["star"])
        return cls(star=star, **data_cp["system"])

    def build_container(self, phase=None, time=None, build_pulsations=True):
        """
        Function returns `OrbitalPositionContainer` with fully built model binary system at user-defined photometric
        phase or time of observation.

        :param time: float; JD
        :param phase: float; photometric phase
        :param build_pulsations: bool;
        :return: elisa.binary_system.container.OrbitalPositionContainer;
        """
        if phase is not None and time is not None:
            raise ValueError('Please specify whether you want to build your container EITHER at given photometric '
                             '`phase` or at given `time`.')
        phase = phase if time is None else utils.jd_to_phase(time, period=self.period, t0=self.reference_time)

        position = self.calculate_lines_of_sight(input_argument=phase, return_nparray=False, calculate_from='phase')[0]
        position_container = SinglePositionContainer.from_single_system(self, position)
        position_container.build(build_pulsations=build_pulsations)

        logger.info(f'Orbital position container was successfully built at photometric phase {phase:.2f}.')
        return position_container

    @classmethod
    def is_property(cls, kwargs):
        """
        Method for checking if keyword arguments are valid properties of this class.

        :param kwargs: Dict;
        """
        is_not = ['`{}`'.format(k) for k in kwargs if k not in cls.ALL_KWARGS]
        if is_not:
            raise AttributeError('Arguments {} are not valid {} properties.'.format(', '.join(is_not), cls.__name__))

    def critical_break_up_radius(self):
        """
        returns critical, break-up equatorial radius for given mass and rotational period

        :return: float
        """
        return np.power(c.G * self.star.mass / np.power(self.angular_velocity, 2), 1.0 / 3.0)

    def critical_break_up_velocity(self):
        """
        returns critical, break-up equatorial rotational velocity for given mass and rotational period

        :return: float
        """
        return np.power(c.G * self.star.mass * self.angular_velocity, 1.0 / 3.0)

    def get_info(self):
        pass

    # _________________________AFTER_REFACTOR___________________________________

    def init(self):
        """
        function to reinitialize SingleSystem class instance after changing parameter(s) of binary system using setters

        :return:
        """
        logger.info(f're/initialising class instance {SingleSystem.__name__}')
        self.__init__(star=self.star, **self.kwargs_serializer())

    def init_orbit(self):
        """
        Initialization of single system orbit class. Similar in functionality of Orbit in Binary system. Simulates
        apparent motion of the observer around single star as the star rotates.
        """
        logger.debug(f"re/initializing orbit in class instance {self.__class__.__name__} / {self.name}")
        orbit_kwargs = {key: getattr(self, key) for key in orbit.Orbit.ALL_KWARGS}
        self.orbit = orbit.Orbit(**orbit_kwargs)

    def setup_discretisation_factor(self):
        if self.star.has_spots():
            for spot in self.star.spots.values():
                if not spot.kwargs.get('discretization_factor'):
                    spot.discretization_factor = self.star.discretization_factor

    @staticmethod
    def is_eccentric():
        """
        Resolve whether system is eccentric.

        :return: bool;
        """
        return False

    def calculate_radii(self):
        """
        Calculates important radii.

        :return: Dict;
        """
        fns = [sradius.calculate_polar_radius, sradius.calculate_equatorial_radius]
        radii = dict(star=dict())
        for fn in fns:
            logger.debug(f'initialising {" ".join(str(fn.__name__).split("_")[1:])} for the star')
            param = f'{"_".join(str(fn.__name__).split("_")[1:])}'
            kwargs = dict(mass=self.star.mass,
                          angular_velocity=self.angular_velocity,
                          surface_potential=self.star.surface_potential)
            try:
                r = fn(**kwargs)
            except Exception as e:
                raise ValueError(f'Function {fn.__name__} was not able to calculate its radius. '
                                 f'Your system is not physical. Exception: {str(e)}')
            radii['star'][param] = r

        return radii

    def setup_radii(self, calculate_equivalent_radius=True):
        """
        Auxiliary function for calculation of important radii.
        """
        radii = self.calculate_radii()
        instance: Star = getattr(self, 'star')

        for key, value in radii['star'].items():
            setattr(instance, key, value)

        if calculate_equivalent_radius:
            setattr(instance, 'equivalent_radius', self.calculate_equivalent_radius())

    @property
    def components(self):
        """
        Return components object in Dict.

        :return: Dict[str, elisa.base.Star]
        """
        return self._components

    def calculate_equipotential_boundary(self):
        """
        calculates a equipotential boundary of star in zx(yz) plane

        :return: Tuple; (np.array, np.array)
        """
        points = []
        angles = np.linspace(0, c.FULL_ARC, 300, endpoint=True)
        init_val = - c.G * self.star.mass / self.star.surface_potential
        scipy_solver_init_value = np.array([init_val])

        for angle in angles:
            precalc_args = (self.star.mass, self.angular_velocity, angle)
            argss = (model.pre_calculate_for_potential_value(*precalc_args), self.star.surface_potential)
            solution, _, ier, _ = fsolve(model.potential_fn, scipy_solver_init_value, full_output=True, args=argss)
            if ier == 1 and not np.isnan(solution[0]):
                solution = solution[0]
            else:
                continue

            points.append([solution * np.sin(angle), solution * np.cos(angle)])
        return np.array(points)

    def properties_serializer(self):
        props = SingleSystemProperties.transform_input(**self.kwargs_serializer())
        props.update({
            "angular_velocity": self.angular_velocity,
        })
        return props

    def transform_input(self, **kwargs):
        """
        Transform and validate input kwargs.
        :param kwargs: Dict
        :return: Dict
        """
        return SingleSystemProperties.transform_input(**kwargs)

    def setup_critical_potential(self):
        self.star.critical_surface_potential = self.calculate_critical_potential()

    def calculate_critical_potential(self):
        """
        Compute and set critical surface potential.
        Critical surface potential is potential when component is stable for give mass and rotaion period.
        """
        precalc_args = self.star.mass, self.angular_velocity, c.HALF_PI
        args = (model.pre_calculate_for_potential_value(*precalc_args), 0.0)

        x0 = - c.G * self.star.mass / self.star.surface_potential
        solution = optimize.newton(model.radial_potential_derivative, x0, args=args[0], tol=1e-12)
        if np.isnan(solution):
            raise ValueError("Iteration process to solve critical potential seems "
                             "to lead nowhere (critical potential solver has failed).")
        return model.potential_fn(solution, *args)

    def check_stability(self):
        """
        Checks if star is rotationally stable.
        """
        if self.star.critical_surface_potential < self.star.surface_potential:
            raise ValueError('Non-physical system. Star rotation is above critical break-up velocity.')

    def get_positions_method(self):
        return self.calculate_lines_of_sight

    def calculate_lines_of_sight(self, input_argument=None, return_nparray=False, calculate_from='phase'):
        """
        Returns vectors oriented in direction star -> observer for given set of phases or azimuths.

        :param calculate_from: str; 'phase' or 'azimuths' parameter based on which the orbital motion should be
                                    calculated
        :param return_nparray: bool; if True positions in form of numpy arrays will be also returned
        :param input_argument: numpy.array;
        :return: Tuple[List[NamedTuple: elisa.const.Position], List[Integer]] or
                 List[NamedTuple: elisa.const.Position]
        """
        input_argument = np.array([input_argument]) if np.isscalar(input_argument) else input_argument
        rotational_motion = self.orbit.rotational_motion(phase=input_argument) if calculate_from == 'phase' \
            else self.orbit.rotational_motion_from_azimuths(azimuth=input_argument)
        idx = np.arange(np.shape(input_argument)[0], dtype=np.int)[:, np.newaxis]
        positions = np.hstack((idx, np.full(idx.shape, np.nan), rotational_motion))

        return positions if return_nparray else [const.Position(*p) for p in positions]

    def calculate_equivalent_radius(self):
        """
        Function returns equivalent radius of the star, i.e. radius of the sphere with the same volume as
        given component.

        :return: float;
        """
        volume = sys_utils.calculate_volume(self)
        return utils.calculate_equiv_radius(volume)

    # light curves *****************************************************************************************************
    def compute_lightcurve(self, **kwargs):
        """
        This function decides which light curve generator function is used.
        Depending on the basic properties of the binary system.

        :param kwargs: Dict; arguments to be passed into light curve generator functions
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** phases ** * - numpy.array
            * ** position_method ** * - method
        :return: Dict; {`passband`: numpy.array, }
        """
        fn_arr = (self._compute_light_curve_without_pulsations, self._compute_light_curve_with_pulsations)

        curve_fn = c_router.resolve_curve_method(self, fn_arr)
        return curve_fn(**kwargs)

    def _compute_light_curve_with_pulsations(self, **kwargs):
        return lc.compute_light_curve_with_pulsations(self, **kwargs)

    def _compute_light_curve_without_pulsations(self, **kwargs):
        return lc.compute_light_curve_without_pulsations(self, **kwargs)

    # radial velocity curves *******************************************************************************************
    def compute_rv(self, **kwargs):
        """
        This function decides which radial velocities generator function is
        used depending on the user-defined method.

        :param kwargs: Dict;
        :**kwargs options**:
            * :method: str; `kinematic` (motion of the centre of mass) or
                            `radiometric` (radiance weighted contribution of each visible element)
            * :position_method: callable; method for obtaining orientation of the star
            * :phases: numpy.array; photometric phases

        :return: Dict; {`star`: numpy.array}
        """
        if kwargs['method'] == 'kinematic':
            return rv.com_radial_velocity(self, **kwargs)
        if kwargs['method'] == 'radiometric':
            fn_arr = (self._compute_rv_curve_without_pulsations,
                      self._compute_rv_curve_with_pulsations)
            curve_fn = c_router.resolve_curve_method(self, fn_arr)

            kwargs = rv_utils.include_passband_data_to_kwargs(**kwargs)
            return curve_fn(**kwargs)

    def _compute_rv_curve_with_pulsations(self, **kwargs):
        return rv.compute_rv_curve_with_pulsations(self, **kwargs)

    def _compute_rv_curve_without_pulsations(self, **kwargs):
        return rv.compute_rv_curve_without_pulsations(self, **kwargs)
