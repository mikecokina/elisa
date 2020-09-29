import numpy as np

from scipy import optimize
from . orbit import orbit
from . curves import lc
from . transform import SingleSystemProperties
from . import (
    model,
    graphic,
    radius as sradius,
    utils as sys_utils
)
from .. logger import getLogger
from .. import const
from .. import (
    utils,
    const as c,
)
from .. base.system import System
from .. opt.fsolver import fsolve

logger = getLogger('single_system.system')


class SingleSystem(System):
    """
    Compute and initialise minmal necessary attributes to be used in light curves computation.
    """

    MANDATORY_KWARGS = ['gamma', 'inclination', 'rotation_period']
    OPTIONAL_KWARGS = ['reference_time', 'phase_shift']
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    STAR_MANDATORY_KWARGS = ['mass', 't_eff', 'gravity_darkening', 'polar_log_g', 'metallicity']
    STAR_OPTIONAL_KWARGS = []
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
        self.setup_radii()
        self.assign_pulsations_amplitudes()
        self.setup_discretisation_factor()

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

    def setup_radii(self):
        """
        Auxiliary function for calculation of important radii.
        """
        fns = [sradius.calculate_polar_radius, sradius.calculate_equatorial_radius]
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
            setattr(self.star, param, r)

        setattr(self.star, 'equivalent_radius', self.calculate_equivalent_radius())

    @property
    def components(self):
        """
        Return components object in Dict.

        :return: Dict[str, elisa.base.Star]
        """
        return self._components

    def kwargs_serializer(self):
        """
        creating dictionary of keyword arguments of SingleSystem class in order to be able to reinitialize the class
        instance in init()

        :return: dict
        """
        serialized_kwargs = {}
        for kwarg in self.ALL_KWARGS:
            serialized_kwargs[kwarg] = getattr(self, kwarg)
        return serialized_kwargs

    def calculate_equipotential_boundary(self):
        """
        calculates a equipotential boundary of star in zx(yz) plane

        :return: tuple; (np.array, np.array)
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
        solution = optimize.newton(model.radial_potential_derivative, x0, args=args, tol=1e-12)
        if not np.isnan(solution):
            return model.potential_fn(solution, *args)
        else:
            raise ValueError("Iteration process to solve critical potential seems "
                             "to lead nowhere (critical potential solver has failed).")

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
        Returns vector oriented in direction star - observer.

        :param calculate_from: str; 'phase' or 'azimuths' parameter based on which orbital motion should be calculated
        :param return_nparray: bool; if True positions in form of numpy arrays will be also returned
        :param input_argument: numpy.array;
        :return: Tuple[List[NamedTuple: elisa.const.SinglePosition], List[Integer]] or
                 List[NamedTuple: elisa.const.SinglePosition]
        """
        input_argument = np.array([input_argument]) if np.isscalar(input_argument) else input_argument
        rotational_motion = self.orbit.rotational_motion(phase=input_argument) if calculate_from == 'phase' \
            else self.orbit.rotational_motion_from_azimuths(azimuth=input_argument)
        idx = np.arange(np.shape(input_argument)[0], dtype=np.int)
        positions = np.hstack((idx[:, np.newaxis], rotational_motion))

        if return_nparray:
            return positions
        else:
            return [const.SinglePosition(*p) for p in positions]

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
            * ** atlas ** * - str
            * ** phases ** * - numpy.array
            * ** position_method ** * - method
        :return: Dict
        """
        if self.star.has_pulsations():
            logger.debug('calculating light curve for a non pulsating single star system')
            return self._compute_light_curve_with_pulsations(**kwargs)
        else:
            logger.debug('calculating light curve for star system with pulsations')
            return self._compute_light_curve_without_pulsations(**kwargs)

    def _compute_light_curve_with_pulsations(self, **kwargs):
        return lc.compute_light_curve_with_pulsations(self, **kwargs)

    def _compute_light_curve_without_pulsations(self, **kwargs):
        return lc.compute_light_curve_without_pulsations(self, **kwargs)
