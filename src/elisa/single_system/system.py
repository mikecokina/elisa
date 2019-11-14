import numpy as np
import scipy

from scipy import optimize
from elisa.single_system.orbit import orbit
from elisa.logger import getLogger
from elisa.single_system.transform import SingleSystemProperties

from elisa import (
    utils,
    const as c,
)
from elisa.base.system import System
from elisa.single_system import (
    build,
    model,
    graphic,
    radius as sradius,
)

logger = getLogger('single_system.system')


class SingleSystem(System):
    """
    Compute and initialise minmal necessary attributes to be used in light curves computation.
    """

    MANDATORY_KWARGS = ['gamma', 'inclination', 'rotation_period']
    OPTIONAL_KWARGS = ['reference_time', 'phase_shift']
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    STAR_MANDATORY_KWARGS = ['mass', 't_eff', 'gravity_darkening', 'polar_log_g']
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

    def build_surface(self, return_surface=False):
        """
        function for building of general system component points and surfaces including spots

        :param return_surface: bool - if true, function returns arrays with all points and faces (surface + spots)
        :type: str
        :return:
        """
        return build.build_surface(self, return_surface=return_surface)

    def build_surface_map(self, colormap=None, return_map=False):
        """
        function calculates surface maps (temperature or gravity acceleration) for star and spot faces and it can return
        them as one array if return_map=True

        :param return_map: if True function returns arrays with surface map including star and spot segments
        :param colormap: str - `temperature` or `gravity`
        :return:
        """
        return build.build_surface_map(self, colormap=colormap, return_map=return_map)

    def get_positions_method(self):
        return self.calculate_lines_of_sight

    def calculate_lines_of_sight(self, phase=None):
        """
        returns vector oriented in direction star - observer

        :param phase: list
        :return: np.array([index, spherical coordinates of line of sight vector])
        """
        idx = np.arange(np.shape(phase)[0], dtype=np.int)

        line_of_sight_spherical = np.empty((np.shape(phase)[0], 3), dtype=np.float)
        line_of_sight_spherical[:, 0] = 1
        line_of_sight_spherical[:, 1] = c.FULL_ARC * phase
        line_of_sight_spherical[:, 2] = self.inclination
        line_of_sight = utils.spherical_to_cartesian(line_of_sight_spherical)
        return np.hstack((idx[:, np.newaxis], line_of_sight))

    def build(self, *args, **kwargs):
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

    def setup_radii(self):
        """
        auxiliary function for calculation of important radii
        :return:
        """
        fns = [sradius.calculate_polar_radius, sradius.calculate_equatorial_radius]
        star = self.star
        for fn in fns:
            logger.debug(f'initialising {" ".join(str(fn.__name__).split("_")[1:])} '
                               f'for the star')
            param = f'{"_".join(str(fn.__name__).split("_")[1:])}'
            kwargs = dict(mass=star.mass,
                          angular_velocity=self.angular_velocity,
                          surface_potential=star.surface_potential)
            try:
                r = fn(**kwargs)
            except:
                raise ValueError(f'Function {fn.__name__} was not able to calculate its radius. '
                                 f'Your system is not physical')
            setattr(star, param, r)

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
        pass

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
            solution, _, ier, _ = scipy.optimize.fsolve(model.potential_fn, scipy_solver_init_value,
                                                        full_output=True, args=argss)
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
        Critical surface potential is for component defined as potential when component fill its Roche lobe.
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
        checks if star is rotationally stable
        :return:
        """
        if self.star.critical_surface_potential < self.star.surface_potential:
            raise ValueError('Non-physical system. Star rotation is above critical break-up velocity.')


