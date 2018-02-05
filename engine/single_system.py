import logging
from engine.system import System
from engine.star import Star
import numpy as np
import scipy
from scipy.spatial import Delaunay
from engine import graphics
from engine import const as c
from astropy import units as u
from engine import units as U
from engine import utils

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : [%(levelname)s] : %(name)s : %(message)s')


class SingleSystem(System):
    KWARGS = ['star', 'gamma', 'inclination', 'rotation_period', 'polar_log_g']
    OPTIONAL_KWARGS = ['discretization_factor']
    ALL_KWARGS = KWARGS + OPTIONAL_KWARGS

    def __init__(self, name=None, **kwargs):
        self.is_property(kwargs)
        super(SingleSystem, self).__init__(name=name, **kwargs)

        # get logger
        self._logger = logging.getLogger(SingleSystem.__name__)
        self._logger.info("Initialising object {}".format(SingleSystem.__name__))

        self._logger.debug("Setting property components "
                           "of class instance {}".format(SingleSystem.__name__))

        # in case of SingleStar system there is no need for user to define stellar component because it is defined here
        self.star = kwargs['star']

        # check if star object doesn't contain any meaningless parameters
        meaningless_params = {'synchronicity': self.star.synchronicity,
                              'backward radius': self.star.backward_radius,
                              'forward_radius': self.star.forward_radius,
                              'side_radius': self.star.side_radius}
        for parameter in meaningless_params:
            if meaningless_params[parameter] is not None:
                meaningless_params[parameter] = None
                self._logger.info('Parameter `{0}` is meaningless in case of single star system.\n '
                                  'Setting parameter `{0}` value to None'.format(parameter))

        # default values of properties
        self._inclination = None
        self._polar_log_g = None
        self._rotation_period = None
        self._discretization_factor = None

        # testing if parameters were initialized
        missing_kwargs = []
        for kwarg in SingleSystem.KWARGS:
            if kwarg not in kwargs:
                missing_kwargs.append("`{}`".format(kwarg))
                self._logger.error("Property {} "
                                   "of class instance {} was not initialized".format(kwarg, SingleSystem.__name__))
            else:
                self._logger.debug("Setting property {} "
                                   "of class instance {} to {}".format(kwarg, SingleSystem.__name__, kwargs[kwarg]))
                setattr(self, kwarg, kwargs[kwarg])

        if len(missing_kwargs) != 0:
            raise ValueError('Mising argument(s): {} in class instance {}'.format(', '.join(missing_kwargs),
                                                                                  SingleSystem.__name__))

        # setting of optional parameters
        if 'discretization_factor' in kwargs:
            setattr(self, 'discretization_factor', kwargs['discretization_factor'])
        else:
            setattr(self, 'discretization_factor', 5)

        # calculation of dependent parameters
        self._angular_velocity = self.angular_velocity(self.rotation_period)
        self.star._polar_log_g = self.polar_log_g
        self.star._polar_gravity_acceleration = np.power(10, self.polar_log_g)  # surface polar gravity
        self.star._polar_radius = self.calculate_polar_radius()
        args = 0,
        self.star._surface_potential = self.surface_potential(self.star.polar_radius, args)[0]
        # this is also check if star surface is closed
        self.star._equatorial_radius = self.calculate_equatorial_radius()

    def init(self):
        """
        function to reinitialize SingleSystem class instance after changing parameter(s) of binary system using setters

        :return:
        """
        self.__init__(mass=self.star.mass, gamma=self.gamma, inclination=self._inclination)

    @classmethod
    def is_property(cls, kwargs):
        """
        method for checking if keyword arguments are valid properties of this class

        :param kwargs: dict
        :return:
        """
        is_not = ['`{}`'.format(k) for k in kwargs if k not in cls.ALL_KWARGS]
        if is_not:
            raise AttributeError('Arguments {} are not valid {} properties.'.format(', '.join(is_not), cls.__name__))

    @property
    def rotation_period(self):
        """
        returns rotation period of single system star in default period unit
        :return: float
        """
        return self._rotation_period

    @rotation_period.setter
    def rotation_period(self, rotation_period):
        """
        setter for rotational period of star in single star system, if unit is not specified, default period unit is
        assumed
        :param rotation_period:
        :return:
        """
        if isinstance(rotation_period, u.quantity.Quantity):
            self._rotation_period = np.float64(rotation_period.to(U.PERIOD_UNIT))
        elif isinstance(rotation_period, (int, np.int, float, np.float)):
            self._rotation_period = np.float64(rotation_period)
        else:
            raise TypeError('Input of variable `rotation_period` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

    @property
    def inclination(self):
        """
        inclination of single star system

        :return: (np.)int, (np.)float, astropy.unit.quantity.Quantity
        """
        return self._inclination

    @inclination.setter
    def inclination(self, inclination):
        """
        set orbit inclination of binary star system, if unit is not specified, default unit is assumed

        :param inclination: (np.)int, (np.)float, astropy.unit.quantity.Quantity
        :return:
        """

        if isinstance(inclination, u.quantity.Quantity):
            self._inclination = np.float64(inclination.to(U.ARC_UNIT))
        elif isinstance(inclination, (int, np.int, float, np.float)):
            self._inclination = np.float64(inclination)
        else:
            raise TypeError('Input of variable `inclination` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

        if not 0 <= self.inclination <= c.PI:
            raise ValueError('Eccentricity value of {} is out of bounds (0, pi).'.format(self.inclination))

        self._logger.debug("Setting property inclination "
                           "of class instance {} to {}".format(SingleSystem.__name__, self._inclination))

    @property
    def polar_log_g(self):
        """
        returns logarythm of polar surface gravity in SI

        :return: float
        """
        return self._polar_log_g

    @polar_log_g.setter
    def polar_log_g(self, log_g):
        """
        setter for polar surface gravity, if unit is not specified in astropy.units format, value in cgs is assumed

        :param log_g:
        :return:
        """
        if isinstance(log_g, u.quantity.Quantity):
            self._polar_log_g = np.float64(log_g.to(U.LOG_ACCELERATION_UNIT))
        elif isinstance(log_g, (int, np.int, float, np.float)):
            self._polar_log_g = np.float64((log_g * u.dex(u.cm / u.s ** 2)).to(U.LOG_ACCELERATION_UNIT))
        else:
            raise TypeError('Input of variable `polar_log_g` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

    @property
    def discretization_factor(self):
        """
        returns one quarter of number of points at star's equator

        :return: float
        """
        return self._discretization_factor

    @discretization_factor.setter
    def discretization_factor(self, discretization_factor):
        """
        setter for discretization factor

        :param log_g:
        :return:
        """
        self._discretization_factor = int(discretization_factor)

    def calculate_polar_radius(self):
        """
        returns polar radius of the star in default units

        :return: float
        """
        return np.power(c.G * self.star.mass / self.star.polar_gravity_acceleration, 0.5)

    def calculate_equatorial_radius(self):
        """
        returns equatorial radius of the star in default units

        :return: float
        """
        args, use = c.HALF_PI, False
        scipy_solver_init_value = np.array([1 / 1000.0])
        solution, _, ier, _ = scipy.optimize.fsolve(self.potential_fn, scipy_solver_init_value,
                                                        full_output=True, args=args)
        # check if star is closed
        if ier == 1 and not np.isnan(solution[0]):
            solution = solution[0]
            if solution <= 0:
                print(solution)
                raise ValueError('Value of single star equatorial radius {} is not valid'.format(solution))
        else:
            raise ValueError('Surface of the star is not closed. Check values of polar gravity an rotation period.')

        return solution

    def surface_potential(self, radius, *args):
        """
        function calculates potential on the given point of the star

        :param radius: (np.)float; spherical variable
        :param args: ((np.)float, (np.)float, (np.)float); (component distance, azimutal angle, polar angle)
        :return: (np.)float
        """
        theta, = args  # latitude angle (0,180)

        return - c.G * self.star.mass / radius - 0.5 * np.power(self._angular_velocity, 2.0) * \
                                                  np.power(radius * np.sin(theta), 2)

    def potential_fn(self, radius, *args):
        """
        implicit potential function

        :param radius: (np.)float; spherical variable
        :param args: ((np.)float, (np.)float, (np.)float); (component distance, azimutal angle, polar angle)
        :return:
        """
        return self.surface_potential(radius, *args) - self.star.surface_potential

    def compute_equipotential_boundary(self):
        """
        calculates a equipotential boundary of star in zx(yz) plane

        :return: tuple (np.array, np.array)
        """
        points = []
        angles = np.linspace(0, c.FULL_ARC, 300, endpoint=True)
        for angle in angles:
            args, use = angle, False
            scipy_solver_init_value = np.array([1 / 1000.0])
            solution, _, ier, _ = scipy.optimize.fsolve(self.potential_fn, scipy_solver_init_value,
                                                        full_output=True, args=args)
            if ier == 1 and not np.isnan(solution[0]):
                solution = solution[0]
                if 30 >= solution >= 0:
                    use = True
            else:
                continue

            points.append([solution * np.sin(angle), solution * np.cos(angle)])
        return np.array(points)

    def angular_velocity(self, rotation_period):
        """
        rotational angular velocity of the star
        :return:
        """
        return c.FULL_ARC / (rotation_period * U.PERIOD_UNIT).to(u.s).value

    def critical_break_up_radius(self):
        """
        returns critical, break-up equatorial radius for given mass and rotational period

        :return: float
        """
        return np.power(c.G * self.star.mass / np.power(self._angular_velocity, 2), 1.0 / 3.0)

    def critical_break_up_velocity(self):
        """
        returns critical, break-up equatorial rotational velocity for given mass and rotational period

        :return: float
        """
        return np.power(c.G * self.star.mass * self._angular_velocity, 1.0 / 3.0)

    def mesh(self, alpha):
        """
        function for creating surface mesh of single star system

        :param alpha: np.float, mean angular distance between vertices, < 90 deg
        :return: numpy.array([[x1 y1 z1],
                              [x2 y2 z2],
                                ...
                              [xN yN zN]])
        """
        if alpha > 90:
            raise ValueError("Invalid value of alpha parameter. Use value less than 90.")

        alpha = np.radians(alpha)
        N = int(c.HALF_PI // alpha)
        characterictic_angle = c.HALF_PI / N
        characterictic_distance = self.star.equatorial_radius * characterictic_angle

        # calculating equatorial part
        r_eq = np.array([self.star.equatorial_radius for ii in range(N)])
        phi_eq = np.array([characterictic_angle*ii for ii in range(N)])
        theta_eq = np.array([c.HALF_PI for ii in range(N)])
        # converting quarter of equator to cartesian
        x_eq, y_eq, z_eq = utils.spherical_to_cartesian(r_eq, phi_eq, theta_eq)

        # calculating radii for each latitude and generating one eighth of surface of the star without poles and equator
        num = int((c.HALF_PI - 2 * characterictic_angle) // characterictic_angle)
        thetas = np.linspace(characterictic_angle, c.HALF_PI-characterictic_angle, num=num, endpoint=True)
        r_q, phi_q, theta_q = [], [], []
        for theta in thetas:
            args, use = theta, False
            scipy_solver_init_value = np.array([1 / 1000.0])
            solution, _, ier, _ = scipy.optimize.fsolve(self.potential_fn, scipy_solver_init_value,
                                                        full_output=True, args=args)
            radius = solution[0]
            num = int(c.HALF_PI * radius * np.sin(theta) // characterictic_distance)
            r_q += [radius for xx in range(num)]
            M = c.HALF_PI/num
            phi_q += [xx*M for xx in range(num)]
            theta_q += [theta for xx in range(num)]

        r_q = np.array(r_q)
        phi_q = np.array(phi_q)
        theta_q = np.array(theta_q)
        # converting this eighth of surface to cartesian coordinates
        x_q, y_q, z_q = utils.spherical_to_cartesian(r_q, phi_q, theta_q)

        # stiching together equator and 8 sectors of stellar surface
        x = np.concatenate((x_eq, -y_eq, -x_eq,  y_eq, x_q, -y_q, -x_q,  y_q,  x_q, -y_q, -x_q,  y_q, np.array([0, 0])))
        y = np.concatenate((y_eq,  x_eq, -y_eq, -x_eq, y_q,  x_q, -y_q, -x_q,  y_q,  x_q, -y_q, -x_q, np.array([0, 0])))
        z = np.concatenate((z_eq,  z_eq,  z_eq,  z_eq, z_q,  z_q,  z_q,  z_q, -z_q, -z_q, -z_q, -z_q,
                            np.array([self.star.polar_radius, -self.star.polar_radius])))
        return np.column_stack((x, y, z))

    def surface(self, vertices):
        """
        calculates triangulation of the given surface points, returns set of triple indices of surface pints that make
        up given triangle

        :param vertices: np.array: numpy.array([[x1 y1 z1],
                                                [x2 y2 z2],
                                                  ...
                                                [xN yN zN]])
        :return: np.array(): numpy.array([[point_index1 point_index2 point_index3],
                                          [...],
                                            ...
                                          [...]])
        """
        triangulation = Delaunay(vertices)
        triangles_indices = triangulation.convex_hull
        return triangles_indices

    def plot(self, descriptor=None, **kwargs):
        """
        universal plot interface for single system class, more detailed documentation for each value of descriptor is
        available in graphics library

        :param descriptor: str (defines type of plot):
                            equpotential - plots orbit in orbital plane
        :param kwargs: dict (depends on descriptor value, see individual functions in graphics.py)
        :return:
        """
        if 'axis_unit' not in kwargs:
            kwargs['axis_unit'] = u.solRad

        if descriptor == 'equipotential':
            KWARGS = ['axis_unit']
            utils.invalid_kwarg_checker(kwargs, KWARGS, SingleSystem.plot)

            method_to_call = graphics.equipotential_single_star
            points = self.compute_equipotential_boundary()

            kwargs['points'] = (points * U.DISTANCE_UNIT).to(kwargs['axis_unit'])

        elif descriptor == 'mesh':
            KWARGS = ['axis_unit', 'alpha']
            method_to_call = graphics.single_star_mesh
            utils.invalid_kwarg_checker(kwargs, KWARGS, SingleSystem.plot)

            if 'alpha' not in kwargs:
                kwargs['alpha'] = 5
            kwargs['mesh'] = self.mesh(alpha=kwargs['alpha'])
            denominator = (1*kwargs['axis_unit'].to(U.DISTANCE_UNIT))
            kwargs['mesh'] /= denominator
            kwargs['equatorial_radius'] = self.star.equatorial_radius*U.DISTANCE_UNIT.to(kwargs['axis_unit'])

        elif descriptor == 'surface':
            KWARGS = ['axis_unit', 'alpha']
            utils.invalid_kwarg_checker(kwargs, KWARGS, SingleSystem.plot)
            method_to_call = graphics.single_star_surface

            if 'alpha' not in kwargs:
                kwargs['alpha'] = 5
            kwargs['mesh'] = self.mesh(alpha=kwargs['alpha'])
            denominator = (1 * kwargs['axis_unit'].to(U.DISTANCE_UNIT))
            kwargs['mesh'] /= denominator

            kwargs['triangles'] = self.surface(vertices=kwargs['mesh'])

            kwargs['equatorial_radius'] = self.star.equatorial_radius * U.DISTANCE_UNIT.to(kwargs['axis_unit'])

        else:
            raise ValueError("Incorrect descriptor `{}`".format(descriptor))

        method_to_call(**kwargs)
