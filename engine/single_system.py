import logging
from engine.system import System
from engine.star import Star
import numpy as np
import scipy
from engine import graphics
from engine import const as c
from astropy import units as u
from engine import units as U

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : [%(levelname)s] : %(name)s : %(message)s')


class SingleSystem(System):
    KWARGS = ['mass', 'gamma', 'inclination', 'rotation_period', 'polar_log_g']

    def __init__(self, name=None, **kwargs):
        self.is_property(kwargs)
        super(SingleSystem, self).__init__(name=name, **kwargs)

        # get logger
        self._logger = logging.getLogger(SingleSystem.__name__)
        self._logger.info("Initialising object {}".format(SingleSystem.__name__))

        self._logger.debug("Setting property components "
                           "of class instance {}".format(SingleSystem.__name__))

        # in case of SingleStar system there is no need for user to define stellar component because it is defined here
        self.star = Star(mass=kwargs['mass'])

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

        # calculation of dependent parameters
        self._angular_velocity = self.angular_velocity(self.rotation_period)
        self.star._polar_log_g = self.polar_log_g
        self.star._polar_gravity_acceleration = np.power(10, self.polar_log_g)  # surface polar gravity
        self.star._polar_radius = self.polar_radius
        args = 0,
        self.star._surface_potential = self.surface_potential(self.star.polar_radius, args)[0]
        self.star._equatorial_radius = self.equatorial_radius

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
        is_not = ['`{}`'.format(k) for k in kwargs if k not in cls.KWARGS]
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
    def polar_radius(self):
        """
        returns polar radius of the star in default units

        :return: float
        """
        return np.power(c.G * self.star.mass / self.star.polar_gravity_acceleration, 0.5)

    @property
    def equatorial_radius(self):
        """
        returns equatorial radius of the star in default units

        :return: float
        """
        args, use = c.HALF_PI, False
        scipy_solver_init_value = np.array([1 / 1000.0])
        solution, _, ier, _ = scipy.optimize.fsolve(self.potential_fn, scipy_solver_init_value,
                                                        full_output=True, args=args)
        if ier == 1 and not np.isnan(solution[0]):
            solution = solution[0]
            if solution <= 0:
                print(solution)
                raise ValueError('Value of single star equatorial radius {} is not valid'.format(solution))

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

    def single_star_mesh(self, N=100):
        characterictic_distance = c.HALF_PI * self.star.equatorial_radius / N
        characterictic_angle = c.HALF_PI / N
        r, phi, theta = [], [], []

        # calculating equatorial part
        r_eq = [self.equatorial_radius for ii in range(N)]
        phi_eq = [characterictic_distance**ii for ii in range(N)]
        theta_eq = [c.HALF_PI for ii in range(N)]

        # adding quarter of the star without pole
        num = int((c.HALF_PI - 2 * characterictic_angle) // characterictic_angle)
        thetas = np.linspace(characterictic_angle, c.HALF_PI-characterictic_angle, num=num, endpoint=True)
        radii_for_thetas = []

        # for tht in thetas:


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
            method_to_call = graphics.equipotential_single_star
            points = self.compute_equipotential_boundary()

            kwargs['points'] = (points * U.DISTANCE_UNIT).to(kwargs['axis_unit'])

        method_to_call(**kwargs)
