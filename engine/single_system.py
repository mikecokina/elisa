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
    KWARGS = ['mass', 'gamma', 'inclination', 'rotation_period', 'log_g']

    def __init__(self, name=None, **kwargs):
        self.is_property(kwargs)
        super(SingleSystem, self).__init__(name=name, **kwargs)

        # get logger
        self._logger = logging.getLogger(SingleSystem.__name__)
        self._logger.info("Initialising object {}".format(SingleSystem.__name__))

        self._logger.debug("Setting property components "
                           "of class instance {}".format(SingleSystem.__name__))

        # in case of SingleStar system there is no need for user to define stellar component because it is defined here
        self._star = Star(mass=kwargs['mass'])
        self._rotation_period = None
        self._log_g = None

        # check if star object doesn't contain any meaningless parameters
        meaningless_params = {'synchronicity': self._star.synchronicity}
        for parameter in meaningless_params:
            if meaningless_params[parameter] is not None:
                meaningless_params[parameter] = None
                self._logger.info('Parameter `{0}` is meaningless in case of single star system.\n '
                                  'Setting parameter `{0}` value to None'.format(parameter))

        # quiet check if star object doesn't contain any meaningless dependent parameters
        meaningless_params = {'backward radius': self._star._backward_radius,
                              'forward_radius': self._star._forward_radius,
                              'side_radius': self._star._side_radius}
        for parameter in meaningless_params:
            if meaningless_params[parameter] is not None:
                meaningless_params[parameter] = None
                self._logger.debug('Parameter `{0}` is meaningless in case of single star system.\n '
                                   'Setting parameter `{0}` value to None'.format(parameter))

        # default values of properties
        self._inclination = None

        # testing if parameters were initialized
        missing_kwargs = []
        for kwarg in SingleSystem.KWARGS:
            if kwarg not in kwargs:
                missing_kwargs.append("`{}`".format(kwarg))
                self._logger.error("Property {} "
                                   "of class instance {} was not initialized".format(kwarg, SingleSystem.__name__))
            else:
                setattr(self, kwarg, kwargs[kwarg])

        if len(missing_kwargs) != 0:
            raise ValueError('Mising argument(s): {} in class instance {}'.format(', '.join(missing_kwargs),
                                                                                  SingleSystem.__name__))

        # calculation of dependent parameters
        self._gravity_acceleration = np.power(10, self.log_g)  # surface polar gravity

    def init(self):
        """
        function to reinitialize BinarySystem class instance after changing parameter(s) of binary system using setters

        :return:
        """
        self.__init__(primary=self.primary, secondary=self.secondary, **self._kwargs_serializer())

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
    def log_g(self):
        """
        returns logarythm of polar surface gravity in SI

        :return: float
        """
        return self._log_g

    @log_g.setter
    def log_g(self, log_g):
        """
        setter for polar surface gravity, if unit is not specified in astropy.units format, value in cgs is assumed

        :param log_g:
        :return:
        """
        if isinstance(log_g, u.quantity.Quantity):
            self._log_g = np.float64(log_g.to(U.LOG_ACCELERATION_UNIT))
        elif isinstance(log_g, (int, np.int, float, np.float)):
            self._log_g = np.float64((log_g*u.dex(u.cm/u.s**2)).to(U.LOG_ACCELERATION_UNIT))
        else:
            raise TypeError('Input of variable `log_g` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

    @property
    def gravity_acceleration(self):
        return self._gravity_acceleration

    # def potential_value(self, radius, *args):
    #     """
    #     function calculates potential for single star (derived from kopal potential F=1, q=0)
    #
    #     :param radius: (np.)float; spherical variable
    #     :param args: ((np.)float, (np.)float, (np.)float); (component distance, azimutal angle, polar angle)
    #     :return: (np.)float
    #     """
    #     theta, = args  # latitude angle (0,180)
    #
    #     block_a = 1.0 / radius
    #     block_d = 0.5 * np.power(radius, 2) * (1 - np.power(np.cos(theta), 2))
    #
    #     return block_a + block_d

    def surface_potential(self, radius, *args):
        """
        function calculates potential on the given point of the star

        :param radius: (np.)float; spherical variable
        :param args: ((np.)float, (np.)float, (np.)float); (component distance, azimutal angle, polar angle)
        :return: (np.)float
        """
        theta, = args  # latitude angle (0,180)

        return - c.G * self._star.mass / radius - 0.5 * np.power(self.angular_velocity(self._rotation_period), 2.0) * \
                                                  np.power(radius * np.sin(theta), 2)

    def potential_fn(self, radius, *args):
        """
        implicit potential function

        :param radius: (np.)float; spherical variable
        :param args: ((np.)float, (np.)float, (np.)float); (component distance, azimutal angle, polar angle)
        :return:
        """
        return self.surface_potential(radius, *args) - self._star.surface_potential

    def compute_equipotential_boundary(self):
        """
         compute a equipotential boundary of star in zx(yz) plane

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
        return c.FULL_ARC / (rotation_period * 86400)

    def critical_break_up_radius(self):
        """
        returns critical, break-up equatorial radius for given mass and rotational period

        :return: float
        """
        return np.power(c.G * self._star.mass / np.power(self.angular_velocity(self.rotation_period), 2), 1.0/3.0)

    def critical_break_up_velocity(self):
        """
        returns critical, break-up equatorial rotational velocity for given mass and rotational period

        :return: float
        """
        return np.power(c.G * self._star.mass * self.angular_velocity(self.rotation_period), 1.0/3.0)

    def plot(self, descriptor=None, **kwargs):
        """
        universal plot interface for single system class, more detailed documentation for each value of descriptor is
        available in graphics library

        :param descriptor: str (defines type of plot):
                            equpotential - plots orbit in orbital plane
        :param kwargs: dict (depends on descriptor value, see individual functions in graphics.py)
        :return:
        """
        if descriptor == 'equipotential':
            method_to_call = graphics.equipotential_single_star
            points = self.compute_equipotential_boundary()

            kwargs['points'] = points

        method_to_call(**kwargs)