'''
             _,--""--,_
        _,,-"          \
    ,-e"                ;
   (*             \     |
    \o\     __,-"  )    |
     `,_   (((__,-"     L___,,--,,__
        ) ,---\  /\    / -- '' -'-' )
      _/ /     )_||   /---,,___  __/
     """"     """"|_ /         ""
                  """"

 ______ _______ ______ _______ ______
|   __ \       |   __ \    ___|   __ \
|   __ <   -   |   __ <    ___|      <
|______/_______|______/_______|___|__|

    Because of funny Polish video

'''

from engine.system import System
from engine.star import Star
from engine.orbit import Orbit
from astropy import units as u
import numpy as np
import logging
from engine import const as c
from scipy.optimize import newton
from engine import utils
from engine import graphics
import scipy

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : [%(levelname)s] : %(name)s : %(message)s')


class BinarySystem(System):
    KWARGS = ['gamma', 'inclination', 'period', 'eccentricity', 'argument_of_periastron', 'primary_minimum_time',
              'phase_shift']

    def __init__(self, primary, secondary, name=None, **kwargs):
        self.is_property(kwargs)
        super(BinarySystem, self).__init__(name=name, **kwargs)

        # get logger
        self._logger = logging.getLogger(BinarySystem.__name__)
        self._logger.info("Initialising object {}".format(BinarySystem.__name__))

        # assign components to binary system
        if not isinstance(primary, Star):
            raise TypeError("Primary component is not instance of class {}".format(Star.__name__))

        if not isinstance(secondary, Star):
            raise TypeError("Secondary component is not instance of class {}".format(Star.__name__))

        self._logger.debug("Setting property components "
                           "of class instance {}".format(BinarySystem.__name__))
        self._primary = primary
        self._secondary = secondary

        # physical properties check
        self._mass_ratio = self.secondary.mass / self.primary.mass

        # default values of properties
        self._inclination = None
        self._period = None
        self._eccentricity = None
        self._argument_of_periastron = None
        self._orbit = None
        self._primary_minimum_time = None
        self._phase_shift = None

        # testing if parameters were initialized
        missing_kwargs = []
        for kwarg in BinarySystem.KWARGS:
            if kwarg not in kwargs:
                missing_kwargs.append("`{}`".format(kwarg))
                self._logger.error("Property {} "
                                   "of class instance {} was not initialized".format(kwarg, BinarySystem.__name__))
            else:
                setattr(self, kwarg, kwargs[kwarg])

        if len(missing_kwargs) != 0:
            raise ValueError('Mising argument(s): {} in class instance {}'.format(', '.join(missing_kwargs),
                                                                                  BinarySystem.__name__))

        # calculation of dependent parameters
        self._semi_major_axis = self.semi_major_axis_from_3rd_kepler_law()

        # orbit initialisation
        self.init_orbit()

        # compute and assing to all radii values to both components

    def init(self):
        """
        function to reinitialize BinarySystem class instance after changing parameter(s) of binary system using setters

        :return:
        """
        self.__init__(primary=self.primary, secondary=self.secondary, **self._kwargs_serializer())

    def _kwargs_serializer(self):
        """
        creating dictionary of keyword arguments of BinarySystem class in order to be able to reinitialize the class
        instance in init()

        :return: dict
        """
        serialized_kwargs = {}
        for kwarg in self.KWARGS:
            serialized_kwargs[kwarg] = getattr(self, kwarg)
        return serialized_kwargs

    def init_orbit(self):
        """
        encapsulating orbit class into binary system

        :return:
        """
        self._logger.debug("Re/Initializing orbit in class instance {} ".format(BinarySystem.__name__))
        orbit_kwargs = {key: getattr(self, key) for key in Orbit.KWARGS}
        self._orbit = Orbit(**orbit_kwargs)

    @property
    def mass_ratio(self):
        """
        returns mass ratio m2/m1 of binary system components

        :return: numpy.float
        """
        return self._mass_ratio

    @mass_ratio.setter
    def mass_ratio(self, value):
        """
        disabled setter for binary system mass ratio

        :param value:
        :return:
        """
        raise Exception("Property ``mass_ratio`` is read-only.")

    @property
    def primary(self):
        """
        encapsulation of primary component into binary system

        :return: class Star
        """
        return self._primary

    @property
    def secondary(self):
        """
        encapsulation of secondary component into binary system

        :return: class Star
        """
        return self._secondary

    @property
    def orbit(self):
        """
        encapsulation of orbit class into binary system

        :return: class Orbit
        """
        return self._orbit

    @property
    def period(self):
        """
        returns orbital period of binary system

        :return: (np.)int, (np.)float, astropy.unit.quantity.Quantity
        """
        return self._period

    @period.setter
    def period(self, period):
        """
        set orbital period of bonary star system, if unit is not specified, days are assumed

        :param period: (np.)int, (np.)float, astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(period, u.quantity.Quantity):
            self._period = np.float64(period.to(self.get_period_unit()))
        elif isinstance(period, (int, np.int, float, np.float)):
            self._period = np.float64(period)
        else:
            raise TypeError('Input of variable `period` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')
        self._logger.debug("Setting property period "
                           "of class instance {} to {}".format(BinarySystem.__name__, self._period))

    @property
    def inclination(self):
        """
        inclination of binary star system

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
            self._inclination = np.float64(inclination.to(self.get_arc_unit()))
        elif isinstance(inclination, (int, np.int, float, np.float)):
            self._inclination = np.float64(inclination)
        else:
            raise TypeError('Input of variable `inclination` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

        if not 0 <= self.inclination <= c.PI:
            raise ValueError('Eccentricity value of {} is out of bounds (0, pi).'.format(self.inclination))

        self._logger.debug("Setting property inclination "
                           "of class instance {} to {}".format(BinarySystem.__name__, self._inclination))

    @property
    def eccentricity(self):
        """
        eccentricity of orbit of binary star system

        :return: (np.)int, (np.)float
        """
        return self._eccentricity

    @eccentricity.setter
    def eccentricity(self, eccentricity):
        """
        set eccentricity

        :param eccentricity: (np.)int, (np.)float
        :return:
        """
        if eccentricity < 0 or eccentricity > 1 or not isinstance(eccentricity, (int, np.int, float, np.float)):
            raise TypeError(
                'Input of variable `eccentricity` is not (np.)int or (np.)float or it is out of boundaries.')
        self._eccentricity = eccentricity
        self._logger.debug("Setting property eccentricity "
                           "of class instance {} to {}".format(BinarySystem.__name__, self._eccentricity))

    @property
    def argument_of_periastron(self):
        """
        argument of periastron

        :return: (np.)int, (np.)float, astropy.unit.quantity.Quantity
        """
        return self._argument_of_periastron

    @argument_of_periastron.setter
    def argument_of_periastron(self, argument_of_periastron):
        """
        setter for argument of periastron

        :param argument_of_periastron: (np.)int, (np.)float, astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(argument_of_periastron, u.quantity.Quantity):
            self._argument_of_periastron = np.float64(argument_of_periastron.to(self.get_arc_unit()))
        elif isinstance(argument_of_periastron, (int, np.int, float, np.float)):
            self._argument_of_periastron = np.float64(argument_of_periastron)
        else:
            raise TypeError('Input of variable `periastron` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

    @property
    def primary_minimum_time(self):
        """
        returns time of primary minimum in default period unit

        :return: numpy.float
        """
        return self._primary_minimum_time

    @primary_minimum_time.setter
    def primary_minimum_time(self, primary_minimum_time):
        """
        setter for time of primary minima

        :param primary_minimum_time: (np.)int, (np.)float, astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(primary_minimum_time, u.quantity.Quantity):
            self._primary_minimum_time = np.float64(primary_minimum_time.to(self.get_period_unit()))
        elif isinstance(primary_minimum_time, (int, np.int, float, np.float)):
            self._primary_minimum_time = np.float64(primary_minimum_time)
        else:
            raise TypeError('Input of variable `primary_minimum_time` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')
        self._logger.debug("Setting property primary_minimum_time "
                           "of class instance {} to {}".format(BinarySystem.__name__, self._primary_minimum_time))

    @property
    def phase_shift(self):
        """
        returns phase shift of the primary eclipse minimum with respect to ephemeris
        true_phase is used during calculations, where: true_phase = phase + phase_shift

        :return: numpy.float
        """
        return self._phase_shift

    @phase_shift.setter
    def phase_shift(self, phase_shift):
        """
        setter for phase shift of the primary eclipse minimum with respect to ephemeris
        this will cause usage of true_phase during calculations, where: true_phase = phase + phase_shift

        :param phase_shift: numpy.float
        :return:
        """
        self._phase_shift = phase_shift
        self._logger.debug("Setting property phase_shift "
                           "of class instance {} to {}".format(BinarySystem.__name__, self._phase_shift))

    @property
    def semi_major_axis(self):
        """
        returns semi major axis of the system in default distance unit

        :return: np.float
        """
        return self._semi_major_axis

    def semi_major_axis_from_3rd_kepler_law(self):
        """
        calculates length semi major axis usin 3rd kepler law

        :return: np.float
        """
        period = (self._period * self.get_period_unit()).to(u.s)
        return (c.G * (self.primary.mass + self.secondary.mass) * period ** 2 / (4 * c.PI ** 2)) ** (1.0 / 3)

    def compute_lc(self):
        pass

    def get_info(self):
        pass

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

    def primary_potential_derivative_x(self, x, *args):
        """
        derivative of potential function perspective of primary component along the x axis

        :param x: (np.)float
        :param args: tuple ((np.)float, (np.)float); (components distance, synchronicity of primary component)
        :return: (np.)float
        """
        d, = args
        r_sqr, rw_sqr = x ** 2, (d - x) ** 2
        return - np.power(x, -2) + ((self.mass_ratio * (d - x)) / rw_sqr ** (
            3.0 / 2.0)) + self.primary.synchronicity ** 2 * (self.mass_ratio + 1) * x - self.mass_ratio / d ** 2

    def secondary_potential_derivative_x(self, x, *args):
        """
        derivative of potential function perspective of secondary component along the x axis

        :param x: (np.)float
        :param args: tuple ((np.)float, (np.)float); (components distance, synchronicity of secondary component)
        :return: (np.)float
        """
        d, = args
        r_sqr, rw_sqr = x ** 2, (d - x) ** 2
        return - np.power(x, -2) + (self.mass_ratio * (d - x) / rw_sqr ** (
            3.0 / 2.0)) - self.secondary.synchronicity ** 2 * (self.mass_ratio + 1) * (1 - x) + (1.0 / d ** 2)

    def potential_value_primary(self, radius, *args):
        """

        :param radius: (np.)float; spherical variable
        :param args: ((np.)float, (np.)float, (np.)float); (component distance, azimutal angle, polar angle)
        :return: (np.)float
        """
        d, phi, theta = args  # distance between components, azimut angle, latitude angle (0,180)

        block_a = 1.0 / radius
        block_b = self.mass_ratio / (np.sqrt(np.power(d, 2) + np.power(radius, 2) - (
            2.0 * radius * np.cos(phi) * np.sin(theta) * d)))
        block_c = (self.mass_ratio * radius * np.cos(phi) * np.sin(theta)) / (np.power(d, 2))
        block_d = 0.5 * np.power(self.primary.synchronicity, 2) * (1 + self.mass_ratio) * np.power(radius, 2) * (
            1 - np.power(np.cos(theta), 2))

        return block_a + block_b - block_c + block_d

    def potential_value_secondary(self, radius, *args):
        """

        :param radius: (np.)float; spherical variable
        :param args: ((np.)float, (np.)float, (np.)float); (component distance, azimutal angle, polar angle)
        :return: (np.)float
        """
        d, phi, theta = args
        inverted_mass_ratio = 1.0 / self.mass_ratio

        block_a = 1. / radius
        block_b = inverted_mass_ratio / (np.sqrt(np.power(d, 2) + np.power(radius, 2) - (
            2 * radius * np.cos(phi) * np.sin(theta) * d)))
        block_c = (inverted_mass_ratio * radius * np.cos(phi) * np.sin(theta)) / (np.power(d, 2))
        block_d = 0.5 * np.power(self.secondary.synchronicity, 2) * (1 + inverted_mass_ratio) * np.power(
            radius, 2) * (1 - np.power(np.cos(theta), 2))

        inverse_potential = (block_a + block_b - block_c + block_d) / inverted_mass_ratio + (
            0.5 * ((inverted_mass_ratio - 1) / inverted_mass_ratio))

        return inverse_potential

    def potential_primary_fn(self, radius, *args):
        """
        implicit potential function from perspective of primary component

        :param radius: (np.)float; spherical variable
        :param args: ((np.)float, (np.)float, (np.)float); (component distance, azimutal angle, polar angle)
        :return:
        """
        return self.potential_value_primary(radius, *args) - self.primary.surface_potential

    def potential_secondary_fn(self, radius, *args):
        """
        implicit potential function from perspective of secondary component

        :param radius: (np.)float; spherical variable
        :param args: ((np.)float, (np.)float, (np.)float); (component distance, azimutal angle, polar angle)
        :return:
        """
        return self.potential_value_secondary(radius, *args) - self.secondary.surface_potential

    def critical_potential(self, component, component_distance):
        # nebude lepsie ak namiesto component distance bude faza?
        """
        return a critical potential for target component

        :param component: str; define target component to compute critical potential; `primary` or `secondary`
        :param component_distance: (np.)float
        :return: (np.)float
        """

        if component == "primary":
            args = component_distance,
            solution = newton(self.primary_potential_derivative_x, 0.001, args=args)
        elif component == "secondary":
            args = component_distance,
            solution = newton(self.secondary_potential_derivative_x, 0.001, args=args)
        else:
            raise ValueError("Parameter `component` has incorrect value. Use `primary` or `secondary`.")

        if not np.isnan(solution):
            if component == "primary":
                args = component_distance, 0.0, np.pi / 2.0
                return abs(self.potential_value_primary(solution, *args))
            else:
                args = (component_distance, 0.0, np.pi / 2.0)
                return abs(self.potential_value_secondary(component_distance - solution, *args))
        else:
            raise ValueError("Iteration process to solve critical potential seems to lead nowhere (critical potential "
                             "solver has failed).")

    def compute_polar_radius(self, component=None, distance=None):
        pass

    def compute_equipotential_boundary(self, phase, plane):
        """
        compute a equipotential boundary of components (partial Hill plane)

        :param phase: (np.)float; phase to obtain a component distance
        :param plane: str; xy, yz, zx
        :return: tuple (np.array, np.array)
        """
        components_distance = self.orbit.orbital_motion(phase=phase)[0][0]

        components = ['primary', 'secondary']
        points_primary, points_secondary = [], []
        fn_map = {'primary': self.potential_primary_fn, 'secondary': self.potential_secondary_fn}

        angles = np.linspace(0, c.FULL_ARC, 300, endpoint=True)
        for component in components:
            for angle in angles:
                if utils.is_plane(plane, 'xy'):
                    args, use = (components_distance, angle, c.HALF_PI), False
                elif utils.is_plane(plane, 'yz'):
                    args, use = (components_distance, c.HALF_PI, angle), False
                elif utils.is_plane(plane, 'zx'):
                    args, use = (components_distance, 0.0, angle), False
                else:
                    raise ValueError('Invalid choice of crossection plane, use only: `xy`, `yz`, `zx`.')

                scipy_solver_init_value = np.array([components_distance / 1000.0])
                solution, _, ier, _ = scipy.optimize.fsolve(fn_map[component], scipy_solver_init_value,
                                                            full_output=True, args=args)

                # check for regular solution
                if ier == 1 and not np.isnan(solution[0]):
                    solution = solution[0]
                    if 30 >= solution >= 0:
                        use = True
                else:
                    continue

                if use:
                    if utils.is_plane(plane, 'yz'):
                        if component == 'primary':
                            points_primary.append([solution * np.sin(angle), solution * np.cos(angle)])
                        elif component == 'secondary':
                            points_secondary.append([solution * np.sin(angle), solution * np.cos(angle)])
                    elif utils.is_plane(plane, 'xz'):
                        if component == 'primary':
                            points_primary.append([solution * np.sin(angle), solution * np.cos(angle)])
                        elif component == 'secondary':
                            points_secondary.append([- (solution * np.sin(angle) - components_distance),
                                                     solution * np.cos(angle)])
                    else:
                        if component == 'primary':
                            points_primary.append([solution * np.cos(angle), solution * np.sin(angle)])
                        elif component == 'secondary':
                            points_secondary.append([- (solution * np.cos(angle) - components_distance),
                                                     solution * np.sin(angle)])

        return np.array(points_primary), np.array(points_secondary)

    def plot(self, descriptor=None, **kwargs):
        """
        universal plot interface for binary system class, more detailed documentation for each value of descriptor is
        available in graphics library

        :param descriptor: str (defines type of plot):
                            orbit - plots orbit in orbital plane
        :param kwargs: dict (depends on descriptor value, see individual functions in graphics.py)
        :return:
        """

        if descriptor == 'orbit':
            KWARGS = ['start_phase', 'stop_phase', 'number_of_points', 'axis_unit']
            utils.invalid_kwarg_checker(kwargs, KWARGS, BinarySystem.plot)

            method_to_call = graphics.orbit
            start_phase = 0 if 'start_phase' not in kwargs else kwargs['start_phase']
            stop_phase = 1.0 if 'stop_phase' not in kwargs else kwargs['stop_phase']
            number_of_points = 300 if 'number_of_points' not in kwargs else kwargs['number_of_points']

            if 'axis_unit' not in kwargs:
                kwargs['axis_unit'] = u.solRad
            elif kwargs['axis_unit'] == 'dimensionless':
                kwargs['axis_unit'] = u.dimensionless_unscaled

            # orbit calculation for given phases
            phases = np.linspace(start_phase, stop_phase, number_of_points)
            ellipse = self.orbit.orbital_motion(phase=phases)
            # if axis are without unit a = 1
            if kwargs['axis_unit'] != u.dimensionless_unscaled:
                a = self._semi_major_axis * self.get_distance_unit().to(kwargs['axis_unit'])
                radius = a * ellipse[:, 0]
            else:
                radius = ellipse[:, 0]
            azimuth = ellipse[:, 1]
            x, y = utils.polar_to_cartesian(radius=radius, phi=azimuth - c.PI / 2.0)
            kwargs['x_data'], kwargs['y_data'] = x, y

        elif descriptor == 'equipotential':
            KWARGS = ['plane', 'phase']
            utils.invalid_kwarg_checker(kwargs, KWARGS, BinarySystem.plot)

            method_to_call = graphics.equipotential

            if 'phase' not in kwargs:
                kwargs['phase'] = 0
            if 'plane' not in kwargs:
                kwargs['plane'] = 'xy'

            # relative distance between components (a = 1)
            if utils.is_plane(kwargs['plane'], 'xy') or utils.is_plane(
                    kwargs['plane'], 'yz') or utils.is_plane(kwargs['plane'], 'zx'):
                points_primary, points_secondary = self.compute_equipotential_boundary(phase=kwargs['phase'],
                                                                                       plane=kwargs['plane'])
            else:
                raise ValueError('Invalid choice of crossection plane, use only: `xy`, `yz`, `zx`.')

            kwargs['points_primary'] = points_primary
            kwargs['points_secondary'] = points_secondary
        else:
            raise ValueError("Incorrect descriptor `{}`".format(descriptor))

        method_to_call(**kwargs)
