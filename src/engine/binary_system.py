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
    https://www.youtube.com/watch?v=YHyOTyTXXdA

'''

import gc
import logging
from copy import copy

import numpy as np
import scipy
from astropy import units as u
from scipy.optimize import newton
from scipy.spatial import Delaunay
from multiprocessing import Pool

from conf import config
from engine import const
from engine import graphics
from engine import ld
from engine import units
from engine import utils
from engine.orbit import Orbit
from engine.star import Star
from engine.system import System

# temporary
from time import time


class BinarySystem(System):
    KWARGS = ['gamma', 'inclination', 'period', 'eccentricity', 'argument_of_periastron', 'primary_minimum_time',
              'phase_shift']
    OPTIONAL_KWARGS = []
    ALL_KWARGS = KWARGS + OPTIONAL_KWARGS

    # this will be removed after full implementation of LD
    LD_COEFF = 0.5

    def __init__(self, primary, secondary, name=None, **kwargs):

        utils.invalid_kwarg_checker(kwargs, BinarySystem.ALL_KWARGS, BinarySystem)
        super(BinarySystem, self).__init__(name=name, **kwargs)

        # get logger
        self._logger = logging.getLogger(BinarySystem.__name__)
        self._logger.info("Initialising object {}".format(BinarySystem.__name__))

        self._logger.debug("Setting property components "
                           "of class instance {}".format(BinarySystem.__name__))

        # assign components to binary system
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
        self._semi_major_axis = None
        self._periastron_phase = None

        params = {
            "primary": self.primary,
            "secondary": self.secondary
        }
        params.update(**kwargs)
        self._star_params_validity_check(**params)
        # set attributes and test whether all parameters were initialized

        utils.check_missing_kwargs(BinarySystem.KWARGS, kwargs, instance_of=BinarySystem)
        # we already ensured that all kwargs are valid and all mandatory kwargs are present so lets set class attributes
        for kwarg in kwargs:
            self._logger.debug("Setting property {} "
                               "of class instance {} to {}".format(kwarg, BinarySystem.__name__, kwargs[kwarg]))
            setattr(self, kwarg, kwargs[kwarg])

        # calculation of dependent parameters
        self._semi_major_axis = self.calculate_semi_major_axis()

        # orbit initialisation (initialise class Orbit from given BinarySystem parameters)
        self.init_orbit()

        # setup critical surface potentials in periastron
        self._setup_critical_potential()
        # binary star morphology estimation
        self.setup_morphology()

        # polar radius of both component in periastron
        self.setup_components_radii(components_distance=self.orbit.periastron_distance)

        # if secondary discretization factor was not set, it will be now with respect to primary component
        if not self.secondary.kwargs.get('discretization_factor'):
            self._logger.info("Setting discretization factor of secondary component according discretization factor "
                              "of primary component.")
            self.secondary.discretization_factor = \
                self.primary.discretization_factor * self.primary.polar_radius / self.secondary.polar_radius * u.rad

            # TODO: retrieval of limb darkenig coefficients

    @property
    def morphology(self):
        """
        morphology of binary star system

        :return: str; detached, semi-detached, over-contact, double-contact
        """
        return self._morphology

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
        set orbital period of bonary star system, if unit is not specified, default period unit is assumed

        :param period: (np.)int, (np.)float, astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(period, u.quantity.Quantity):
            self._period = np.float64(period.to(units.PERIOD_UNIT))
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
            self._inclination = np.float64(inclination.to(units.ARC_UNIT))
        elif isinstance(inclination, (int, np.int, float, np.float)):
            self._inclination = np.float64((inclination * u.deg).to(units.ARC_UNIT))
        else:
            raise TypeError('Input of variable `inclination` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

        if not 0 <= self.inclination <= const.PI:
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
        if eccentricity < 0 or eccentricity >= 1 or not isinstance(eccentricity, (int, np.int, float, np.float)):
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
        setter for argument of periastron, if unit is not supplied, value in degrees is assumed

        :param argument_of_periastron: (np.)int, (np.)float, astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(argument_of_periastron, u.quantity.Quantity):
            self._argument_of_periastron = np.float64(argument_of_periastron.to(units.ARC_UNIT))
        elif isinstance(argument_of_periastron, (int, np.int, float, np.float)):
            self._argument_of_periastron = np.float64((argument_of_periastron * u.deg).to(units.ARC_UNIT))
        else:
            raise TypeError('Input of variable `periastron` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

        if not 0 <= self._argument_of_periastron <= const.FULL_ARC:
            self._argument_of_periastron %= const.FULL_ARC
        self._logger.debug("Setting property argument of periastron "
                           "of class instance {} to {}".format(BinarySystem.__name__, self._argument_of_periastron))

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
            self._primary_minimum_time = np.float64(primary_minimum_time.to(units.PERIOD_UNIT))
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

    def calculate_semi_major_axis(self):
        """
        calculates length semi major axis using 3rd kepler law

        :return: np.float
        """
        period = (self._period * units.PERIOD_UNIT).to(u.s)
        return (const.G * (self.primary.mass + self.secondary.mass) * period ** 2 / (4 * const.PI ** 2)) ** (1.0 / 3)

    def setup_components_radii(self, components_distance):
        fns = [self.calculate_polar_radius, self.calculate_side_radius, self.calculate_backward_radius]
        components = ['primary', 'secondary']

        for component in components:
            component_instance = getattr(self, component)
            for fn in fns:
                self._logger.debug('Initialising {} for {} component'.format(
                    ' '.join(str(fn.__name__).split('_')[1:]),
                    component
                ))
                param = '_{}'.format('_'.join(str(fn.__name__).split('_')[1:]))
                radius = fn(component, components_distance)
                setattr(component_instance, param, radius)
                if self.morphology != 'over-contact':
                    radius = self.calculate_forward_radius(component, components_distance)
                    setattr(component_instance, '_forward_radius', radius)

    def _evaluate_spots_mesh(self, components_distance, component=None):
        """
        compute points of each spots and assigns values to spot container instance

        :param components_distance: float
        :return:
        """

        def solver_condition(x, *_args):
            if isinstance(x, np.ndarray):
                x = x[0]
            point = utils.spherical_to_cartesian([x, _args[1], _args[2]])
            point[0] = point[0] if component == "primary" else components_distance - point[0]
            # ignore also spots where one of points is situated just on the neck
            if self.morphology == "over-contact":
                if (component == "primary" and point[0] >= neck_position) or \
                        (component == "secondary" and point[0] <= neck_position):
                    return False
            return True

        components = self._component_to_list(component)
        fns = {
            "primary": (self.potential_primary_fn, self.pre_calculate_for_potential_value_primary),
            "secondary": (self.potential_secondary_fn, self.pre_calculate_for_potential_value_secondary)
        }
        fns = {_component: fns[_component] for _component in components}

        # in case of wuma system, get separation and make additional test of location of each point (if primary
        # spot doesn't intersect with secondary, if does, then such spot will be skipped completly)
        neck_position = self.calculate_neck_position() if self.morphology == "over-contact" else 1e10

        for component, functions in fns.items():
            potential_fn, precalc_fn = functions
            self._logger.info("Evaluating spots for {} component".format(component))
            component_instance = getattr(self, component)

            if not component_instance.spots:
                self._logger.info("No spots to evaluate for {} component. Continue.".format(component))
                continue

            # iterate over spots
            for spot_index, spot_instance in list(component_instance.spots.items()):
                # lon -> phi, lat -> theta
                lon, lat = spot_instance.longitude, spot_instance.latitude

                self._setup_spot_instance_discretization_factor(spot_instance, spot_index, component_instance)
                alpha = spot_instance.discretization_factor
                diameter = spot_instance.angular_diameter

                # initial containers for current spot
                boundary_points, spot_points = list(), list()

                # initial radial vector
                radial_vector = np.array([1.0, lon, lat])  # unit radial vector to the center of current spot
                center_vector = utils.spherical_to_cartesian([1.0, lon, lat])

                args1, use = (components_distance, radial_vector[1], radial_vector[2]), False
                args2 = precalc_fn(*args1)
                kwargs = {'original_kwargs': args1}
                solution, use = self._solver(potential_fn, solver_condition, *args2, **kwargs)

                if not use:
                    # in case of spots, each point should be usefull, otherwise remove spot from
                    # component spot list and skip current spot computation
                    self._logger.warning("Center of spot {} doesn't satisfy reasonable conditions and "
                                         "entire spot will be omitted.".format(spot_instance.kwargs_serializer()))

                    component_instance.remove_spot(spot_index=spot_index)
                    continue

                spot_center_r = solution
                spot_center = utils.spherical_to_cartesian([spot_center_r, lon, lat])

                # compute euclidean distance of two points on spot (x0)
                # we have to obtain distance between center and 1st point in 1st inner ring of spot
                args1, use = (components_distance, lon, lat + alpha), False
                args2 = precalc_fn(*args1)
                kwargs = {'original_kwargs': args1}
                solution, use = self._solver(potential_fn, solver_condition, *args2, **kwargs)

                if not use:
                    # in case of spots, each point should be usefull, otherwise remove spot from
                    # component spot list and skip current spot computation
                    self._logger.warning("First inner ring of spot {} doesn't satisfy reasonable conditions and "
                                         "entire spot will be omitted".format(spot_instance.kwargs_serializer()))

                    component_instance.remove_spot(spot_index=spot_index)
                    continue

                x0 = np.sqrt(spot_center_r ** 2 + solution ** 2 - (2.0 * spot_center_r * solution * np.cos(alpha)))

                # number of points in latitudal direction
                # + 1 to obtain same discretization as object itself
                num_radial = int(np.round((diameter * 0.5) / alpha)) + 1
                self._logger.debug('Number of rings in spot {} is {}'
                                   ''.format(spot_instance.kwargs_serializer(), num_radial))
                thetas = np.linspace(lat, lat + (diameter * 0.5), num=num_radial, endpoint=True)

                num_azimuthal = [1 if i == 0 else int(i * 2.0 * np.pi * x0 // x0) for i in range(0, len(thetas))]
                deltas = [np.linspace(0., const.FULL_ARC, num=num, endpoint=False) for num in num_azimuthal]

                try:
                    for theta_index, theta in enumerate(thetas):
                        # first point of n-th ring of spot (counting start from center)
                        default_spherical_vector = [1.0, lon % const.FULL_ARC, theta]

                        for delta_index, delta in enumerate(deltas[theta_index]):
                            # rotating default spherical vector around spot center vector and thus generating concentric
                            # circle of points around centre of spot
                            delta_vector = utils.arbitrary_rotation(theta=delta, omega=center_vector,
                                                                    vector=utils.spherical_to_cartesian(
                                                                        default_spherical_vector),
                                                                    degrees=False)

                            spherical_delta_vector = utils.cartesian_to_spherical(delta_vector)

                            args1 = (components_distance, spherical_delta_vector[1], spherical_delta_vector[2])
                            args2 = precalc_fn(*args1)
                            kwargs = {'original_kwargs': args1}
                            solution, use = self._solver(potential_fn, solver_condition, *args2, **kwargs)

                            if not use:
                                component_instance.remove_spot(spot_index=spot_index)
                                raise StopIteration

                            spot_point = utils.spherical_to_cartesian([solution, spherical_delta_vector[1],
                                                                       spherical_delta_vector[2]])
                            spot_points.append(spot_point)

                            if theta_index == len(thetas) - 1:
                                boundary_points.append(spot_point)

                except StopIteration:
                    self._logger.warning("At least 1 point of spot {} doesn't satisfy reasonable conditions and "
                                         "entire spot will be omitted.".format(spot_instance.kwargs_serializer()))
                    continue

                boundary_center = spot_points[0]

                # max size from barycenter of boundary to boundary
                spot_instance.max_size = max([np.linalg.norm(np.array(boundary_center) - np.array(b))
                                              for b in boundary_points])

                if component == "primary":
                    spot_instance.points = np.array(spot_points)
                    spot_instance.boundary = np.array(boundary_points)
                    spot_instance.center = np.array(spot_center)
                else:
                    spot_instance.points = np.array([np.array([components_distance - point[0], -point[1], point[2]])
                                                     for point in spot_points])

                    spot_instance.boundary = np.array([np.array([components_distance - point[0], -point[1], point[2]])
                                                       for point in boundary_points])

                    spot_instance.center = np.array([components_distance - spot_center[0], -spot_center[1],
                                                     spot_center[2]])
                gc.collect()

    def _star_params_validity_check(self, **kwargs):
        """
        checking if star instances have all additional atributes set properly

        :param kwargs: list
        :return:
        """

        if not isinstance(kwargs.get("primary"), Star):
            raise TypeError("Primary component is not instance of class {}".format(Star.__name__))

        if not isinstance(kwargs.get("secondary"), Star):
            raise TypeError("Secondary component is not instance of class {}".format(Star.__name__))

        # checking if stellar components have all mandatory parameters initialised
        # these parameters are not mandatory in single star system, so validity check cannot be provided
        # on whole set of KWARGS in star object
        star_mandatory_kwargs = ['mass', 'surface_potential', 'synchronicity', 'albedo']
        missing_kwargs = []
        for component in [self.primary, self.secondary]:
            for kwarg in star_mandatory_kwargs:
                if getattr(component, kwarg) is None:
                    missing_kwargs.append("`{}`".format(kwarg))

            component_name = 'primary' if component == self.primary else 'secondary'
            if len(missing_kwargs) != 0:
                raise ValueError('Mising argument(s): {} in {} component Star class'.format(
                    ', '.join(missing_kwargs), component_name))

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

    def _setup_critical_potential(self):
        """
        compute and set critical surface potential for both components

        :return:
        """
        self.primary.critical_surface_potential = self.critical_potential(
            component="primary", components_distance=1 - self.eccentricity
        )
        self.secondary.critical_surface_potential = self.critical_potential(
            component="secondary", components_distance=1 - self.eccentricity
        )

    @staticmethod
    def compute_filling_factor(surface_potential, lagrangian_points):
        """

        :param surface_potential:
        :param lagrangian_points: list; lagrangian points in `order` (in order to ensure that L2)
        :return:
        """
        return (lagrangian_points[1] - surface_potential) / (lagrangian_points[1] - lagrangian_points[2])

    def setup_morphology(self):
        """
        Setup binary star class property `morphology`
        :return:
        """
        __PRECISSION__ = 1e-8
        __SETUP_VALUE__ = None
        if (self.primary.synchronicity == 1 and self.secondary.synchronicity == 1) and self.eccentricity == 0.0:
            lp = self.libration_potentials()

            self.primary.filling_factor = self.compute_filling_factor(self.primary.surface_potential, lp)
            self.secondary.filling_factor = self.compute_filling_factor(self.secondary.surface_potential, lp)

            if ((1 > self.secondary.filling_factor > 0) or (1 > self.primary.filling_factor > 0)) and \
                    (abs(self.primary.filling_factor - self.secondary.filling_factor) > __PRECISSION__):
                raise ValueError("Detected over-contact binary system, but potentials of components are not the same.")
            if self.primary.filling_factor > 1 or self.secondary.filling_factor > 1:
                raise ValueError("Non-Physical system: primary_filling_factor or "
                                 "secondary_filling_factor is greater then 1. Filling factor is obtained as following:"
                                 "(Omega_{inner} - Omega) / (Omega_{inner} - Omega_{outter})")

            if (abs(self.primary.filling_factor) < __PRECISSION__ and self.secondary.filling_factor < 0) or (
                    self.primary.filling_factor < 0 and abs(
                self.secondary.filling_factor) < __PRECISSION__) or (
                    abs(self.primary.filling_factor) < __PRECISSION__ and
                    abs(self.secondary.filling_factor) < __PRECISSION__):
                __SETUP_VALUE__ = "semi-detached"
            elif self.primary.filling_factor < 0 and self.secondary.filling_factor < 0:
                __SETUP_VALUE__ = "detached"
            elif 1 >= self.primary.filling_factor > 0:
                __SETUP_VALUE__ = "over-contact"
            elif self.primary.filling_factor > 1 or self.secondary.filling_factor > 1:
                raise ValueError("Non-Physical system: potential of components is to low.")

        else:
            self.primary.filling_factor, self.secondary.filling_factor = None, None
            if (abs(self.primary.surface_potential - self.primary.critical_surface_potential) < __PRECISSION__) and \
                    (abs(
                        self.secondary.surface_potential - self.secondary.critical_surface_potential) < __PRECISSION__):
                __SETUP_VALUE__ = "double-contact"

            elif (not (not (abs(
                    self.primary.surface_potential - self.primary.critical_surface_potential) < __PRECISSION__) or not (
                    self.secondary.surface_potential > self.secondary.critical_surface_potential))) or \
                    ((abs(
                        self.secondary.surface_potential - self.secondary.critical_surface_potential) < __PRECISSION__)
                     and (self.primary.surface_potential > self.primary.critical_surface_potential)):
                __SETUP_VALUE__ = "semi-detached"

            elif (self.primary.surface_potential > self.primary.critical_surface_potential) and (
                    self.secondary.surface_potential > self.secondary.critical_surface_potential):
                __SETUP_VALUE__ = "detached"

            else:
                raise ValueError("Non-Physical system. Change stellar parameters.")
        self._morphology = __SETUP_VALUE__

    def init_orbit(self):
        """
        encapsulating orbit class into binary system

        :return:
        """
        self._logger.debug("Re/Initializing orbit in class instance {} ".format(BinarySystem.__name__))
        orbit_kwargs = {key: getattr(self, key) for key in Orbit.KWARGS}
        self._orbit = Orbit(**orbit_kwargs)

    def compute_lc(self):
        pass

    def get_info(self):
        pass

    def primary_potential_derivative_x(self, x, *args):
        """
        derivative of potential function perspective of primary component along the x axis

        :param x: (np.)float
        :param args: tuple ((np.)float, (np.)float); (components distance, synchronicity of primary component)
        :return: (np.)float
        """
        d, = args
        r_sqr, rw_sqr = x ** 2, (d - x) ** 2
        return - (x / r_sqr ** (3.0 / 2.0)) + ((self.mass_ratio * (d - x)) / rw_sqr ** (
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
        return - (x / r_sqr ** (3.0 / 2.0)) + ((self.mass_ratio * (d - x)) / rw_sqr ** (
                3.0 / 2.0)) - self.secondary.synchronicity ** 2 * (self.mass_ratio + 1) * (d - x) + (1.0 / d ** 2)

    def pre_calculate_for_potential_value_primary(self, *args):
        """
        function calculates auxiliary values for calculation of primary component potential, and therefore they don't
        need to be wastefully recalculated every iteration in solver

        :param args: (component distance, azimut angle (0, 2pi), latitude angle (0, pi)
        :return: tuple: (b, c, d, e) such that: Psi1 = 1/r + a/sqrt(b+r^2+c*r) - d*r + e*x^2
        """
        distance, phi, theta = args  # distance between components, azimuth angle, latitude angle (0,180)

        cs = np.cos(phi) * np.sin(theta)

        b = np.power(distance, 2)
        c = 2 * distance * cs
        d = (self.mass_ratio * cs) / b
        e = 0.5 * np.power(self.primary.synchronicity, 2) * (1 + self.mass_ratio) * (1 - np.power(np.cos(theta), 2))

        if np.isscalar(phi):
            return b, c, d, e
        else:
            bb = b * np.ones(np.shape(phi))
            return np.column_stack((bb, c, d, e))

    def potential_value_primary(self, radius, *args):
        """
        calculates modified kopal potential from point of view of primary component

        :param radius: (np.)float; spherical variable
        :param args: tuple: (B, C, D, E) such that: Psi1 = 1/r + A/sqrt(B+r^2+Cr) - D*r + E*x^2
        :return: (np.)float
        """

        B, C, D, E = args  # auxiliary values pre-calculated in pre_calculate_for_potential_value_primary()
        radius2 = np.power(radius, 2)

        return 1 / radius + self.mass_ratio / np.sqrt(B + radius2 - C * radius) - D * radius + E * radius2

    def pre_calculate_for_potential_value_primary_cylindrical(self, *args):
        """
        function calculates auxiliary values for calculation of primary component potential  in cylindrical symmetry,
        and therefore they don't need to be wastefully recalculated every iteration in solver

        :param args: (azimut angle (0, 2pi), z_n (cylindrical, identical with cartesian x))
        :return: tuple: (A, B, C, D, E, F) such that: Psi1 = 1/sqrt(A+r^2) + q/sqrt(B + r^2) - C + D*(E+F*r^2)
        """
        phi, z = args

        qq = self.mass_ratio / (1 + self.mass_ratio)

        A = np.power(z, 2)
        B = np.power(1 - z, 2)
        C = 0.5 * self.mass_ratio * qq
        D = 0.5 + 0.5 * self.mass_ratio
        E = np.power(qq - z, 2)
        F = np.power(np.sin(phi), 2)

        return A, B, C, D, E, F

    def potential_value_primary_cylindrical(self, radius, *args):
        """
        calculates modified kopal potential from point of view of primary component in cylindrical coordinates
        r_n, phi_n, z_n, where z_n = x and heads along z axis, this function is intended for generation of ``necks``
        of W UMa systems, therefore components distance = 1 an synchronicity = 1 is assumed

        :param radius: np.float
        :param args: tuple: (A, B, C, D, E, F) such that: Psi1 = 1/sqrt(A+r^2) + q/sqrt(B + r^2) - C + D*(E+F*r^2)
        :return:
        """
        A, B, C, D, E, F = args

        radius2 = np.power(radius, 2)
        return 1 / np.sqrt(A + radius2) + self.mass_ratio / np.sqrt(B + radius2) - C + D * (E + F * radius2)

    def pre_calculate_for_potential_value_secondary(self, *args):
        """
        function calculates auxiliary values for calculation of secondary component potential, and therefore they don't
        need to be wastefully recalculated every iteration in solver

        :param args: (component distance, azimut angle (0, 2pi), latitude angle (0, pi)
        :return: tuple: (B, C, D, E, F) such that: Psi2 = q/r + 1/sqrt(B+r^2+Cr) - D*r + E*x^2 + F
        """
        distance, phi, theta = args  # distance between components, azimut angle, latitude angle (0,180)

        cs = np.cos(phi) * np.sin(theta)

        B = np.power(distance, 2)
        C = 2 * distance * cs
        D = cs / B
        E = 0.5 * np.power(self.secondary.synchronicity, 2) * (1 + self.mass_ratio) * (1 - np.power(np.cos(theta), 2))
        F = 0.5 - 0.5 * self.mass_ratio

        if np.isscalar(phi):
            return B, C, D, E, F
        else:
            BB = B * np.ones(np.shape(phi))
            FF = F * np.ones(np.shape(phi))
            return np.column_stack((BB, C, D, E, FF))

    def potential_value_secondary(self, radius, *args):
        """
        calculates modified kopal potential from point of view of secondary component

        :param radius: np.float; spherical variable
        :param args: tuple: (B, C, D, E, F) such that: Psi2 = q/r + 1/sqrt(B+r^2-Cr) - D*r + E*x^2 + F
        :return: np.float
        """
        B, C, D, E, F = args
        radius2 = np.power(radius, 2)

        return self.mass_ratio / radius + 1. / np.sqrt(B + radius2 - C * radius) - D * radius + E * radius2 + F

    def pre_calculate_for_potential_value_secondary_cylindrical(self, *args):
        """
        function calculates auxiliary values for calculation of secondary component potential in cylindrical symmetry,
        and therefore they don't need to be wastefully recalculated every iteration in solver

        :param args: (azimut angle (0, 2pi), z_n (cylindrical, identical with cartesian x))
        :return: tuple: (A, B, C, D, E, F, G) such that: Psi2 = q/sqrt(A+r^2) + 1/sqrt(B + r^2) - C + D*(E+F*r^2) + G
        """
        phi, z = args

        qq = 1 / (1 + self.mass_ratio)

        A = np.power(z, 2)
        B = np.power(1 - z, 2)
        C = 0.5 / qq
        D = np.power(qq - z, 2)
        E = np.power(np.sin(phi), 2)
        F = 0.5 - 0.5 * self.mass_ratio - 0.5 * qq

        return A, B, C, D, E, F

    def potential_value_secondary_cylindrical(self, radius, *args):
        """
        calculates modified kopal potential from point of view of secondary component in cylindrical coordinates
        r_n, phi_n, z_n, where z_n = x and heads along z axis, this function is intended for generation of ``necks``
        of W UMa systems, therefore components distance = 1 an synchronicity = 1 is assumed

        :param radius: np.float
        :param args: tuple: (A, B, C, D, E, F, G) such that: Psi2 = q/sqrt(A+r^2) + 1/sqrt(B+r^2) - C + D*(E+F*r^2) + G
        :return:
        """
        A, B, C, D, E, F = args

        radius2 = np.power(radius, 2)
        return self.mass_ratio / np.sqrt(A + radius2) + 1. / np.sqrt(B + radius2) + C * (D + E * radius2) + F

    def potential_primary_fn(self, radius, *args):
        """
        implicit potential function from perspective of primary component

        :param radius: np.float; spherical variable
        :param args: (np.float, np.float, np.float); (component distance, azimutal angle, polar angle)
        :return:
        """
        return self.potential_value_primary(radius, *args) - self.primary.surface_potential

    def potential_primary_cylindrical_fn(self, radius, *args):
        """
        implicit potential function from perspective of primary component given in cylindrical coordinates

        :param radius: np.float
        :param args: tuple: (phi, z) - polar coordinates
        :return:
        """
        return self.potential_value_primary_cylindrical(radius, *args) - self.primary.surface_potential

    def potential_secondary_fn(self, radius, *args):
        """
        implicit potential function from perspective of secondary component

        :param radius: np.float; spherical variable
        :param args: (np.float, np.float, np.float); (component distance, azimutal angle, polar angle)
        :return: np.float
        """
        return self.potential_value_secondary(radius, *args) - self.secondary.surface_potential

    def potential_secondary_cylindrical_fn(self, radius, *args):
        """
        implicit potential function from perspective of secondary component given in cylindrical coordinates

        :param radius: np.float
        :param args: tuple: (phi, z) - polar coordinates
        :return: np.float
        """
        return self.potential_value_secondary_cylindrical(radius, *args) - self.secondary.surface_potential

    def critical_potential(self, component, components_distance):
        """
        return a critical potential for target component

        :param component: str; define target component to compute critical potential; `primary` or `secondary`
        :param components_distance: np.float
        :return: np.float
        """
        args = components_distance,
        if component == "primary":
            solution = newton(self.primary_potential_derivative_x, 0.000001, args=args, tol=1e-12)
        elif component == "secondary":
            solution = newton(self.secondary_potential_derivative_x, 0.000001, args=args, tol=1e-12)
        else:
            raise ValueError("Parameter `component` has incorrect value. Use `primary` or `secondary`.")

        if not np.isnan(solution):
            args = components_distance, 0.0, const.HALF_PI
            if component == "primary":
                args = self.pre_calculate_for_potential_value_primary(*args)
                return abs(self.potential_value_primary(solution, *args))
            elif component == 'secondary':
                args = self.pre_calculate_for_potential_value_secondary(*args)
                return abs(self.potential_value_secondary(components_distance - solution, *args))
        else:
            raise ValueError("Iteration process to solve critical potential seems to lead nowhere (critical potential "
                             "_solver has failed).")

    def calculate_potential_gradient(self, component, components_distance, points=None):
        """
        return outter gradients in each point of star surface or defined points

        :param component: str, `primary` or `secondary`
        :param components_distance: float, in SMA distance
        :param points: list/numpy.array or None
        :return: numpy.array
        """
        component_instance = getattr(self, component)
        points = component_instance.points if points is None else points

        r3 = np.power(np.linalg.norm(points, axis=1), 3)
        r_hat3 = np.power(np.linalg.norm(points - np.array([components_distance, 0., 0]), axis=1), 3)
        if component == 'primary':
            F2 = np.power(self.primary.synchronicity, 2)
            domega_dx = - points[:, 0] / r3 + self.mass_ratio * (
                    components_distance - points[:, 0]) / r_hat3 + F2 * (
                                self.mass_ratio + 1) * points[:, 0] - self.mass_ratio / np.power(components_distance, 2)
        elif component == 'secondary':
            F2 = np.power(self.secondary.synchronicity, 2)
            domega_dx = - points[:, 0] / r3 + self.mass_ratio * (
                    components_distance - points[:, 0]) / r_hat3 - F2 * (
                                self.mass_ratio + 1) * (components_distance - points[:, 0]) * points[:,
                                                                                              0] + 1 / np.power(
                components_distance, 2)
        else:
            raise ValueError('Invalid value `{}` of argument `component`. Use `primary` or `secondary`.'
                             .format(component))
        domega_dy = - points[:, 1] * (1 / r3 + self.mass_ratio / r_hat3 - F2 * (self.mass_ratio + 1))
        domega_dz = - points[:, 2] * (1 / r3 + self.mass_ratio / r_hat3)
        return -np.column_stack((domega_dx, domega_dy, domega_dz))

    def calculate_face_magnitude_gradient(self, component, components_distance, points=None, faces=None):
        """
        return array of face mean of magnitude gradients

        :param component:
        :param components_distance:
        :param points: points in which to calculate magnitude of gradient, if False/None take star points
        :param faces: faces corresponding to given points
        :return: np.array
        """
        if points is not None and faces is None:
            raise TypeError('Specify faces corresponding to given points')

        component_instance = getattr(self, component)
        if component_instance.spots:
            faces = component_instance.faces if faces is None else faces
            points = component_instance.points if points is None else points
        else:
            faces = component_instance.faces[:component_instance.base_symmetry_faces_number] if faces is None \
                else faces
            points = component_instance.points[:component_instance.base_symmetry_points_number] if points is None \
                else points

        gradients = self.calculate_potential_gradient(component, components_distance, points=points)
        domega_dx, domega_dy, domega_dz = gradients[:, 0], gradients[:, 1], gradients[:, 2]
        points_gradients = np.power(np.power(domega_dx, 2) + np.power(domega_dy, 2) + np.power(domega_dz, 2), 0.5)

        return np.mean(points_gradients[faces], axis=1) if component_instance.spots \
            else np.mean(points_gradients[faces], axis=1)[component_instance.face_symmetry_vector]

    def calculate_polar_potential_gradient_magnitude(self, component=None, components_distance=None):
        """
        returns magnitude of polar potential gradient

        :param component: str, `primary` or `secondary`
        :param components_distance: float, in SMA distance
        :return: numpy.array
        """
        component_instance = getattr(self, component)
        points = np.array([0., 0., component_instance.polar_radius]) if component == 'primary' \
            else np.array([components_distance, 0., component_instance.polar_radius])
        r3 = np.power(np.linalg.norm(points), 3)
        r_hat3 = np.power(np.linalg.norm(points - np.array([components_distance, 0., 0.])), 3)
        if component == 'primary':
            domega_dx = self.mass_ratio * components_distance / r_hat3 \
                        - self.mass_ratio / np.power(components_distance, 2)
        elif component == 'secondary':
            domega_dx = - points[0] / r3 + self.mass_ratio * (components_distance - points[0]) / r_hat3 \
                        + 1. / np.power(components_distance, 2)
        else:
            raise ValueError('Invalid value `{}` of argument `component`. Use `primary` or `secondary`.'
                             .format(component))
        domega_dz = - points[2] * (1. / r3 + self.mass_ratio / r_hat3)
        return np.power(np.power(domega_dx, 2) + np.power(domega_dz, 2), 0.5)

    def calculate_radius(self, *args):
        """
        function calculates radius of the star in given direction of arbitrary direction vector (in spherical
        coordinates) starting from the centre of the star

        :param args: tuple - (component: str - `primary` or `secondary`,
                              components_distance: float - distance between components in SMA units,
                              phi: float - longitudonal angle of direction vector measured from point under L_1 in
                                           positive direction (in radians)
                              omega: float - latitudonal angle of direction vector measured from north pole (in radians)
                              )
        :return: float - radius
        """
        if args[0] == 'primary':
            fn = self.potential_primary_fn
            precalc = self.pre_calculate_for_potential_value_primary
        elif args[0] == 'secondary':
            fn = self.potential_secondary_fn
            precalc = self.pre_calculate_for_potential_value_secondary
        else:
            raise ValueError('Invalid value of `component` argument {}. Expecting `primary` or `secondary`.'
                             .format(args[0]))

        scipy_solver_init_value = np.array([args[1] / 1e4])
        argss = precalc(*args[1:])
        solution, a, ier, b = scipy.optimize.fsolve(fn, scipy_solver_init_value,
                                                    full_output=True, args=argss, xtol=1e-10)

        # check for regular solution
        if ier == 1 and not np.isnan(solution[0]) and 30 >= solution[0] >= 0:
            return solution[0]
        else:
            if 0 < solution[0] < 1.0:
                return solution[0]
            else:
                raise ValueError('Invalid value of radius {} was calculated.'.format(solution))

    def calculate_polar_radius(self, component, components_distance):
        """
        calculates polar radius in the similar manner as in BinarySystem.compute_equipotential_boundary method

        :param component: str; `primary` or `secondary`
        :param components_distance: float
        :return: float; polar radius
        """
        args = (component, components_distance, 0.0, 0.0)
        return self.calculate_radius(*args)

    def calculate_side_radius(self, component, components_distance):
        """
        calculates side radius in the similar manner as in BinarySystem.compute_equipotential_boundary method

        :param component: str; `primary` or `secondary`
        :param components_distance: float
        :return: float; side radius
        """
        args = (component, components_distance, const.HALF_PI, const.HALF_PI)
        return self.calculate_radius(*args)

    def calculate_backward_radius(self, component, components_distance):
        """
        calculates backward radius in the similar manner as in BinarySystem.compute_equipotential_boundary method

        :param component: str; `primary` or `secondary`
        :param components_distance: float
        :return: float; polar radius
        """

        args = (component, components_distance, const.PI, const.HALF_PI)
        return self.calculate_radius(*args)

    def calculate_forward_radius(self, component, components_distance):
        """
        calculates forward radius in the similar manner as in BinarySystem.compute_equipotential_boundary method,
        NOT very usable in over-contact systems

        :param component:
        :param components_distance:
        :return:
        """
        args = (component, components_distance, 0.0, const.HALF_PI)
        return self.calculate_radius(*args)

    def compute_equipotential_boundary(self, components_distance, plane):
        """
        compute a equipotential boundary of components (crossection of Hill plane)

        :param components_distance: (np.)float
        :param plane: str; xy, yz, zx
        :return: tuple (np.array, np.array)
        """

        components = ['primary', 'secondary']
        points_primary, points_secondary = [], []
        fn_map = {'primary': (self.potential_primary_fn, self.pre_calculate_for_potential_value_primary),
                  'secondary': (self.potential_secondary_fn, self.pre_calculate_for_potential_value_secondary)}

        angles = np.linspace(-3 * const.HALF_PI, const.HALF_PI, 300, endpoint=True)
        for component in components:
            for angle in angles:
                if utils.is_plane(plane, 'xy'):
                    args, use = (components_distance, angle, const.HALF_PI), False
                elif utils.is_plane(plane, 'yz'):
                    args, use = (components_distance, const.HALF_PI, angle), False
                elif utils.is_plane(plane, 'zx'):
                    args, use = (components_distance, 0.0, angle), False
                else:
                    raise ValueError('Invalid choice of crossection plane, use only: `xy`, `yz`, `zx`.')

                scipy_solver_init_value = np.array([components_distance / 10000.0])
                args = fn_map[component][1](*args)
                solution, _, ier, _ = scipy.optimize.fsolve(fn_map[component][0], scipy_solver_init_value,
                                                            full_output=True, args=args, xtol=1e-12)

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

    def lagrangian_points(self):
        """

        :return: list; x-valeus of libration points [L3, L1, L2] respectively
        """

        def potential_dx(x, *args):
            """
            general potential in case of primary.synchornicity = secondary.synchronicity = 1.0 and eccentricity = 0.0

            :param x: (np.)float
            :param args: tuple; periastron distance of components
            :return: (np.)float
            """
            d, = args
            r_sqr, rw_sqr = x ** 2, (d - x) ** 2
            return - (x / r_sqr ** (3.0 / 2.0)) + ((self.mass_ratio * (d - x)) / rw_sqr ** (
                    3.0 / 2.0)) + (self.mass_ratio + 1) * x - self.mass_ratio / d ** 2

        periastron_distance = self.orbit.periastron_distance
        xs = np.linspace(- periastron_distance * 3.0, periastron_distance * 3.0, 100)

        args_val = periastron_distance,
        round_to = 10
        points, lagrange = [], []

        for x_val in xs:
            try:
                # if there is no valid value (in case close to x=0.0, potential_dx diverge)
                np.seterr(divide='raise', invalid='raise')
                potential_dx(round(x_val, round_to), *args_val)
                np.seterr(divide='print', invalid='print')
            except Exception as e:
                self._logger.debug("Invalid value passed to potential, exception: {0}".format(str(e)))
                continue

            try:
                solution, _, ier, _ = scipy.optimize.fsolve(potential_dx, x_val, full_output=True, args=args_val,
                                                            xtol=1e-12)
                if ier == 1:
                    if round(solution[0], 5) not in points:
                        try:
                            value_dx = abs(round(potential_dx(solution[0], *args_val), 4))
                            use = True if value_dx == 0 else False
                        except Exception as e:
                            self._logger.debug(
                                "Skipping sollution for x: {0} due to exception: {1}".format(x_val, str(e)))
                            use = False

                        if use:
                            points.append(round(solution[0], 5))
                            lagrange.append(solution[0])
                            if len(lagrange) == 3:
                                break
            except Exception as e:
                self._logger.debug("Solution for x: {0} lead to nowhere, exception: {1}".format(x_val, str(e)))
                continue

        return sorted(lagrange) if self.mass_ratio < 1.0 else sorted(lagrange, reverse=True)

    def libration_potentials(self):
        """
        return potentials in L3, L1, L2 respectively

        :return: list; [Omega(L3), Omega(L1), Omega(L2)]
        """

        def potential(radius):
            theta, d = const.HALF_PI, self.orbit.periastron_distance
            if isinstance(radius, (float, int, np.float, np.int)):
                radius = [radius]
            elif not isinstance(radius, (list, np.array)):
                raise ValueError("Incorrect value of variable `radius`")

            p_values = []
            for r in radius:
                phi, r = (0.0, r) if r >= 0 else (const.PI, abs(r))

                block_a = 1.0 / r
                block_b = self.mass_ratio / (np.sqrt(np.power(d, 2) + np.power(r, 2) - (
                        2.0 * r * np.cos(phi) * np.sin(theta) * d)))
                block_c = (self.mass_ratio * r * np.cos(phi) * np.sin(theta)) / (np.power(d, 2))
                block_d = 0.5 * (1 + self.mass_ratio) * np.power(r, 2) * (
                        1 - np.power(np.cos(theta), 2))

                p_values.append(block_a + block_b - block_c + block_d)
            return p_values

        lagrangian_points = self.lagrangian_points()
        return potential(lagrangian_points)

    def mesh_detached(self, component, components_distance, symmetry_output=False):
        """
        creates surface mesh of given binary star component in case of detached (semi-detached) system

        :param symmetry_output: bool - if true, besides surface points are returned also `symmetry_vector`,
                                       `base_symmetry_points_number`, `inverse_symmetry_matrix`
        :param component: str - `primary` or `secondary`
        :param components_distance: np.float
        :return: numpy.array([[x1 y1 z1],
                              [x2 y2 z2],
                                ...
                              [xN yN zN]]) - array of surface points if symmetry_output = False, else:
                 numpy.array([[x1 y1 z1],
                              [x2 y2 z2],
                                ...
                              [xN yN zN]]) - array of surface points,
                 numpy.array([indices_of_symmetrical_points]) - array which remapped surface points to symmetrical one
                                                                quarter of surface,
                 numpy.float - number of points included in symmetrical one quarter of surface,
                 numpy.array([quadrant[indexes_of_remapped_points_in_quadrant]) - matrix of four sub matrices that
                                                                                mapped basic symmetry quadrant to all
                                                                                others quadrants
        """
        component_instance = getattr(self, component)
        if component_instance.discretization_factor > const.HALF_PI:
            raise ValueError("Invalid value of alpha parameter. Use value less than 90.")

        alpha = component_instance.discretization_factor

        if component == 'primary':
            fn = self.potential_primary_fn
            precalc = self.pre_calculate_for_potential_value_primary
        elif component == 'secondary':
            fn = self.potential_secondary_fn
            precalc = self.pre_calculate_for_potential_value_secondary
        else:
            raise ValueError('Invalid value of `component` argument: `{}`. Expecting '
                             '`primary` or `secondary`.'.format(component))

        # pre calculating azimuths for surface points on quarter of the star surface
        phi, theta, separator = self.pre_calc_azimuths_for_detached_points(alpha)

        # calculating mesh in cartesian coordinates for quarter of the star
        args = phi, theta, components_distance, precalc, fn
        # points_q = self.get_surface_points_multithread(*args)
        points_q = self.get_surface_points(*args)

        equator = points_q[:separator[0], :]
        # assigning equator points and nearside and farside points A and B
        x_a, x_eq, x_b = equator[0, 0], equator[1: -1, 0], equator[-1, 0]
        y_a, y_eq, y_b = equator[0, 1], equator[1: -1, 1], equator[-1, 1]
        z_a, z_eq, z_b = equator[0, 2], equator[1: -1, 2], equator[-1, 2]

        # calculating points on phi = 0 meridian
        meridian = points_q[separator[0]: separator[1], :]
        x_meridian, y_meridian, z_meridian = meridian[:, 0], meridian[:, 1], meridian[:, 2]

        # the rest of the surface
        quarter = points_q[separator[1]:, :]
        x_q, y_q, z_q = quarter[:, 0], quarter[:, 1], quarter[:, 2]

        # stiching together 4 quarters of stellar surface in order:
        # north hemisphere: left_quadrant (from companion point of view):
        #                   nearside_point, farside_point, equator, quarter, meridian
        #                   right_quadrant:
        #                   quadrant, equator
        # south hemisphere: right_quadrant:
        #                   quadrant, meridian
        #                   left_quadrant:
        #                   quadrant
        x = np.array([x_a, x_b])
        y = np.array([y_a, y_b])
        z = np.array([z_a, z_b])
        x = np.concatenate((x, x_eq, x_q, x_meridian, x_q, x_eq, x_q, x_meridian, x_q))
        y = np.concatenate((y, y_eq, y_q, y_meridian, -y_q, -y_eq, -y_q, -y_meridian, y_q))
        z = np.concatenate((z, z_eq, z_q, z_meridian, z_q, z_eq, -z_q, -z_meridian, -z_q))

        x = -x + components_distance if component == 'secondary' else x
        points = np.column_stack((x, y, z))
        if symmetry_output:
            equator_length = np.shape(x_eq)[0]
            meridian_length = np.shape(x_meridian)[0]
            quarter_length = np.shape(x_q)[0]
            quadrant_start = 2 + equator_length
            base_symmetry_points_number = 2 + equator_length + quarter_length + meridian_length
            symmetry_vector = np.concatenate((np.arange(base_symmetry_points_number),  # 1st quadrant
                                              np.arange(quadrant_start, quadrant_start + quarter_length),
                                              np.arange(2, quadrant_start),  # 2nd quadrant
                                              np.arange(quadrant_start, base_symmetry_points_number),  # 3rd quadrant
                                              np.arange(quadrant_start, quadrant_start + quarter_length)
                                              ))

            points_length = np.shape(x)[0]
            inverse_symmetry_matrix = \
                np.array([np.arange(base_symmetry_points_number),  # 1st quadrant
                          np.concatenate(([0, 1],
                                          np.arange(base_symmetry_points_number + quarter_length,
                                                    base_symmetry_points_number + quarter_length + equator_length),
                                          np.arange(base_symmetry_points_number,
                                                    base_symmetry_points_number + quarter_length),
                                          np.arange(base_symmetry_points_number - meridian_length,
                                                    base_symmetry_points_number))),  # 2nd quadrant
                          np.concatenate(([0, 1],
                                          np.arange(base_symmetry_points_number + quarter_length,
                                                    base_symmetry_points_number + quarter_length + equator_length),
                                          np.arange(base_symmetry_points_number + quarter_length + equator_length,
                                                    base_symmetry_points_number + 2 * quarter_length + equator_length +
                                                    meridian_length))),  # 3rd quadrant
                          np.concatenate((np.arange(2 + equator_length),
                                          np.arange(points_length - quarter_length, points_length),
                                          np.arange(base_symmetry_points_number + 2 * quarter_length + equator_length,
                                                    base_symmetry_points_number + 2 * quarter_length + equator_length +
                                                    meridian_length)))  # 4th quadrant
                          ])

            return points, symmetry_vector, base_symmetry_points_number, inverse_symmetry_matrix
        else:
            return points

    @staticmethod
    def pre_calc_azimuths_for_detached_points(alpha):
        """
        returns azimuths for the whole quarter surface in specific order (near point, equator, far point and the rest)
        separator gives you information about position of these sections

        :param alpha:
        :return:
        """
        separator = []

        # azimuths for points on equator
        num = int(const.PI // alpha)
        phi = np.linspace(0., const.PI, num=num + 1)
        theta = np.array([const.HALF_PI for _ in phi])
        separator.append(np.shape(theta)[0])

        # azimuths for points on meridian
        num = int(const.HALF_PI // alpha)
        phi_meridian = np.array([const.PI for _ in range(num - 1)] + [0 for _ in range(num)])
        theta_meridian = np.concatenate((np.linspace(const.HALF_PI - alpha, alpha, num=num - 1),
                                         np.linspace(0., const.HALF_PI, num=num, endpoint=False)))

        phi = np.concatenate((phi, phi_meridian))
        theta = np.concatenate((theta, theta_meridian))
        separator.append(np.shape(theta)[0])

        # azimuths for rest of the quarter
        num = int(const.HALF_PI // alpha)
        thetas = np.linspace(alpha, const.HALF_PI, num=num - 1, endpoint=False)
        phi_q, theta_q = [], []
        for tht in thetas:
            alpha_corrected = alpha / np.sin(tht)
            num = int(const.PI // alpha_corrected)
            alpha_corrected = const.PI / (num + 1)
            phi_q_add = [alpha_corrected * ii for ii in range(1, num + 1)]
            phi_q += phi_q_add
            theta_q += [tht for _ in phi_q_add]

        phi = np.concatenate((phi, phi_q))
        theta = np.concatenate((theta, theta_q))

        return phi, theta, separator

    @staticmethod
    def get_surface_points(*argss):
        """
        function solves radius for given azimuths that are passed in *argss

        :param argss:
        :return:
        """
        phi, theta, components_distance, precalc, fn = argss

        pre_calc_vals = precalc(*(components_distance, phi, theta))

        solver_init_value = np.array([1. / 10000.])
        r = []
        for ii, phii in enumerate(phi):
            # args = (components_distance, phii, theta[ii])
            # args = precalc(*args)
            args = tuple(pre_calc_vals[ii, :])
            solution, _, ier, _ = scipy.optimize.fsolve(fn, solver_init_value, full_output=True, args=args, xtol=1e-12)
            r.append(solution[0])

        r = np.array(r)
        return utils.spherical_to_cartesian(np.column_stack((r, phi, theta)))

    @staticmethod
    def get_surface_point_multithread(*args):
        """
        function solves radius for given azimuths that are passed in *argss

        :param argss:
        :return:
        """
        fn = args[0]
        solver_init_value = np.array([1. / 10000.])
        solution, _, ier, _ = scipy.optimize.fsolve(fn, solver_init_value, full_output=True, args=args[1:], xtol=1e-12)
        print(type(solution[0]))
        return solution[0]

    def get_surface_points_multithread(self, *argss):
        """
        function solves radius for given azimuths that are passed in *argss via multithreading approach

        :param argss:
        :return:
        """

        phi, theta, components_distance, precalc, fn = argss

        precalc_vals = list(precalc(*(components_distance, phi, theta)))
        precalc_vals[0] = precalc_vals[0] * np.ones(np.shape(phi))

        aux = [[precalc_vals[ii][jj] for ii in range(len(precalc_vals))] for jj in range(len(precalc_vals[1]))]
        args = [tuple([fn] + ax) for ax in aux]

        pool = Pool(processes=config.NUMBER_OF_THREADS)

        p = pool.apply_async(self.r_solving_thread, args=args)
        radius = p.get(timeout=1)
        print(radius)

        return utils.spherical_to_cartesian(np.column_stack((radius, phi, theta)))

    def r_solving_thread(self, *args):
        # phi, theta, components_distance, precalc, fn = args

        points_thread = self.get_surface_points(*args)
        return points_thread

    def mesh_over_contact(self, component=None, symmetry_output=False):
        """
        creates surface mesh of given binary star component in case of over-contact system

        :param symmetry_output: bool - if true, besides surface points are returned also `symmetry_vector`,
                                       `base_symmetry_points_number`, `inverse_symmetry_matrix`
        :param component: str - `primary` or `secondary`
        :return: numpy.array([[x1 y1 z1],
                              [x2 y2 z2],
                                ...
                              [xN yN zN]]) - array of surface points if symmetry_output = False, else:
                 numpy.array([[x1 y1 z1],
                              [x2 y2 z2],
                                ...
                              [xN yN zN]]) - array of surface points,
                 numpy.array([indices_of_symmetrical_points]) - array which remapped surface points to symmetrical one
                                                                quarter of surface,
                 numpy.float - number of points included in symmetrical one quarter of surface,
                 numpy.array([quadrant[indexes_of_remapped_points_in_quadrant]) - matrix of four sub matrices that
                                                                                mapped basic symmetry quadrant to all
                                                                                others quadrants
        """
        component_instance = getattr(self, component)
        if component_instance.discretization_factor > const.HALF_PI:
            raise ValueError("Invalid value of alpha parameter. Use value less than 90.")

        alpha = component_instance.discretization_factor
        scipy_solver_init_value = np.array([1. / 10000.])

        # calculating distance between components
        components_distance = self.orbit.orbital_motion(phase=0)[0][0]

        if component == 'primary':
            fn = self.potential_primary_fn
            fn_cylindrical = self.potential_primary_cylindrical_fn
            precalc = self.pre_calculate_for_potential_value_primary
            precal_cylindrical = self.pre_calculate_for_potential_value_primary_cylindrical
        elif component == 'secondary':
            fn = self.potential_secondary_fn
            fn_cylindrical = self.potential_secondary_cylindrical_fn
            precalc = self.pre_calculate_for_potential_value_secondary
            precal_cylindrical = self.pre_calculate_for_potential_value_secondary_cylindrical
        else:
            raise ValueError('Invalid value of `component` argument: `{}`. '
                             'Expecting `primary` or `secondary`.'.format(component))

        # calculating points on farside equator
        num = int(const.HALF_PI // alpha)
        r_eq1 = []
        phi_eq1 = np.linspace(const.HALF_PI, const.PI, num=num + 1)
        theta_eq1 = np.array([const.HALF_PI for _ in phi_eq1])
        for phi in phi_eq1:
            args = (components_distance, phi, const.HALF_PI)
            args = precalc(*args)
            solution, _, ier, _ = scipy.optimize.fsolve(fn, scipy_solver_init_value, full_output=True, args=args,
                                                        xtol=1e-12)
            r_eq1.append(solution[0])
        r_eq1 = np.array(r_eq1)
        equator1 = utils.spherical_to_cartesian(np.column_stack((r_eq1, phi_eq1, theta_eq1)))
        # assigning equator points and point A
        x_eq1, x_a = equator1[: -1, 0], equator1[-1, 0],
        y_eq1, y_a = equator1[: -1, 1], equator1[-1, 1],
        z_eq1, z_a = equator1[: -1, 2], equator1[-1, 2],

        # calculating points on phi = pi meridian
        r_meridian1 = []
        num = int(const.HALF_PI // alpha)
        phi_meridian1 = np.array([const.PI for _ in range(num)])
        theta_meridian1 = np.linspace(0., const.HALF_PI - alpha, num=num)
        for ii, theta in enumerate(theta_meridian1):
            args = (components_distance, phi_meridian1[ii], theta)
            args = precalc(*args)
            solution, _, ier, _ = scipy.optimize.fsolve(fn, scipy_solver_init_value, full_output=True, args=args,
                                                        xtol=1e-12)
            r_meridian1.append(solution[0])
        r_meridian1 = np.array(r_meridian1)
        meridian1 = utils.spherical_to_cartesian(np.column_stack((r_meridian1, phi_meridian1, theta_meridian1)))
        x_meridian1, y_meridian1, z_meridian1 = meridian1[:, 0], meridian1[:, 1], meridian1[:, 2]

        # calculating points on phi = pi/2 meridian, perpendicular to component`s radius vector
        r_meridian2 = []
        num = int(const.HALF_PI // alpha) - 1
        phi_meridian2 = np.array([const.HALF_PI for _ in range(num)])
        theta_meridian2 = np.linspace(alpha, const.HALF_PI, num=num, endpoint=False)
        for ii, theta in enumerate(theta_meridian2):
            args = (components_distance, phi_meridian2[ii], theta)
            args = precalc(*args)
            solution, _, ier, _ = scipy.optimize.fsolve(fn, scipy_solver_init_value, full_output=True, args=args,
                                                        xtol=1e-12)
            r_meridian2.append(solution[0])
        r_meridian2 = np.array(r_meridian2)
        meridian2 = utils.spherical_to_cartesian(np.column_stack((r_meridian2, phi_meridian2, theta_meridian2)))
        x_meridian2, y_meridian2, z_meridian2 = meridian2[:, 0], meridian2[:, 1], meridian2[:, 2]

        # calculating the rest of the surface on farside
        thetas = np.linspace(alpha, const.HALF_PI, num=num, endpoint=False)
        r_q1, phi_q1, theta_q1 = [], [], []
        for theta in thetas:
            alpha_corrected = alpha / np.sin(theta)
            num = int(const.HALF_PI // alpha_corrected)
            alpha_corrected = const.HALF_PI / (num + 1)
            phi_q_add = [const.HALF_PI + alpha_corrected * ii for ii in range(1, num + 1)]
            phi_q1 += phi_q_add
            for phi in phi_q_add:
                theta_q1.append(theta)
                args = (components_distance, phi, theta)
                args = precalc(*args)
                solution, _, ier, _ = scipy.optimize.fsolve(fn, scipy_solver_init_value, full_output=True, args=args,
                                                            xtol=1e-12)
                r_q1.append(solution[0])
        r_q1, phi_q1, theta_q1 = np.array(r_q1), np.array(phi_q1), np.array(theta_q1)
        quarter = utils.spherical_to_cartesian(np.column_stack((r_q1, phi_q1, theta_q1)))
        x_q1, y_q1, z_q1 = quarter[:, 0], quarter[:, 1], quarter[:, 2]

        # generating the neck
        neck_position, neck_polynome = self.calculate_neck_position(return_polynomial=True)
        # lets define cylindrical coordinate system r_n, phi_n, z_n for our neck where z_n = x, phi_n = 0 heads along
        # z axis
        delta_z = alpha * self.calculate_polar_radius(component=component, components_distance=1)
        if component == 'primary':
            num = 15 * int(
                neck_position // (component_instance.polar_radius * component_instance.discretization_factor))
            # position of z_n adapted to the slope of the neck, gives triangles with more similar areas
            x_curve = np.linspace(0., neck_position, num=num, endpoint=True)
            z_curve = np.polyval(neck_polynome, x_curve)
            curve = np.column_stack((x_curve, z_curve))
            neck_lengths = np.sqrt(np.sum(np.diff(curve, axis=0) ** 2, axis=1))
            neck_length = np.sum(neck_lengths)
            segment = neck_length / (int(neck_length // delta_z) + 1)

            k = 1
            z_ns, line_sum = [], 0.0
            for ii in range(num - 2):
                line_sum += neck_lengths[ii]
                if line_sum > k * segment:
                    z_ns.append(x_curve[ii + 1])
                    k += 1
            z_ns.append(neck_position)
            z_ns = np.array(z_ns)
            # num = int(neck_position // delta_z) + 1
            # z_ns = np.linspace(delta_z, neck_position, num=num, endpoint=True)
        else:
            num = 15 * int(
                (1 - neck_position) // (component_instance.polar_radius * component_instance.discretization_factor))
            # position of z_n adapted to the slope of the neck, gives triangles with more similar areas
            x_curve = np.linspace(neck_position, 1, num=num, endpoint=True)
            z_curve = np.polyval(neck_polynome, x_curve)
            curve = np.column_stack((x_curve, z_curve))
            neck_lengths = np.sqrt(np.sum(np.diff(curve, axis=0) ** 2, axis=1))
            neck_length = np.sum(neck_lengths)
            segment = neck_length / (int(neck_length // delta_z) + 1)

            k = 1
            z_ns, line_sum = [1 - neck_position], 0.0
            for ii in range(num - 2):
                line_sum += neck_lengths[ii]
                if line_sum > k * segment:
                    z_ns.append(1 - x_curve[ii + 1])
                    k += 1

            z_ns = np.array(z_ns)

            # num = int((1 - neck_position) // delta_z) + 1
            # z_ns = np.linspace(delta_z, 1.0 - neck_position, num=num, endpoint=True)

        # generating equatorial, polar part and rest of the neck
        r_eqn, phi_eqn, z_eqn = [], [], []
        r_meridian_n, phi_meridian_n, z_meridian_n = [], [], []
        r_n, phi_n, z_n = [], [], []
        for z in z_ns:
            z_eqn.append(z)
            phi_eqn.append(const.HALF_PI)
            args = (const.HALF_PI, z)
            args = precal_cylindrical(*args)
            solution, _, ier, _ = scipy.optimize.fsolve(fn_cylindrical, scipy_solver_init_value, full_output=True,
                                                        args=args, xtol=1e-12)
            r_eqn.append(solution[0])

            z_meridian_n.append(z)
            phi_meridian_n.append(0.)
            args = (0., z)
            args = precal_cylindrical(*args)
            solution, _, ier, _ = scipy.optimize.fsolve(fn_cylindrical, scipy_solver_init_value, full_output=True,
                                                        args=args, xtol=1e-12)
            r_meridian_n.append(solution[0])

            num = int(const.HALF_PI * r_eqn[-1] // delta_z)
            num = 1 if num == 0 else num
            start_val = const.HALF_PI / num
            phis = np.linspace(start_val, const.HALF_PI, num=num - 1, endpoint=False)
            for phi in phis:
                z_n.append(z)
                phi_n.append(phi)
                args = (phi, z)
                args = precal_cylindrical(*args)
                solution, _, ier, _ = scipy.optimize.fsolve(fn_cylindrical, scipy_solver_init_value, full_output=True,
                                                            args=args, xtol=1e-12)
                r_n.append(solution[0])

        r_eqn = np.array(r_eqn)
        z_eqn = np.array(z_eqn)
        phi_eqn = np.array(phi_eqn)
        z_eqn, y_eqn, x_eqn = utils.cylindrical_to_cartesian(r_eqn, phi_eqn, z_eqn)

        r_meridian_n = np.array(r_meridian_n)
        z_meridian_n = np.array(z_meridian_n)
        phi_meridian_n = np.array(phi_meridian_n)
        z_meridian_n, y_meridian_n, x_meridian_n = \
            utils.cylindrical_to_cartesian(r_meridian_n, phi_meridian_n, z_meridian_n)

        r_n = np.array(r_n)
        z_n = np.array(z_n)
        phi_n = np.array(phi_n)
        z_n, y_n, x_n = utils.cylindrical_to_cartesian(r_n, phi_n, z_n)

        # building point blocks similar to those in detached system (equator pts, meridian pts and quarter pts)
        x_eq = np.concatenate((x_eqn, x_eq1), axis=0)
        y_eq = np.concatenate((y_eqn, y_eq1), axis=0)
        z_eq = np.concatenate((z_eqn, z_eq1), axis=0)
        x_q = np.concatenate((x_n, x_meridian2, x_q1), axis=0)
        y_q = np.concatenate((y_n, y_meridian2, y_q1), axis=0)
        z_q = np.concatenate((z_n, z_meridian2, z_q1), axis=0)
        x_meridian = np.concatenate((x_meridian_n, x_meridian1), axis=0)
        y_meridian = np.concatenate((y_meridian_n, y_meridian1), axis=0)
        z_meridian = np.concatenate((z_meridian_n, z_meridian1), axis=0)

        x = np.array([x_a])
        y = np.array([y_a])
        z = np.array([z_a])
        x = np.concatenate((x, x_eq, x_q, x_meridian, x_q, x_eq, x_q, x_meridian, x_q))
        y = np.concatenate((y, y_eq, y_q, y_meridian, -y_q, -y_eq, -y_q, -y_meridian, y_q))
        z = np.concatenate((z, z_eq, z_q, z_meridian, z_q, z_eq, -z_q, -z_meridian, -z_q))

        x = -x + components_distance if component == 'secondary' else x
        points = np.column_stack((x, y, z))
        if symmetry_output:
            equator_length = np.shape(x_eq)[0]
            meridian_length = np.shape(x_meridian)[0]
            quarter_length = np.shape(x_q)[0]
            quadrant_start = 1 + equator_length
            base_symmetry_points_number = 1 + equator_length + quarter_length + meridian_length
            symmetry_vector = np.concatenate((np.arange(base_symmetry_points_number),  # 1st quadrant
                                              np.arange(quadrant_start, quadrant_start + quarter_length),
                                              np.arange(1, quadrant_start),  # 2nd quadrant
                                              np.arange(quadrant_start, base_symmetry_points_number),  # 3rd quadrant
                                              np.arange(quadrant_start, quadrant_start + quarter_length)
                                              ))

            points_length = np.shape(x)[0]
            inverse_symmetry_matrix = \
                np.array([np.arange(base_symmetry_points_number),  # 1st quadrant
                          np.concatenate(([0],
                                          np.arange(base_symmetry_points_number + quarter_length,
                                                    base_symmetry_points_number + quarter_length + equator_length),
                                          np.arange(base_symmetry_points_number,
                                                    base_symmetry_points_number + quarter_length),
                                          np.arange(base_symmetry_points_number - meridian_length,
                                                    base_symmetry_points_number))),  # 2nd quadrant
                          np.concatenate(([0],
                                          np.arange(base_symmetry_points_number + quarter_length,
                                                    base_symmetry_points_number + quarter_length + equator_length),
                                          np.arange(base_symmetry_points_number + quarter_length + equator_length,
                                                    base_symmetry_points_number + 2 * quarter_length + equator_length +
                                                    meridian_length))),  # 3rd quadrant
                          np.concatenate((np.arange(1 + equator_length),
                                          np.arange(points_length - quarter_length, points_length),
                                          np.arange(base_symmetry_points_number + 2 * quarter_length + equator_length,
                                                    base_symmetry_points_number + 2 * quarter_length + equator_length +
                                                    meridian_length)))  # 4th quadrant
                          ])

            return points, symmetry_vector, base_symmetry_points_number, inverse_symmetry_matrix
        else:
            return points

    def detached_system_surface(self, component=None, points=None, components_distance=None):
        """
        calculates surface faces from the given component's points in case of detached or semi-contact system

        :param components_distance: float
        :param points:
        :param component: str
        :return: np.array - N x 3 array of vertices indices
        """
        component_instance = getattr(self, component)
        if points is None:
            points = component_instance.points

        if not np.any(points):
            raise ValueError("{} component, with class instance name {} do not contain any valid surface point "
                             "to triangulate".format(component, component_instance.name))
        # there is a problem with triangulation of near over-contact system, delaunay is not good with pointy surfaces
        critical_pot = self.primary.critical_surface_potential if component == 'primary' \
            else self.secondary.critical_surface_potential
        potential = self.primary.surface_potential if component == 'primary' \
            else self.secondary.surface_potential
        if potential - critical_pot > 0.01:
            self._logger.debug('Triangulating surface of {} component using standard method.'.format(component))
            triangulation = Delaunay(points)
            triangles_indices = triangulation.convex_hull
        else:
            self._logger.debug('Surface of {} component is near or at critical potential. Therefore custom '
                               'triangulation method for (near)critical potential surfaces will be '
                               'used.'.format(component))
            # calculating closest point to the barycentre
            r_near = np.max(points[:, 0]) if component == 'primary' else np.min(points[:, 0])
            # projection of component's far side surface into ``sphere`` with radius r1
            projected_points = np.empty(np.shape(points), dtype=float)

            points_to_transform = copy(points)
            if component == 'secondary':
                points_to_transform[:, 0] -= components_distance
            projected_points = \
                r_near * points_to_transform / np.linalg.norm(points_to_transform, axis=1)[:, None]
            if component == 'secondary':
                projected_points[:, 0] += components_distance

            triangulation = Delaunay(projected_points)
            triangles_indices = triangulation.convex_hull

        return triangles_indices

    def over_contact_surface(self, component=None, points=None, components_distance=None):
        """
        calculates surface faces from the given component's points in case of over-contact system

        :param components_distance: float - not used
        :param points: numpy.array - points to triangulate
        :param component: str - `primary` or `secondary`
        :return: np.array - N x 3 array of vertice indices
        """
        component_instance = getattr(self, component)
        if points is None:
            points = component_instance.points
        if np.isnan(points).any():
            raise ValueError("{} component, with class instance name {} contain any valid point "
                             "to triangulate".format(component, component_instance.name))
        # calculating position of the neck
        neck_x = np.max(points[:, 0]) if component == 'primary' else np.min(points[:, 0])
        # parameter k is used later to transform inner surface to quasi sphere (convex object) which will be then
        # triangulated
        k = neck_x / (neck_x + 0.01) if component == 'primary' else neck_x / ((1 - neck_x) + 0.01)

        # projection of component's far side surface into ``sphere`` with radius r1
        projected_points = np.empty(np.shape(points), dtype=float)

        # outside facing points are just inflated to match with transformed inner surface
        # condition to select outward facing points
        outside_points_test = points[:, 0] <= 0 if component == 'primary' else points[:, 0] >= 1
        outside_points = points[outside_points_test]
        if component == 'secondary':
            outside_points[:, 0] -= 1
        projected_points[outside_points_test] = \
            neck_x * outside_points / np.linalg.norm(outside_points, axis=1)[:, None]
        if component == 'secondary':
            projected_points[:, 0] += 1

        # condition to select outward facing points
        inside_points_test = (points[:, 0] > 0)[:-1] if component == 'primary' else (points[:, 0] < 1)[:-1]
        # if auxiliary point was used than  it is not appended to list of inner points to be transformed
        # (it would cause division by zero error)
        inside_points_test = np.append(inside_points_test, False) if \
            np.array_equal(points[-1], np.array([neck_x, 0, 0])) else np.append(inside_points_test, True)
        inside_points = points[inside_points_test]
        # scaling radii for each point in cylindrical coordinates
        r = (neck_x ** 2 - (k * inside_points[:, 0]) ** 2) ** 0.5 if component == 'primary' else \
            (neck_x ** 2 - (k * (1 - inside_points[:, 0])) ** 2) ** 0.5

        length = np.linalg.norm(inside_points[:, 1:], axis=1)
        projected_points[inside_points_test, 0] = inside_points[:, 0]
        projected_points[inside_points_test, 1:] = r[:, None] * inside_points[:, 1:] / length[:, None]
        # if auxiliary point was used, than it will be appended to list of transformed points
        if np.array_equal(points[-1], np.array([neck_x, 0, 0])):
            projected_points[-1] = points[-1]

        triangulation = Delaunay(projected_points)
        triangles_indices = triangulation.convex_hull

        # removal of faces on top of the neck
        neck_test = ~(np.equal(points[triangles_indices][:, :, 0], neck_x).all(-1))
        new_triangles_indices = triangles_indices[neck_test]

        return new_triangles_indices

    def calculate_neck_position(self, return_polynomial=False):
        """
        function calculates x-coordinate of the `neck` (the narrowest place) of an over-contact system
        :return: np.float (0.1)
        """
        neck_position = None
        components_distance = 1.0
        components = ['primary', 'secondary']
        points_primary, points_secondary = [], []
        fn_map = {'primary': (self.potential_primary_fn, self.pre_calculate_for_potential_value_primary),
                  'secondary': (self.potential_secondary_fn, self.pre_calculate_for_potential_value_secondary)}

        # generating only part of the surface that I'm interested in (neck in xy plane for x between 0 and 1)
        angles = np.linspace(0., const.HALF_PI, 100, endpoint=True)
        for component in components:
            for angle in angles:
                args, use = (components_distance, angle, const.HALF_PI), False

                scipy_solver_init_value = np.array([components_distance / 10000.0])
                args = fn_map[component][1](*args)
                solution, _, ier, _ = scipy.optimize.fsolve(fn_map[component][0], scipy_solver_init_value,
                                                            full_output=True, args=args, xtol=1e-12)

                # check for regular solution
                if ier == 1 and not np.isnan(solution[0]):
                    solution = solution[0]
                    if 30 >= solution >= 0:
                        use = True
                else:
                    continue

                if use:
                    if component == 'primary':
                        points_primary.append([solution * np.cos(angle), solution * np.sin(angle)])
                    elif component == 'secondary':
                        points_secondary.append([- (solution * np.cos(angle) - components_distance),
                                                 solution * np.sin(angle)])

        neck_points = np.array(points_secondary + points_primary)
        # fitting of the neck with polynomial in order to find minimum
        polynomial_fit = np.polyfit(neck_points[:, 0], neck_points[:, 1], deg=15)
        polynomial_fit_differentiation = np.polyder(polynomial_fit)
        roots = np.roots(polynomial_fit_differentiation)
        roots = [np.real(xx) for xx in roots if np.imag(xx) == 0]
        # choosing root that is closest to the middle of the system, should work...
        # idea is to rule out roots near 0 or 1
        comparision_value = 1
        for root in roots:
            new_value = abs(0.5 - root)
            if new_value < comparision_value:
                comparision_value = new_value
                neck_position = root
        if return_polynomial:
            return neck_position, polynomial_fit
        else:
            return neck_position

    def build_mesh(self, component=None, components_distance=None):
        """
        build points of surface for primary or/and secondary component !!! w/o spots yet !!!

        :param component: str or empty
        :param components_distance: float
        :return:
        """
        if components_distance is None:
            raise ValueError('Argument `component_distance` was not supplied.')
        component = self._component_to_list(component)

        component_x_center = {'primary': 0.0, 'secondary': components_distance}
        for _component in component:
            component_instance = getattr(self, _component)
            # in case of spoted surface, symmetry is not used
            _a, _b, _c, _d = self.mesh_over_contact(component=_component, symmetry_output=True) \
                if self.morphology == 'over-contact' \
                else self.mesh_detached(
                component=_component, components_distance=components_distance, symmetry_output=True
            )
            component_instance.points = _a
            component_instance.point_symmetry_vector = _b
            component_instance.base_symmetry_points_number = _c
            component_instance.inverse_point_symmetry_matrix = _d

            component_instance = getattr(self, _component)
            self._evaluate_spots_mesh(components_distance=components_distance, component=_component)
            if self.morphology == 'over-contact':
                self._incorporate_spots_overcontact_mesh(component_instance=component_instance,
                                                         component_com=component_x_center[_component])
            else:
                self._incorporate_spots_mesh(component_instance=component_instance,
                                             component_com=component_x_center[_component])

    def _incorporate_spots_overcontact_mesh(self, component_instance=None, component_com=None):
        if not component_instance.spots:
            return
        self._logger.info("Incorporating spot points to component {} mesh".format(component_instance.name))

        if component_instance is None:
            raise ValueError('Object instance was not given.')

        if component_com is None:
            raise ValueError('Object centre of mass was not given.')

        vertices_map = [{"enum": -1} for _ in component_instance.points]
        # `all_component_points` do not contain points of any spot
        all_component_points = copy(component_instance.points)
        neck = np.min(all_component_points[:component_instance.base_symmetry_points_number, 0])

        for spot_index, spot in component_instance.spots.items():
            # average spacing in spot points
            vertices_to_remove, vertices_test = [], []

            cos_max_angle_point = np.cos(0.5 * spot.angular_diameter + 0.30 * spot.discretization_factor)

            spot_center = spot.boundary_center - np.array([component_com, 0., 0.])

            # removing star points in spot
            # for dist, ix in zip(distances, indices):
            for ix, pt in enumerate(all_component_points):
                surface_point = all_component_points[ix] - np.array([component_com, 0., 0.])
                cos_angle = np.inner(spot_center, surface_point) / (
                        np.linalg.norm(spot_center) * np.linalg.norm(surface_point)
                )

                if cos_angle < cos_max_angle_point or pt[0] == neck:
                    continue
                vertices_to_remove.append(ix)

            # simplices of target object for testing whether point lying inside or not of spot boundary, removing
            # duplicate points on the spot border
            # kedze vo vertice_map nie su body skvrny tak toto tu je zbytocne viac menej
            vertices_to_remove = list(set(vertices_to_remove))

            # points and vertices_map update
            if vertices_to_remove:
                _points, _vertices_map = list(), list()

                for ix, vertex in list(zip(range(0, len(all_component_points)), all_component_points)):
                    if ix in vertices_to_remove:
                        # skip point if is marked for removal
                        continue

                    # append only points of currrent object that do not intervent to spot
                    # [current, since there should be already spot from previous iteration step]
                    _points.append(vertex)
                    _vertices_map.append({"enum": vertices_map[ix]["enum"]})

                for vertex in spot.points:
                    _points.append(vertex)
                    _vertices_map.append({"enum": spot_index})

                all_component_points = copy(_points)
                vertices_map = copy(_vertices_map)

        separated_points = self.split_points_of_spots_and_component(all_component_points, vertices_map,
                                                                    component_instance)
        self.setup_component_instance_points(component_instance, separated_points)

    def build_faces(self, component=None, components_distance=None):
        """
        function creates faces of the star surface for given components provided you already calculated surface points
        of the component

        :type components_distance: float
        :param component: `primary` or `secondary` if not supplied both components are calculated
        :return:
        """
        if not components_distance:
            raise ValueError('components_distance value was not provided.')

        component = self._component_to_list(component)
        for _component in component:
            component_instance = getattr(self, _component)
            self.build_surface_with_spots(_component, components_distance=components_distance) \
                if component_instance.spots \
                else self.build_surface_with_no_spots(_component, components_distance=components_distance)

    def build_surface(self, components_distance=None, component=None, return_surface=False):
        """
        function for building of general binary star component surfaces including spots

        :param return_surface: bool - if true, function returns dictionary of arrays with all points and faces
                                      (surface + spots) for each component
        :param components_distance: distance between components
        :param component: specify component, use `primary` or `secondary`
        :return:
        """
        if not components_distance:
            raise ValueError('components_distance value was not provided.')

        component = self._component_to_list(component)
        ret_points, ret_faces = {}, {}

        for _component in component:
            component_instance = getattr(self, _component)

            # build mesh and incorporate spots points to given obtained object mesh
            self.build_mesh(component=_component, components_distance=components_distance)

            if not component_instance.spots:
                self.build_surface_with_no_spots(_component, components_distance=components_distance)
                if return_surface:
                    ret_points[_component] = copy(component_instance.points)
                    ret_faces[_component] = copy(component_instance.faces)
                continue
            else:
                self.build_surface_with_spots(_component, components_distance=components_distance)

            if return_surface:
                ret_points[_component], ret_faces[_component] = component_instance.return_whole_surface()

        return (ret_points, ret_faces) if return_surface else None

    def build_surface_with_no_spots(self, component=None, components_distance=None):
        """
        function for building binary star component surfaces without spots

        :param components_distance: float
        :param component:
        :return:
        """
        component = self._component_to_list(component)

        for _component in component:
            component_instance = getattr(self, _component)
            # triangulating only one quarter of the star

            if self.morphology != 'over-contact':
                points_to_triangulate = component_instance.points[:component_instance.base_symmetry_points_number, :]
                triangles = self.detached_system_surface(component=_component, points=points_to_triangulate,
                                                         components_distance=components_distance)

            else:
                neck = np.max(component_instance.points[:, 0]) if component[0] == 'primary' \
                    else np.min(component_instance.points[:, 0])
                points_to_triangulate = \
                    np.append(component_instance.points[:component_instance.base_symmetry_points_number, :],
                              np.array([[neck, 0, 0]]), axis=0)
                triangles = self.over_contact_surface(component=_component, points=points_to_triangulate)
                # filtering out triangles containing last point in `points_to_triangulate`
                triangles = triangles[(triangles < component_instance.base_symmetry_points_number).all(1)]

            # filtering out faces on xy an xz planes
            y0_test = ~np.isclose(points_to_triangulate[triangles][:, :, 1], 0).all(1)
            z0_test = ~np.isclose(points_to_triangulate[triangles][:, :, 2], 0).all(1)
            triangles = triangles[np.logical_and(y0_test, z0_test)]

            component_instance.base_symmetry_faces_number = np.int(np.shape(triangles)[0])
            # lets exploit axial symmetry and fill the rest of the surface of the star
            all_triangles = [inv[triangles] for inv in component_instance.inverse_point_symmetry_matrix]
            component_instance.base_symmetry_faces = triangles
            component_instance.faces = np.concatenate(all_triangles, axis=0)

            base_face_symmetry_vector = np.arange(component_instance.base_symmetry_faces_number)
            component_instance.face_symmetry_vector = np.concatenate([base_face_symmetry_vector for _ in range(4)])

    def _get_surface_builder_fn(self):
        """
        returns suitable triangulation function depending on morphology
        :return: function instance that performs generation surface faces
        """
        return self.over_contact_surface if self.morphology == "over-contact" else self.detached_system_surface

    def build_surface_with_spots(self, component=None, components_distance=None):
        """
        function capable of triangulation of spotty stellar surfaces, it merges all surface points, triangulates them
        and then sorts the resulting surface faces under star or spot
        :param components_distance: float
        :param component: str `primary` or `secondary`
        :return:
        """
        component = self._component_to_list(component)
        component_com = {'primary': 0.0, 'secondary': components_distance}
        for _component in component:
            component_instance = getattr(self, _component)
            points, vertices_map = self._return_all_points(component_instance, return_vertices_map=True)

            surface_fn = self._get_surface_builder_fn()
            faces = surface_fn(component=_component, points=points, components_distance=components_distance)
            model, spot_candidates = self._initialize_model_container(vertices_map)
            model = self._split_spots_and_component_faces(
                points, faces, model, spot_candidates, vertices_map, component_instance,
                component_com=component_com[_component]
            )
            self._remove_overlaped_spots(vertices_map, component_instance)
            self._remap_surface_elements(model, component_instance, points)

    @staticmethod
    def _component_to_list(component):
        """
        converts component name string into list

        :param component: if None, `['primary', 'secondary']` will be returned
                          otherwise `primary` and `secondary` will be converted into lists [`primary`] and [`secondary`]
        :return:
        """
        if not component:
            component = ['primary', 'secondary']
        elif component in ['primary', 'secondary']:
            component = [component]
        else:
            raise ValueError('Invalid name of the component. Use `primary` or `secondary`.')
        return component

    def plot(self, descriptor=None, **kwargs):
        """
        universal plot interface for binary system class, more detailed documentation for each value of descriptor is
        available in graphics library

        :param descriptor: str (defines type of plot):
                            orbit - plots orbit in orbital plane
                            equipotential - plots crossections of surface Hill planes in xy,yz,zx planes
                            mesh - plot surface points
                            surface - plot stellar surfaces
        :param kwargs: dict (depends on descriptor value, see individual functions in graphics.py)
        :return:
        """

        if descriptor == 'orbit':
            KWARGS = ['start_phase', 'stop_phase', 'number_of_points', 'axis_unit', 'frame_of_reference']
            utils.invalid_kwarg_checker(kwargs, KWARGS, BinarySystem.plot)

            method_to_call = graphics.orbit
            start_phase = kwargs.get('start_phase', 0.0)
            stop_phase = kwargs.get('stop_phase', 1.0)
            number_of_points = kwargs.get('number_of_points', 300)

            kwargs['axis_unit'] = kwargs.get('axis_unit', u.solRad)
            kwargs['frame_of_reference'] = kwargs.get('frame_of_reference', 'primary_component')

            if kwargs['axis_unit'] == 'dimensionless':
                kwargs['axis_unit'] = u.dimensionless_unscaled

            # orbit calculation for given phases
            phases = np.linspace(start_phase, stop_phase, number_of_points)
            ellipse = self.orbit.orbital_motion(phase=phases)
            # if axis are without unit a = 1
            if kwargs['axis_unit'] != u.dimensionless_unscaled:
                a = self._semi_major_axis * units.DISTANCE_UNIT.to(kwargs['axis_unit'])
                radius = a * ellipse[:, 0]
            else:
                radius = ellipse[:, 0]
            azimuth = ellipse[:, 1]
            x, y = utils.polar_to_cartesian(radius=radius, phi=azimuth - const.PI / 2.0)
            if kwargs['frame_of_reference'] == 'barycentric':
                kwargs['x1_data'] = - self.mass_ratio * x / (1 + self.mass_ratio)
                kwargs['y1_data'] = - self.mass_ratio * y / (1 + self.mass_ratio)
                kwargs['x2_data'] = x / (1 + self.mass_ratio)
                kwargs['y2_data'] = y / (1 + self.mass_ratio)
            elif kwargs['frame_of_reference'] == 'primary_component':
                kwargs['x_data'], kwargs['y_data'] = x, y

        elif descriptor == 'equipotential':
            KWARGS = ['plane', 'phase']
            utils.invalid_kwarg_checker(kwargs, KWARGS, BinarySystem.plot)

            method_to_call = graphics.equipotential

            kwargs['phase'] = kwargs.get('phase', 0.0)
            kwargs['plane'] = kwargs.get('plane', 'xy')

            # relative distance between components (a = 1)
            if utils.is_plane(kwargs['plane'], 'xy') or utils.is_plane(
                    kwargs['plane'], 'yz') or utils.is_plane(kwargs['plane'], 'zx'):
                components_distance = self.orbit.orbital_motion(phase=kwargs['phase'])[0][0]
                points_primary, points_secondary = \
                    self.compute_equipotential_boundary(components_distance=components_distance, plane=kwargs['plane'])
            else:
                raise ValueError('Invalid choice of crossection plane, use only: `xy`, `yz`, `zx`.')

            kwargs['points_primary'] = points_primary
            kwargs['points_secondary'] = points_secondary

        elif descriptor == 'mesh':
            KWARGS = ['phase', 'components_to_plot', 'plot_axis', 'inclination', 'azimuth']
            utils.invalid_kwarg_checker(kwargs, KWARGS, BinarySystem.plot)
            method_to_call = graphics.binary_mesh

            kwargs['phase'] = kwargs.get('phase', 0)
            kwargs['components_to_plot'] = kwargs.get('components_to_plot', 'both')
            kwargs['plot_axis'] = kwargs.get('plot_axis', True)
            kwargs['inclination'] = kwargs.get('inclination', np.degrees(self.inclination))

            components_distance, azim = self.orbit.orbital_motion(phase=kwargs['phase'])[0][:2]
            kwargs['azimuth'] = kwargs.get('azimuth', np.degrees(azim) - 90)

            if kwargs['components_to_plot'] in ['primary', 'both']:
                points, _ = self.build_surface(component='primary', components_distance=components_distance,
                                               return_surface=True)
                kwargs['points_primary'] = points['primary']

            if kwargs['components_to_plot'] in ['secondary', 'both']:
                points, _ = self.build_surface(component='secondary', components_distance=components_distance,
                                               return_surface=True)
                kwargs['points_secondary'] = points['secondary']

        elif descriptor == 'wireframe':
            KWARGS = ['phase', 'components_to_plot', 'plot_axis', 'inclination', 'azimuth']
            utils.invalid_kwarg_checker(kwargs, KWARGS, BinarySystem.plot)
            method_to_call = graphics.binary_wireframe

            kwargs['phase'] = kwargs.get('phase', 0)
            kwargs['components_to_plot'] = kwargs.get('components_to_plot', 'both')
            kwargs['plot_axis'] = kwargs.get('plot_axis', True)
            kwargs['inclination'] = kwargs.get('inclination', np.degrees(self.inclination))

            components_distance, azim = self.orbit.orbital_motion(phase=kwargs['phase'])[0][:2]
            kwargs['azimuth'] = kwargs.get('azimuth', np.degrees(azim) - 90)

            if kwargs['components_to_plot'] in ['primary', 'both']:
                points, faces = self.build_surface(component='primary', components_distance=components_distance,
                                                   return_surface=True)
                kwargs['points_primary'] = points['primary']
                kwargs['primary_triangles'] = faces['primary']
            if kwargs['components_to_plot'] in ['secondary', 'both']:
                points, faces = self.build_surface(component='secondary', components_distance=components_distance,
                                                   return_surface=True)
                kwargs['points_secondary'] = points['secondary']
                kwargs['secondary_triangles'] = faces['secondary']

        elif descriptor == 'surface':
            KWARGS = ['phase', 'components_to_plot', 'normals', 'edges', 'colormap', 'plot_axis', 'face_mask_primary',
                      'face_mask_secondary', 'inclination', 'azimuth']
            utils.invalid_kwarg_checker(kwargs, KWARGS, BinarySystem.plot)

            method_to_call = graphics.binary_surface

            kwargs['phase'] = kwargs.get('phase', 0)
            kwargs['components_to_plot'] = kwargs.get('components_to_plot', 'both')
            kwargs['normals'] = kwargs.get('normals', False)
            kwargs['edges'] = kwargs.get('edges', False)
            kwargs['colormap'] = kwargs.get('colormap', None)
            kwargs['plot_axis'] = kwargs.get('plot_axis', True)
            kwargs['face_mask_primary'] = kwargs.get('face_mask_primary', None)
            kwargs['face_mask_secondary'] = kwargs.get('face_mask_secondary', None)
            kwargs['inclination'] = kwargs.get('inclination', np.degrees(self.inclination))

            components_distance, azim = self.orbit.orbital_motion(phase=kwargs['phase'])[0][:2]
            kwargs['azimuth'] = kwargs.get('azimuth', np.degrees(azim) - 90)

            # this part decides if both components need to be calculated at once (due to reflection effect)
            # if kwargs['colormap'] == 'temperature' and self.reflection_effect_iterations != 0:
            if kwargs['colormap'] == 'temperature':
                points, faces = self.build_surface(components_distance=components_distance,
                                                   return_surface=True)
                kwargs['points_primary'] = points['primary']
                kwargs['primary_triangles'] = faces['primary']
                kwargs['points_secondary'] = points['secondary']
                kwargs['secondary_triangles'] = faces['secondary']

                cmap = self.build_surface_map(colormap=kwargs['colormap'],
                                              components_distance=components_distance, return_map=True)
                kwargs['primary_cmap'] = cmap['primary']
                kwargs['secondary_cmap'] = cmap['secondary']

                if kwargs['face_mask_primary'] is not None:
                    kwargs['primary_triangles'] = kwargs['primary_triangles'][kwargs['face_mask_primary']]
                    kwargs['primary_cmap'] = kwargs['primary_cmap'][kwargs['face_mask_primary']]
                if kwargs['face_mask_secondary'] is not None:
                    kwargs['secondary_triangles'] = kwargs['secondary_triangles'][kwargs['face_mask_secondary']]
                    kwargs['secondary_cmap'] = kwargs['secondary_cmap'][kwargs['face_mask_secondary']]

                if kwargs['normals']:
                    kwargs['primary_centres'] = self.primary.calculate_surface_centres(
                        kwargs['points_primary'], kwargs['primary_triangles'])
                    kwargs['primary_arrows'] = self.primary.calculate_normals(
                        kwargs['points_primary'], kwargs['primary_triangles'], com=0)
                    kwargs['secondary_centres'] = self.secondary.calculate_surface_centres(
                        kwargs['points_secondary'], kwargs['secondary_triangles'])
                    kwargs['secondary_arrows'] = self.secondary.calculate_normals(
                        kwargs['points_secondary'], kwargs['secondary_triangles'], com=components_distance)
            else:
                if kwargs['components_to_plot'] in ['primary', 'both']:
                    points, faces = self.build_surface(component='primary', components_distance=components_distance,
                                                       return_surface=True)
                    kwargs['points_primary'] = points['primary']
                    kwargs['primary_triangles'] = faces['primary']

                    if kwargs['colormap']:
                        cmap = self.build_surface_map(colormap=kwargs['colormap'], component='primary',
                                                      components_distance=components_distance, return_map=True)
                        kwargs['primary_cmap'] = cmap['primary']

                    if kwargs['normals']:
                        kwargs['primary_centres'] = self.primary.calculate_surface_centres(
                            kwargs['points_primary'], kwargs['primary_triangles'])
                        kwargs['primary_arrows'] = self.primary.calculate_normals(
                            kwargs['points_primary'], kwargs['primary_triangles'], com=0)

                    if kwargs['face_mask_primary'] is not None:
                        kwargs['primary_triangles'] = kwargs['primary_triangles'][kwargs['face_mask_primary']]
                        kwargs['primary_cmap'] = kwargs['primary_cmap'][kwargs['face_mask_primary']]

                if kwargs['components_to_plot'] in ['secondary', 'both']:
                    points, faces = self.build_surface(component='secondary', components_distance=components_distance,
                                                       return_surface=True)
                    kwargs['points_secondary'] = points['secondary']
                    kwargs['secondary_triangles'] = faces['secondary']

                    if kwargs['colormap']:
                        cmap = self.build_surface_map(colormap=kwargs['colormap'], component='secondary',
                                                      components_distance=components_distance, return_map=True)
                        kwargs['secondary_cmap'] = cmap['secondary']

                    if kwargs['normals']:
                        kwargs['secondary_centres'] = self.secondary.calculate_surface_centres(
                            kwargs['points_secondary'], kwargs['secondary_triangles'])
                        kwargs['secondary_arrows'] = self.secondary.calculate_normals(
                            kwargs['points_secondary'], kwargs['secondary_triangles'], com=components_distance)

                    if kwargs['face_mask_secondary'] is not None:
                        kwargs['secondary_triangles'] = kwargs['secondary_triangles'][kwargs['face_mask_secondary']]
                        kwargs['secondary_cmap'] = kwargs['secondary_cmap'][kwargs['face_mask_secondary']]

        else:
            raise ValueError("Incorrect descriptor `{}`".format(descriptor))

        method_to_call(**kwargs)

    def build_surface_map(self, colormap=None, component=None, components_distance=None, return_map=False):
        """
        function calculates surface maps (temperature or gravity acceleration) for star and spot faces and it can return
        them as one array if return_map=True

        :param return_map: if True function returns arrays with surface map including star and spot segments
        :param colormap: switch for `temperature` or `gravity` colormap to create
        :param component: `primary` or `secondary` component surface map to calculate, if not supplied
        :param components_distance: distance between components
        :return:
        """
        if colormap is None:
            raise ValueError('Specify colormap to calculate (`temperature` or `gravity_acceleration`).')
        if components_distance is None:
            raise ValueError('Component distance value was not supplied.')

        component = self._component_to_list(component)

        for _component in component:
            component_instance = getattr(self, _component)

            # compute and assign surface areas of elements if missing
            self._logger.debug('Computing surface areas of {} elements.'.format(_component))
            component_instance.areas = component_instance.calculate_areas()

            # compute and assign polar radius if missing
            self._logger.debug('Computing polar radius of {} component.'.format(_component))
            component_instance._polar_radius = self.calculate_polar_radius(
                component=_component, components_distance=components_distance)

            # compute and assign potential gradient magnitudes for elements if missing
            self._logger.debug('Computing potential gradient magnitudes distribution of {} component.'
                               ''.format(_component))
            component_instance.potential_gradient_magnitudes = self.calculate_face_magnitude_gradient(
                component=_component, components_distance=components_distance)

            self._logger.debug('Computing magnitude of {} polar potential gradient.'.format(_component))
            component_instance.polar_potential_gradient_magnitude = \
                self.calculate_polar_potential_gradient_magnitude(
                    component=_component, components_distance=components_distance)

            # compute and assign temperature of elements
            if colormap == 'temperature':
                self._logger.debug('Computing effective temprature distibution of {} component.'.format(_component))
                component_instance.temperatures = component_instance.calculate_effective_temperatures()
                if component_instance.pulsations:
                    self._logger.debug('Adding pulsations to surface temperature distribution '
                                       'of the {} component.'.format(_component))
                    component_instance.temperatures = component_instance.add_pulsations()

            if component_instance.spots:
                for spot_index, spot in component_instance.spots.items():
                    self._logger.debug('Calculating surface areas of {} component / {} spot.'.format(_component,
                                                                                                     spot_index))
                    spot.areas = spot.calculate_areas()

                    self._logger.debug('Calculating distribution of potential gradient magnitudes of {} component / '
                                       '{} spot.'.format(_component, spot_index))
                    spot.potential_gradient_magnitudes = self.calculate_face_magnitude_gradient(
                        component=_component,
                        components_distance=components_distance,
                        points=spot.points, faces=spot.faces)

                    if colormap == 'temperature':
                        self._logger.debug('Computing temperature distribution of {} component / {} spot'
                                           ''.format(_component, spot_index))
                        spot.temperatures = spot.temperature_factor * \
                                            component_instance.calculate_effective_temperatures(
                                                gradient_magnitudes=spot.potential_gradient_magnitudes)
                        if component_instance.pulsations:
                            self._logger.debug('Adding pulsations to temperature distribution of {} component / {} spot'
                                               ''.format(_component, spot_index))
                            spot.temperatures = component_instance.add_pulsations(points=spot.points, faces=spot.faces,
                                                                                  temperatures=spot.temperatures)

                if colormap == 'temperature':
                    self._logger.debug('Renormalizing temperature map of {0} component due to presence of spots'
                                       ''.format(component))
                    component_instance.renormalize_temperatures()

        # implementation of reflection effect
        if colormap == 'temperature':
            if len(component) == 2:
                com = {'primary': 0, 'secondary': components_distance}
                for _component in component:
                    component_instance = getattr(self, _component)
                    component_instance.set_all_surface_centres()
                    component_instance.set_all_normals(com=com[_component])

                self.reflection_effect(iterations=config.REFLECTION_EFFECT_ITERATIONS,
                                       components_distance=components_distance)
            else:
                self._logger.debug('Reflection effect can be calculated only when surface map of both components is '
                                   'calculated. Skipping calculation of reflection effect.')

        if return_map:
            return_map = {}
            for _component in component:
                component_instance = getattr(self, _component)
                if colormap == 'gravity_acceleration':
                    return_map[_component] = copy(component_instance.potential_gradient_magnitudes)
                elif colormap == 'temperature':
                    return_map[_component] = copy(component_instance.temperatures)

                if component_instance.spots:
                    for spot_index, spot in component_instance.spots.items():
                        if colormap == 'gravity_acceleration':
                            return_map[_component] = np.append(return_map[_component],
                                                               spot.potential_gradient_magnitudes)
                        elif colormap == 'temperature':
                            return_map[_component] = np.append(return_map[_component], spot.temperatures)
            return return_map
        return

    @classmethod
    def is_property(cls, kwargs):
        """
        method for checking if keyword arguments are valid properties of this class

        :param kwargs: dict
        :return:
        """
        is_not = ['`{}`'.format(k) for k in kwargs if k not in cls.ALL_KWARGS]
        if is_not:
            raise AttributeError('Arguments {} are not valid {} properties.'.format(', '.join(is_not),
                                                                                    cls.__name__))

    def faces_visibility_x_limits(self, components_distance):
        # this section calculates the visibility of each surface face
        # don't forget to treat self visibility of faces on the same star in over-contact system

        # if stars are too close and with too different radii, you can see more (less) than a half of the stellare
        # surface, calculating excess angle

        sin_theta = np.abs(self.primary.polar_radius - self.secondary.polar_radius) / components_distance
        x_corr_primary = self.primary.polar_radius * sin_theta
        x_corr_secondary = self.secondary.polar_radius * sin_theta

        # visibility of faces is given by their x position
        xlim = {}
        (xlim['primary'], xlim['secondary']) = (x_corr_primary, 1 + x_corr_secondary) \
            if self.primary.polar_radius > self.secondary.polar_radius else (- x_corr_primary, 1 - x_corr_secondary)
        return xlim

    # noinspection PyTypeChecker
    def reflection_effect(self, iterations=None, components_distance=None):
        if not config.REFLECTION_EFFECT:
            self._logger.debug('Reflection effect is switched off.')
            return
        if iterations is None:
            raise ValueError('Number of iterations for reflection effect was not specified.')
        elif iterations == 0:
            self._logger.debug('Number of reflections in reflection effect was set to zero. Reflection effect will '
                               'not be calculated.')
            return

        if components_distance is None:
            raise ValueError('Components distance was not supplied.')

        component = self._component_to_list(None)

        xlim = self.faces_visibility_x_limits(components_distance=components_distance)

        # this tests if you can use surface symmetries
        use_quarter_star_test = self.primary.spots is None and self.secondary.spots is None
        vis_test_symmetry = {}

        # declaring variables
        centres, vis_test, gamma, normals, faces, points, temperatures, areas = {}, {}, {}, {}, {}, {}, {}, {}
        # centres - dict with all centres concatenated (star and spot) into one matrix for convenience
        # vis_test - dict with bool map for centres to select only faces visible from any face on companion
        # companion
        # gamma is of dimensions num_of_visible_faces_primary x num_of_visible_faces_secondary

        # selecting faces that have a chance to be visible from other component
        for _component in component:
            component_instance = getattr(self, _component)

            points[_component], faces[_component], centres[_component], normals[_component], temperatures[_component], \
            areas[_component] = self.init_surface_variables(component_instance)

            # test for visibility of star faces
            vis_test[_component], vis_test_symmetry[_component] = \
                self.get_visibility_tests(centres[_component], use_quarter_star_test, xlim[_component], _component)

            if component_instance.spots:
                # including spots into overall surface
                for spot_index, spot in component_instance.spots.items():
                    vis_test_spot = self.visibility_test(spot.face_centres, xlim[_component], _component)

                    # merge surface and spot face parameters into one variable
                    centres[_component], normals[_component], temperatures[_component], areas[_component], \
                    vis_test[_component] = \
                        self.include_spot_to_surface_variables(centres[_component], spot.face_centres,
                                                               normals[_component], spot.normals,
                                                               temperatures[_component], spot.temperatures,
                                                               areas[_component], spot.areas,
                                                               vis_test[_component], vis_test_spot)

        # calculating C_A = (albedo_A / D_intB) - scalar
        # D_intB - bolometric limb darkening factor
        d_int = {
            'primary': ld.calculate_bolometric_limb_darkening_factor(config.LIMB_DARKENING_LAW, self.LD_COEFF),
            'secondary': ld.calculate_bolometric_limb_darkening_factor(config.LIMB_DARKENING_LAW, self.LD_COEFF)
        }

        _c = {
            'primary': (self.primary.albedo / d_int['secondary']),
            'secondary': (self.secondary.albedo / d_int['primary'])
        }

        # setting reflection factor R = 1 + F_irradiated / F_original, initially equal to one everywhere - vector
        reflection_factor = {
            _component: np.ones((np.sum(vis_test[_component]),), dtype=np.float) for _component in component
        }

        counterpart = {'primary': 'secondary', 'secondary': 'primary'}

        # for faster convergence, reflection effect is calculated first on cooler component
        components = ['primary', 'secondary'] if self.primary.t_eff <= self.secondary.t_eff else \
            ['secondary', 'primary']

        if use_quarter_star_test:
            # calculating distances and distance vectors between, join vector is already normalized
            _shape, _shape_reduced = self.get_distance_matrix_shape(vis_test)

            distance, join_vector = self.get_symmetrical_distance_matrix(_shape, _shape_reduced, centres, vis_test,
                                                                         vis_test_symmetry)

            # calculating cos of angle gamma between face normal and join vector
            # initialising gammma matrices
            gamma = self.get_symmetrical_gammma(_shape[:2], _shape_reduced, normals, join_vector, vis_test,
                                                vis_test_symmetry)

            # testing mutual visibility of faces by assigning 0 to non visible face combination
            self.check_symmetric_gamma_for_negative_num(gamma, _shape_reduced)

            # calculating QAB = (cos gamma_a)*cos(gamma_b)/d**2
            q_ab = self.get_symmetrical_q_ab(_shape[:2], _shape_reduced, gamma, distance)

            # calculating limb darkening factor for each combination of surface faces
            d_gamma = self.get_symmetrical_d_gamma(_shape[:2], _shape_reduced, normals, join_vector, vis_test)

            # calculating limb darkening factors for each combination of faces shape
            # (N_faces_primary * N_faces_secondary)
            # precalculating matrix part of reflection effect correction
            matrix_to_sum2 = {
                'primary': q_ab[:_shape_reduced[0], :] * d_gamma['secondary'][:_shape_reduced[0], :],
                'secondary': q_ab[:, :_shape_reduced[1]] * d_gamma['primary'][:, :_shape_reduced[1]]
            }
            symmetry_to_use = {'primary': _shape_reduced[0], 'secondary': _shape_reduced[1]}
            for _ in range(iterations):
                for _component in components:
                    component_instance = getattr(self, _component)
                    counterpart = 'primary' if _component == 'secondary' else 'secondary'

                    # calculation of reflection effect correction as
                    # 1 + (c / t_effi) * sum_j(r_j * Q_ab * t_effj^4 * D(gamma_j) * areas_j)
                    # calculating vector part of reflection effect correction
                    vector_to_sum1 = reflection_factor[counterpart] * \
                                     np.power(temperatures[counterpart][vis_test[counterpart]], 4) * \
                                     areas[counterpart][vis_test[counterpart]]
                    counterpart_to_sum = np.matmul(vector_to_sum1, matrix_to_sum2['secondary']) \
                        if _component == 'secondary' else np.matmul(matrix_to_sum2['primary'], vector_to_sum1)
                    reflection_factor[_component][:symmetry_to_use[_component]] = \
                        1 + (_c[_component] / np.power(temperatures[_component][vis_test_symmetry[_component]], 4)) * \
                        counterpart_to_sum

                    # using symmetry to redistribute reflection factor R
                    refl_fact_aux = np.empty(shape=np.shape(temperatures[_component]))
                    refl_fact_aux[vis_test_symmetry[_component]] = \
                        reflection_factor[_component][:symmetry_to_use[_component]]
                    refl_fact_aux = refl_fact_aux[component_instance.face_symmetry_vector]
                    reflection_factor[_component] = refl_fact_aux[vis_test[_component]]

            for _component in components:
                component_instance = getattr(self, _component)
                # assigning new temperatures according to last iteration as
                # teff_new = teff_old * reflection_factor^0.25
                temperatures[_component][vis_test_symmetry[_component]] = \
                    temperatures[_component][vis_test_symmetry[_component]] * \
                    np.power(reflection_factor[_component][:symmetry_to_use[_component]], 0.25)
                temperatures[_component] = temperatures[_component][component_instance.face_symmetry_vector]

        else:
            # calculating distances and distance vectors between, join vector is already normalized
            distance, join_vector = utils.calculate_distance_matrix(points1=centres['primary'][vis_test['primary']],
                                                                    points2=centres['secondary'][vis_test['secondary']],
                                                                    return_join_vector_matrix=True)

            # calculating cos of angle gamma between face normal and join vector
            gamma = {'primary':
                         np.sum(np.multiply(normals['primary'][vis_test['primary']][:, None, :], join_vector), axis=2),
                     'secondary':
                         -np.sum(np.multiply(normals['secondary'][vis_test['secondary']][None, :, :], join_vector),
                                 axis=2)}
            # negative sign is there because of reversed distance vector used for secondary component

            # testing mutual visibility of faces by assigning 0 to non visible face combination
            gamma['primary'][gamma['primary'] < 0] = 0.
            gamma['secondary'][gamma['secondary'] < 0] = 0.

            # calculating QAB = (cos gamma_a)*cos(gamma_b)/d**2
            q_ab = np.divide(np.multiply(gamma['primary'], gamma['secondary']), np.power(distance, 2))

            # calculating limb darkening factors for each combination of faces shape
            # (N_faces_primary * N_faces_secondary)
            d_gamma = \
                {'primary': ld.limb_darkening_factor(normal_vector=normals['primary'][vis_test['primary'], None, :],
                                                     line_of_sight=join_vector,
                                                     coefficients=self.LD_COEFF,
                                                     limb_darkening_law=config.LIMB_DARKENING_LAW),
                 'secondary': ld.limb_darkening_factor(normal_vector=normals['secondary'][None, vis_test['secondary'], :
                                                                     ],
                                                       line_of_sight=-join_vector,
                                                       coefficients=self.LD_COEFF,
                                                       limb_darkening_law=config.LIMB_DARKENING_LAW)
                 }

            # precalculating matrix part of reflection effect correction
            matrix_to_sum2 = {_component: q_ab * d_gamma[counterpart[_component]] for _component in component}
            for _ in range(iterations):
                for _component in components:
                    counterpart = 'primary' if _component == 'secondary' else 'secondary'

                    # calculation of reflection effect correction as
                    # 1 + (c / t_effi) * sum_j(r_j * Q_ab * t_effj^4 * D(gamma_j) * areas_j)
                    # calculating vector part of reflection effect correction
                    vector_to_sum1 = reflection_factor[counterpart] * np.power(
                        temperatures[counterpart][vis_test[counterpart]], 4) * areas[counterpart][vis_test[counterpart]]
                    counterpart_to_sum = np.matmul(vector_to_sum1, matrix_to_sum2['secondary']) \
                        if _component == 'secondary' else np.matmul(matrix_to_sum2['primary'], vector_to_sum1)
                    reflection_factor[_component] = 1 + (_c[_component] / np.power(
                        temperatures[_component][vis_test[_component]], 4)) \
                                                    * counterpart_to_sum

            for _component in components:
                # assigning new temperatures according to last iteration as
                # teff_new = teff_old * reflection_factor^0.25
                temperatures[_component][vis_test[_component]] = \
                    temperatures[_component][vis_test[_component]] * np.power(reflection_factor[_component], 0.25)

        # redistributing temperatures back to the parent objects
        self.redistribute_temperatures(temperatures)

    def build_log_of_cgs_surface_gravity(self, component=None, components_distance=None):
        # todo: consider to put this function into build surface gravity
        # what it is for
        if components_distance is None:
            raise ValueError('Component distance value was not supplied.')

        component = self._component_to_list(component)
        for _component in component:
            self._logger.debug('Computing surface cgs unit gravity of {} elements.'.format(_component))
            component_instance = getattr(self, _component)
            cgs_polar_gravity = self.polar_gravity_acceleration(_component, components_distance)

            polar_gradient_magnitude = self.calculate_polar_potential_gradient_magnitude(_component,
                                                                                         components_distance)
            gravity_scalling_factor = cgs_polar_gravity / polar_gradient_magnitude

            component_instance._log_g = np.log10(
                gravity_scalling_factor * component_instance.potential_gradient_magnitudes)

            if component_instance.spots:
                for spot_index, spot in component_instance.spots.items():
                    self._logger.debug('Calculating surface cgs unit gravity of {} component / {} spot.'
                                       ''.format(_component, spot_index))
                    self._logger.debug('Calculating distribution of potential gradient magnitudes of {} component / '
                                       '{} spot.'.format(_component, spot_index))

                    spot.log_g = np.log10(gravity_scalling_factor * spot.potential_gradient_magnitudes)

    @staticmethod
    def init_surface_variables(component_instance):
        """
        function copies basic parameters of the stellar surface (points, faces, normals, temperatures and areas) of
        given star instance into new arrays during calculation of reflection effect

        :param component_instance:
        :return:
        """
        points, faces = component_instance.return_whole_surface()
        centres = copy(component_instance.face_centres)
        normals = copy(component_instance.normals)
        temperatures = copy(component_instance.temperatures)
        areas = copy(component_instance.areas)
        return points, faces, centres, normals, temperatures, areas

    def get_visibility_tests(self, centres, q_test, xlim, component):
        """
        function calculates tests for visibilities of faces from other component, used in reflection effect

        :param centres: np.array of face centres
        :param q_test: use_quarter_star_test
        :param xlim: visibility threshold in x axis for given component
        :param component: `primary` or `secondary`
        :return: visual tests for normal and symmetrical star
        """
        if q_test:
            y_test, z_test = centres[:, 1] > 0, centres[:, 2] > 0
            # this branch is activated in case of clean surface where symmetries can be used
            # excluding quadrants that can be mirrored using symmetries
            quadrant_exclusion = np.logical_or(y_test, z_test) \
                if self.morphology == 'over-contfact' \
                else np.array([True] * len(centres))

            single_quadrant = np.logical_and(y_test, z_test)
            # excluding faces on far sides of components
            test1 = self.visibility_test(centres, xlim, component)
            # this variable contains faces that can seen from base symmetry part of the other star
            vis_test = np.logical_and(test1, quadrant_exclusion)
            vis_test_symmetry = np.logical_and(test1, single_quadrant)

        else:
            vis_test = centres[:, 0] >= xlim if \
                component == 'primary' else centres[:, 0] <= xlim
            vis_test_symmetry = None

        return vis_test, vis_test_symmetry

    @staticmethod
    def visibility_test(centres, xlim, component):
        """
        tests if given faces are visible from the other star

        :param centres:
        :param xlim: visibility threshold in x axis for given component
        :return:
        """
        return centres[:, 0] >= xlim if component == 'primary' else centres[:, 0] <= xlim

    @staticmethod
    def include_spot_to_surface_variables(centres, spot_centres, normals, spot_normals,
                                          temperatures, spot_temperatures, areas, spot_areas, vis_test, vis_test_spot):
        """
        function includes surface parameters of spot faces into global arrays containing parameters from whole surface
        used in reflection effect

        :param centres:
        :param spot_centres: spot centres to append to `centres`
        :param normals:
        :param spot_normals: spot normals to append to `normals`
        :param temperatures:
        :param spot_temperatures: spot temperatures to append to `temperatures`
        :param areas:
        :param spot_areas: spot areas to append to `areas`
        :param vis_test:
        :param vis_test_spot: spot visibility test to append to `vis_test`
        :return:
        """
        centres = np.append(centres, spot_centres, axis=0)
        normals = np.append(normals, spot_normals, axis=0)
        temperatures = np.append(temperatures, spot_temperatures, axis=0)
        areas = np.append(areas, spot_areas, axis=0)
        vis_test = np.append(vis_test, vis_test_spot, axis=0)

        return centres, normals, temperatures, areas, vis_test

    def get_distance_matrix_shape(self, vis_test):
        """
        calculates shapes of distance and join vector matrices along with shapes of symetrical parts of those matrices
        used in reflection effect

        :param vis_test:
        :return:
        """
        shape = (np.sum(vis_test['primary']), np.sum(vis_test['secondary']), 3)
        shape_reduced = (np.sum(vis_test['primary'][:self.primary.base_symmetry_faces_number]),
                         np.sum(vis_test['secondary'][:self.secondary.base_symmetry_faces_number]))
        return shape, shape_reduced

    @staticmethod
    def get_symmetrical_distance_matrix(shape, shape_reduced, centres, vis_test, vis_test_symmetry):
        """
        function uses symmetries of the stellar component in order to reduce time in calculation distance matrix
        :param shape: desired shape of join vector matrix
        :param shape_reduced: shape of the surface symmetries, (faces above those indices are symmetrical to the ones
        below)
        :param centres:
        :param vis_test:
        :param vis_test_symmetry:
        :return: distance - distance matrix
                 join vector - matrix of unit vectors pointing between each two faces on opposite stars
        """
        distance = np.empty(shape=shape[:2], dtype=np.float)
        join_vector = np.empty(shape=shape, dtype=np.float)

        # in case of symmetries, you need to calculate only minority part of distance matrix connected with base
        # symmetry part of the both surfaces
        distance[:shape_reduced[0], :], join_vector[:shape_reduced[0], :, :] = \
            utils.calculate_distance_matrix(points1=centres['primary'][vis_test_symmetry['primary']],
                                            points2=centres['secondary'][vis_test['secondary']],
                                            return_join_vector_matrix=True)

        aux = centres['primary'][vis_test['primary']]
        distance[shape_reduced[0]:, :shape_reduced[1]], join_vector[shape_reduced[0]:, :shape_reduced[1], :] = \
            utils.calculate_distance_matrix(points1=aux[shape_reduced[0]:],
                                            points2=centres['secondary'][vis_test_symmetry['secondary']],
                                            return_join_vector_matrix=True)
        return distance, join_vector

    @staticmethod
    def get_symmetrical_gammma(shape, shape_reduced, normals, join_vector, vis_test, vis_test_symmetry):
        """
        function uses surface symmetries to calculate cosine of angles between join vector and surface normals

        :param shape: desired shape of gamma
        :param shape_reduced: shape of the surface symmetries, (faces above those indices are symmetrical to the ones
        below)
        :param normals:
        :param join_vector:
        :param vis_test:
        :param vis_test_symmetry:
        :return: gamma - cos(angle(normal, join_vector))
        """
        gamma = {'primary': np.empty(shape=shape, dtype=np.float),
                 'secondary': np.empty(shape=shape, dtype=np.float)}

        # calculating only necessary components of the matrix (near left and upper edge) because of surface symmetry
        gamma['primary'][:, :shape_reduced[1]] = \
            np.sum(np.multiply(normals['primary'][vis_test['primary']][:, None, :],
                               join_vector[:, :shape_reduced[1], :]), axis=2)
        gamma['primary'][:shape_reduced[0], shape_reduced[1]:] = \
            np.sum(np.multiply(normals['primary'][vis_test_symmetry['primary']][:, None, :],
                               join_vector[:shape_reduced[0], shape_reduced[1]:, :]), axis=2)

        gamma['secondary'][:shape_reduced[0], :] = \
            - np.sum(np.multiply(normals['secondary'][vis_test['secondary']][None, :, :],
                                 join_vector[:shape_reduced[0], :, :]), axis=2)
        gamma['secondary'][shape_reduced[0]:, :shape_reduced[1]] = \
            - np.sum(np.multiply(normals['secondary'][vis_test_symmetry['secondary']][None, :, :],
                                 join_vector[shape_reduced[0]:, :shape_reduced[1], :]), axis=2)
        return gamma

    @staticmethod
    def check_symmetric_gamma_for_negative_num(gamma, shape_reduced):
        """
        if cos < 0 it will be redefined as 0
        :param gamma:
        :param shape_reduced:
        :return:
        """
        gamma['primary'][:, :shape_reduced[1]][gamma['primary'][:, :shape_reduced[1]] < 0] = 0.
        gamma['primary'][:shape_reduced[0], shape_reduced[1]:][gamma['primary'][:shape_reduced[0],
                                                               shape_reduced[1]:] < 0] = 0.
        gamma['secondary'][:shape_reduced[0], :][gamma['secondary'][:shape_reduced[0], :] < 0] = 0.
        gamma['secondary'][shape_reduced[0]:, :shape_reduced[1]][gamma['secondary'][shape_reduced[0]:,
                                                                 :shape_reduced[1]] < 0] = 0.

    @staticmethod
    def get_symmetrical_q_ab(shape, shape_reduced, gamma, distance):
        """
        function uses surface symmetries to calculate parameter QAB = (cos gamma_a)*cos(gamma_b)/d**2 in reflection
        effect

        :param shape: desired shape of q_ab
        :param shape_reduced: shape of the surface symmetries, (faces above those indices are symmetrical to the ones
        below)
        :param gamma:
        :param distance:
        :return:
        """
        q_ab = np.empty(shape=shape, dtype=np.float)
        q_ab[:, :shape_reduced[1]] = \
            np.divide(np.multiply(gamma['primary'][:, :shape_reduced[1]],
                                  gamma['secondary'][:, :shape_reduced[1]]),
                      np.power(distance[:, :shape_reduced[1]], 2))
        q_ab[:shape_reduced[0], shape_reduced[1]:] = \
            np.divide(np.multiply(gamma['primary'][:shape_reduced[0], shape_reduced[1]:],
                                  gamma['secondary'][:shape_reduced[0], shape_reduced[1]:]),
                      np.power(distance[:shape_reduced[0], shape_reduced[1]:], 2))
        return q_ab

    def get_symmetrical_d_gamma(self, shape, shape_reduced, normals, join_vector, vis_test):
        """
        function uses surface symmetries to calculate limb darkening factor matrices for each components that are used
        in reflection effect

        :param shape: desired shape of limb darkening matrices d_gamma
        :param shape_reduced: shape of the surface symmetries, (faces above those indices are symmetrical to the ones
        below)
        :param normals:
        :param join_vector:
        :param vis_test:
        :return:
        """
        d_gamma = {'primary': np.empty(shape=shape, dtype=np.float),
                   'secondary': np.empty(shape=shape, dtype=np.float)}

        cos_theta = np.sum(normals['primary'][vis_test['primary'], None, :] *
                           join_vector[:, :shape_reduced[1], :], axis=-1)
        d_gamma['primary'][:, :shape_reduced[1]] = ld.limb_darkening_factor(
            coefficients=self.LD_COEFF,
            limb_darkening_law=config.LIMB_DARKENING_LAW,
            cos_theta=cos_theta)

        aux_normals = normals['primary'][vis_test['primary']]
        cos_theta = np.sum(aux_normals[:shape_reduced[0], None, :] *
                           join_vector[:shape_reduced[0], shape_reduced[1]:, :], axis=-1)
        d_gamma['primary'][:shape_reduced[0], shape_reduced[1]:] = ld.limb_darkening_factor(
            coefficients=self.LD_COEFF,
            limb_darkening_law=config.LIMB_DARKENING_LAW,
            cos_theta=cos_theta)

        cos_theta = np.sum(normals['secondary'][None, vis_test['secondary'], :] *
                           join_vector[:shape_reduced[0], :, :], axis=-1)
        d_gamma['secondary'][:shape_reduced[0], :] = ld.limb_darkening_factor(
            coefficients=self.LD_COEFF,
            limb_darkening_law=config.LIMB_DARKENING_LAW,
            cos_theta=cos_theta)

        aux_normals = normals['secondary'][vis_test['secondary']]
        cos_theta = np.sum(aux_normals[None, :shape_reduced[1], :] *
                           join_vector[shape_reduced[0]:, :shape_reduced[1], :], axis=-1)
        d_gamma['secondary'][shape_reduced[0]:, :shape_reduced[1]] = ld.limb_darkening_factor(
            coefficients=self.LD_COEFF,
            limb_darkening_law=config.LIMB_DARKENING_LAW,
            cos_theta=cos_theta)

        return d_gamma

    def redistribute_temperatures(self, temperatures):
        for _component in ['primary', 'secondary']:
            component_instance = getattr(self, _component)
            counter = len(component_instance.temperatures)
            component_instance.temperatures = temperatures[_component][:counter]
            if component_instance.spots:
                for spot_index, spot in component_instance.spots.items():
                    spot.temperatures = temperatures[_component][counter: counter + len(spot.temperatures)]
                    counter += len(spot.temperatures)

    def build_surface_gravity(self, component=None, components_distance=None):
        """
        function calculates gravity potential gradient magnitude (surface gravity) for each face

        :param component: `primary` or `secondary`
        :param components_distance: float
        :return:
        """

        if components_distance is None:
            raise ValueError('Component distance value was not supplied.')

        component = self._component_to_list(component)
        for _component in component:
            component_instance = getattr(self, _component)

            self._logger.debug('Computing surface areas of {} elements.'.format(_component))
            component_instance.areas = component_instance.calculate_areas()

            # consider to recompute polar radius (or remember consider this problem in case of eccentric orbit)
            # edit: you should do that far before this function

            # compute and assign potential gradient magnitudes for elements if missing
            self._logger.debug('Computing potential gradient magnitudes distribution of {} component.'
                               ''.format(_component))
            component_instance.potential_gradient_magnitudes = self.calculate_face_magnitude_gradient(
                component=_component, components_distance=components_distance)

            self._logger.debug('Computing magnitude of {} polar potential gradient.'.format(_component))
            component_instance.polar_potential_gradient_magnitude = \
                self.calculate_polar_potential_gradient_magnitude(
                    component=_component, components_distance=components_distance)

            if component_instance.spots:
                for spot_index, spot in component_instance.spots.items():
                    self._logger.debug('Calculating surface areas of {} component / {} spot.'.format(_component,
                                                                                                     spot_index))
                    spot.areas = spot.calculate_areas()

                    self._logger.debug('Calculating distribution of potential gradient magnitudes of {} component / '
                                       '{} spot.'.format(_component, spot_index))
                    spot.potential_gradient_magnitudes = self.calculate_face_magnitude_gradient(
                        component=_component,
                        components_distance=components_distance,
                        points=spot.points, faces=spot.faces)

    def build_temperature_distribution(self, component=None, components_distance=None):
        """
        function calculates temperature distribution on across all faces

        :param components_distance:
        :param component: `primary` or `secondary`
        :return:
        """
        component = self._component_to_list(component)

        for _component in component:
            component_instance = getattr(self, _component)

            self._logger.debug('Computing effective temprature distibution on {} component.'.format(_component))
            component_instance.temperatures = component_instance.calculate_effective_temperatures()
            if component_instance.pulsations:
                self._logger.debug('Adding pulsations to surface temperature distribution '
                                   'of the {} component.'.format(_component))
                component_instance.temperatures = component_instance.add_pulsations()

            if component_instance.spots:
                for spot_index, spot in component_instance.spots.items():
                    self._logger.debug('Computing temperature distribution of {} component / {} spot'
                                       ''.format(_component, spot_index))
                    spot.temperatures = spot.temperature_factor * component_instance.calculate_effective_temperatures(
                        gradient_magnitudes=spot.potential_gradient_magnitudes)
                    if component_instance.pulsations:
                        self._logger.debug('Adding pulsations to temperature distribution of {} component / {} spot'
                                           ''.format(_component, spot_index))
                        spot.temperatures = component_instance.add_pulsations(points=spot.points, faces=spot.faces,
                                                                              temperatures=spot.temperatures)

            self._logger.debug('Renormalizing temperature map of {0} component due to presence of spots'
                               ''.format(component))
            component_instance.renormalize_temperatures()

        if 'primary' in component and 'secondary' in component:
            self.reflection_effect(iterations=config.REFLECTION_EFFECT_ITERATIONS,
                                   components_distance=components_distance)

    def build_faces_orientation(self, component=None, components_distance=None):
        component = self._component_to_list(component)
        com_x = {'primary': 0, 'secondary': components_distance}

        for _component in component:
            component_instance = getattr(self, _component)
            component_instance.set_all_surface_centres()
            component_instance.set_all_normals(com=com_x[_component])

    def get_critical_inclination(self, components_distance=None):
        if components_distance is None:
            raise ValueError('Component distance value was not supplied.')

        if self.morphology != 'over-contact':
            radius1 = np.mean([self.primary.side_radius, self.primary.forward_radius, self.primary.backward_radius,
                               self.primary.polar_radius])
            radius2 = np.mean([self.secondary.side_radius, self.secondary.forward_radius,
                               self.secondary.backward_radius, self.secondary.polar_radius])
            cos_i_critical = (radius1 + radius2) / components_distance
            return np.degrees(np.arccos(cos_i_critical))

    def get_eclipse_boundaries(self, components_distance=None):
        if components_distance is None:
            raise ValueError('Component distance value was not supplied.')

        # check whether the inclination is high enough to enable eclipses
        if self.morphology != 'over-contact':
            radius1 = np.mean([self.primary.side_radius, self.primary.forward_radius, self.primary.backward_radius,
                               self.primary.polar_radius])
            radius2 = np.mean([self.secondary.side_radius, self.secondary.forward_radius,
                               self.secondary.backward_radius, self.secondary.polar_radius])
            sin_i_critical = (radius1 + radius2) / components_distance
            sin_i = np.sin(self.inclination)
            if sin_i < sin_i_critical:
                self._logger.debug('Inclination is not sufficient to produce eclipses.')
                return None
            radius1 = self.primary.forward_radius
            radius2 = self.secondary.forward_radius
            sin_i_critical = (radius1 + radius2) / components_distance
            azimuth = np.arcsin(np.sqrt(np.power(sin_i_critical, 2) - np.power(np.cos(self.inclination), 2)))
            azimuths = np.array([const.FULL_ARC - azimuth, azimuth, const.PI - azimuth, const.PI + azimuth])
            return azimuths

    def polar_gravity_acceleration(self, component=None, components_distance=None):
        if components_distance is None:
            raise ValueError('Component distance value was not supplied.')

        component = self._component_to_list(component)

        for _componet in component:
            components_instance = getattr(self, _componet)

            mass_ratio = self.mass_ratio if _componet == "primary" else 1.0 / self.mass_ratio

            polar_radius = components_instance.polar_radius
            x_com = (mass_ratio * components_distance) / (1.0 + mass_ratio)
            semi_major_axis = self.semi_major_axis.value

            primary_mass, secondary_mass = self.primary.mass, self.secondary.mass
            if _componet == "secondary":
                primary_mass, secondary_mass = secondary_mass, primary_mass

            r_vector = np.array([0.0, 0.0, polar_radius * semi_major_axis])
            centrifugal_distance = np.array([x_com * semi_major_axis, 0.0, 0.0])
            actual_distance = np.array([components_distance * semi_major_axis, 0., 0.])
            h_vector = r_vector - actual_distance
            angular_velocity = self.angular_velocity(components_distance=components_distance)

            block_a = - ((const.G * primary_mass) / np.linalg.norm(r_vector) ** 3) * r_vector
            block_b = - ((const.G * secondary_mass) / np.linalg.norm(h_vector) ** 3) * h_vector
            block_c = - (angular_velocity ** 2) * centrifugal_distance

            g = block_a + block_b + block_c

            # magnitude of polar gravity acceleration in physical CGS units
            return np.linalg.norm(g) * 1e2

    def angular_velocity(self, components_distance=None):
        if components_distance is None:
            raise ValueError('Component distance value was not supplied.')

        return ((2.0 * np.pi) / (self.period * 86400.0 * (components_distance ** 2))) * np.sqrt(
            (1.0 - self.eccentricity) * (1.0 + self.eccentricity))  # $\rad.sec^{-1}$
