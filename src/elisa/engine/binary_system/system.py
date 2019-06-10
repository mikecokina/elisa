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

from copy import copy
from multiprocessing.pool import Pool

import numpy as np
import scipy
from astropy import units as u
from scipy.optimize import newton
from scipy.spatial.qhull import Delaunay

from elisa.conf import config
from elisa.engine import const, logger
from elisa.engine import graphics
from elisa.engine import ld
from elisa.engine import units
from elisa.engine import utils
# from elisa.engine.binary_system import static, build, mp, lc
from elisa.engine.binary_system import static, build, mp, lc, geo
from elisa.engine.binary_system.plot import Plot
from elisa.engine.orbit import Orbit
from elisa.engine.base.star import Star
from elisa.engine.base.system import System


class BinarySystem(System):
    MANDATORY_KWARGS = ['gamma', 'inclination', 'period', 'eccentricity', 'argument_of_periastron',
                        'primary_minimum_time', 'phase_shift']
    OPTIONAL_KWARGS = []
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    # this will be removed after full implementation of LD
    LD_COEFF = 0.5

    def __init__(self, primary, secondary, name=None, suppress_logger=False, **kwargs):
        utils.invalid_kwarg_checker(kwargs, BinarySystem.ALL_KWARGS, BinarySystem)
        super(BinarySystem, self).__init__(name=name, suppress_logger=suppress_logger, **kwargs)

        self.initial_kwargs.update(
            dict(
                primary=Star(name="star.dump.primary", suppress_logger=True, **primary.initial_kwargs),
                secondary=Star(name="star.dump.secondary", suppress_logger=True, **secondary.initial_kwargs),
                suppress_logger=True
            )
        )

        # get logger
        self._logger = logger.getLogger(name=BinarySystem.__name__, suppress=suppress_logger)
        self._logger.info("initialising object {}".format(BinarySystem.__name__))
        self._suppress_logger = suppress_logger

        self._logger.debug("setting property components of class instance {}".format(BinarySystem.__name__))

        self.plot = Plot(self)

        # assign components to binary system
        self._primary = primary
        self._secondary = secondary

        # physical properties check
        self._mass_ratio = self.secondary.mass / self.primary.mass

        # default values of properties
        self._period = None
        self._eccentricity = None
        self._argument_of_periastron = None
        self._orbit = None
        self._primary_minimum_time = None
        self._phase_shift = None
        self._semi_major_axis = None
        self._periastron_phase = None
        self._morphology = None
        self._inclination = None

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
            self._logger.debug("setting property {} "
                               "of class instance {} to {}".format(kwarg, BinarySystem.__name__, kwargs[kwarg]))
            setattr(self, kwarg, kwargs[kwarg])

        # calculation of dependent parameters
        self._logger.debug("computing semi-major axis")
        self._semi_major_axis = self.calculate_semi_major_axis()

        # orbit initialisation (initialise class Orbit from given BinarySystem parameters)
        self._logger.debug("Initializing orbit instance in BinarySystem")
        self.init_orbit()

        # setup critical surface potentials in periastron
        self._logger.debug("setting up critical surface potentials of components in periastron")
        self._setup_periastron_critical_potential()

        # binary star morphology estimation
        self._logger.debug("setting up morphological classification of binary system")
        self.setup_morphology()

        # polar radius of both component in periastron
        self.setup_components_radii(components_distance=self.orbit.periastron_distance)

        # if secondary discretization factor was not set, it will be now with respect to primary component
        if not self.secondary.kwargs.get('discretization_factor'):
            self.secondary.discretization_factor = \
                self.primary.discretization_factor * self.primary.polar_radius / self.secondary.polar_radius * u.rad
            self._logger.info("Setting discretization factor of secondary component "
                              "according discretization factor of primary component "
                              "to: {:.2f} degrees.".format(np.degrees(self.secondary.discretization_factor)))

    def init(self):
        """
        function to reinitialize BinarySystem class instance after changing parameter(s) of binary system using setters

        :return:
        """
        self.__init__(primary=self.primary, secondary=self.secondary, **self._kwargs_serializer())

    def init_orbit(self):
        """
        encapsulating orbit class into binary system

        :return:
        """
        self._logger.debug("re/initializing orbit in class instance {} / {}"
                           "".format(BinarySystem.__name__, BinarySystem.name))
        orbit_kwargs = {key: getattr(self, key) for key in Orbit.MANDATORY_KWARGS}
        self._orbit = Orbit(suppress_logger=self._suppress_logger, **orbit_kwargs)

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
        set orbital period of binary star system, if unit is not specified, default period unit is assumed

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
        self._logger.debug("setting property argument of periastron of class instance {} to {}"
                           "".format(BinarySystem.__name__, self._argument_of_periastron))

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
        self._logger.debug("setting property primary_minimum_time of class instance {} to {}"
                           "".format(BinarySystem.__name__, self._primary_minimum_time))

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
        self._logger.debug("setting property phase_shift of class instance {} to {}"
                           "".format(BinarySystem.__name__, self._phase_shift))

    @property
    def inclination(self):
        """
        returns inclination of binary system orbit

        :return: numpy.float
        """
        return self._inclination

    @inclination.setter
    def inclination(self, inclination):
        """
        setter for inclination of binary system orbit

        :param inclination: numpy.float
        :return:
        """
        if isinstance(inclination, u.quantity.Quantity):
            self._inclination = np.float64(inclination.to(units.ARC_UNIT))
        elif isinstance(inclination, (int, np.int, float, np.float)):
            self._inclination = np.float64(inclination)
        else:
            raise TypeError('Input of variable `inclination` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

        if not 0 <= self.inclination <= const.PI:
            raise ValueError('Eccentricity value of {} is out of bounds (0, pi).'.format(self.inclination))

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
        period = np.float64((self._period * units.PERIOD_UNIT).to(u.s))
        return (const.G * (self.primary.mass + self.secondary.mass) * period ** 2 / (4 * const.PI ** 2)) ** (1.0 / 3)

    def setup_components_radii(self, components_distance):
        fns = [self.calculate_polar_radius, self.calculate_side_radius, self.calculate_backward_radius]
        components = ['primary', 'secondary']

        for component in components:
            component_instance = getattr(self, component)
            for fn in fns:
                self._logger.debug('initialising {} for {} component'.format(
                    ' '.join(str(fn.__name__).split('_')[1:]), component))

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

        components = static.component_to_list(component)
        fns = {
            "primary": (self.potential_primary_fn, self.pre_calculate_for_potential_value_primary),
            "secondary": (self.potential_secondary_fn, self.pre_calculate_for_potential_value_secondary)
        }
        fns = {_component: fns[_component] for _component in components}

        # in case of wuma system, get separation and make additional test of location of each point (if primary
        # spot doesn't intersect with secondary, if does, then such spot will be skipped completly)
        neck_position = self.calculate_neck_position() if self.morphology == "over-contact" else 1e10

        for component, functions in fns.items():
            self._logger.info("evaluating spots for {} component".format(component))
            potential_fn, precalc_fn = functions
            component_instance = getattr(self, component)

            if not component_instance.spots:
                self._logger.info("no spots to evaluate for {} component - continue.".format(component))
                continue

            # iterate over spots
            for spot_index, spot_instance in list(component_instance.spots.items()):
                # lon -> phi, lat -> theta
                lon, lat = spot_instance.longitude, spot_instance.latitude

                component_instance.setup_spot_instance_discretization_factor(spot_instance, spot_index)
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
                    self._logger.warning("center of spot {} doesn't satisfy reasonable "
                                         "conditions and entire spot will be omitted"
                                         "".format(spot_instance.kwargs_serializer()))

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
                    self._logger.warning("first inner ring of spot {} doesn't satisfy reasonable "
                                         "conditions and entire spot will be omitted"
                                         "".format(spot_instance.kwargs_serializer()))

                    component_instance.remove_spot(spot_index=spot_index)
                    continue

                x0 = np.sqrt(spot_center_r ** 2 + solution ** 2 - (2.0 * spot_center_r * solution * np.cos(alpha)))

                # number of points in latitudal direction
                # + 1 to obtain same discretization as object itself
                num_radial = int(np.round((diameter * 0.5) / alpha)) + 1
                self._logger.debug('number of rings in spot {} is {}'
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
                                                                    degrees=False,
                                                                    omega_normalized=True)

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
                    self._logger.warning("at least 1 point of spot {} doesn't satisfy "
                                         "reasonable conditions and entire spot will be omitted"
                                         "".format(spot_instance.kwargs_serializer()))
                    continue
                if component == "primary":
                    spot_instance.points = np.array(spot_points)
                    spot_instance.boundary = np.array(boundary_points)
                    spot_instance.center = np.array(spot_center)
                else:
                    spot_instance.points = np.array([np.array([components_distance - point[0], -point[1], point[2]])
                                                     for point in spot_points])

                    spot_instance.boundary = np.array([np.array([components_distance - point[0], -point[1], point[2]])
                                                       for point in boundary_points])

                    spot_instance.center = \
                        np.array([components_distance - spot_center[0], -spot_center[1], spot_center[2]])
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
        star_mandatory_kwargs = ['mass', 'surface_potential', 'synchronicity', 'albedo', 'metallicity']
        missing_kwargs = []
        for component in [self.primary, self.secondary]:
            for kwarg in star_mandatory_kwargs:
                if np.isnan(getattr(component, kwarg)):
                    missing_kwargs.append(f"`{kwarg}`")

            component_name = 'primary' if component == self.primary else 'secondary'
            if len(missing_kwargs) != 0:
                raise ValueError('Mising argument(s): {} in {} component Star class'
                                 ''.format(', '.join(missing_kwargs), component_name))

    def _kwargs_serializer(self):
        """
        creating dictionary of keyword arguments of BinarySystem class in order to be able to reinitialize the class
        instance in init()

        :return: dict
        """
        serialized_kwargs = {}
        for kwarg in self.ALL_KWARGS:
            serialized_kwargs[kwarg] = getattr(self, kwarg)
        return serialized_kwargs

    def _setup_periastron_critical_potential(self):
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

    def setup_morphology(self):
        """
        Setup binary star class property `morphology`
        :return:
        """
        __PRECISSION__ = 1e-8
        __SETUP_VALUE__ = None
        if (self.primary.synchronicity == 1 and self.secondary.synchronicity == 1) and self.eccentricity == 0.0:
            lp = self.libration_potentials()

            self.primary.filling_factor = static.compute_filling_factor(self.primary.surface_potential, lp)
            self.secondary.filling_factor = static.compute_filling_factor(self.secondary.surface_potential, lp)

            if ((1 > self.secondary.filling_factor > 0) or (1 > self.primary.filling_factor > 0)) and \
                    (abs(self.primary.filling_factor - self.secondary.filling_factor) > __PRECISSION__):
                raise ValueError("Detected over-contact binary system, but potentials of components are not the same.")
            if self.primary.filling_factor > 1 or self.secondary.filling_factor > 1:
                raise ValueError("Non-Physical system: primary_filling_factor or "
                                 "secondary_filling_factor is greater then 1\n"
                                 "Filling factor is obtained as following:"
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

        b, c, d, e = args  # auxiliary values pre-calculated in pre_calculate_for_potential_value_primary()
        radius2 = np.power(radius, 2)

        return 1 / radius + self.mass_ratio / np.sqrt(b + radius2 - c * radius) - d * radius + e * radius2

    def pre_calculate_for_potential_value_primary_cylindrical(self, *args):
        """
        function calculates auxiliary values for calculation of primary component potential  in cylindrical symmetry,
        and therefore they don't need to be wastefully recalculated every iteration in solver

        :param args: (azimut angle (0, 2pi), z_n (cylindrical, identical with cartesian x))
        :return: tuple: (a, b, c, d, e, f) such that: Psi1 = 1/sqrt(a+r^2) + q/sqrt(b + r^2) - c + d*(e+f*r^2)
        """
        phi, z = args

        qq = self.mass_ratio / (1 + self.mass_ratio)

        a = np.power(z, 2)
        b = np.power(1 - z, 2)
        c = 0.5 * self.mass_ratio * qq
        d = 0.5 + 0.5 * self.mass_ratio
        e = np.power(qq - z, 2)
        f = np.power(np.sin(phi), 2)

        if np.isscalar(phi):
            return a, b, c, d, e, f
        else:
            cc = c * np.ones(np.shape(phi))
            dd = d * np.ones(np.shape(phi))
            return np.column_stack((a, b, cc, dd, e, f))

    def potential_value_primary_cylindrical(self, radius, *args):
        """
        calculates modified kopal potential from point of view of primary component in cylindrical coordinates
        r_n, phi_n, z_n, where z_n = x and heads along z axis, this function is intended for generation of ``necks``
        of W UMa systems, therefore components distance = 1 an synchronicity = 1 is assumed

        :param radius: np.float
        :param args: tuple: (a, b, c, d, e, f) such that: Psi1 = 1/sqrt(a+r^2) + q/sqrt(b + r^2) - c + d*(e+f*r^2)
        :return:
        """
        a, b, c, d, e, f = args

        radius2 = np.power(radius, 2)
        return 1 / np.sqrt(a + radius2) + self.mass_ratio / np.sqrt(b + radius2) - c + d * (e + f * radius2)

    def pre_calculate_for_potential_value_secondary(self, *args):
        """
        function calculates auxiliary values for calculation of secondary component potential, and therefore they don't
        need to be wastefully recalculated every iteration in solver

        :param args: (component distance, azimut angle (0, 2pi), latitude angle (0, pi)
        :return: tuple: (b, c, d, e, f) such that: Psi2 = q/r + 1/sqrt(b+r^2+Cr) - d*r + e*x^2 + f
        """
        distance, phi, theta = args  # distance between components, azimut angle, latitude angle (0,180)

        cs = np.cos(phi) * np.sin(theta)

        b = np.power(distance, 2)
        c = 2 * distance * cs
        d = cs / b
        e = 0.5 * np.power(self.secondary.synchronicity, 2) * (1 + self.mass_ratio) * (1 - np.power(np.cos(theta), 2))
        f = 0.5 - 0.5 * self.mass_ratio

        if np.isscalar(phi):
            return b, c, d, e, f
        else:
            bb = b * np.ones(np.shape(phi))
            ff = f * np.ones(np.shape(phi))
            return np.column_stack((bb, c, d, e, ff))

    def potential_value_secondary(self, radius, *args):
        """
        calculates modified kopal potential from point of view of secondary component

        :param radius: np.float; spherical variable
        :param args: tuple: (b, c, d, e, f) such that: Psi2 = q/r + 1/sqrt(b+r^2-Cr) - d*r + e*x^2 + f
        :return: np.float
        """
        b, c, d, e, f = args
        radius2 = np.power(radius, 2)

        return self.mass_ratio / radius + 1. / np.sqrt(b + radius2 - c * radius) - d * radius + e * radius2 + f

    def pre_calculate_for_potential_value_secondary_cylindrical(self, *args):
        """
        function calculates auxiliary values for calculation of secondary component potential in cylindrical symmetry,
        and therefore they don't need to be wastefully recalculated every iteration in solver

        :param args: (azimut angle (0, 2pi), z_n (cylindrical, identical with cartesian x))
        :return: tuple: (a, b, c, d, e, f, G) such that: Psi2 = q/sqrt(a+r^2) + 1/sqrt(b + r^2) - c + d*(e+f*r^2) + G
        """
        phi, z = args

        qq = 1.0 / (1 + self.mass_ratio)

        a = np.power(z, 2)
        b = np.power(1 - z, 2)
        c = 0.5 / qq
        d = np.power(qq - z, 2)
        e = np.power(np.sin(phi), 2)
        f = 0.5 - 0.5 * self.mass_ratio - 0.5 * qq

        if np.isscalar(phi):
            return a, b, c, d, e, f
        else:
            cc = c * np.ones(np.shape(phi))
            ff = f * np.ones(np.shape(phi))
            return np.column_stack((a, b, cc, d, e, ff))

    def potential_value_secondary_cylindrical(self, radius, *args):
        """
        calculates modified kopal potential from point of view of secondary component in cylindrical coordinates
        r_n, phi_n, z_n, where z_n = x and heads along z axis, this function is intended for generation of ``necks``
        of W UMa systems, therefore components distance = 1 an synchronicity = 1 is assumed

        :param radius: np.float
        :param args: tuple: (a, b, c, d, e, f, G) such that: Psi2 = q/sqrt(a+r^2) + 1/sqrt(b+r^2) - c + d*(e+f*r^2) + G
        :return:
        """
        a, b, c, d, e, f = args

        radius2 = np.power(radius, 2)
        return self.mass_ratio / np.sqrt(a + radius2) + 1. / np.sqrt(b + radius2) + c * (d + e * radius2) + f

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
            raise ValueError("Iteration process to solve critical potential seems "
                             "to lead nowhere (critical potential _solver has failed).")

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
            f2 = np.power(self.primary.synchronicity, 2)
            domega_dx = - points[:, 0] / r3 + self.mass_ratio * (
                components_distance - points[:, 0]) / r_hat3 + f2 * (
                self.mass_ratio + 1) * points[:, 0] - self.mass_ratio / np.power(components_distance, 2)
        elif component == 'secondary':
            f2 = np.power(self.secondary.synchronicity, 2)
            domega_dx = - points[:, 0] / r3 + self.mass_ratio * (
                components_distance - points[:, 0]) / r_hat3 - f2 * (
                self.mass_ratio + 1) * (
                components_distance - points[:, 0]) * points[:, 0] + 1 / np.power(
                components_distance, 2)
        else:
            raise ValueError('Invalid value `{}` of argument `component`. '
                             'Use `primary` or `secondary`.'.format(component))

        domega_dy = - points[:, 1] * (1 / r3 + self.mass_ratio / r_hat3 - f2 * (self.mass_ratio + 1))
        domega_dz = - points[:, 2] * (1 / r3 + self.mass_ratio / r_hat3)
        return -np.column_stack((domega_dx, domega_dy, domega_dz))

    def calculate_face_magnitude_gradient(self, component, components_distance, points=None, faces=None):
        """
        return array of face magnitude gradients calculated as a mean of magnitude gradients on vertices

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
            raise ValueError('Invalid value `{}` of argument `component`. '
                             'Use `primary` or `secondary`.'.format(component))
        domega_dz = - points[2] * (1. / r3 + self.mass_ratio / r_hat3)
        return np.power(np.power(domega_dx, 2) + np.power(domega_dz, 2), 0.5)

    def calculate_polar_gravity_acceleration(self, component, components_distance, logg=False):
        """
        calculates polar gravity acceleration for component of binary system, calculated from gradient of roche
        potential d_Omega/dr using transformation g = d_Psi/dr = (GM_component/semi_major_axis**2) * d_Omega/dr
        ( * 1/q in case of secondary component )
        :param component: `primary` or `secondary`
        :param components_distance: float (in SMA units)
        :param logg: if true log g is returned
        :return: surface gravity or log10 of surface gravity
        """
        component_instance = getattr(self, component)
        component_instance.polar_potential_gradient_magnitude = \
            self.calculate_polar_potential_gradient_magnitude(component=component,
                                                              components_distance=components_distance)
        gradient = \
            const.G * component_instance.mass * component_instance.polar_potential_gradient_magnitude / \
            np.power(self.semi_major_axis, 2)
        gradient = gradient / self.mass_ratio if component == 'secondary' else gradient
        return np.log10(gradient) if logg else gradient

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
            raise ValueError('Invalid value of `component` argument {}. '
                             'Expecting `primary` or `secondary`.'.format(args[0]))

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

    def mesh_detached(self, component, components_distance, symmetry_output=False, **kwargs):
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
        suppress_parallelism = kwargs.get("suppress_parallelism", True)
        component_instance = getattr(self, component)
        if component_instance.discretization_factor > const.HALF_PI:
            raise ValueError("Invalid value of alpha parameter. Use value less than 90.")

        alpha = component_instance.discretization_factor

        if component == 'primary':
            potential_fn = self.potential_primary_fn
            precalc_fn = self.pre_calculate_for_potential_value_primary
        elif component == 'secondary':
            potential_fn = self.potential_secondary_fn
            precalc_fn = self.pre_calculate_for_potential_value_secondary
        else:
            raise ValueError('Invalid value of `component` argument: `{}`. Expecting '
                             '`primary` or `secondary`.'.format(component))

        # pre calculating azimuths for surface points on quarter of the star surface
        phi, theta, separator = static.pre_calc_azimuths_for_detached_points(alpha)

        # calculating mesh in cartesian coordinates for quarter of the star
        args = phi, theta, components_distance, precalc_fn, potential_fn

        if config.NUMBER_OF_THREADS == 1 or suppress_parallelism:
            self._logger.debug('calculating surface points of {0} component in mesh_detached '
                               'function using single process method'.format(component))

            points_q = static.get_surface_points(*args)
        else:
            self._logger.debug('calculating surface points of {0} component in mesh_detached '
                               'function using multi process method'.format(component))
            points_q = self.get_surface_points_multiproc(*args)

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

    def get_surface_points_multiproc(self, *args):
        """
        function solves radius for given azimuths that are passed in *argss via multithreading approach

        :param args:
        :return:
        """

        phi, theta, components_distance, precalc_fn, potential_fn = args
        precalc_vals = precalc_fn(*(components_distance, phi, theta))
        preacalc_vals_args = [tuple(precalc_vals[i, :]) for i in range(np.shape(precalc_vals)[0])]

        if potential_fn == self.potential_primary_fn:
            potential_fn_str_repr = "static_potential_primary_fn"
            surface_potential = self.primary.surface_potential
        else:
            potential_fn_str_repr = "static_potential_secondary_fn"
            surface_potential = self.secondary.surface_potential

        pool = Pool(processes=config.NUMBER_OF_THREADS)
        res = [pool.apply_async(mp.get_surface_points_worker,
                                (potential_fn_str_repr, args))
               for args in mp.prepare_get_surface_points_args(preacalc_vals_args,  self.mass_ratio, surface_potential)]

        pool.close()
        pool.join()
        result_list = [np.array(r.get()) for r in res]
        r = np.array(sorted(result_list, key=lambda x: x[0])).T[1]
        return utils.spherical_to_cartesian(np.column_stack((r, phi, theta)))

    def get_surface_points_multiproc_cylindrical(self, *args):
        """
        function solves radius for given azimuths that are passed in *argss via multithreading approach

        :param args:
        :return:
        """

        phi, z, precalc_fn, potential_fn = args
        precalc_vals = precalc_fn(*(phi, z))
        preacalc_vals_args = [tuple(precalc_vals[i, :]) for i in range(np.shape(precalc_vals)[0])]

        if potential_fn == self.potential_primary_cylindrical_fn:
            potential_fn_str_repr = "static_potential_primary_cylindrical_fn"
            surface_potential = self.primary.surface_potential
        else:
            potential_fn_str_repr = "static_potential_secondary_cylindrical_fn"
            surface_potential = self.secondary.surface_potential

        pool = Pool(processes=config.NUMBER_OF_THREADS)
        res = [pool.apply_async(mp.get_surface_points_worker,
                                (potential_fn_str_repr, args))
               for args in mp.prepare_get_surface_points_args(preacalc_vals_args,  self.mass_ratio, surface_potential)]

        pool.close()
        pool.join()
        result_list = [np.array(r.get()) for r in res]
        r = np.array(sorted(result_list, key=lambda x: x[0])).T[1]
        return utils.cylindrical_to_cartesian(np.column_stack((r, phi, z)))

    def mesh_over_contact(self, component=None, symmetry_output=False, **kwargs):
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
        suppress_parallelism = kwargs.get("suppress_parallelism", True)
        component_instance = getattr(self, component)
        if component_instance.discretization_factor > const.HALF_PI:
            raise ValueError("Invalid value of alpha parameter. Use value less than 90.")

        alpha = component_instance.discretization_factor

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

        # precalculating azimuths for farside points
        phi_farside, theta_farside, separator_farside = static.pre_calc_azimuths_for_overcontact_farside_points(alpha)

        # generating the azimuths for neck
        neck_position, neck_polynomial = self.calculate_neck_position(return_polynomial=True)
        phi_neck, z_neck, separator_neck = \
            static.pre_calc_azimuths_for_overcontact_neck_points(alpha, neck_position, neck_polynomial,
                                                                 polar_radius=component_instance.polar_radius,
                                                                 component=component)

        # solving points on farside
        args = phi_farside, theta_farside, components_distance, precalc, fn
        # here implement multiprocessing
        if config.NUMBER_OF_THREADS == 1 or suppress_parallelism:
            self._logger.debug('calculating farside points of {0} component in mesh_overcontact '
                               'function using single process method'.format(component))
            points_farside = static.get_surface_points(*args)
        else:
            self._logger.debug('calculating farside points of {0} component in mesh_overcontact '
                               'function using multi process method'.format(component))
            points_farside = self.get_surface_points_multiproc(*args)

        # assigning equator points and point A (the point on the tip of the farside equator)
        equator_farside = points_farside[:separator_farside[0], :]
        x_eq1, x_a = equator_farside[: -1, 0], equator_farside[-1, 0]
        y_eq1, y_a = equator_farside[: -1, 1], equator_farside[-1, 1]
        z_eq1, z_a = equator_farside[: -1, 2], equator_farside[-1, 2]

        # assigning points on phi = pi
        meridian_farside1 = points_farside[separator_farside[0]: separator_farside[1], :]
        x_meridian1, y_meridian1, z_meridian1 = \
            meridian_farside1[:, 0], meridian_farside1[:, 1], meridian_farside1[:, 2]

        # assigning points on phi = pi/2 meridian, perpendicular to component`s distance vector
        meridian_farside2 = points_farside[separator_farside[1]: separator_farside[2], :]
        x_meridian2, y_meridian2, z_meridian2 = \
            meridian_farside2[:, 0], meridian_farside2[:, 1], meridian_farside2[:, 2]

        # assigning the rest of the surface on farside
        quarter = points_farside[separator_farside[2]:, :]
        x_q1, y_q1, z_q1 = quarter[:, 0], quarter[:, 1], quarter[:, 2]

        # solving points on neck
        args = phi_neck, z_neck, precal_cylindrical, fn_cylindrical
        if config.NUMBER_OF_THREADS == 1 or suppress_parallelism:
            self._logger.debug('calculating neck points of {0} component in mesh_overcontact '
                               'function using single process method'.format(component))
            points_neck = static.get_surface_points_cylindrical(*args)
        else:
            self._logger.debug('calculating neck points of {0} component in mesh_overcontact '
                               'function using multi process method'.format(component))
            points_neck = self.get_surface_points_multiproc_cylindrical(*args)

        # assigning equator points on neck
        r_eqn = points_neck[:separator_neck[0], :]
        z_eqn, y_eqn, x_eqn = r_eqn[:, 0], r_eqn[:, 1], r_eqn[:, 2]

        # assigning points on phi = 0 meridian, perpendicular to component`s distance vector
        r_meridian_n = points_neck[separator_neck[0]: separator_neck[1], :]
        z_meridian_n, y_meridian_n, x_meridian_n = r_meridian_n[:, 0], r_meridian_n[:, 1], r_meridian_n[:, 2]

        # assigning the rest of the surface on neck
        r_n = points_neck[separator_neck[1]:, :]
        z_n, y_n, x_n = r_n[:, 0], r_n[:, 1], r_n[:, 2]

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
            raise ValueError("{} component, with class instance name {} do not "
                             "contain any valid surface point to triangulate"
                             "".format(component, component_instance.name))
        # there is a problem with triangulation of near over-contact system, delaunay is not good with pointy surfaces
        critical_pot = self.primary.critical_surface_potential if component == 'primary' \
            else self.secondary.critical_surface_potential
        potential = self.primary.surface_potential if component == 'primary' \
            else self.secondary.surface_potential
        if potential - critical_pot > 0.01:
            self._logger.debug('triangulating surface of {} component using standard method'.format(component))
            triangulation = Delaunay(points)
            triangles_indices = triangulation.convex_hull
        else:
            self._logger.debug('surface of {} component is near or at critical potential; '
                               'therefore custom triangulation method for (near)critical '
                               'potential surfaces will be used.'.format(component))
            # calculating closest point to the barycentre
            r_near = np.max(points[:, 0]) if component == 'primary' else np.min(points[:, 0])
            # projection of component's far side surface into ``sphere`` with radius r1
            # fixme: unused variable, remove it
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

    def over_contact_system_surface(self, component=None, points=None, **kwargs):
        # do not remove kwargs, keep compatible interface w/ detached where components distance has to be provided
        # in this case,m components distance is sinked in kwargs and not used
        """
        calculates surface faces from the given component's points in case of over-contact system

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

        fn_map = {
            'primary': (self.potential_primary_fn, self.pre_calculate_for_potential_value_primary),
            'secondary': (self.potential_secondary_fn, self.pre_calculate_for_potential_value_secondary)
        }

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

    def _get_surface_builder_fn(self):
        """
        returns suitable triangulation function depending on morphology
        :return: function instance that performs generation surface faces
        """
        return self.over_contact_system_surface if self.morphology == "over-contact" else self.detached_system_surface

    @classmethod
    def is_property(cls, kwargs):
        """
        method for checking if keyword arguments are valid properties of this class

        :param kwargs: dict
        :return:
        """
        is_not = ['`{}`'.format(k) for k in kwargs if k not in cls.ALL_KWARGS]
        if is_not:
            raise AttributeError('Arguments {} are not valid {} properties.'
                                 ''.format(', '.join(is_not), cls.__name__))

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
            if self.primary.polar_radius > self.secondary.polar_radius else (-x_corr_primary, 1 - x_corr_secondary)
        return xlim

    # noinspection PyTypeChecker
    def reflection_effect(self, iterations=None, components_distance=None):
        if not config.REFLECTION_EFFECT:
            self._logger.debug('reflection effect is switched off')
            return
        if iterations is None:
            raise ValueError('Number of iterations for reflection effect was not specified.')
        elif iterations == 0:
            self._logger.debug('number of reflections in reflection effect was set to zero. '
                               'Reflection effect will not be calculated')
            return

        if components_distance is None:
            raise ValueError('Components distance was not supplied.')

        component = static.component_to_list(None)

        xlim = self.faces_visibility_x_limits(components_distance=components_distance)

        # this tests if you can use surface symmetries
        use_quarter_star_test = not self.primary.has_spots() and not self.secondary.has_spots()
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
                areas[_component] = static.init_surface_variables(component_instance)

            # test for visibility of star faces
            vis_test[_component], vis_test_symmetry[_component] = \
                self.get_visibility_tests(centres[_component], use_quarter_star_test, xlim[_component], _component)

            if component_instance.spots:
                # including spots into overall surface
                for spot_index, spot in component_instance.spots.items():
                    vis_test_spot = static.visibility_test(spot.face_centres, xlim[_component], _component)

                    # merge surface and spot face parameters into one variable
                    centres[_component], normals[_component], temperatures[_component], areas[_component], \
                        vis_test[_component] = \
                        static.include_spot_to_surface_variables(centres[_component], spot.face_centres,
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

            distance, join_vector = static.get_symmetrical_distance_matrix(_shape, _shape_reduced, centres, vis_test,
                                                                           vis_test_symmetry)

            # calculating cos of angle gamma between face normal and join vector
            # initialising gammma matrices
            gamma = static.get_symmetrical_gammma(_shape[:2], _shape_reduced, normals, join_vector, vis_test,
                                                  vis_test_symmetry)

            # testing mutual visibility of faces by assigning 0 to non visible face combination
            static.check_symmetric_gamma_for_negative_num(gamma, _shape_reduced)

            # calculating QAB = (cos gamma_a)*cos(gamma_b)/d**2
            q_ab = static.get_symmetrical_q_ab(_shape[:2], _shape_reduced, gamma, distance)

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
                    vector_to_sum1 = reflection_factor[counterpart] * np.power(
                        temperatures[counterpart][vis_test[counterpart]], 4) * areas[counterpart][vis_test[counterpart]]
                    counterpart_to_sum = np.matmul(vector_to_sum1, matrix_to_sum2['secondary']) \
                        if _component == 'secondary' else np.matmul(matrix_to_sum2['primary'], vector_to_sum1)
                    reflection_factor[_component][:symmetry_to_use[_component]] = \
                        1 + (_c[_component] / np.power(
                            temperatures[_component][vis_test_symmetry[_component]], 4)) * counterpart_to_sum

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
            test1 = static.visibility_test(centres, xlim, component)
            # this variable contains faces that can seen from base symmetry part of the other star
            vis_test = np.logical_and(test1, quadrant_exclusion)
            vis_test_symmetry = np.logical_and(test1, single_quadrant)

        else:
            vis_test = centres[:, 0] >= xlim if \
                component == 'primary' else centres[:, 0] <= xlim
            vis_test_symmetry = None

        return vis_test, vis_test_symmetry

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
        """
        in this function array of `temperatures` is parsed into chunks that belong to stellar surface and spots

        :param temperatures: np.array - temperatures from the whole surface, ordered: surface, spot1, spot2...
        :return:
        """
        for _component in ['primary', 'secondary']:
            component_instance = getattr(self, _component)
            counter = len(component_instance.temperatures)
            component_instance.temperatures = temperatures[_component][:counter]
            if component_instance.spots:
                for spot_index, spot in component_instance.spots.items():
                    spot.temperatures = temperatures[_component][counter: counter + len(spot.temperatures)]
                    counter += len(spot.temperatures)

    def angular_velocity(self, components_distance=None):
        if components_distance is None:
            raise ValueError('Component distance value was not supplied.')

        return ((2.0 * np.pi) / (self.period * 86400.0 * (components_distance ** 2))) * np.sqrt(
            (1.0 - self.eccentricity) * (1.0 + self.eccentricity))  # $\rad.sec^{-1}$

    def get_positions_method(self):
        return self.calculate_orbital_motion

    def calculate_orbital_motion(self, phase=None, return_nparray=False):
        orbital_motion = self.orbit.orbital_motion(phase=phase)
        idx = np.arange(np.shape(phase)[0])
        positions = np.hstack((idx[:, np.newaxis], orbital_motion))
        retval = [const.BINARY_POSITION_PLACEHOLDER(*p) for p in positions]
        return retval, positions if return_nparray else retval

    # todo: are you planning to use that?? no usages for now...
    def calculate_lines_of_sight(self, phase=None):
        """
        returns indices and positions of secondary component relative to primary component
        :param phase: list - list of phases at which to calculate the positions
        :return:
        """
        orbital_motion = self.orbit.orbital_motion(phase=phase)
        idx = np.arange(np.shape(phase)[0])

        line_of_sight_spherical = np.empty((np.shape(phase)[0], 3), dtype=np.float)
        line_of_sight_spherical[:, 0] = 1
        line_of_sight_spherical[:, 1] = -1.0 * orbital_motion[:, 2]
        line_of_sight_spherical[:, 2] = const.PI - self.inclination
        line_of_sight = utils.spherical_to_cartesian(line_of_sight_spherical)
        return np.hstack((idx[:, np.newaxis], line_of_sight))

    def compute_lightcurve(self, **kwargs):
        """
        this function decides which light curve generator function is used, depending on the basic properties of the
        binary system

        :param kwargs: - arguments to be passed into light curve generator functions
        :return:
        """
        if self.eccentricity == 0 and self.primary.synchronicity == 1 and self.secondary.synchronicity == 1:
            self._logger.debug('Implementing light curve generator function for synchronous binary system with '
                               'circular orbit.')
            return self._compute_circular_synchronous_lightcurve(**kwargs)
        elif self.eccentricity == 0 and (self.primary.synchronicity != 1 or self.secondary.synchronicity != 1) \
                and (self.primary.has_spots() or self.secondary.has_spots()):
            self._logger.debug('Implementing light curve generator function for asynchronous binary system with '
                               'circular orbit.')
            return self._compute_circular_spotify_asynchronous_lightcurve(**kwargs)
        elif 1 > self.eccentricity > 0:
            self._logger.debug('Implementing light curve generator function for eccentric orbit.')
            return self._compute_eccentric_lightcurve(**kwargs)
        raise NotImplementedError("Orbit type not implemented or invalid")

    def _compute_circular_synchronous_lightcurve(self, **kwargs):
        return lc.compute_circular_synchronous_lightcurve(self, **kwargs)

    def _compute_circular_spotify_asynchronous_lightcurve(self, *args, **kwargs):
        pass

    def _compute_eccentric_lightcurve(self, *args, **kwargs):
        # todo: just for testing, remove
        return lc.compute_eccentric_lightcurve(self, **kwargs)

    # ### build methods
    # todo/idea: remove these definitions and call methods from `build` modul

    def build_surface_gravity(self, component: str or list = None, components_distance: float or np.dtype = None):
        return build.build_surface_gravity(self, component, components_distance)

    def build_faces_orientation(self, component: str or list = None, components_distance: float = None):
        return build.build_faces_orientation(self, component, components_distance)

    def build_temperature_distribution(self, component=None, components_distance=None):
        return build.build_temperature_distribution(self, component, components_distance)

    def build_surface_map(self, colormap=None, component=None, components_distance=None, return_map=False):
        return build.build_surface_map(self, colormap, component, components_distance, return_map)

    def build_mesh(self, component=None, components_distance=None, **kwargs):
        return build.build_mesh(self, component, components_distance)

    def build_faces(self, component=None, components_distance=None):
        return build.build_faces(self, component, components_distance)

    def build_surface(self, component=None, components_distance=None, **kwargs):
        return build.build_surface(self, component, components_distance, **kwargs)

    def build_surface_with_no_spots(self, component=None, components_distance=None):
        return build.build_surface_with_no_spots(self, component, components_distance)

    def build_surface_with_spots(self, component=None, components_distance=None):
        return build.build_surface_with_spots(self, component, components_distance)

    # this makes no sence but don't have a time to make it better
    def build_surface_areas(self, component=None):
        return build.compute_all_surface_areas(self, component=component)

    def build(self, component=None, components_distance=None):
        self.build_mesh(component, components_distance)
        self.build_faces(component, components_distance)
        self.build_surface_areas(component)
        self.build_faces_orientation(component, components_distance)
        self.build_surface_gravity(component, components_distance)
        self.build_temperature_distribution(component, components_distance)

    def prepare_system_positions_container(self, orbital_motion, ecl_boundaries):
        return geo.SystemOrbitalPosition(self.primary, self.secondary, self.inclination, orbital_motion, ecl_boundaries)
