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
from engine import units
import scipy
from scipy.spatial import Delaunay
from copy import copy
from scipy.spatial import KDTree

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

        params = {"primary": self.primary, "secondary": self.secondary}
        params.update(**kwargs)
        self._params_validity_check(**params)
        # set attributes and test whether all parameters were initialized
        missing_kwargs = []
        for kwarg in BinarySystem.KWARGS:
            if kwarg not in kwargs:
                missing_kwargs.append("`{}`".format(kwarg))
                self._logger.error("Property {} "
                                   "of class instance {} was not initialized".format(kwarg, BinarySystem.__name__))
            else:
                setattr(self, kwarg, kwargs[kwarg])

        # will show all missing kwargs from KWARGS
        if missing_kwargs:
            raise ValueError('Mising argument(s): {} in class instance {}'.format(', '.join(missing_kwargs),
                                                                                  BinarySystem.__name__))

        # calculation of dependent parameters
        self._semi_major_axis = self.calculate_semi_major_axis()

        # orbit initialisation
        self.init_orbit()

        # binary star morphology estimation
        self._morphology = self._estimate_morphology()

        # todo: compute and assign to all radii values to both components

        # evaluate spots of both components
        self._evaluate_spots(components_distance=1.0)

    def _evaluate_spots(self, components_distance):
        """
        compute points of each spots and assigns values to spot container instance

        :param components_distance: float
        :return:
        """
        def solver_condition(x, *_args, **_kwargs):
            x, _, _ = utils.spherical_to_cartesian(x, _args[1], _args[2])
            x = x if component == "primary" else components_distance - x
            # ignore also spots where one of points is situated just on the neck
            if (component == "primary" and x >= neck_position) or (component == "secondary" and x <= neck_position):
                return False
            return True

        fns = {"primary": self.potential_primary_fn, "secondary": self.potential_secondary_fn}

        neck_position = 1e10
        # in case of wuma system, get separation and make additional test of location of each point (if primary
        # spot doesn't intersect with secondary, if does, then such spot will be skiped completly)
        if self.morphology == "over-contact":
            neck_position = self.calculate_neck_position()

        for component, fn in fns.items():
            self._logger.info("Evaluating spots for {} component".format(component))
            component_instance = getattr(self, component)

            if not component_instance.spots:
                self._logger.info("No spots to evaluate for {} component. Skipping.".format(component))
                continue

            # iterate over spots
            for spot_index, spot_instance in list(component_instance.spots.items()):
                # lon -> phi, lat -> theta
                lon, lat = spot_instance.longitude, spot_instance.latitude
                alpha, diameter = spot_instance.angular_density, spot_instance.angular_diameter

                # initial containers for current spot
                boundary_points, spot_points = [], []

                # initial radial vector
                radial_vector = np.array([1.0, lon, lat])  # unit radial vector to the center of current spot
                center_vector = np.array(utils.spherical_to_cartesian(1.0, lon, lat))

                args, use = (components_distance, radial_vector[1], radial_vector[2]), False
                solution, use = self.solver(fn, solver_condition, *args)

                if not use:
                    # in case of spots, each point should be usefull, otherwise remove spot from
                    # component spot list and skip current spot computation
                    self._logger.info("Center of spot {} doesn't satisfy reasonable conditions and "
                                      "entire spot will be omitted. Your spot probably lies close to the "
                                      "neck.".format(spot_instance.kwargs_serializer()))

                    component_instance.remove_spot(spot_index=spot_index)
                    continue

                spot_center_r = solution
                spot_center = np.array(utils.spherical_to_cartesian(spot_center_r, lon, lat))

                # compute euclidean distance of two points on spot (x0)
                # we have to obtain distance between center and 1st point in 1st ring of spot
                args, use = (components_distance, lon, lat + alpha), False
                solution, use = self.solver(fn, solver_condition, *args)

                if not use:
                    # in case of spots, each point should be usefull, otherwise remove spot from
                    # component spot list and skip current spot computation
                    self._logger.info("First ring of spot {} doesn't satisfy reasonable conditions and "
                                      "entire spot will be omitted".format(spot_instance.kwargs_serializer()))

                    component_instance.remove_spot(spot_index=spot_index)
                    continue
                x0 = np.sqrt(spot_center_r ** 2 + solution ** 2 - (2.0 * spot_center_r * solution * np.cos(alpha)))

                # number of points in latitudal direction
                num_radial = int((diameter * 0.5) // alpha)
                thetas = np.linspace(lat, lat + (diameter * 0.5), num=num_radial, endpoint=True)

                num_azimuthal = [1 if i == 0 else int(i * 2.0 * np.pi * x0 // x0) for i in range(0, len(thetas))]
                deltas = [np.linspace(0., c.FULL_ARC, num=num, endpoint=False) for num in num_azimuthal]

                # todo: add condition to die
                try:
                    for theta_index, theta in enumerate(thetas):
                        # first point of n-th ring of spot (counting start from center)
                        default_spherical_vector = [1.0, lon % c.FULL_ARC, theta]

                        for delta_index, delta in enumerate(deltas[theta_index]):
                            # rotating default spherical vector around spot center vector and thus generating concentric
                            # circle of points around centre of spot
                            delta_vector = utils.arbitrary_rotation(theta=delta, omega=center_vector,
                                                                    vector=utils.spherical_to_cartesian(
                                                                        default_spherical_vector[0],
                                                                        default_spherical_vector[1],
                                                                        default_spherical_vector[2]),
                                                                    degrees=False)

                            spherical_delta_vector = utils.cartesian_to_sphetical(delta_vector[0],
                                                                                  delta_vector[1],
                                                                                  delta_vector[2])

                            args = (components_distance, spherical_delta_vector[1], spherical_delta_vector[2])
                            solution, use = self.solver(fn, solver_condition, *args)

                            if not use:
                                raise StopIteration

                            spot_point = np.array(utils.spherical_to_cartesian(solution, spherical_delta_vector[1],
                                                                               spherical_delta_vector[2]))
                            spot_points.append(spot_point)

                            if theta_index == len(thetas) - 1:
                                boundary_points.append(spot_point)

                except StopIteration:
                    self._logger.info("Any point of spot {} doesn't satisfy reasonable conditions and "
                                      "entire spot will be omitted.".format(spot_instance.kwargs_serializer()))
                    # what about this?: component_instance.remove_spot(spot_index=spot_index)
                    return

                boundary_com = np.sum(np.array(boundary_points), axis=0) / len(boundary_points)
                boundary_com = utils.cartesian_to_sphetical(*boundary_com)
                solution, _ = self.solver(fn, solver_condition, *(components_distance, boundary_com[1], boundary_com[2]))
                boundary_center = np.array(utils.spherical_to_cartesian(solution, boundary_com[1], boundary_com[2]))

                # first point will be always barycenter of boundary
                spot_points[0] = boundary_center

                # max size from barycenter of boundary to boundary
                # todo: make sure this value is correct = make an unittests for spots
                spot_instance.max_size = max([np.linalg.norm(np.array(boundary_center) - np.array(b))
                                              for b in boundary_points])

                if component == "primary":
                    spot_instance.points = np.array(spot_points)
                    spot_instance.boundary = np.array(boundary_points)
                    spot_instance.boundary_center = np.array(boundary_center)
                    spot_instance.center = np.array(spot_center)
                else:
                    spot_instance.points = np.array([np.array([components_distance - point[0], -point[1], point[2]])
                                                     for point in spot_points])

                    spot_instance.boundary = np.array([np.array([components_distance - point[0], -point[1], point[2]])
                                                       for point in boundary_points])

                    spot_instance.boundary_center = np.array([components_distance - boundary_center[0],
                                                             -boundary_center[1], boundary_center[2]])

                    spot_instance.center = np.array([components_distance - spot_center[0], -spot_center[1][1],
                                                    spot_center[1][2]])

                spot_instance.normals = self.calculate_potential_gradient(component=component,
                                                                          components_distance=components_distance,
                                                                          points=np.array(spot_instance.points))

    def solver(self, fn, condition, *args, **kwargs):
        """
        will solve fn implicit function taking args by using scipy.optimize.fsolve method and return
        solution if satisfy conditional function

        :param fn: function
        :param condition: function
        :param args: tuple
        :return: float (np.nan), bool
        """
        solution, use = np.nan, False
        scipy_solver_init_value = np.array([1. / 10000.])
        try:
            solution, _, ier, _ = scipy.optimize.fsolve(fn, scipy_solver_init_value, full_output=True, args=args,
                                                        xtol=1e-12)
            if ier == 1 and not np.isnan(solution[0]):
                solution = solution[0]
                use = True if 1 > solution > 0 else False
        except Exception as e:
            self._logger.debug("Attempt to solve function {} finished w/ exception: {}".format(fn.__name__, str(e)))
            use = False

        return (solution, use) if condition(solution, *args, **kwargs) else (np.nan, False)

    def _params_validity_check(self, **kwargs):

        if not isinstance(kwargs.get("primary"), Star):
            raise TypeError("Primary component is not instance of class {}".format(Star.__name__))

        if not isinstance(kwargs.get("secondary"), Star):
            raise TypeError("Secondary component is not instance of class {}".format(Star.__name__))

        # checking if stellar components have all necessary parameters initialised
        # tehese parameters are not mandatory in single star system, so validity check cannot be provided
        # on whole set of KWARGS in star object
        star_mandatory_kwargs = ['mass', 'surface_potential', 'synchronicity']
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

    def _estimate_morphology(self):
        """
        Setup binary star class property `morphology`
        :return:
        """
        PRECISSION = 1e-8

        # fixme: probably should be better to create a new function like setup_critical_potentials()

        primary_critical_potential = self.critical_potential(component="primary",
                                                             components_distance=1-self.eccentricity)
        secondary_critical_potential = self.critical_potential(component="secondary",
                                                               components_distance=1-self.eccentricity)

        self.primary.critical_surface_potential = primary_critical_potential
        self.secondary.critical_surface_potential = secondary_critical_potential

        if self.primary.synchronicity == 1 and self.secondary.synchronicity == 1 and self.eccentricity == 0.0:
            lp = self.libration_potentials()
            self._primary_filling_factor = (lp[1] - self.primary.surface_potential) / (lp[1] - lp[2])
            self._secondary_filling_factor = (lp[1] - self.secondary.surface_potential) / (lp[1] - lp[2])

            if ((1 > self.secondary_filling_factor > 0) or (1 > self.primary_filling_factor > 0)) and \
                    (self.primary_filling_factor - self.secondary_filling_factor > PRECISSION):
                raise ValueError("Detected over-contact binary system, but potentials of components are not the same.")
            if self.primary_filling_factor > 1 or self.secondary_filling_factor > 1:
                raise ValueError("Non-Physical system: primary_filling_factor or "
                                 "secondary_filling_factor is greater then 1. Filling factor is obtained as following:"
                                 "(Omega_{inner} - Omega) / (Omega_{inner} - Omega_{outter})")

            if (abs(self.primary_filling_factor) < PRECISSION and self.secondary_filling_factor < 0) or (
                            self.primary_filling_factor < 0 and abs(self.secondary_filling_factor) < PRECISSION):
                return "semi-detached"
            elif self.primary_filling_factor < 0 and self.secondary_filling_factor < 0:
                return "detached"
            elif 1 >= self.primary_filling_factor > 0:
                return "over-contact"
            elif self.primary_filling_factor > 1 or self.secondary_filling_factor > 1:
                raise ValueError("Non-Physical system: potential of components is to low.")

        else:
            self._primary_filling_factor, self._secondary_filling_factor = None, None
            if abs(self.primary.surface_potential - primary_critical_potential) < PRECISSION and \
               abs(self.secondary.surface_potential - secondary_critical_potential) < PRECISSION:
                return "double-contact"

            elif self.primary.surface_potential > primary_critical_potential and (
                        self.secondary.surface_potential > secondary_critical_potential):
                return "detached"

            else:
                raise ValueError("Non-Physical system. Change stellar parameters.")

    def init_orbit(self):
        """
        encapsulating orbit class into binary system

        :return:
        """
        self._logger.debug("Re/Initializing orbit in class instance {} ".format(BinarySystem.__name__))
        orbit_kwargs = {key: getattr(self, key) for key in Orbit.KWARGS}
        self._orbit = Orbit(**orbit_kwargs)

    @property
    def primary_filling_factor(self):
        """
        filling factor for primary components

        :return: (np.)float
        """
        return self._primary_filling_factor

    @property
    def secondary_filling_factor(self):
        """
        fillinf catro for secondary component

        :return: (np.)float
        """
        return self._secondary_filling_factor

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
            self._argument_of_periastron = np.float64(argument_of_periastron.to(units.ARC_UNIT))
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
        calculates length semi major axis usin 3rd kepler law

        :return: np.float
        """
        period = (self._period * units.PERIOD_UNIT).to(u.s)
        return (c.G * (self.primary.mass + self.secondary.mass) * period ** 2 / (4 * c.PI ** 2)) ** (1.0 / 3)

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

    def potential_value_primary(self, radius, *args):
        """
        calculates modified kopal potential from point of view of primary component

        :param radius: (np.)float; spherical variable
        :param args: ((np.)float, (np.)float, (np.)float); (component distance, azimuthal angle, polar angle)
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

    def potential_value_primary_cylindrical(self, radius, *args):
        """
        calculates modified kopal potential from point of view of primary component in cylindrical coordinates
        r_n, phi_n, z_n, where z_n = x and heads along z axis, this function is intended for generation of ``necks``
        of W UMa systems, therefore components distance = 1 an synchronicity = 1 is assumed

        :param radius: np.float
        :param args: tuple (np.float, np.float) - phi, z (polar coordinates)
        :return:
        """
        phi, z = args

        block_a = 1 / np.power(np.power(z, 2) + np.power(radius, 2), 0.5)
        block_b = self.mass_ratio / np.power(np.power(1 - z, 2) + np.power(radius, 2), 0.5)
        block_c = 0.5 * np.power(self.mass_ratio, 2) / (self.mass_ratio + 1)
        block_d = 0.5 * (self.mass_ratio + 1) * (np.power(self.mass_ratio / (self.mass_ratio + 1) - z, 2)
                                                 + np.power(radius * np.sin(phi), 2))

        return block_a + block_b - block_c + block_d

    def potential_value_secondary(self, radius, *args):
        """
        calculates modified kopal potential from point of view of secondary component

        :param radius: np.float; spherical variable
        :param args: (np.float, np.float, np.float); (component distance, azimutal angle, polar angle)
        :return: np.float
        """
        d, phi, theta = args
        inverted_mass_ratio = 1.0 / self.mass_ratio

        block_a = 1.0 / radius
        block_b = inverted_mass_ratio / (np.sqrt(np.power(d, 2) + np.power(radius, 2) - (
            2.0 * radius * np.cos(phi) * np.sin(theta) * d)))
        block_c = (inverted_mass_ratio * radius * np.cos(phi) * np.sin(theta)) / (np.power(d, 2))
        block_d = 0.5 * np.power(self.secondary.synchronicity, 2) * (1 + inverted_mass_ratio) * np.power(
            radius, 2) * (1 - np.power(np.cos(theta), 2))

        inverse_potential = (block_a + block_b - block_c + block_d) / inverted_mass_ratio + (
            0.5 * ((inverted_mass_ratio - 1) / inverted_mass_ratio))

        return inverse_potential

    def potential_value_secondary_cylindrical(self, radius, *args):
        """
        calculates modified kopal potential from point of view of secondary component in cylindrical coordinates
        r_n, phi_n, z_n, where z_n = x and heads along z axis, this function is intended for generation of ``necks``
        of W UMa systems, therefore components distance = 1 an synchronicity = 1 is assumed

        :param radius: np.float
        :param args: tuple (np.float, np.float) - phi, z (polar coordinates)
        :return:
        """
        phi, z = args
        inverted_mass_ratio = 1.0 / self.mass_ratio

        block_a = 1 / np.power(np.power(z, 2) + np.power(radius, 2), 0.5)
        block_b = inverted_mass_ratio / np.power(np.power(1 - z, 2) + np.power(radius, 2), 0.5)
        block_c = 0.5 * np.power(inverted_mass_ratio, 2) / (inverted_mass_ratio + 1)
        block_d = 0.5 * (inverted_mass_ratio + 1) * (np.power(inverted_mass_ratio / (inverted_mass_ratio + 1) - z, 2)
                                                     + np.power(radius * np.sin(phi), 2))

        return (block_a + block_b - block_c + block_d) / inverted_mass_ratio + (
                0.5 * ((inverted_mass_ratio - 1) / inverted_mass_ratio))

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
            if component == "primary":
                args = components_distance, 0.0, c.HALF_PI
                return abs(self.potential_value_primary(solution, *args))
            else:
                args = (components_distance, 0.0, c.HALF_PI)
                return abs(self.potential_value_secondary(components_distance - solution, *args))
        else:
            raise ValueError("Iteration process to solve critical potential seems to lead nowhere (critical potential "
                             "solver has failed).")

    def calculate_potential_gradient(self, component, components_distance, points=None):
        """
        return outter gradients in each point of star surface or defined points

        :param component: str, `primary` or `secondary`
        :param components_distance: float, in SMA distance
        :param points: list/numpy.array
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
                self.mass_ratio + 1) * (components_distance - points[:, 0]) * points[:, 0] + 1 / np.power(
                components_distance, 2)
        else:
            raise ValueError('Invalid value `{}` of argument `component`. Use `primary` or `secondary`.'
                             .format(component))
        domega_dy = - points[:, 1] * (1 / r3 + self.mass_ratio / r_hat3 - F2 * (self.mass_ratio + 1))
        domega_dz = - points[:, 2] * (1 / r3 + self.mass_ratio / r_hat3)
        return -np.column_stack((domega_dx, domega_dy, domega_dz))

    def calculate_face_magnitude_gradient(self, component, components_distance):
        """
        return array of face mean of magnitude gradients

        :param component:
        :param components_distance:
        :return: np.array
        """
        component_instance = getattr(self, component)
        gradients = self.calculate_potential_gradient(component, components_distance)
        domega_dx, domega_dy, domega_dz = gradients[:, 0], gradients[:, 1], gradients[:, 2]
        points_gradients = np.power(np.power(domega_dx, 2) + np.power(domega_dy, 2) + np.power(domega_dz, 2), 0.5)
        return np.mean(points_gradients[component_instance.faces], axis=1)

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

    def calculate_polar_radius(self, component=None, components_distance=None):
        """
        calculates polar radius in the similar manner as in BinarySystem.compute_equipotential_boundary method

        :param component: str - `primary` or `secondary`
        :param components_distance: float
        :return: float - polar radius
        """
        if component == 'primary':
            fn = self.potential_primary_fn
        elif component == 'secondary':
            fn = self.potential_secondary_fn
        else:
            raise ValueError('Invalid value of `component` argument {}. Expecting `primary` or `secondary`.'
                             .format(component))
        args = (components_distance, 0., 0)
        scipy_solver_init_value = np.array([components_distance / 10000.0])
        solution, _, ier, _ = scipy.optimize.fsolve(fn, scipy_solver_init_value,
                                                    full_output=True, args=args, xtol=1e-12)

        # check for regular solution
        if ier == 1 and not np.isnan(solution[0]) and 30 >= solution[0] >= 0:
            return solution[0]
        else:
            raise ValueError('Invalid value of polar radius {} was calculated.'.format(solution))

    def calculate_side_radius(self, component=None, components_distance=None):
        """
        calculates side radius in the similar manner as in BinarySystem.compute_equipotential_boundary method

        :param component: str - `primary` or `secondary`
        :param components_distance: float - photometric phase
        :return: float - polar radius
        """
        if component == 'primary':
            fn = self.potential_primary_fn
        elif component == 'secondary':
            fn = self.potential_secondary_fn
        else:
            raise ValueError('Invalid value of `component` argument {}. Expecting `primary` or `secondary`.'
                             .format(component))
        args = (components_distance, c.HALF_PI, c.HALF_PI)
        scipy_solver_init_value = np.array([components_distance / 10000.0])
        solution, _, ier, _ = scipy.optimize.fsolve(fn, scipy_solver_init_value,
                                                    full_output=True, args=args, xtol=1e-12)

        # check for regular solution
        if ier == 1 and not np.isnan(solution[0]) and 30 >= solution[0] >= 0:
            return solution[0]
        else:
            raise ValueError('Invalid value of polar radius {} was calculated.'.format(solution))

    def compute_equipotential_boundary(self, components_distance, plane):
        """
        compute a equipotential boundary of components (crossection of Hill plane)

        :param components_distance: (np.)float
        :param plane: str; xy, yz, zx
        :return: tuple (np.array, np.array)
        """

        components = ['primary', 'secondary']
        points_primary, points_secondary = [], []
        fn_map = {'primary': self.potential_primary_fn, 'secondary': self.potential_secondary_fn}

        angles = np.linspace(-3*c.HALF_PI, c.HALF_PI, 300, endpoint=True)
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

                scipy_solver_init_value = np.array([components_distance / 10000.0])
                solution, _, ier, _ = scipy.optimize.fsolve(fn_map[component], scipy_solver_init_value,
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
            theta, d = c.HALF_PI, self.orbit.periastron_distance
            if isinstance(radius, (float, int, np.float, np.int)):
                radius = [radius]
            elif not isinstance(radius, (list, np.array)):
                raise ValueError("Incorrect value of variable `radius`")

            p_values = []
            for r in radius:
                phi, r = (0.0, r) if r >= 0 else (c.PI, abs(r))

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

    def mesh_detached(self, component, components_distance):
        """
        creates surface mesh of given binary star component in case of detached (semi-detached) system

        :param component: str - `primary` or `secondary`
        :param components_distance: np.float
        :return: numpy.array - set of points in shape numpy.array([[x1 y1 z1],
                                                                     [x2 y2 z2],
                                                                      ...
                                                                     [xN yN zN]])
        """
        component_instance = getattr(self, component)
        if component_instance.discretization_factor > 90:
            raise ValueError("Invalid value of alpha parameter. Use value less than 90.")

        alpha = np.radians(component_instance.discretization_factor)
        scipy_solver_init_value = np.array([1. / 10000.])

        if component == 'primary':
            fn = self.potential_primary_fn
        elif component == 'secondary':
            fn = self.potential_secondary_fn
        else:
            raise ValueError('Invalid value of `component` argument: `{}`. Expecting '
                             '`primary` or `secondary`.'.format(component))

        # calculating points on equator
        num = int(c.PI // alpha)
        r_eq = []
        phi_eq = np.linspace(0., c.PI, num=num + 1)
        theta_eq = np.array([c.HALF_PI for _ in phi_eq])
        for phi in phi_eq:
            args = (components_distance, phi, c.HALF_PI)
            solution, _, ier, _ = scipy.optimize.fsolve(fn, scipy_solver_init_value, full_output=True, args=args,
                                                        xtol=1e-12)
            r_eq.append(solution[0])
        r_eq = np.array(r_eq)
        x_eq, y_eq, z_eq = utils.spherical_to_cartesian(r_eq, phi_eq, theta_eq)

        # calculating points on phi = 0 meridian
        r_meridian = []
        num = int(c.HALF_PI // alpha)
        phi_meridian = np.array([c.PI for _ in range(num - 1)] + [0 for _ in range(num)])
        theta_meridian = np.concatenate((np.linspace(c.HALF_PI - alpha, alpha, num=num - 1),
                                         np.linspace(0., c.HALF_PI, num=num, endpoint=False)))
        for ii, theta in enumerate(theta_meridian):
            args = (components_distance, phi_meridian[ii], theta)
            solution, _, ier, _ = scipy.optimize.fsolve(fn, scipy_solver_init_value, full_output=True, args=args,
                                                        xtol=1e-12)
            r_meridian.append(solution[0])
        r_meridian = np.array(r_meridian)
        x_meridian, y_meridian, z_meridian = utils.spherical_to_cartesian(r_meridian, phi_meridian, theta_meridian)

        # calculating the rest (quarter) of the surface
        thetas = np.linspace(alpha, c.HALF_PI, num=num, endpoint=False)
        r_q, phi_q, theta_q = [], [], []
        for theta in thetas:
            alpha_corrected = alpha / np.sin(theta)
            num = int(c.PI // alpha_corrected)
            alpha_corrected = c.PI / (num + 1)
            phi_q_add = [alpha_corrected * ii for ii in range(1, num + 1)]
            phi_q += phi_q_add
            for phi in phi_q_add:
                theta_q.append(theta)
                args = (components_distance, phi, theta)
                solution, _, ier, _ = scipy.optimize.fsolve(fn, scipy_solver_init_value, full_output=True, args=args,
                                                            xtol=1e-12)
                r_q.append(solution[0])

        r_q, phi_q, theta_q = np.array(r_q), np.array(phi_q), np.array(theta_q)
        x_q, y_q, z_q = utils.spherical_to_cartesian(r_q, phi_q, theta_q)

        x = np.concatenate((x_eq,  x_eq[1:-1], x_meridian,  x_meridian, x_q,  x_q,  x_q,  x_q))
        y = np.concatenate((y_eq, -y_eq[1:-1], y_meridian,  y_meridian, y_q, -y_q,  y_q, -y_q))
        z = np.concatenate((z_eq,  z_eq[1:-1], z_meridian, -z_meridian, z_q,  z_q, -z_q, -z_q))
        x = -x + components_distance if component == 'secondary' else x
        points = np.column_stack((x, y, z))

        return points

    def calculate_neck_position(self, return_polynomial=False):
        """
        function calculates x-coordinate of the `neck` (the narrowest place) of an over-contact system
        :return: np.float (0.1)
        """
        neck_position = None
        components_distance = 1.0
        components = ['primary', 'secondary']
        points_primary, points_secondary = [], []
        fn_map = {'primary': self.potential_primary_fn, 'secondary': self.potential_secondary_fn}

        # generating only part of the surface that I'm interested in (neck in xy plane for x between 0 and 1)
        angles = np.linspace(0., c.HALF_PI, 100, endpoint=True)
        for component in components:
            for angle in angles:
                args, use = (components_distance, angle, c.HALF_PI), False

                scipy_solver_init_value = np.array([components_distance / 10000.0])
                solution, _, ier, _ = scipy.optimize.fsolve(fn_map[component], scipy_solver_init_value,
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

    def mesh_over_contact(self, component):
        """
        creates surface mesh of given binary star component in case of over-contact system

        :param component: str - `primary` or `secondary`
        :return: numpy.array - set of points in shape numpy.array([[x1 y1 z1],
                                                                     [x2 y2 z2],
                                                                      ...
                                                                     [xN yN zN]])
        """
        component_instance = getattr(self, component)
        if component_instance.discretization_factor > 90:
            raise ValueError("Invalid value of alpha parameter. Use value less than 90.")

        alpha = np.radians(component_instance.discretization_factor)
        scipy_solver_init_value = np.array([1. / 10000.])

        # calculating distance between components
        components_distance = self.orbit.orbital_motion(phase=0)[0][0]

        if component == 'primary':
            fn = self.potential_primary_fn
            fn_cylindrical = self.potential_primary_cylindrical_fn
        elif component == 'secondary':
            fn = self.potential_secondary_fn
            fn_cylindrical = self.potential_secondary_cylindrical_fn
        else:
            raise ValueError('Invalid value of `component` argument: `{}`. '
                             'Expecting `primary` or `secondary`.'.format(component))

        # calculating points on farside equator
        num = int(c.HALF_PI // alpha)
        r_eq = []
        phi_eq = np.linspace(c.HALF_PI, c.PI, num=num + 1)
        theta_eq = np.array([c.HALF_PI for _ in phi_eq])
        for phi in phi_eq:
            args = (components_distance, phi, c.HALF_PI)
            solution, _, ier, _ = scipy.optimize.fsolve(fn, scipy_solver_init_value, full_output=True, args=args,
                                                        xtol=1e-12)
            r_eq.append(solution[0])
        r_eq = np.array(r_eq)
        x_eq, y_eq, z_eq = utils.spherical_to_cartesian(r_eq, phi_eq, theta_eq)

        # calculating points on phi = pi meridian
        r_meridian = []
        num = int(c.HALF_PI // alpha)
        phi_meridian = np.array([c.PI for _ in range(num)])
        theta_meridian = np.linspace(c.HALF_PI - alpha, 0., num=num)
        for ii, theta in enumerate(theta_meridian):
            args = (components_distance, phi_meridian[ii], theta)
            solution, _, ier, _ = scipy.optimize.fsolve(fn, scipy_solver_init_value, full_output=True, args=args,
                                                        xtol=1e-12)
            r_meridian.append(solution[0])
        r_meridian = np.array(r_meridian)
        x_meridian, y_meridian, z_meridian = utils.spherical_to_cartesian(r_meridian, phi_meridian, theta_meridian)

        # calculating points on phi = pi/2 meridian
        r_meridian2 = []
        num = int(c.HALF_PI // alpha) - 1
        phi_meridian2 = np.array([c.HALF_PI for _ in range(num)])
        theta_meridian2 = np.linspace(alpha, c.HALF_PI, num=num, endpoint=False)
        for ii, theta in enumerate(theta_meridian2):
            args = (components_distance, phi_meridian2[ii], theta)
            solution, _, ier, _ = scipy.optimize.fsolve(fn, scipy_solver_init_value, full_output=True, args=args,
                                                        xtol=1e-12)
            r_meridian2.append(solution[0])
        r_meridian2 = np.array(r_meridian2)
        x_meridian2, y_meridian2, z_meridian2 = utils.spherical_to_cartesian(r_meridian2, phi_meridian2,
                                                                             theta_meridian2)

        # calculating the rest of the surface on farside
        thetas = np.linspace(alpha, c.HALF_PI, num=num, endpoint=False)
        r_q, phi_q, theta_q = [], [], []
        for theta in thetas:
            alpha_corrected = alpha / np.sin(theta)
            num = int(c.HALF_PI // alpha_corrected)
            alpha_corrected = c.HALF_PI / (num + 1)
            phi_q_add = [c.HALF_PI + alpha_corrected * ii for ii in range(1, num + 1)]
            phi_q += phi_q_add
            for phi in phi_q_add:
                theta_q.append(theta)
                args = (components_distance, phi, theta)
                solution, _, ier, _ = scipy.optimize.fsolve(fn, scipy_solver_init_value, full_output=True, args=args,
                                                            xtol=1e-12)
                r_q.append(solution[0])
        r_q, phi_q, theta_q = np.array(r_q), np.array(phi_q), np.array(theta_q)
        x_q, y_q, z_q = utils.spherical_to_cartesian(r_q, phi_q, theta_q)

        # generating the neck
        neck_position, neck_polynome = self.calculate_neck_position(return_polynomial=True)
        # lets define cylindrical coordinate system r_n, phi_n, z_n for our neck where z_n = x, phi_n = 0 heads along
        # z axis
        num = 100
        delta_z = alpha * self.calculate_polar_radius(component=component, components_distance=1-self.eccentricity)
        if component == 'primary':
            # position of z_n adapted to the slope of the neck, gives triangles with more similar areas
            x_curve = np.linspace(0., neck_position, num=num, endpoint=True)
            z_curve = np.polyval(neck_polynome, x_curve)
            curve = np.column_stack((x_curve, z_curve))
            neck_lengths = np.sqrt(np.sum(np.diff(curve, axis=0)**2, axis=1))
            neck_length = np.sum(neck_lengths)
            segment = neck_length / (int(neck_length // delta_z) + 1)

            k = 1
            z_ns, line_sum = [], 0.0
            for ii in range(num-1):
                line_sum += neck_lengths[ii]
                if line_sum > k * segment:
                    z_ns.append(x_curve[ii+1])
                    k += 1
            z_ns.append(neck_position)
            z_ns = np.array(z_ns)
            # num = int(neck_position // delta_z) + 1
            # z_ns = np.linspace(delta_z, neck_position, num=num, endpoint=True)
        else:
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
        r_n, phi_n, z_n = [], [], []
        for z in z_ns:
            z_eqn.append(z)
            phi_eqn.append(0.0)
            args = (0.0, z)
            solution, _, ier, _ = scipy.optimize.fsolve(fn_cylindrical, scipy_solver_init_value, full_output=True,
                                                        args=args, xtol=1e-12)
            r_eqn.append(solution[0])

            z_eqn.append(z)
            phi_eqn.append(c.HALF_PI)
            args = (c.HALF_PI, z)
            solution, _, ier, _ = scipy.optimize.fsolve(fn_cylindrical, scipy_solver_init_value, full_output=True,
                                                        args=args, xtol=1e-12)
            r_eqn.append(solution[0])

            num = int(c.HALF_PI * r_eqn[-1] // delta_z)
            start_val = c.HALF_PI / (num + 1)
            phis = np.linspace(start_val, c.HALF_PI, num=num, endpoint=False)
            for phi in phis:
                z_n.append(z)
                phi_n.append(phi)
                args = (phi, z)
                solution, _, ier, _ = scipy.optimize.fsolve(fn_cylindrical, scipy_solver_init_value, full_output=True,
                                                            args=args, xtol=1e-12)
                r_n.append(solution[0])

        r_eqn = np.array(r_eqn)
        z_eqn = np.array(z_eqn)
        phi_eqn = np.array(phi_eqn)
        z_eqn, y_eqn, x_eqn = utils.cylindrical_to_cartesian(r_eqn, phi_eqn, z_eqn)

        r_n = np.array(r_n)
        z_n = np.array(z_n)
        phi_n = np.array(phi_n)
        z_n, y_n, x_n = utils.cylindrical_to_cartesian(r_n, phi_n, z_n)

        x = np.concatenate((x_eq,  x_eq[:-1], x_meridian,  x_meridian, x_meridian2,  x_meridian2,  x_meridian2,
                            x_meridian2, x_q,  x_q,  x_q,  x_q, x_eqn,  x_eqn, x_n,  x_n,  x_n,  x_n))
        y = np.concatenate((y_eq, -y_eq[:-1], y_meridian,  y_meridian, y_meridian2,  y_meridian2, -y_meridian2,
                            -y_meridian2, y_q, -y_q,  y_q, -y_q, y_eqn, -y_eqn, y_n, -y_n, -y_n,  y_n))
        z = np.concatenate((z_eq,  z_eq[:-1], z_meridian, -z_meridian, z_meridian2, -z_meridian2,  z_meridian2,
                            -z_meridian2, z_q,  z_q, -z_q, -z_q, z_eqn, -z_eqn, z_n,  z_n, -z_n, -z_n))
        x = -x + components_distance if component == 'secondary' else x
        points = np.column_stack((x, y, z))

        return points

    def detached_system_surface(self, component):
        """
        calculates surface faces from the given component's points in case of detached or semi-contact system

        :param component: str
        :return: np.array - N x 3 array of vertice indices
        """
        component_instance = getattr(self, component)
        triangulation = Delaunay(component_instance.points)
        triangles_indices = triangulation.convex_hull
        return triangles_indices

    def over_contact_surface(self, component):
        """
        calculates surface faces from the given component's points in case of over-contact system

        :param component: str - `primary` or `secondary`
        :return: np.array - N x 3 array of vertice indices
        """
        component_instance = getattr(self, component)
        neck_x = self.calculate_neck_position()

        # projection of component's far side surface into ``sphere`` with radius r1
        r1 = neck_x  # radius of the sphere and cylinder
        projected_points = []
        if component == 'primary':
            k = r1 / (neck_x + 0.01)
            for point in component_instance.points:
                if point[0] <= 0:
                    projected_points.append(r1 * point / np.linalg.norm(point))
                else:
                    r = (r1**2 - (k * point[0])**2)**0.5
                    length = np.linalg.norm(point[1:])
                    new_point = np.array([point[0], r * point[1] / length, r * point[2] / length])
                    projected_points.append(new_point)
        else:
            for point in component_instance.points:
                if point[0] >= 1:
                    point_copy = np.array(point)
                    point_copy[0] -= 1
                    new_val = r1 * point_copy / np.linalg.norm(point_copy)
                    new_val[0] += 1
                    projected_points.append(new_val)
                else:
                    k = r1 / ((1 - neck_x) + 0.01)
                    r = (r1**2 - (k * (1 - point[0]))**2)**0.5
                    length = np.linalg.norm(point[1:])
                    new_point = np.array([point[0], r * point[1] / length, r * point[2] / length])
                    projected_points.append(new_point)

        projected_points = np.array(projected_points)

        # triangulation of this now convex object
        triangulation = Delaunay(projected_points)
        triangles_indices = triangulation.convex_hull

        # removal of faces on top of the neck
        new_triangles_indices = []
        for indices in triangles_indices:
            min_x = min([component_instance.points[ii, 0] for ii in indices])
            max_x = max([component_instance.points[ii, 0] for ii in indices])
            if abs(max_x - min_x) > 1e-8:
                new_triangles_indices.append(indices)
            elif not 0 < min_x < 1:
                new_triangles_indices.append(indices)

        return np.array(new_triangles_indices)

    def build_mesh(self, component, components_distance=None):
        if components_distance is None:
            components_distance = 1 - self.eccentricity
        component_instance = getattr(self, component)
        component_instance.points = self.mesh_over_contact(component=component) if self.morphology == 'over-contact' \
            else self.mesh_detached(component=component, components_distance=components_distance)

    def build_surface(self, component='None'):
        component_instance = getattr(self, component)
        component_instance.faces = self.over_contact_surface(component=component) if self.morphology == 'over-contact' \
            else self.detached_system_surface(component=component)

    def evaluate_normals(self, component, component_distance):
        """
        evaluate normals for both components using potential gradient (useful before triangulation)

        :param component:
        :param component_distance:
        :return:
        """
        component_instance = getattr(self, component)
        component_instance.normals = self.calculate_potential_gradient(component=component,
                                                                       components_distance=component_distance)

    def build_colormap(self, component=None, components_distance=None):
        """
        auxiliary function for plot function with descriptor value `surface` in case of temperature colormap turned on

        :param component: str - `primary` or `secondary`
        :param components_distance: float
        :return:
        """
        component_instance = getattr(self, component)
        if component_instance.areas is None:
            component_instance.areas = component_instance.calculate_areas()
        if component_instance.polar_radius is None:
            component_instance.polar_radius = self.calculate_polar_radius(component=component,
                                                                          components_distance=components_distance)
        if component_instance.potential_gradients is None:
            component_instance.potential_gradients = \
                self.calculate_face_magnitude_gradient(component=component, components_distance=components_distance)
            component_instance.polar_potential_gradient = \
                self.calculate_polar_potential_gradient_magnitude(component=component,
                                                                  components_distance=components_distance)
        if component_instance.temperatures is None:
            component_instance.temperatures = component_instance.calculate_effective_temperatures()

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

        # fixme: alpha argument is not used anymore

        if descriptor == 'orbit':
            KWARGS = ['start_phase', 'stop_phase', 'number_of_points', 'axis_unit', 'frame_of_reference']
            utils.invalid_kwarg_checker(kwargs, KWARGS, BinarySystem.plot)

            method_to_call = graphics.orbit
            start_phase = 0 if 'start_phase' not in kwargs else kwargs['start_phase']
            stop_phase = 1.0 if 'stop_phase' not in kwargs else kwargs['stop_phase']
            number_of_points = 300 if 'number_of_points' not in kwargs else kwargs['number_of_points']

            if 'axis_unit' not in kwargs:
                kwargs['axis_unit'] = u.solRad
            elif kwargs['axis_unit'] == 'dimensionless':
                kwargs['axis_unit'] = u.dimensionless_unscaled

            if 'frame_of_reference' not in kwargs:
                kwargs['frame_of_reference'] = 'primary_component'

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
            x, y = utils.polar_to_cartesian(radius=radius, phi=azimuth - c.PI / 2.0)
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

            if 'phase' not in kwargs:
                kwargs['phase'] = 0
            if 'plane' not in kwargs:
                kwargs['plane'] = 'xy'

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
            KWARGS = ['phase', 'components_to_plot']
            utils.invalid_kwarg_checker(kwargs, KWARGS, BinarySystem.plot)

            method_to_call = graphics.binary_mesh

            if 'phase' not in kwargs:
                kwargs['phase'] = 0
            if 'components_to_plot' not in kwargs:
                kwargs['components_to_plot'] = 'both'

            components_distance = self.orbit.orbital_motion(phase=kwargs['phase'])[0][0]

            if kwargs['components_to_plot'] in ['primary', 'both']:
                if self.primary.points is None:
                    self.build_mesh(component='primary', components_distance=components_distance)
                kwargs['points_primary'] = self.primary.points

            if kwargs['components_to_plot'] in ['secondary', 'both']:
                if self.secondary.points is None:
                    self.build_mesh(component='secondary', components_distance=components_distance)
                kwargs['points_secondary'] = self.secondary.points

        elif descriptor == 'surface':
            KWARGS = ['phase', 'components_to_plot', 'normals', 'edges', 'colormap']
            utils.invalid_kwarg_checker(kwargs, KWARGS, BinarySystem.plot)

            method_to_call = graphics.binary_surface

            if 'phase' not in kwargs:
                kwargs['phase'] = 0
            if 'components_to_plot' not in kwargs:
                kwargs['components_to_plot'] = 'both'
            if 'normals' not in kwargs:
                kwargs['normals'] = False
            if 'edges' not in kwargs:
                kwargs['edges'] = False
            if 'colormap' not in kwargs:
                kwargs['colormap'] = None

            components_distance = self.orbit.orbital_motion(phase=kwargs['phase'])[0][0]

            if kwargs['components_to_plot'] in ['primary', 'both']:
                if self.primary.points is None:
                    self.build_mesh(component='primary', components_distance=components_distance)
                kwargs['points_primary'] = self.primary.points
                if self.primary.faces is None:
                    self.build_surface(component='primary')
                kwargs['primary_triangles'] = self.primary.faces

                if kwargs['normals']:
                    kwargs['primary_centres'] = self.primary.calculate_surface_centres(
                        kwargs['points_primary'], kwargs['primary_triangles'])
                    kwargs['primary_arrows'] = self.primary.calculate_normals(
                        kwargs['points_primary'], kwargs['primary_triangles'])

                if kwargs['colormap'] == 'temperature':
                    self.build_colormap(component='primary', components_distance=components_distance)
                    kwargs['primary_cmap'] = self.primary.temperatures

            if kwargs['components_to_plot'] in ['secondary', 'both']:
                if self.secondary.points is None:
                    self.build_mesh(component='secondary', components_distance=components_distance)
                kwargs['points_secondary'] = self.secondary.points
                if self.secondary.faces is None:
                    self.build_surface(component='secondary')
                kwargs['secondary_triangles'] = self.secondary.faces

                if kwargs['normals']:
                    kwargs['secondary_centres'] = self.secondary.calculate_surface_centres(
                        kwargs['points_secondary'], kwargs['secondary_triangles'])
                    kwargs['secondary_arrows'] = self.secondary.calculate_normals(
                        kwargs['points_secondary'], kwargs['secondary_triangles'])

                if kwargs['colormap'] == 'temperature':
                    self.build_colormap(component='secondary', components_distance=components_distance)
                    kwargs['secondary_cmap'] = self.secondary.temperatures

        else:
            raise ValueError("Incorrect descriptor `{}`".format(descriptor))

        method_to_call(**kwargs)

    def surface(self, component):
        # todo: add info
        """

        :param component: specify component, use `primary` or `secondary`
        :type: str
        :return:
        """

        if component not in ["primary", "secondary"]:
            raise ValueError("Incorrect component value, `primary` or `secondary` allowed.")
        component_instance = getattr(self, component)

        # build surface if there is no spot specified
        if not component_instance.spots:
            self.build_surface(component)
            return

        vertices_map = [{"type": "object", "enum": -1} for _ in component_instance.points]
        points = copy(component_instance.points)
        normals = copy(component_instance.normals)

        # average spacing of component surface points
        avsp = utils.average_spacing(data=component_instance.points, neighbours=6)

        for spot_index, spot in component_instance.spots.items():
            avsp_spot = utils.average_spacing(data=spot.points, neighbours=6)
            vertices_to_remove, vertices_test = [], []

            # find nearest points to spot alt center
            tree = KDTree(points)
            distances, indices = tree.query(spot.boundary_center, k=len(points))

            max_dist_to_object_point = spot.max_size + (0.25 * avsp)
            max_dist_to_spot_point = spot.max_size + (0.1 * avsp_spot)

            for dist, ix in zip(distances, indices):
                if dist > max_dist_to_object_point:
                    # break, because distancies are ordered by size, so there is no more points of object that
                    # have to be removed
                    break

                # distance exeption for spots points
                # we keep such point [it is point in innner ring]
                if vertices_map[ix]["type"] == "spot" and dist > max_dist_to_spot_point:
                    continue
                # norms 0 belong to boundary center
                if np.dot(spot.normals[0], normals[ix]) > 0:
                    vertices_to_remove.append(ix)

            # simplices of target object for testing whether point lying inside or not of spot boundary
            vertices_to_remove = list(set(vertices_to_remove))

            # points, norms and vertices_map update
            if len(vertices_to_remove) != 0:
                # test if index to remove from all current points from vertices_map belongs to any of spots
                spot_indices, star_indices = [], []
                for item in vertices_to_remove:
                    # that cannot occurred in firt step of loop, since there is no spot
                    if vertices_map[item]["type"] == "spot":
                        spot_indices.append(item)
                    else:
                        star_indices.append(item)

                _points, _normals = [], []
                _vertices_map = {}
                m_ix = 0

                for ix, vertex, norm in list(zip(range(0, len(points)), points, normals)):
                    if ix in vertices_to_remove:
                        # skip point if is marked for removal
                        continue

                    # append only points of currrent object that do not intervent to spot
                    # [current, since there should be already spot from previous iteration step]
                    _points.append(vertex)
                    _normals.append(norm)

                    _vertices_map[m_ix] = {"type": vertices_map[ix]["type"], "enum": vertices_map[ix]["enum"]}
                    m_ix += 1

                shift = len(_points)
                for i, vertex, norm in list(zip(range(shift, shift + len(spot.points)), spot.points, spot.normals)):
                    _points.append(vertex)
                    _normals.append(norm)
                    _vertices_map[i] = {"type": "spot", "enum": spot_index}

                points = copy(_points)
                vertices_map = copy(_vertices_map)
                normals = copy(_normals)

                del (_points, _vertices_map, _normals)

        points, normals = np.array(points), np.array(normals)
        component_instance.points = np.array(points)

        # triangulation process
        self.build_surface(component)

        spots_instance_indices = list(set([vertices_map[ix]["enum"]
                                      for ix in vertices_map if vertices_map[ix]["type"] == "spot"]))

        model = {"object": [], "spots": {}}
        spot_candidates = {"simplex": {}, "com": {}, "3rd_enum": {}, "ix": {}}

        # init variables
        for spot_index in spots_instance_indices:
            model["spots"][spot_index] = []
            for key in ["com", "3rd_enum", "ix"]:
                spot_candidates[key][spot_index] = []

        # iterate over triagnulation
        # simplex (2d simplex over triangulatio e.g. [100, 25, 36]), I mentioned 2d, since in 3d, simplex is tetrahedron
        # face (point representation of triangle (contain real coordinates, not just indices))
        for simplex, face, ix in list(zip(component_instance.faces,
                                          component_instance.points[component_instance.faces],
                                          range(component_instance.faces.shape[0]))):

            # test if each point belongs to spot
            if vertices_map[simplex[0]]["type"] == "spot" and vertices_map[simplex[1]]["type"] == "spot" \
                    and vertices_map[simplex[2]]["type"] == "spot":

                # if each point belongs to the same spot, then it is for sure face of that spot
                if vertices_map[simplex[0]]["enum"] == vertices_map[simplex[1]]["enum"] == vertices_map[simplex[2]]["enum"]:
                    model["spots"][vertices_map[simplex[0]]["enum"]].append(np.array(simplex))

                else:
                    # if at least one of points of face belongs to different spot, we have to test
                    # which one of those spots current face belongs to

                    reference_to_spot, trd_enum = None, None
                    # variable trd_enum is enum index of 3rd corner of face;

                    if vertices_map[simplex[-1]]["enum"] == vertices_map[simplex[0]]["enum"]:
                        reference_to_spot = vertices_map[simplex[-1]]["enum"]
                        trd_enum = vertices_map[simplex[1]]["enum"]
                    elif vertices_map[simplex[0]]["enum"] == vertices_map[simplex[1]]["enum"]:
                        reference_to_spot = vertices_map[simplex[0]]["enum"]
                        trd_enum = vertices_map[simplex[-1]]["enum"]
                    elif vertices_map[simplex[1]]["enum"] == vertices_map[simplex[-1]]["enum"]:
                        reference_to_spot = vertices_map[simplex[1]]["enum"]
                        trd_enum = vertices_map[simplex[0]]["enum"]

                    if reference_to_spot is not None:
                        spot_candidates["com"][reference_to_spot].append(np.average(face, axis=0))
                        spot_candidates["3rd_enum"][reference_to_spot].append(trd_enum)
                        spot_candidates["ix"][reference_to_spot].append(ix)

            # if at least one of points belongs to star body, then it is for sure star body face
            elif vertices_map[simplex[0]]["type"] == "t_object" or vertices_map[simplex[1]]["type"] == "t_object" \
                    or vertices_map[simplex[2]]["type"] == "t_object":
                model["object"].append(np.array(simplex))
            else:
                model["object"].append(np.array(simplex))

        if spot_candidates["com"]:
            for spot_index in spot_candidates["com"].keys():
                # get center and size of current spot candidate
                center, size = component_instance.spots[spot_index].boundary_center, \
                               component_instance.spots[spot_index].max_size

                # compute distance of all center of mass of faces of current
                # spot candidate to the center of this candidate
                dists = [np.linalg.norm(np.array(com) - np.array(center)) for com in spot_candidates["com"][spot_index]]

                # test if dist is smaller as current spot size;
                # if dist is smaller, then current face belongs to spots otherwise face belongs to t_object itself

                for idx, dist in enumerate(dists):
                    simplex_index = spot_candidates["ix"][spot_index][idx]
                    if dist < size:
                        model["spots"][spot_index].append(np.array(component_instance.faces[simplex_index]))
                    else:
                        # make the same computation for 3rd vertex of face
                        # it might be confusing, but spot candidate is spot where 2 of 3 vertex of one face belong to first spot,
                        # and the 3rd index belongs to another (neighbour) spot
                        # it has to be alos tested, whether face finally do not belongs to spot candidate;

                        trd_spot_index = spot_candidates["3rd_enum"][spot_index][idx]

                        trd_center = component_instance.spots[trd_spot_index].boundary_center
                        trd_size = component_instance.spots[trd_spot_index].max_size

                        com = spot_candidates["com"][spot_index][idx]
                        dist = np.linalg.norm(np.array(com) - np.array(trd_center))

                        if dist < trd_size:
                            model["spots"][trd_spot_index].append(np.array(component_instance.faces[simplex_index]))
                        else:
                            model["object"].append(np.array(component_instance.faces[simplex_index]))

        # remove spots that are totaly overlaped
        for spot_index, _ in list(component_instance.spots.items()):
            if spot_index not in spots_instance_indices:
                self._logger.warning("Spot with index {} doesn't contain any face and will be removed "
                                     "from component {} spot list".format(spot_index, component_instance.name))
                component_instance.remove_spot(spot_index=spot_index)
            else:
                self._logger.debug(
                    "Changing value of parameter points of spot {} / "
                    "component {}".format(spot_index, component_instance.name))
                # get points currently belong to the given spot
                indices = list(set(np.array(model["spots"][spot_index]).flatten()))
                component_instance.spots[spot_index].points = np.array(component_instance.points[indices])

                self._logger.debug(
                    "Changing value of parameter faces of spot {} / "
                    "component {}".format(spot_index, component_instance.name))
                component_instance.spots[spot_index].faces = model["spots"][spot_index]
                remap_dict = {idx[1]: idx[0] for idx in enumerate(indices)}
                component_instance.spots[spot_index].faces = \
                    np.array(utils.remap(component_instance.spots[spot_index].faces, remap_dict))

        self._logger.debug("Changing value of parameter points of object {}".format(component_instance.name))
        indices = list(set(np.array(model["object"]).flatten()))
        component_instance.points = component_instance.points[indices]

        self._logger.debug("Changing value of parameter faces of object {}".format(component_instance.name))
        remap_dict = {idx[1]: idx[0] for idx in enumerate(indices)}

        component_instance.faces = np.array(utils.remap(model["object"], remap_dict))

        # todo: recompute normals????

        # graphics.binary_surface(components_to_plot="primary",
        #                         primary_triangles=component_instance.faces,
        #                         points_primary=component_instance.points, edges=True)

    def is_property(self, kwargs):
        """
        method for checking if keyword arguments are valid properties of this class

        :param kwargs: dict
        :return:
        """
        is_not = ['`{}`'.format(k) for k in kwargs if k not in dir(self)]
        if is_not:
            raise AttributeError('Arguments {} are not valid {} properties.'.format(', '.join(is_not),
                                                                                    BinarySystem.__name__))
