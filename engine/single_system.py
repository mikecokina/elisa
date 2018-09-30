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
from copy import copy

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : [%(levelname)s] : %(name)s : %(message)s')


class SingleSystem(System):
    KWARGS = ['star', 'gamma', 'inclination', 'rotation_period', 'polar_log_g']
    OPTIONAL_KWARGS = []
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

        for kwarg in SingleSystem.OPTIONAL_KWARGS:
            if kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])

        # setting of optional parameters

        # calculation of dependent parameters
        self._angular_velocity = self.angular_velocity(self.rotation_period)
        self.star._polar_log_g = self.polar_log_g
        self.star._polar_gravity_acceleration = np.power(10, self.polar_log_g)  # surface polar gravity
        # this is also check if star surface is closed
        self.init_radii()
        self._evaluate_spots()

    def init_radii(self):
        """
        auxiliary function for calculation of important radii
        :return:
        """
        self._logger.debug('Calculating polar radius.')
        self.star._polar_radius = self.calculate_polar_radius()
        self._logger.debug('Calculating surface potential.')
        args = 0,
        self.star._surface_potential = self.surface_potential(self.star.polar_radius, args)[0]
        self._logger.debug('Calculating equatorial radius.')
        self.star._equatorial_radius = self.calculate_equatorial_radius()

    def init(self):
        """
        function to reinitialize SingleSystem class instance after changing parameter(s) of binary system using setters

        :return:
        """
        self._logger.info('Reinitialising class instance {}'.format(SingleSystem.__name__))
        self.__init__(**self._kwargs_serializer())

    def _kwargs_serializer(self):
        """
        creating dictionary of keyword arguments of SingleSystem class in order to be able to reinitialize the class
        instance in init()

        :return: dict
        """
        serialized_kwargs = {}
        for kwarg in self.KWARGS:
            serialized_kwargs[kwarg] = getattr(self, kwarg)
        return serialized_kwargs

    def _evaluate_spots(self):
        """
        compute points of each spots and assigns values to spot container instance

        :return:
        """

        # fixme: it's not crutial, but this function and same function in binary system should on the same place
        def solver_condition(x, *_args, **_kwargs):
            return True

        self._logger.info("Evaluating spots.")

        if not self.star.spots:
            self._logger.info("No spots to evaluate.")
            return

        # iterate over spots
        for spot_index, spot_instance in list(self.star.spots.items()):
            # lon -> phi, lat -> theta
            lon, lat = spot_instance.longitude, spot_instance.latitude
            if spot_instance.angular_density is None:
                self._logger.debug('Angular density of the spot {0} was not supplied and discretization factor of star '
                                   '{1} was used.'.format(spot_index, self.star.discretization_factor))
                spot_instance.angular_density = 0.9 * self.star.discretization_factor * U.ARC_UNIT
            alpha, diameter = spot_instance.angular_density, spot_instance.angular_diameter

            # initial containers for current spot
            boundary_points, spot_points = [], []

            # initial radial vector
            radial_vector = np.array([1.0, lon, lat])  # unit radial vector to the center of current spot
            center_vector = utils.spherical_to_cartesian([1.0, lon, lat])

            args, use = (radial_vector[2],), False

            solution, use = self._solver(self.potential_fn, solver_condition, *args)

            if not use:
                # in case of spots, each point should be usefull, otherwise remove spot from
                # component spot list and skip current spot computation
                self._logger.info("Center of spot {} doesn't satisfy reasonable conditions and "
                                  "entire spot will be omitted.".format(spot_instance.kwargs_serializer()))

                self.star.remove_spot(spot_index=spot_index)
                continue

            spot_center_r = solution
            spot_center = utils.spherical_to_cartesian([spot_center_r, lon, lat])

            # compute euclidean distance of two points on spot (x0)
            # we have to obtain distance between center and 1st point in 1st ring of spot
            args, use = (lat + alpha,), False
            solution, use = self._solver(self.potential_fn, solver_condition, *args)
            if not use:
                # in case of spots, each point should be usefull, otherwise remove spot from
                # component spot list and skip current spot computation
                self._logger.info("First ring of spot {} doesn't satisfy reasonable conditions and "
                                  "entire spot will be omitted".format(spot_instance.kwargs_serializer()))

                self.star.remove_spot(spot_index=spot_index)
                continue

            x0 = np.sqrt(spot_center_r ** 2 + solution ** 2 - (2.0 * spot_center_r * solution * np.cos(alpha)))

            # number of points in latitudal direction
            num_radial = int(np.round((diameter * 0.5) / alpha)) + 1
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
                                                                    default_spherical_vector),
                                                                degrees=False)

                        spherical_delta_vector = utils.cartesian_to_spherical(delta_vector)

                        args = (spherical_delta_vector[2],)
                        solution, use = self._solver(self.potential_fn, solver_condition, *args)

                        if not use:
                            self.star.remove_spot(spot_index=spot_index)
                            raise StopIteration

                        spot_point = utils.spherical_to_cartesian([solution, spherical_delta_vector[1],
                                                                   spherical_delta_vector[2]])
                        spot_points.append(spot_point)

                        if theta_index == len(thetas) - 1:
                            boundary_points.append(spot_point)

            except StopIteration:
                self._logger.info("At least 1 point of spot {} doesn't satisfy reasonable conditions and "
                                  "entire spot will be omitted.".format(spot_instance.kwargs_serializer()))
                return

            boundary_com = np.sum(np.array(boundary_points), axis=0) / len(boundary_points)
            boundary_com = utils.cartesian_to_spherical(boundary_com)
            solution, _ = self._solver(self.potential_fn, solver_condition, *(boundary_com[2],))
            boundary_center = utils.spherical_to_cartesian([solution, boundary_com[1], boundary_com[2]])

            # first point will be always barycenter of boundary
            spot_points[0] = boundary_center

            # max size from barycenter of boundary to boundary
            # todo: make sure this value is correct = make an unittests for spots
            spot_instance.max_size = max([np.linalg.norm(np.array(boundary_center) - np.array(b))
                                          for b in boundary_points])

            spot_instance.points = np.array(spot_points)
            spot_instance.boundary = np.array(boundary_points)
            spot_instance.boundary_center = np.array(boundary_center)
            spot_instance.center = np.array(spot_center)

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
        if self._rotation_period <= 0:
            raise ValueError('Period of rotation must be non-zero positive value. Your value: {0}.'
                             .format(rotation_period))

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
        returns logarithm of polar surface gravity in SI

        :return: float
        """
        return self._polar_log_g

    @polar_log_g.setter
    def polar_log_g(self, log_g):
        """
        setter for polar surface gravity, if unit is not specified in astropy.units format, value in m/s^2 is assumed

        :param log_g:
        :return:
        """
        if isinstance(log_g, u.quantity.Quantity):
            self._polar_log_g = np.float64(log_g.to(U.LOG_ACCELERATION_UNIT))
        elif isinstance(log_g, (int, np.int, float, np.float)):
            # self._polar_log_g = np.float64((log_g * u.dex(u.cm / u.s ** 2)).to(U.LOG_ACCELERATION_UNIT))
            self._polar_log_g = np.float64(log_g)
        else:
            raise TypeError('Input of variable `polar_log_g` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

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

        return - c.G * self.star.mass / radius - 0.5 * np.power(self._angular_velocity * radius * np.sin(theta), 2.0)

    def calculate_potential_gradient_magnitudes(self, points=None, faces=None):
        """
        returns array of absolute values of potential gradients for corresponding faces

        :return: np.array
        """
        if points is not None and faces is None:
            raise TypeError('Specify faces corresponding to given points')
        if self.star.spots:
            face = self.star.faces if faces is None else faces
            point = self.star.points if points is None else points
        else:
            face = self.star.faces[:self.star.base_symmetry_faces_number] if faces is None else faces
            point = self.star.points[:self.star.base_symmetry_points_number] if points is None else points

        r3 = np.power(np.linalg.norm(point, axis=1), 3)
        domega_dx = c.G * self.star.mass * point[:, 0] / r3 \
                    - np.power(self._angular_velocity, 2) * point[:, 0]
        domega_dy = c.G * self.star.mass * point[:, 1] / r3 \
                    - np.power(self._angular_velocity, 2) * point[:, 1]
        domega_dz = c.G * self.star.mass * point[:, 2] / r3
        points_gradients = np.power(np.power(domega_dx, 2) + np.power(domega_dy, 2) + np.power(domega_dz, 2), 0.5)

        return np.mean(points_gradients[face], axis=1) if self.star.spots \
            else np.mean(points_gradients[face], axis=1)[self.star.face_symmetry_vector]

    def calculate_polar_potential_gradient_magnitude(self):
        """
        returns matgnitude of polar gradient of gravitational potential

        :return:
        """
        points_z = self.calculate_polar_radius()
        r3 = np.power(points_z, 3)
        domega_dz = c.G * self.star.mass * points_z / r3
        return domega_dz

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

    def mesh(self, symmetry_output=False):
        """
        function for creating surface mesh of single star system

        :return: numpy.array([[x1 y1 z1],
                              [x2 y2 z2],
                                ...
                              [xN yN zN]]) - array of surface points if symmetry_output = False, else:
                 numpy.array([[x1 y1 z1],
                              [x2 y2 z2],
                                ...
                              [xN yN zN]]) - array of surface points,
                 numpy.array([indices_of_symmetrical_points]) - array which remapped surface points to symmetrical one
                                                                eighth of surface,
                 numpy.float - number of points included in symmetrical one eighth of surface,
                 numpy.array([octants[indexes_of_remapped_points_in_octants]) - matrix of eight sub matrices that mapped
                                                                                basic symmetry quadrant to all others
                                                                                octants

        """
        if self.star.discretization_factor > c.HALF_PI:
            raise ValueError("Invalid value of alpha parameter. Use value less than 90.")

        alpha = self.star.discretization_factor
        N = int(c.HALF_PI // alpha)
        characterictic_angle = c.HALF_PI / N
        characterictic_distance = self.star.equatorial_radius * characterictic_angle

        # calculating equatorial part
        r_eq = np.array([self.star.equatorial_radius for ii in range(N)])
        phi_eq = np.array([characterictic_angle*ii for ii in range(N)])
        theta_eq = np.array([c.HALF_PI for ii in range(N)])
        # converting quarter of equator to cartesian
        equator = utils.spherical_to_cartesian(np.column_stack((r_eq, phi_eq, theta_eq)))
        x_eq, y_eq, z_eq = equator[:, 0], equator[:, 1], equator[:, 2]

        # calculating radii for each latitude and generating one eighth of surface of the star without poles and equator
        num = int((c.HALF_PI - 2 * characterictic_angle) // characterictic_angle)
        thetas = np.linspace(characterictic_angle, c.HALF_PI-characterictic_angle, num=num, endpoint=True)
        r_q, phi_q, theta_q = [], [], []
        # also generating meridian line
        r_mer, phi_mer, theta_mer = [], [], []
        for theta in thetas:
            args, use = theta, False
            scipy_solver_init_value = np.array([1 / 1000.0])
            solution, _, ier, _ = scipy.optimize.fsolve(self.potential_fn, scipy_solver_init_value,
                                                        full_output=True, args=args)
            radius = solution[0]
            num = int(c.HALF_PI * radius * np.sin(theta) // characterictic_distance)
            r_q += [radius for xx in range(1, num)]
            M = c.HALF_PI/num
            phi_q += [xx*M for xx in range(1, num)]
            theta_q += [theta for xx in range(1, num)]

            r_mer.append(radius)
            phi_mer.append(0)
            theta_mer.append(theta)

        r_q = np.array(r_q)
        phi_q = np.array(phi_q)
        theta_q = np.array(theta_q)
        r_mer = np.array(r_mer)
        phi_mer = np.array(phi_mer)
        theta_mer = np.array(theta_mer)

        # converting this eighth of surface to cartesian coordinates
        quarter = utils.spherical_to_cartesian(np.column_stack((r_q, phi_q, theta_q)))
        meridian = utils.spherical_to_cartesian(np.column_stack((r_mer, phi_mer, theta_mer)))
        x_q, y_q, z_q = quarter[:, 0], quarter[:, 1], quarter[:, 2]
        x_mer, y_mer, z_mer = meridian[:, 0], meridian[:, 1], meridian[:, 2]

        # stitching together equator and 8 sectors of stellar surface
        # in order: north hemisphere: north pole, x_meridian, xy_equator, xy_quarter, y_meridian, y-x_equator,
        #                             y-x_quarter, -x_meridian, -x-y_equator, -x-y_quarter, -y_meridian, -yx_equator,
        #                             -yx_quarter
        #           south hemisphere: south_pole, x_meridian, xy_quarter, y_meridian, y-x_quarter, -x_meridian,
        #                             -x-y_quarter, -y_meridian, -yx_quarter

        x = np.concatenate((np.array([0]), x_mer, x_eq, x_q, -y_mer, -y_eq, -y_q, -x_mer, -x_eq, -x_q,  y_mer,  y_eq,
                            y_q, np.array([0]), x_mer,  x_q, -y_mer, -y_q, -x_mer, -x_q,  y_mer,  y_q))
        y = np.concatenate((np.array([0]), y_mer, y_eq, y_q,  x_mer,  x_eq,  x_q, -y_mer, -y_eq, -y_q, -x_mer, -x_eq,
                            -x_q, np.array([0]), y_mer,  y_q,  x_mer,  x_q, -y_mer, -y_q, -x_mer, -x_q))
        z = np.concatenate((np.array([self.star.polar_radius]), z_mer, z_eq, z_q,  z_mer,  z_eq,  z_q,  z_mer,  z_eq,
                            z_q,  z_mer,  z_eq,  z_q, np.array([-self.star.polar_radius]), -z_mer, -z_q, -z_mer, -z_q,
                            -z_mer, -z_q, -z_mer, -z_q))

        if symmetry_output:
            quarter_equator_length = len(x_eq)
            meridian_length = len(x_mer)
            quarter_length = len(x_q)
            base_symmetry_points_number = 1 + meridian_length + quarter_equator_length + quarter_length + \
                                          meridian_length
            symmetry_vector = np.concatenate((np.arange(base_symmetry_points_number),  # 1st quadrant
                                              # stray point on equator
                                              [base_symmetry_points_number],
                                              # 2nd quadrant
                                              np.arange(2 + meridian_length, base_symmetry_points_number),
                                              # 3rd quadrant
                                              np.arange(1 + meridian_length, base_symmetry_points_number),
                                              # 4rd quadrant
                                              np.arange(1 + meridian_length, base_symmetry_points_number -
                                                        meridian_length),
                                              # south hemisphere
                                              np.arange(1 + meridian_length),
                                              np.arange(1 + meridian_length + quarter_equator_length,
                                                        base_symmetry_points_number),  # 1st quadrant
                                              np.arange(1 + meridian_length + quarter_equator_length,
                                                        base_symmetry_points_number),  # 2nd quadrant
                                              np.arange(1 + meridian_length + quarter_equator_length,
                                                        base_symmetry_points_number),  # 3nd quadrant
                                              np.arange(1 + meridian_length + quarter_equator_length,
                                                        base_symmetry_points_number - meridian_length)))

            south_pole_index = 4*(base_symmetry_points_number - meridian_length) - 3
            reduced_bspn = base_symmetry_points_number-meridian_length  # auxiliary variable1
            reduced_bspn2 = base_symmetry_points_number - quarter_equator_length
            inverse_symmetry_matrix = \
                np.array([
                    np.arange(base_symmetry_points_number+1),  # 1st quadrant (north hem)
                    # 2nd quadrant (north hem)
                    np.concatenate(([0], np.arange(reduced_bspn, 2*base_symmetry_points_number-meridian_length))),
                    # 3rd quadrant (north hem)
                    np.concatenate(([0], np.arange(2*reduced_bspn - 1, 3*reduced_bspn + meridian_length -1))),
                    # 4th quadrant (north hem)
                    np.concatenate(([0], np.arange(3*reduced_bspn - 2, 4*reduced_bspn - 3),
                                    np.arange(1, meridian_length + 2))),
                    # 1st quadrant (south hemisphere)
                    np.concatenate((np.arange(south_pole_index, meridian_length + 1 + south_pole_index),
                                    np.arange(1 + meridian_length, 1 + meridian_length + quarter_equator_length),
                                    np.arange(meridian_length + 1 + south_pole_index,
                                              base_symmetry_points_number - quarter_equator_length + south_pole_index),
                                    [base_symmetry_points_number])),
                    # 2nd quadrant (south hem)
                    np.concatenate(([south_pole_index],
                                    np.arange(reduced_bspn2 - meridian_length + south_pole_index,
                                              reduced_bspn2 + south_pole_index),
                                    np.arange(base_symmetry_points_number,
                                              base_symmetry_points_number + quarter_equator_length),
                                    np.arange(reduced_bspn2 + south_pole_index, 2*reduced_bspn2 - meridian_length - 1 +
                                              south_pole_index),
                                    [2*base_symmetry_points_number-meridian_length-1])),
                    # 3rd quadrant (south hem)
                    np.concatenate(([south_pole_index],
                                    np.arange(2*reduced_bspn2 - 2*meridian_length - 1 + south_pole_index,
                                              2*reduced_bspn2 - meridian_length - 1 + south_pole_index),
                                    np.arange(2*base_symmetry_points_number - meridian_length - 1,
                                              2*base_symmetry_points_number - meridian_length + quarter_equator_length
                                              - 1),
                                    np.arange(2*reduced_bspn2 - meridian_length - 1 + south_pole_index,
                                              3*reduced_bspn2 - 2*meridian_length - 2 + south_pole_index),
                                    [3*reduced_bspn + meridian_length - 2])),
                    # 4th quadrant (south hem)
                    np.concatenate(([south_pole_index],
                                    np.arange(3*reduced_bspn2 - 3*meridian_length - 2 + south_pole_index,
                                              3*reduced_bspn2 - 2*meridian_length - 2 + south_pole_index),
                                    np.arange(3*reduced_bspn + meridian_length - 2,
                                              3*reduced_bspn + meridian_length - 2 +
                                              quarter_equator_length),
                                    np.arange(3*reduced_bspn2 - 2*meridian_length - 2 + south_pole_index, len(x)),
                                    np.arange(1 + south_pole_index, meridian_length + south_pole_index + 1),
                                    [1 + meridian_length]
                                    ))
                          ])

            return np.column_stack((x, y, z)), symmetry_vector, base_symmetry_points_number + 1, inverse_symmetry_matrix
        else:
            return np.column_stack((x, y, z))

    def single_surface(self, points=None):
        """
        calculates triangulation of given set of points, if points are not given, star surface points are used. Returns
        set of triple indices of surface pints that make up given triangle

        :param points: np.array: numpy.array([[x1 y1 z1],
                                                [x2 y2 z2],
                                                  ...
                                                [xN yN zN]])
        :return: np.array(): numpy.array([[point_index1 point_index2 point_index3],
                                          [...],
                                            ...
                                          [...]])
        """
        if points is None:
            points = self.star.points
        triangulation = Delaunay(points)
        triangles_indices = triangulation.convex_hull
        return triangles_indices

    def build_surface_with_no_spots(self):
        """
        function is calling surface building function for single systems without spots and assigns star's surface to
        star object as its property
        :return:
        """
        points_length = np.shape(self.star.points[:self.star.base_symmetry_points_number, :])[0]
        # triangulating only one eighth of the star
        points_to_triangulate = np.append(self.star.points[:self.star.base_symmetry_points_number, :],
                                          [[0, 0, 0]], axis=0)
        triangles = self.single_surface(points=points_to_triangulate)
        # removing faces from triangulation, where origin point is included
        triangles = triangles[~(triangles >= points_length).any(1)]
        triangles = triangles[~((points_to_triangulate[triangles] == 0.).all(1)).any(1)]
        # setting number of base symmetry faces
        self.star.base_symmetry_faces_number = np.int(np.shape(triangles)[0])
        # lets exploit axial symmetry and fill the rest of the surface of the star
        all_triangles = [inv[triangles] for inv in self.star.inverse_point_symmetry_matrix]
        self.star.faces = np.concatenate(all_triangles, axis=0)

        base_face_symmetry_vector = np.arange(self.star.base_symmetry_faces_number)
        self.star.face_symmetry_vector = np.concatenate([base_face_symmetry_vector for _ in range(8)])

    def build_surface_with_spots(self):
        """
        function for triangulation of surface with spots

        :return:
        """
        self.star.faces = self.single_surface()

    def plot(self, descriptor=None, **kwargs):
        """
        universal plot interface for single system class, more detailed documentation for each value of descriptor is
        available in graphics library

        :param descriptor: str (defines type of plot):
                               equpotential - plots orbit in orbital plane
                               mesh - plots surface points mesh
                               surface - plots stellar surface
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
            KWARGS = ['axis_unit', 'plot_axis']
            method_to_call = graphics.single_star_mesh
            utils.invalid_kwarg_checker(kwargs, KWARGS, SingleSystem.plot)

            kwargs['plot_axis'] = kwargs.get('plot_axis', True)

            kwargs['mesh'], _ = self.build_surface(return_surface=True)  # potom tu daj ked bude vediet skvrny
            denominator = (1*kwargs['axis_unit'].to(U.DISTANCE_UNIT))
            kwargs['mesh'] /= denominator
            kwargs['equatorial_radius'] = self.star.equatorial_radius*U.DISTANCE_UNIT.to(kwargs['axis_unit'])

        elif descriptor == 'wireframe':
            KWARGS = ['axis_unit', 'plot_axis']
            method_to_call = graphics.single_star_wireframe
            utils.invalid_kwarg_checker(kwargs, KWARGS, SingleSystem.plot)

            kwargs['plot_axis'] = kwargs.get('plot_axis', True)

            kwargs['mesh'], kwargs['triangles'] = self.build_surface(return_surface=True)
            denominator = (1 * kwargs['axis_unit'].to(U.DISTANCE_UNIT))
            kwargs['mesh'] /= denominator
            kwargs['equatorial_radius'] = self.star.equatorial_radius * U.DISTANCE_UNIT.to(kwargs['axis_unit'])

        elif descriptor == 'surface':
            KWARGS = ['axis_unit', 'edges', 'normals', 'colormap', 'plot_axis']
            utils.invalid_kwarg_checker(kwargs, KWARGS, SingleSystem.plot)
            method_to_call = graphics.single_star_surface

            kwargs['edges'] = kwargs.get('edges', False)
            kwargs['normals'] = kwargs.get('normals', False)
            kwargs['colormap'] = kwargs.get('colormap', None)
            kwargs['plot_axis'] = kwargs.get('plot_axis', True)

            output = self.build_surface(return_surface=True)
            kwargs['mesh'], kwargs['triangles'] = copy(output[0]), copy(output[1])
            denominator = (1 * kwargs['axis_unit'].to(U.DISTANCE_UNIT))
            kwargs['mesh'] /= denominator
            kwargs['equatorial_radius'] = self.star.equatorial_radius * U.DISTANCE_UNIT.to(kwargs['axis_unit'])

            if kwargs['colormap'] is not None:
                kwargs['cmap'] = self.build_surface_map(colormap=kwargs['colormap'], return_map=True)
            if kwargs['normals']:
                kwargs['arrows'] = self.star.calculate_normals(points=kwargs['mesh'], faces=kwargs['triangles'])
                kwargs['centres'] = self.star.calculate_surface_centres(points=kwargs['mesh'],
                                                                        faces=kwargs['triangles'])

        else:
            raise ValueError("Incorrect descriptor `{}`".format(descriptor))

        method_to_call(**kwargs)

    def build_faces(self):
        """
        function creates faces of the star surface provided you already calculated surface points of the star

        :return:
        """
        # build surface if there is no spot specified
        if not self.star.spots:
            self.build_surface_with_no_spots()

        self.incorporate_spots_to_surface(component_instance=self.star, surface_fn=self.build_surface_with_spots)

    def build_surface(self, return_surface=False):
        """
        function for building of general system component points and surfaces including spots

        :param return_surface: bool - if true, function returns arrays with all points and faces (surface + spots)
        :param component: specify component, use `primary` or `secondary`
        :type: str
        :return:
        """
        self.build_mesh()

        # build surface if there is no spot specified
        if not self.star.spots:
            self.build_surface_with_no_spots()
            if return_surface:
                return self.star.points, self.star.faces
            else:
                return

        # saving one eighth of the star without spots to be used as reference for faces unaffected by spots
        # self.star.base_symmetry_points = copy(self.star.points[:self.star.base_symmetry_points_number])
        # self.star.base_symmetry_faces = copy(self.star.faces[:self.star.base_symmetry_faces_number])
        self.incorporate_spots_to_surface(component_instance=self.star, surface_fn=self.build_surface_with_spots)
        if return_surface:
            ret_points = copy(self.star.points)
            ret_faces = copy(self.star.faces)
            for spot_index, spot in self.star.spots.items():
                n_points = np.shape(ret_points)[0]
                ret_faces = np.append(ret_faces, spot.faces+n_points, axis=0)
                ret_points = np.append(ret_points, spot.points, axis=0)
            return ret_points, ret_faces

    def build_surface_map(self, colormap=None, return_map=False):
        """
        function calculates surface maps (temperature or gravity acceleration) for star and spot faces and it can return
        them as one array if return_map=True

        :param return_map: if True function returns arrays with surface map including star and spot segments
        :param colormap: str - `temperature` or `gravity`
        :return:
        """
        if colormap is None:
            raise ValueError('Specify colormap to calculate (`temperature` or `gravity_acceleration`).')

        self._logger.debug('Computing surface areas of stellar surface.')
        self.star.areas = self.star.calculate_areas()
        self._logger.debug('Computing polar radius')
        self.star._polar_radius = self.calculate_polar_radius()
        self._logger.debug('Computing potential gradient magnitudes distribution accros the stellar surface')
        self.star.potential_gradient_magnitudes = self.calculate_potential_gradient_magnitudes()
        self._logger.debug('Computing magnitude of polar potential gradient.')
        self.star.polar_potential_gradient_magnitude = self.calculate_polar_potential_gradient_magnitude()

        if colormap == 'temperature':
            self._logger.debug('Computing effective temprature distibution of stellar surface.')
            self.star.temperatures = self.star.calculate_effective_temperatures()
            if self.star.pulsations:
                self._logger.debug('Adding pulsations to surface temperature distribution of the star.')
                self.star.temperatures = self.star.add_pulsations()

        if self.star.spots:
            for spot_index, spot in self.star.spots.items():
                self._logger.debug('Calculating surface areas of spot: {}'.format(spot_index))
                spot.areas = spot.calculate_areas()

                self._logger.debug('Calculating distribution of potential gradient magnitudes of spot:'
                                   ' {}'.format(spot_index))
                spot.potential_gradient_magnitudes = self.calculate_potential_gradient_magnitudes(points=spot.points,
                                                                                                  faces=spot.faces)

                if colormap == 'temperature':
                    self._logger.debug('Computing temperature distribution of spot: {}'.format(spot_index))
                    spot.temperatures = spot.temperature_factor * \
                                        self.star.calculate_effective_temperatures(gradient_magnitudes=
                                                                                   spot.potential_gradient_magnitudes)
                    if self.star.pulsations:
                        self._logger.debug('Adding pulsations to temperature distribution of spot: '
                                           '{}'.format(spot_index))
                        spot.temperatures = self.star.add_pulsations(points=spot.points, faces=spot.faces,
                                                                     temperatures=spot.temperatures)
            self._logger.debug('Renormalizing temperature map of star surface.')
            self.star.renormalize_temperatures()

        if return_map:
            if colormap == 'temperature':
                ret_list = copy(self.star.temperatures)
            elif colormap == 'gravity_acceleration':
                ret_list = copy(self.star.potential_gradient_magnitudes)

            if self.star.spots:
                for spot_index, spot in self.star.spots.items():
                    if colormap == 'temperature':
                        ret_list = np.append(ret_list, spot.temperatures)
                    elif colormap == 'gravity_acceleration':
                        ret_list = np.append(ret_list, spot.potential_gradient_magnitudes)
            return ret_list
        return

    def build_mesh(self):
        """
        build points of surface for star!!! w/o spots yet !!!
        """
        if not self.star.spots:
            self.star.points, self.star.point_symmetry_vector, self.star.base_symmetry_points_number, \
                self.star.inverse_point_symmetry_matrix = self.mesh(symmetry_output=True)
        else:
            self.star.points = self.mesh(symmetry_output=False)

    def compute_lc(self):
        pass

    def get_info(self):
        pass
