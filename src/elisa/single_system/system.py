import numpy as np
import scipy
from astropy import units as u
from scipy.spatial.qhull import Delaunay

from elisa import logger, utils, const as c, units as eunits
from elisa.base.system import System
from elisa.single_system import build, lc
from elisa.single_system import static
from elisa.single_system.plot import Plot
from elisa.single_system.animation import Animation
from elisa.base import error


class SingleSystem(System):
    MANDATORY_KWARGS = ['gamma', 'inclination', 'rotation_period']
    OPTIONAL_KWARGS = ['reference_time']
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    STAR_MANDATORY_KWARGS = ['mass', 't_eff', 'gravity_darkening', 'polar_log_g']
    STAR_OPTIONAL_KWARGS = []
    STAR_ALL_KWARGS = STAR_MANDATORY_KWARGS + STAR_OPTIONAL_KWARGS

    def __init__(self, star, name=None, suppress_logger=False, **kwargs):
        utils.invalid_kwarg_checker(kwargs, SingleSystem.ALL_KWARGS, self.__class__)
        utils.check_missing_kwargs(SingleSystem.MANDATORY_KWARGS, kwargs, instance_of=SingleSystem)
        self.stars = dict(star=star)
        self._object_params_validity_check(self.stars, self.STAR_MANDATORY_KWARGS)

        super(SingleSystem, self).__init__(name, self.__class__.__name__, suppress_logger, **kwargs)

        self._logger.info(f"initialising object {self.__class__.__name__}")
        self._logger.debug(f"setting properties of a star in class instance {self.__class__.__name__}")

        self.plot = Plot(self)
        self.animation = Animation(self)

        self.star = star

        # default values of properties
        self._rotation_period = None
        self._reference_time = 0
        self._angular_velocity = None
        self._period = self.rotation_period

        # set attributes and test whether all parameters were initialized
        # we already ensured that all kwargs are valid and all mandatory kwargs are present so lets set class attributes
        self.init_properties(**kwargs)

        # calculation of dependent parameters
        self._angular_velocity = static.angular_velocity(self.rotation_period)

        # this is also check if star surface is closed
        self.init_radii()

    def init(self):
        """
        function to reinitialize SingleSystem class instance after changing parameter(s) of binary system using setters

        :return:
        """
        self._logger.info(f're/initialising class instance {SingleSystem.__name__}')
        self.__init__(star=self.star, **self.kwargs_serializer())

    def init_radii(self):
        """
        auxiliary function for calculation of important radii
        :return:
        """
        self._logger.debug('calculating polar radius')
        self.star._polar_radius = self.calculate_polar_radius()
        self._logger.debug('calculating surface potential')
        args = 0,
        p_args = self.pre_calculate_for_potential_value(*args)
        self.star._surface_potential = static.potential(self.star.polar_radius, *p_args)
        self._logger.debug('calculating equatorial radius')
        self.star._equatorial_radius = self.calculate_equatorial_radius()

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
            self._rotation_period = np.float64(rotation_period.to(eunits.PERIOD_UNIT))
        elif isinstance(rotation_period, (int, np.int, float, np.float)):
            self._rotation_period = np.float64(rotation_period)
        else:
            raise TypeError('input of variable `rotation_period` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')
        if self._rotation_period <= 0:
            raise ValueError(f'period of rotation must be non-zero positive value. Your value: {rotation_period}')

    @property
    def reference_time(self):
        """
        Returns reference time in default period unit. Time corresponding for rotation phase 0.

        :return: float
        """
        return self._reference_time

    @reference_time.setter
    def reference_time(self, reference_time):
        """
        Setter for reference time.

        :param reference_time: (numpy.)int, (numpy.)float, astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(reference_time, u.quantity.Quantity):
            self._reference_time = np.float64(reference_time.to(eunits.PERIOD_UNIT))
        elif isinstance(reference_time, (int, np.int, float, np.float)):
            self._reference_time = np.float64(reference_time)
        else:
            raise TypeError('Input of variable `reference_time` is not (numpy.)int or (numpy.)float '
                            'nor astropy.unit.quantity.Quantity instance.')
        self._logger.debug(f"setting property `reference_time` of class instance "
                           f"{self.__class__.__name__} to {self._reference_time}")

    def _evaluate_spots(self):
        """
        compute points of each spots and assigns values to spot container instance

        :return:
        """

        # fixme: it's not crutial, but this function and same function in binary system should on the same place
        def solver_condition(x, *_args, **_kwargs):
            return True

        self._logger.info("Evaluating spots.")

        if not self.star.has_spots():
            self._logger.info("No spots to evaluate.")
            return

        # iterate over spots
        for spot_index, spot_instance in list(self.star.spots.items()):
            # lon -> phi, lat -> theta
            lon, lat = spot_instance.longitude, spot_instance.latitude
            if spot_instance.angular_density is None:
                self._logger.debug('Angular density of the spot {0} was not supplied and discretization factor of star '
                                   '{1} was used.'.format(spot_index, self.star.discretization_factor))
                spot_instance.angular_density = 0.9 * self.star.discretization_factor * eunits.ARC_UNIT
            alpha, spot_radius = spot_instance.angular_density, spot_instance.angular_radius

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
            num_radial = int(np.round(spot_radius / alpha)) + 1
            thetas = np.linspace(lat, lat + spot_radius, num=num_radial, endpoint=True)

            num_azimuthal = [1 if i == 0 else int(i * 2.0 * np.pi * x0 // x0) for i in range(0, len(thetas))]
            deltas = [np.linspace(0., c.FULL_ARC, num=num, endpoint=False) for num in num_azimuthal]
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
                                                                degrees=False,
                                                                omega_normalized=True)

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

    def calculate_polar_radius(self):
        """
        returns polar radius of the star in default units

        :return: float
        """
        polar_gravity_acceleration = np.power(10, self.star.polar_log_g)
        return np.power(c.G * self.star.mass / polar_gravity_acceleration, 0.5)

    def calculate_equatorial_radius(self):
        """
        returns equatorial radius of the star in default units

        :return: float
        """
        args, use = (c.HALF_PI, ), False
        p_args = (self.pre_calculate_for_potential_value(*args), self.star.surface_potential)
        scipy_solver_init_value = np.array([1 / 1000.0])
        solution, _, ier, _ = scipy.optimize.fsolve(self.potential_fn, scipy_solver_init_value,
                                                    full_output=True, args=p_args)
        # check if star is closed
        if ier == 1 and not np.isnan(solution[0]):
            solution = solution[0]
            if solution <= 0:
                print(solution)
                raise ValueError('Value of single star equatorial radius {} is not valid'.format(solution))
        else:
            raise ValueError('Surface of the star is not closed. Check values of polar gravity an rotation period.')

        return solution

    def calculate_face_magnitude_gradient(self, points=None, faces=None):
        """
        returns array of absolute values of potential gradients for corresponding faces

        :return: np.array
        """
        if points is not None and faces is None:
            raise TypeError('Specify faces corresponding to given points')
        if self.star.has_spots():
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

        return np.mean(points_gradients[face], axis=1) if self.star.has_spots() \
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

    def pre_calculate_for_potential_value(self, *args, return_as_tuple=False):
        """
        Function calculates auxiliary values for calculation of primary component potential,
        and therefore they don't need to be wastefully recalculated every iteration in solver.

        :param return_as_tuple: return coefficients as a tuple of numpy vectors instead of numpy matrix
        :type return_as_tuple: bool
        :param args: tuple; (latitude angle (0, pi))
        :return: tuple: (a, b) such that: Psi = -a/r - b*r**2
        """
        theta, = args

        a = c.G * self.star.mass
        b = 0.5 * np.power(self._angular_velocity * np.sin(theta), 2)

        if np.isscalar(theta):
            return a, b
        else:
            aa = a * np.ones(np.shape(theta))
            return (aa, b) if return_as_tuple else np.column_stack((aa, b))

    def potential_fn(self, radius, *args):
        """
        implicit potential function

        :param radius: (np.)float; spherical variable
        :param args: ((np.)float, (np.)float, (np.)float); (component distance, azimutal angle, polar angle)
        :return:
        """
        target_potential = args[1]
        return static.potential(radius, *args[0]) - target_potential

    def calculate_equipotential_boundary(self):
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

    def mesh(self, symmetry_output=False, **kwargs):
        """
        function for creating surface mesh of single star system

        :return:

        ::

                numpy.array([[x1 y1 z1],
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

        potential_fn = self.potential_fn
        precalc_fn = self.pre_calculate_for_potential_value
        potential_derivative_fn = static.radial_potential_derivative

        N = int(c.HALF_PI // alpha)
        characterictic_angle = c.HALF_PI / N
        characterictic_distance = self.star.equatorial_radius * characterictic_angle

        # calculating equatorial part
        x_eq, y_eq, z_eq = self.calculate_equator_points(N)

        # axial symmetry, therefore calculating latitudes
        thetas = static.pre_calc_latitudes(characterictic_angle)

        x0 = 0.5 * (self.star.equatorial_radius + self.star.polar_radius)
        args = thetas, x0, precalc_fn, potential_fn, potential_derivative_fn, self.star.surface_potential
        radius = static.get_surface_points_radii(*args)

        # converting this eighth of surface to cartesian coordinates
        x_q, y_q, z_q = static.calculate_points_on_quarter_surface(radius, thetas, characterictic_distance)
        x_mer, y_mer, z_mer = static.calculate_points_on_meridian(radius, thetas)

        # stitching together equator and 8 sectors of stellar surface
        # in order: north hemisphere: north pole, x_meridian, xy_equator, xy_quarter, y_meridian, y-x_equator,
        #                             y-x_quarter, -x_meridian, -x-y_equator, -x-y_quarter, -y_meridian, -yx_equator,
        #                             -yx_quarter
        #           south hemisphere: south_pole, x_meridian, xy_quarter, y_meridian, y-x_quarter, -x_meridian,
        #                             -x-y_quarter, -y_meridian, -yx_quarter

        x = np.concatenate((np.array([0]), x_mer, x_eq, x_q, -y_mer, -y_eq, -y_q, -x_mer, -x_eq, -x_q, y_mer, y_eq,
                            y_q, np.array([0]), x_mer, x_q, -y_mer, -y_q, -x_mer, -x_q, y_mer, y_q))
        y = np.concatenate((np.array([0]), y_mer, y_eq, y_q, x_mer, x_eq, x_q, -y_mer, -y_eq, -y_q, -x_mer, -x_eq,
                            -x_q, np.array([0]), y_mer, y_q, x_mer, x_q, -y_mer, -y_q, -x_mer, -x_q))
        z = np.concatenate((np.array([self.star.polar_radius]), z_mer, z_eq, z_q, z_mer, z_eq, z_q, z_mer, z_eq,
                            z_q, z_mer, z_eq, z_q, np.array([-self.star.polar_radius]), -z_mer, -z_q, -z_mer, -z_q,
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

            south_pole_index = 4 * (base_symmetry_points_number - meridian_length) - 3
            reduced_bspn = base_symmetry_points_number - meridian_length  # auxiliary variable1
            reduced_bspn2 = base_symmetry_points_number - quarter_equator_length
            inverse_symmetry_matrix = \
                np.array([
                    np.arange(base_symmetry_points_number + 1),  # 1st quadrant (north hem)
                    # 2nd quadrant (north hem)
                    np.concatenate(([0], np.arange(reduced_bspn, 2 * base_symmetry_points_number - meridian_length))),
                    # 3rd quadrant (north hem)
                    np.concatenate(([0], np.arange(2 * reduced_bspn - 1, 3 * reduced_bspn + meridian_length - 1))),
                    # 4th quadrant (north hem)
                    np.concatenate(([0], np.arange(3 * reduced_bspn - 2, 4 * reduced_bspn - 3),
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
                                    np.arange(reduced_bspn2 + south_pole_index,
                                              2 * reduced_bspn2 - meridian_length - 1 +
                                              south_pole_index),
                                    [2 * base_symmetry_points_number - meridian_length - 1])),
                    # 3rd quadrant (south hem)
                    np.concatenate(([south_pole_index],
                                    np.arange(2 * reduced_bspn2 - 2 * meridian_length - 1 + south_pole_index,
                                              2 * reduced_bspn2 - meridian_length - 1 + south_pole_index),
                                    np.arange(2 * base_symmetry_points_number - meridian_length - 1,
                                              2 * base_symmetry_points_number - meridian_length + quarter_equator_length
                                              - 1),
                                    np.arange(2 * reduced_bspn2 - meridian_length - 1 + south_pole_index,
                                              3 * reduced_bspn2 - 2 * meridian_length - 2 + south_pole_index),
                                    [3 * reduced_bspn + meridian_length - 2])),
                    # 4th quadrant (south hem)
                    np.concatenate(([south_pole_index],
                                    np.arange(3 * reduced_bspn2 - 3 * meridian_length - 2 + south_pole_index,
                                              3 * reduced_bspn2 - 2 * meridian_length - 2 + south_pole_index),
                                    np.arange(3 * reduced_bspn + meridian_length - 2,
                                              3 * reduced_bspn + meridian_length - 2 +
                                              quarter_equator_length),
                                    np.arange(3 * reduced_bspn2 - 2 * meridian_length - 2 + south_pole_index, len(x)),
                                    np.arange(1 + south_pole_index, meridian_length + south_pole_index + 1),
                                    [1 + meridian_length]
                                    ))
                ])

            return np.column_stack((x, y, z)), symmetry_vector, base_symmetry_points_number + 1, inverse_symmetry_matrix
        else:
            return np.column_stack((x, y, z))

    def calculate_equator_points(self, n):
        """
        function calculates points on equator of rotating single star
        :param n: int; number of points on one quarter of equator
        :return: tuple; x, y, z cartesian coordinates of equator points
        """
        r_eq = self.star.equatorial_radius * np.ones(n)
        phi_eq = np.linspace(0, c.HALF_PI, num=n, endpoint=False)
        theta_eq = c.HALF_PI * np.ones(n)
        # converting quarter of equator to cartesian
        equator = utils.spherical_to_cartesian(np.column_stack((r_eq, phi_eq, theta_eq)))
        return equator[:, 0], equator[:, 1], equator[:, 2]

    def single_surface(self, points=None):
        """
        calculates triangulation of given set of points, if points are not given, star surface points are used. Returns
        set of triple indices of surface pints that make up given triangle

        :param points: np.array:

        ::

            numpy.array([[x1 y1 z1],
                         [x2 y2 z2],
                         ...
                        [xN yN zN]])

        :return: np.array():

        ::

            numpy.array([[point_index1 point_index2 point_index3],
                         [...],
                          ...
                         [...]])
        """
        if points is None:
            points = self.star.points
        triangulation = Delaunay(points)
        triangles_indices = triangulation.convex_hull
        return triangles_indices

    def _evaluate_spots_mesh(self):
        """
        compute points of each spots and assigns values to spot container instance

        :return:
        """

        def solver_condition(x, *_args, **_kwargs):
            return True

        self._logger.info("Evaluating spots.")

        if not self.star.has_spots():
            self._logger.info("No spots to evaluate.")
            return

        potential_fn = self.potential_fn
        precalc_fn = self.pre_calculate_for_potential_value
        potential_derivative_fn = static.radial_potential_derivative

        # iterate over spots
        for spot_index, spot_instance in list(self.star.spots.items()):
            # lon -> phi, lat -> theta
            lon, lat = spot_instance.longitude, spot_instance.latitude
            self.star.setup_spot_instance_discretization_factor(spot_instance, spot_index, self.star)

            alpha, spot_radius = spot_instance.discretization_factor, spot_instance.angular_radius

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
            num_radial = int(np.round(spot_radius / alpha)) + 1
            thetas = np.linspace(lat, lat + spot_radius, num=num_radial, endpoint=True)

            num_azimuthal = [1 if i == 0 else int(i * 2.0 * np.pi * x0 // x0) for i in range(0, len(thetas))]
            deltas = [np.linspace(0., c.FULL_ARC, num=num, endpoint=False) for num in num_azimuthal]

            spot_phi, spot_theta = [], []
            for theta_index, theta in enumerate(thetas):
                # first point of n-th ring of spot (counting start from center)
                default_spherical_vector = [1.0, lon % c.FULL_ARC, theta]

                for delta_index, delta in enumerate(deltas[theta_index]):
                    # rotating default spherical vector around spot center vector and thus generating concentric
                    # circle of points around centre of spot
                    delta_vector = utils.arbitrary_rotation(theta=delta, omega=center_vector,
                                                            vector=utils.spherical_to_cartesian(
                                                                default_spherical_vector),
                                                            degrees=False,
                                                            omega_normalized=True)

                    spherical_delta_vector = utils.cartesian_to_spherical(delta_vector)

                    spot_phi.append(spherical_delta_vector[1])
                    spot_theta.append(spherical_delta_vector[2])

            spot_phi, spot_theta = np.array(spot_phi), np.array(spot_theta)
            args = spot_phi, spot_theta, spot_center_r, precalc_fn, potential_fn, potential_derivative_fn, \
                   self.star.surface_potential
            try:
                spot_points = static.get_surface_points(*args)
            except error.MaxIterationError:
                self._logger.warning(f"at least 1 point of spot {spot_instance.kwargs_serializer()} "
                                     f"doesn't satisfy reasonable conditions and entire spot will be omitted")
                component_instance.remove_spot(spot_index=spot_index)
                continue

            spot_instance.points = np.array(spot_points)
            spot_instance.boundary = np.array(boundary_points)
            spot_instance.boundary_center = spot_points[0]
            spot_instance.center = np.array(spot_center)

    def build_temperature_distribution(self):
        """
        function calculates temperature distribution on across all faces

        :return:
        """
        self._logger.debug('Computing effective temprature distibution on the star.')
        self.star.temperatures = self.star.calculate_effective_temperatures()
        if self.star.pulsations:
            self._logger.debug('Adding pulsations to surface temperature distribution ')
            self.star.temperatures = self.star.add_pulsations()

        if self.star.has_spots():
            for spot_index, spot in self.star.spots.items():
                self._logger.debug('Computing temperature distribution of {} spot'.format(spot_index))
                spot.temperatures = spot.temperature_factor * self.star.calculate_effective_temperatures(
                    gradient_magnitudes=spot.potential_gradient_magnitudes)
                if self.star.pulsations:
                    self._logger.debug('Adding pulsations to temperature distribution of {} spot'.format(spot_index))
                    spot.temperatures = self.star.add_pulsations(points=spot.points, faces=spot.faces,
                                                                 temperatures=spot.temperatures)

    def get_info(self):
        pass

    def build_faces(self):
        """
            function creates faces of the star surface provided you already calculated surface points of the star

            :return:
            """
        build.build_faces(self)

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

    def build_mesh(self, **kwargs):
        """
        build points of surface for including spots
        """
        build.build_mesh(self, **kwargs)

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

    def compute_lightcurve(self, **kwargs):
        """
        This function decides which light curve generator function is used.
        Depending on the basic properties of the single system.

        :param kwargs: Dict; arguments to be passed into light curve generator functions
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** atlas ** * - str
            * ** phases ** * - numpy.array
            * ** position_method ** * - method
        :return: Dict
        """
        self._logger.info('Applying light curve generator function for single system.')
        return self._compute_general_lightcurve(**kwargs)

    def _compute_general_lightcurve(self, **kwargs):
        return lc.compute_general_lightcurve(self, **kwargs)

    def build(self, do_pulsations=False, phase=False, **kwargs):
        """
        Main method to build single star system from parameters given on init of SingleStar.

        called following methods::

            - build_mesh
            - build_faces
            - build_surface_areas
            - build_faces_orientation
            - build_surface_gravity
            - build_temperature_distribution

        :param phase: float; phase to build system on
        :param do_pulsations: bool; switch to incorporate pulsations
        :return:
        """
        self.build_mesh(**kwargs)
        self.build_from_points(do_pulsations, phase)

    def build_from_points(self, do_pulsations=False, phase=False):
        # TODO: not properly implemented
        self.build_faces()
        self.build_surface_areas()
        self.build_faces_orientation()
        self.build_surface_gravity()
        self.build_temperature_distribution(do_pulsations=do_pulsations, phase=phase)
