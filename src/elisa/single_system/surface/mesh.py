import numpy as np
from elisa import const as c
from elisa.logger import getLogger

logger = getLogger("single-system-mesh-module")


def build_mesh(system_container):
    """
    build points of surface for including spots
    """
    _a, _b, _c, _d = system_container.mesh(symmetry_output=True)

    system_container.star.points = _a
    system_container.star.point_symmetry_vector = _b
    system_container.star.base_symmetry_points_number = _c
    system_container.star.inverse_point_symmetry_matrix = _d

    system_container.mesh_spots()
    system_container.star.incorporate_spots_mesh(component_com=0)


def mesh(system_container, symmetry_output=False):
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
    star_container = getattr(system_container, 'star')
    discretization_factor = star_container.discretization_factor
    if discretization_factor > c.HALF_PI:
        raise ValueError("Invalid value of alpha parameter. Use value less than 90.")

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