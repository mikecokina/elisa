# noinspection PyTypeChecker
import numpy as np

from copy import copy
from .. surface import faces as bsfaces
from .. import utils as bsutils
from ... logger import getLogger
from ... import settings
from ... utils import is_empty
from ... base.surface import temperature as btemperature
from ... pulse import pulsations
from ... import (
    umpy as up,
    ld,
    utils
)

logger = getLogger("binary_system.surface.temperature")


def redistribute_temperatures(in_system, temperatures):
    """
    In this function array of `temperatures` is parsed into chunks that belong to stellar surface and spots.

    :param in_system: instance to redistribute temperatures
    :param temperatures: numpy.array; temperatures from the whole surface, ordered: surface, spot1, spot2...
    :param in_system: instance to redistribute temperatures
    """
    for component in ['primary', 'secondary']:
        star = getattr(in_system, component)
        counter = len(star.temperatures)
        star.temperatures = temperatures[component][:counter]
        if star.has_spots():
            for spot_index, spot in star.spots.items():
                spot.temperatures = temperatures[component][counter: counter + len(spot.temperatures)]
                counter += len(spot.temperatures)
    return in_system


def reflection_effect(system, components_distance, iterations):
    """
    Alter temperatures of components to involve reflection effect.

    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :param iterations: int; iterations of reflection effect counts
    :param components_distance: float; components distance in SMA units
    :return: system; elisa.binary_system.contaier.OrbitalPositionContainer; instance
    """

    if not settings.REFLECTION_EFFECT:
        logger.debug('reflection effect is switched off')
        return system
    if iterations <= 0:
        logger.debug('number of reflections in reflection effect was set to zero or negative; '
                     'reflection effect will not be calculated')
        return system

    components = bsutils.component_to_list(component='all')

    xlim = bsfaces.faces_visibility_x_limits(system.primary.polar_radius,
                                             system.secondary.polar_radius,
                                             components_distance)

    # this tests if you can use surface symmetries
    not_pulsation_test = not system.has_pulsations()
    not_spot_test = not system.has_spots()
    use_quarter_star_test = not_pulsation_test and not_spot_test
    vis_test_symmetry = {}

    # declaring variables
    centres, vis_test, gamma, normals = {}, {}, {}, {}
    faces, points, temperatures, areas, log_g = {}, {}, {}, {}, {}
    # centres - dict with all centres concatenated (star and spot) into one matrix for convenience
    # vis_test - dict with bool map for centres to select only faces visible from any face on companion
    # companion
    # gamma is of dimensions num_of_visible_faces_primary x num_of_visible_faces_secondary

    # selecting faces that have a chance to be visible from other component
    for component in components:
        star = getattr(system, component)

        points[component], faces[component], centres[component], normals[component], temperatures[component], \
            areas[component], log_g[component] = init_surface_variables(star)

        # test for visibility of star faces
        vis_test[component], vis_test_symmetry[component] = bsfaces.get_visibility_tests(centres[component],
                                                                                         use_quarter_star_test,
                                                                                         xlim[component],
                                                                                         component,
                                                                                         system.morphology)
        if star.has_spots():
            # including spots into overall surface
            for spot_index, spot in star.spots.items():
                vis_test_spot = bsfaces.visibility_test(spot.face_centres, xlim[component], component)

                # merge surface and spot face parameters into one variable
                centres[component], normals[component], temperatures[component], areas[component], \
                    vis_test[component], log_g[component] = \
                    include_spot_to_surface_variables(centres[component], spot.face_centres,
                                                      normals[component], spot.normals,
                                                      temperatures[component], spot.temperatures,
                                                      areas[component], spot.areas, log_g[component],
                                                      spot.log_g, vis_test[component], vis_test_spot)

    # limb darkening coefficients for each face of each component
    ldc = {cmp: ld.get_bolometric_ld_coefficients(temperatures[cmp], log_g[cmp],
                                                  getattr(system, cmp).metallicity) for cmp in components}

    # calculating C_A = (albedo_A / D_intB) - scalar
    # D_intB - bolometric limb darkening factor
    d_int = {cmp: ld.calculate_bolometric_limb_darkening_factor(settings.LIMB_DARKENING_LAW, ldc[cmp])
             for cmp in components}
    _c = {
        'primary': (system.primary.albedo / d_int['primary']),
        'secondary': (system.secondary.albedo / d_int['secondary'])
    }

    # setting reflection factor R = 1 + F_irradiated / F_original, initially equal to one everywhere - vector
    reflection_factor = {cmp: np.ones(np.sum(vis_test[cmp]), dtype=np.float) for cmp in components}
    counterpart = settings.BINARY_COUNTERPARTS

    # for faster convergence, reflection effect is calculated first on cooler component
    components = ['primary', 'secondary'] if system.primary.t_eff <= system.secondary.t_eff else \
        ['secondary', 'primary']

    if use_quarter_star_test:
        # calculating distances and distance vectors between, join vector is already normalized
        shp, shp_reduced = get_distance_matrix_shape(system, vis_test)

        distance, join_vector = get_symmetrical_distance_matrix(shp, shp_reduced, centres, vis_test, vis_test_symmetry)

        # calculating cos of angle gamma between face normal and join vector
        # initialising gammma matrices
        gamma = get_symmetrical_gammma(shp[:2], shp_reduced, normals, join_vector, vis_test, vis_test_symmetry)

        # testing mutual visibility of faces by assigning 0 to non visible face combination
        check_symmetric_gamma_for_negative_num(gamma, shp_reduced)

        # calculating QAB = (cos gamma_a)*cos(gamma_b)/d**2
        q_ab = get_symmetrical_q_ab(shp[:2], shp_reduced, gamma, distance)

        # calculating limb darkening factor for each combination of surface faces
        d_gamma = get_symmetrical_d_gamma(shp[:2], shp_reduced, ldc, gamma)

        # calculating limb darkening factors for each combination of faces shape
        # (N_faces_primary * N_faces_secondary)
        # precalculating matrix part of reflection effect correction
        matrix_to_sum2 = {
            'primary': q_ab[:shp_reduced[0], :] * d_gamma['secondary'][:shp_reduced[0], :],
            'secondary': q_ab[:, :shp_reduced[1]] * d_gamma['primary'][:, :shp_reduced[1]]
        }
        symmetry_to_use = {'primary': shp_reduced[0], 'secondary': shp_reduced[1]}
        for _ in range(iterations):
            for component in components:
                star = getattr(system, component)
                counterpart = 'primary' if component == 'secondary' else 'secondary'

                # calculation of reflection effect correction as
                # 1 + (c / t_effi) * sum_j(r_j * Q_ab * t_effj^4 * D(gamma_j) * areas_j)
                # calculating vector part of reflection effect correction
                vector_to_sum1 = reflection_factor[counterpart] * up.power(
                    temperatures[counterpart][vis_test[counterpart]], 4) * areas[counterpart][vis_test[counterpart]]
                counterpart_to_sum = up.matmul(vector_to_sum1, matrix_to_sum2['secondary']) \
                    if component == 'secondary' else up.matmul(matrix_to_sum2['primary'], vector_to_sum1)
                reflection_factor[component][:symmetry_to_use[component]] = \
                    1 + (_c[component][vis_test_symmetry[component]] / up.power(
                        temperatures[component][vis_test_symmetry[component]], 4)) * counterpart_to_sum

                # using symmetry to redistribute reflection factor R
                refl_fact_aux = np.empty(shape=np.shape(temperatures[component]))
                refl_fact_aux[vis_test_symmetry[component]] = reflection_factor[component][:symmetry_to_use[component]]
                refl_fact_aux = refl_fact_aux[star.face_symmetry_vector]
                reflection_factor[component] = refl_fact_aux[vis_test[component]]

        for component in components:
            star = getattr(system, component)
            # assigning new temperatures according to last iteration as
            # teff_new = teff_old * reflection_factor^0.25
            temperatures[component][vis_test_symmetry[component]] = \
                temperatures[component][vis_test_symmetry[component]] * \
                up.power(reflection_factor[component][:symmetry_to_use[component]], 0.25)
            temperatures[component] = temperatures[component][star.face_symmetry_vector]

    else:
        # calculating distances and distance vectors between, join vector is already normalized
        distance, join_vector = utils.calculate_distance_matrix(points1=centres['primary'][vis_test['primary']],
                                                                points2=centres['secondary'][vis_test['secondary']],
                                                                return_join_vector_matrix=True)

        # calculating cos of angle gamma between face normal and join vector
        gamma = \
            {'primary': np.sum(up.multiply(normals['primary'][vis_test['primary']][:, None, :], join_vector), axis=2),
             'secondary': np.sum(up.multiply(normals['secondary'][vis_test['secondary']][None, :, :], -join_vector),
                                 axis=2)}
        # negative sign is there because of reversed distance vector used for secondary component

        # testing mutual visibility of faces by assigning 0 to non visible face combination
        gamma['primary'][gamma['primary'] < 0] = 0.
        gamma['secondary'][gamma['secondary'] < 0] = 0.

        # calculating QAB = (cos gamma_a)*cos(gamma_b)/d**2
        q_ab = up.divide(up.multiply(gamma['primary'], gamma['secondary']), up.power(distance, 2))

        # calculating limb darkening factors for each combination of faces shape
        # (N_faces_primary * N_faces_secondary)

        # coefficients_primary = ld.interpolate_on_ld_grid()
        d_gamma = \
            {'primary': ld.limb_darkening_factor(coefficients=ldc['primary'][:, vis_test['primary']].T,
                                                 limb_darkening_law=settings.LIMB_DARKENING_LAW,
                                                 cos_theta=gamma['primary']),
             'secondary': ld.limb_darkening_factor(coefficients=ldc['secondary'][:, vis_test['secondary']].T,
                                                   limb_darkening_law=settings.LIMB_DARKENING_LAW,
                                                   cos_theta=gamma['secondary'].T).T
             }

        # precalculating matrix part of reflection effect correction
        matrix_to_sum2 = {cmp: q_ab * d_gamma[counterpart[cmp]] for cmp in components}

        for _ in range(iterations):
            for component in components:
                counterpart = settings.BINARY_COUNTERPARTS[component]

                # calculation of reflection effect correction as
                # 1 + (c / t_effi) * sum_j(r_j * Q_ab * t_effj^4 * D(gamma_j) * areas_j)
                # calculating vector part of reflection effect correction
                vector_to_sum1 = reflection_factor[counterpart] * up.power(
                    temperatures[counterpart][vis_test[counterpart]], 4) * areas[counterpart][vis_test[counterpart]]
                counterpart_to_sum = up.matmul(vector_to_sum1, matrix_to_sum2['secondary']) \
                    if component == 'secondary' else up.matmul(matrix_to_sum2['primary'], vector_to_sum1)
                reflection_factor[component] = 1 + (_c[component][vis_test[component]] / up.power(
                    temperatures[component][vis_test[component]], 4)) * counterpart_to_sum

        for component in components:
            # assigning new temperatures according to last iteration as
            # teff_new = teff_old * reflection_factor^0.25
            temperatures[component][vis_test[component]] = \
                temperatures[component][vis_test[component]] * up.power(reflection_factor[component], 0.25)

    # redistributing temperatures back to the parent objects
    redistribute_temperatures(system, temperatures)
    return system


def renormalize_temperatures(star):
    """
    In case of spot presence, renormalize temperatures to fit effective temperature again,
    since spots disrupt effective temperature of Star as entity.
    """
    # no need to calculate surfaces they had to be calculated already, otherwise there is nothing to renormalize
    total_surface = np.sum(star.areas)
    if star.has_spots():
        for spot_index, spot in star.spots.items():
            total_surface += np.sum(spot.areas)
    desired_flux_value = total_surface * np.power(star.t_eff, 4)

    current_flux = np.sum(star.areas * np.power(star.temperatures, 4))
    if star.spots:
        for spot_index, spot in star.spots.items():
            current_flux += np.sum(spot.areas * np.power(spot.temperatures, 4))

    coefficient = up.power(desired_flux_value / current_flux, 0.25)
    logger.debug(f'surface temperature map renormalized by a factor {coefficient}')
    star.temperatures *= coefficient
    if star.spots:
        for spot_index, spot in star.spots.items():
            spot.temperatures *= coefficient


def build_temperature_distribution(system, components_distance, component="all"):
    """
    Function calculates temperature distribution on across all faces.

    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :param components_distance: str;
    :param component: `primary` or `secondary`
    :return: system: elisa.binary_system.contaier.OrbitalPositionContainer; instance
    """
    if is_empty(component):
        logger.debug("no component set to build temperature distribution")
        return system

    components = bsutils.component_to_list(component)

    for component in components:
        star = getattr(system, component)

        logger.debug(f'computing effective temperature distribution '
                     f'on {component} component name: {star.name}')

        temperatures = btemperature.calculate_effective_temperatures(star, star.potential_gradient_magnitudes)
        setattr(star, "temperatures", temperatures)

        if star.has_spots():
            for spot_index, spot in star.spots.items():
                logger.debug(f'computing temperature distribution of spot {spot_index} / {component} component')

                pgms = spot.potential_gradient_magnitudes
                spot_temperatures = spot.temperature_factor * btemperature.calculate_effective_temperatures(star, pgms)
                setattr(spot, "temperatures", spot_temperatures)

        logger.debug(f'renormalizing temperature of components due to '
                     f'presence of spots in case of component {component}')
        renormalize_temperatures(star)

    if 'primary' in components and 'secondary' in components:
        logger.debug(f'calculating reflection effect with {settings.REFLECTION_EFFECT_ITERATIONS} '
                     f'iterations.')
        reflection_effect(system, components_distance, settings.REFLECTION_EFFECT_ITERATIONS)
    return system


def init_surface_variables(star):
    """
    Function copies basic parameters of the stellar surface (points, faces, normals, temperatures, areas and log_g) of
    given star instance into new arrays during calculation of reflection effect.

    :param star: elisa.base.container.StarContainer;
    :return: Tuple; (points, faces, centres, normals, temperatures, areas)
    """
    points, faces = star.surface_serializer()
    centres = copy(star.face_centres)
    normals = copy(star.normals)
    temperatures = copy(star.temperatures)
    log_g = copy(star.log_g)
    areas = copy(star.areas)
    return points, faces, centres, normals, temperatures, areas, log_g


def include_spot_to_surface_variables(centres, spot_centres, normals, spot_normals, temperatures,
                                      spot_temperatures, areas, spot_areas, log_g, spot_log_g, vis_test, vis_test_spot):
    """
    Function includes surface parameters of spot faces into global arrays containing parameters from whole surface
    used in reflection effect.

    :param spot_log_g: numpy.array;
    :param log_g: numpy.array;
    :param centres: numpy.array;
    :param spot_centres: numpy.array; spot centres to append to `centres`
    :param normals: numpy.array;
    :param spot_normals: numpy.array; spot normals to append to `normals`
    :param temperatures: numpy.array;
    :param spot_temperatures: numpy.array; spot temperatures to append to `temperatures`
    :param areas: numpy.array;
    :param spot_areas: numpy.array; spot areas to append to `areas`
    :param vis_test: numpy.array;
    :param vis_test_spot: numpy.array; spot visibility test to append to `vis_test`
    :return: Tuple; (centres, normals, temperatures, areas, vis_test)
    """
    centres = np.append(centres, spot_centres, axis=0)
    normals = np.append(normals, spot_normals, axis=0)
    temperatures = np.append(temperatures, spot_temperatures, axis=0)
    areas = np.append(areas, spot_areas, axis=0)
    log_g = np.append(log_g, spot_log_g, axis=0)
    vis_test = np.append(vis_test, vis_test_spot, axis=0)

    return centres, normals, temperatures, areas, vis_test, log_g


def get_symmetrical_distance_matrix(shape, shape_reduced, centres, vis_test, vis_test_symmetry):
    """
    Function uses symmetries of the stellar component in order to reduce time in calculation distance matrix.

    :param shape: Tuple[int]; desired shape of join vector matrix
    :param shape_reduced: Tuple[int]; shape of the surface symmetries,
                         (faces above those indices are symmetrical to the ones below)
    :param centres: Dict;
    :param vis_test: Dict[str, numpy.array];
    :param vis_test_symmetry: Dict[str, numpy.array];
    :return: Tuple; (distance, join vector)

    ::

        distance - distance matrix
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


def get_symmetrical_gammma(shape, shape_reduced, normals, join_vector, vis_test, vis_test_symmetry):
    """
    Function uses surface symmetries to calculate cosine of angles between join vector and surface normals.

    :param shape: Tuple[int]; desired shape of gamma
    :param shape_reduced: Tuple[int]; shape of the surface symmetries, (faces above those
                                      indices are symmetrical to the ones below)
    :param normals: Dict[str, numpy.array];
    :param join_vector: Dict[str, numpy.array];
    :param vis_test: Dict[str, numpy.array];
    :param vis_test_symmetry: Dict[str, numpy.array];
    :return: gamma: Dict[str, numpy.array]; cos(angle(normal, join_vector))
    """
    gamma = {'primary': np.empty(shape=shape, dtype=np.float),
             'secondary': np.empty(shape=shape, dtype=np.float)}

    # calculating only necessary components of the matrix (near left and upper edge) because of surface symmetry
    gamma['primary'][:, :shape_reduced[1]] = \
        np.sum(up.multiply(normals['primary'][vis_test['primary']][:, None, :],
                           join_vector[:, :shape_reduced[1], :]), axis=2)
    gamma['primary'][:shape_reduced[0], shape_reduced[1]:] = \
        np.sum(up.multiply(normals['primary'][vis_test_symmetry['primary']][:, None, :],
                           join_vector[:shape_reduced[0], shape_reduced[1]:, :]), axis=2)

    gamma['secondary'][:shape_reduced[0], :] = \
        np.sum(up.multiply(normals['secondary'][vis_test['secondary']][None, :, :],
                           -join_vector[:shape_reduced[0], :, :]), axis=2)
    gamma['secondary'][shape_reduced[0]:, :shape_reduced[1]] = \
        np.sum(up.multiply(normals['secondary'][vis_test_symmetry['secondary']][None, :, :],
                           -join_vector[shape_reduced[0]:, :shape_reduced[1], :]), axis=2)
    return gamma


def check_symmetric_gamma_for_negative_num(gamma, shape_reduced):
    """
    If cos < 0 it will be redefined as 0 are inplaced.

    :param gamma: Dict[str, numpy.array];
    :param shape_reduced: Tuple[int];
    """
    gamma['primary'][:, :shape_reduced[1]][gamma['primary'][:, :shape_reduced[1]] < 0] = 0.
    gamma['primary'][:shape_reduced[0], shape_reduced[1]:][gamma['primary'][:shape_reduced[0],
                                                           shape_reduced[1]:] < 0] = 0.
    gamma['secondary'][:shape_reduced[0], :][gamma['secondary'][:shape_reduced[0], :] < 0] = 0.
    gamma['secondary'][shape_reduced[0]:, :shape_reduced[1]][gamma['secondary'][shape_reduced[0]:,
                                                             :shape_reduced[1]] < 0] = 0.


def get_symmetrical_d_gamma(shape, shape_reduced, ldc, gamma):
    """
    Function uses surface symmetries to calculate limb darkening factor matrices
    for each components that are used in reflection effect.

    :param ldc: Dict; arrays of limb darkening coefficients for each face of each component
    :param shape: Tuple; desired shape of limb darkening matrices d_gamma
    :param shape_reduced: Tuple; shape of the surface symmetries, (faces above those
                                 indices are symmetrical to the ones below)
    :param gamma: fixme;
    """
    d_gamma = {'primary': np.empty(shape=shape, dtype=np.float),
               'secondary': np.empty(shape=shape, dtype=np.float)}

    cos_theta = gamma['primary'][:, :shape_reduced[1]]
    d_gamma['primary'][:, :shape_reduced[1]] = ld.limb_darkening_factor(
        coefficients=ldc['primary'][:, :shape[0]].T,
        limb_darkening_law=settings.LIMB_DARKENING_LAW,
        cos_theta=cos_theta)

    cos_theta = gamma['primary'][:shape_reduced[0], shape_reduced[1]:]
    d_gamma['primary'][:shape_reduced[0], shape_reduced[1]:] = ld.limb_darkening_factor(
        coefficients=ldc['primary'][:, :shape_reduced[0]].T,
        limb_darkening_law=settings.LIMB_DARKENING_LAW,
        cos_theta=cos_theta)

    cos_theta = gamma['secondary'][:shape_reduced[0], :]
    d_gamma['secondary'][:shape_reduced[0], :] = ld.limb_darkening_factor(
        coefficients=ldc['secondary'][:, :shape[1]].T,
        limb_darkening_law=settings.LIMB_DARKENING_LAW,
        cos_theta=cos_theta.T).T

    cos_theta = gamma['secondary'][shape_reduced[0]:, :shape_reduced[1]]
    d_gamma['secondary'][shape_reduced[0]:, :shape_reduced[1]] = ld.limb_darkening_factor(
        coefficients=ldc['secondary'][:, :shape_reduced[1]].T,
        limb_darkening_law=settings.LIMB_DARKENING_LAW,
        cos_theta=cos_theta.T).T

    return d_gamma


def get_symmetrical_q_ab(shape, shape_reduced, gamma, distance):
    """
    Function uses surface symmetries to calculate parameter::

        QAB = (cos gamma_a)*cos(gamma_b)/d**2

    in reflection effect.

    :param shape: Tuple[int]; desired shape of q_ab
    :param shape_reduced: Tuple[int]; shape of the surface symmetries,
                                     (faces above those indices are symmetrical to the ones below)
    :param gamma: Dict[str, numpy.array];
    :param distance: numpy.array;
    :return: numpy.array;
    """
    q_ab = np.empty(shape=shape, dtype=np.float)
    q_ab[:, :shape_reduced[1]] = \
        up.divide(up.multiply(gamma['primary'][:, :shape_reduced[1]],
                              gamma['secondary'][:, :shape_reduced[1]]),
                  up.power(distance[:, :shape_reduced[1]], 2))
    q_ab[:shape_reduced[0], shape_reduced[1]:] = \
        up.divide(up.multiply(gamma['primary'][:shape_reduced[0], shape_reduced[1]:],
                              gamma['secondary'][:shape_reduced[0], shape_reduced[1]:]),
                  up.power(distance[:shape_reduced[0], shape_reduced[1]:], 2))
    return q_ab


def get_distance_matrix_shape(system, vis_test):
    """
    Calculates shapes of distance and join vector matrices along with shapes
    of symetrical parts of those matrices used in reflection effect.

    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :param vis_test: numpy.array;
    :return: Tuple;
    """
    shape = (np.sum(vis_test['primary']), np.sum(vis_test['secondary']), 3)
    shape_reduced = (np.sum(vis_test['primary'][:system.primary.base_symmetry_faces_number]),
                     np.sum(vis_test['secondary'][:system.secondary.base_symmetry_faces_number]))
    return shape, shape_reduced


def build_temperature_perturbations(system, components_distance, component):
    """
    adds position perturbations to container mesh

    :param system: elisa.binary_system.contaier.OrbitalPositionContainer; instance
    :param components_distance: float;
    :param component: Union[str, None];
    :return:
    """
    components = bsutils.component_to_list(component)
    for component in components:
        star = getattr(system, component)
        if star.has_pulsations():
            phase = bsutils.calculate_rotational_phase(system, component)
            com_x = 0 if component == 'primary' else components_distance
            star = pulsations.incorporate_temperature_perturbations(star, com_x=com_x, phase=phase, time=system.time)
    return system


