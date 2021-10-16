import numpy as np
from scipy.interpolate import Akima1DInterpolator

from .. import dynamic
from .. curves import (
    utils as crv_utils,
    c_managed
)
from ... logger import getLogger
from ... import utils, settings, const
from ... binary_system.orbit.container import OrbitalSupplements
from ... observer.mp_manager import manage_observations
from ... binary_system.orbit.orbit import (
    component_distance_from_mean_anomaly,
    get_approx_ecl_angular_width
)


logger = getLogger("binary_system.curves.c_appx_router")


def look_for_approximation(not_pulsations_test):
    """
    This condition checks if even to attempt to utilize apsidal line symmetry approximations.

    :param not_pulsations_test: bool;
    :return: bool;
    """
    valid_nu_separation = settings.MAX_NU_SEPARATION > 0 and settings.MAX_NU_SEPARATION is not None
    valid_max_d_flux = settings.MAX_D_FLUX > 0 and settings.MAX_D_FLUX is not None

    interp_approx = valid_nu_separation and settings.USE_INTERPOLATION_APPROXIMATION
    symmetrical_approx = settings.USE_SYMMETRICAL_COUNTERPARTS_APPROXIMATION
    neighbour_approx = valid_max_d_flux and settings.USE_SIMILAR_NEIGHBOURS_APPROXIMATION
    appx = interp_approx or neighbour_approx or symmetrical_approx
    return appx and not_pulsations_test


def resolve_ecc_approximation_method(
        binary,
        phases,
        position_method,
        try_to_find_appx,
        phases_span_test,
        crv_labels,
        curve_fn,
        **kwargs
):
    """
    Resolve and return approximation method to compute lightcurve in case of eccentric orbit.
    Return value is lambda function with already prepared params.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param phases: numpy.array;
    :param position_method: function;
    :param try_to_find_appx: bool;
    :param phases_span_test: bool; test if phases coverage is sufiicient for phases mirroring along apsidal line
    :param crv_labels: labels of the calculated curves (passbands, components,...)
    :param curve_fn: curve function
    :param kwargs: Dict;
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
    :return: callable;
    """
    approx_method_list = [
        integrate_eccentric_curve_exactly,
        integrate_eccentric_curve_interp_appx,
        integrate_eccentric_curve_symmetrical_counterparts_appx,
        integrate_eccentric_curve_similar_neighbours_appx
    ]

    params = dict(input_argument=phases, return_nparray=True, calculate_from='phase')
    all_orbital_pos_arr = position_method(**params)
    all_orbital_pos = utils.convert_binary_orbital_motion_arr_to_positions(all_orbital_pos_arr)
    potentials = binary.correct_potentials(phases, component="all", iterations=2)

    # APPX ZERO ********************************************************************************************************
    if not try_to_find_appx:
        args = binary, all_orbital_pos, potentials, crv_labels, curve_fn
        return 'zero', lambda: approx_method_list[0](*args, **kwargs)

    # calculating components radii for each orbital position
    radii = crv_utils.forward_radii_from_distances(binary, all_orbital_pos_arr[:, 1], potentials)

    azimuths = all_orbital_pos_arr[:, 2]
    reduced_phase_ids, counterpart_postion_arr, reduced_phase_mask = \
        crv_utils.prepare_apsidaly_symmetric_orbit(binary, azimuths, phases)

    # spliting orbital motion into two separate groups on different sides of apsidal line
    reduced_orbit_arr, reduced_orbit_supplement_arr = \
        crv_utils.split_orbit_by_apse_line(all_orbital_pos_arr, reduced_phase_mask)

    # APPX ONE *********************************************************************************************************
    args = (binary, phases_span_test, reduced_orbit_arr, counterpart_postion_arr, reduced_orbit_supplement_arr)
    interp_appx, reduced_orbit_arr, counterpart_postion_arr = eval_interpolation_approximation(*args)

    if interp_appx:
        args = (binary, radii, phases, reduced_orbit_arr, counterpart_postion_arr, potentials, crv_labels, curve_fn)
        return 'one', lambda: approx_method_list[1](*args, **kwargs)

    # APPX TWO *********************************************************************************************************
    args = binary, radii, reduced_orbit_arr, reduced_orbit_supplement_arr, phases_span_test
    symmetrical_counterparts_appx, orbital_supplements = eval_symmetrical_counterparts_approximation(*args)

    if symmetrical_counterparts_appx:
        args = (binary, radii, phases, orbital_supplements, potentials, crv_labels, curve_fn)
        return 'two', lambda: approx_method_list[2](*args, **kwargs)

    # APPX THREE *******************************************************************************************************
    _args = (binary, radii, all_orbital_pos_arr)
    similar_neighbours_appx, new_geometry_mask, sorted_positions = eval_similar_neighbours_approximation(*_args)
    if similar_neighbours_appx:
        args = (binary, sorted_positions, new_geometry_mask, potentials, crv_labels, curve_fn)
        return 'three', lambda: approx_method_list[3](*args, **kwargs)

    args = binary, all_orbital_pos, potentials, crv_labels, curve_fn
    return 'zero', lambda: approx_method_list[0](*args, **kwargs)


# *******************************************evaluate_approximations****************************************************
def eval_interpolation_approximation(
        binary,
        phases_span_test,
        reduced_orbit_array,
        counterpart_position_array,
        reduced_orbit_supplement_arr
):
    """
    Test if it is possible to compute eccentric binary system with interpolation approximation.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param phases_span_test: bool; test for sufficient phase span of observations
    :param reduced_orbit_array: numpy.array; orbital positions defined by user on one side of the apsidal line
    :param counterpart_position_array: numpy.array; symmetrical counterparts to the 'reduced_orbit_array'
    :param reduced_orbit_supplement_arr: numpy.array; orbital positions defined by user on the opposite side as
                                         'reduced_orbit_array' side of the apsidal line
    :return: bool;
    """
    # true anomalies of orbital positions modelled by interpolation approximation
    true_anomalies_supplements = reduced_orbit_supplement_arr[:, 3]

    # base test to establish, if curve contains enough points
    max_nu_sep = np.max(np.diff(np.sort(true_anomalies_supplements)))
    if max_nu_sep > settings.MAX_NU_SEPARATION or 0 > settings.MAX_NU_SEPARATION or not phases_span_test or \
            not settings.USE_INTERPOLATION_APPROXIMATION:
        logger.debug('Orbit is not sufficiently populated to implement interpolation approximation 1')
        return False, reduced_orbit_array, counterpart_position_array

    # component distance during eclipses
    ecl_true_anomalies = np.array([binary.orbit.conjunctions[f'{component}_eclipse']['true_anomaly']
                                  for component in settings.BINARY_COUNTERPARTS])
    distances_at_ecl = component_distance_from_mean_anomaly(binary.eccentricity, ecl_true_anomalies)

    # angular width of each eclipse
    angular_ecl_widths = [get_approx_ecl_angular_width(binary.primary.forward_radius,
                                                       binary.secondary.forward_radius,
                                                       distance, binary.inclination)
                          for distance in distances_at_ecl]

    for ii, ecl_nu in enumerate(ecl_true_anomalies):
        if angular_ecl_widths[ii][0] == 0.0:
            continue

        # including adjacent points to the eclipse to ensure smoothness
        d_nu1 = crv_utils.adjust_eclipse_width(true_anomalies_supplements, ecl_nu - angular_ecl_widths[ii][0])
        d_nu2 = crv_utils.adjust_eclipse_width(true_anomalies_supplements, ecl_nu + angular_ecl_widths[ii][0])

        bottom, top = ecl_nu - angular_ecl_widths[ii][0] - d_nu1, ecl_nu + angular_ecl_widths[ii][0] + d_nu2
        points_ecl_mask_suplements = np.logical_and(true_anomalies_supplements > bottom,
                                                    true_anomalies_supplements < top)
        # treating eclipses on boundaries of 0, 2pi interval
        if bottom < 0.0:
            points_ecl_mask_suplements = np.logical_or(points_ecl_mask_suplements,
                                                       true_anomalies_supplements > bottom + const.FULL_ARC)
        elif top > const.FULL_ARC:
            points_ecl_mask_suplements = np.logical_or(points_ecl_mask_suplements,
                                                       true_anomalies_supplements < top - const.FULL_ARC)

        # number of points in eclipse
        points_in_ecl_suplements = np.sum(points_ecl_mask_suplements)

        # subtraction of the central plateau (taking into account only descent and ascent part)
        plateau_factor = 1 - angular_ecl_widths[ii][1] / angular_ecl_widths[ii][0]

        if plateau_factor * points_in_ecl_suplements < settings.MIN_POINTS_IN_ECLIPSE:
            rowstack_args = (reduced_orbit_array, reduced_orbit_supplement_arr[points_ecl_mask_suplements])
            reduced_orbit_array = np.row_stack(rowstack_args)

            rowstack_args = (counterpart_position_array, np.full((points_in_ecl_suplements, 5), np.nan))
            counterpart_position_array = np.row_stack(rowstack_args)
        else:
            # interpolation approximation causes artifacts in case of the very
            # flat plateaus in the bottom of the eclipse
            return False, reduced_orbit_array, counterpart_position_array

    # removing duplicite entries
    _, indices = np.unique(reduced_orbit_array[:, 0], return_index=True)
    reduced_orbit_array = reduced_orbit_array[indices]
    counterpart_position_array = counterpart_position_array[indices]

    return True, reduced_orbit_array, counterpart_position_array


def eval_symmetrical_counterparts_approximation(binary, radii, base_orbit_arr, orbit_supplement_arr, phases_span_test):
    """
    Test if it is possible to compute eccentric binary system with symmetrical counterparts approximation.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param radii: numpy.array; forward potentials
    :param base_orbit_arr: numpy.array;
    :param orbit_supplement_arr: numpy.array;
    :param phases_span_test: bool;
    :return: Tuple; approximation test, orbital supplements
    """
    if not phases_span_test or not settings.USE_SYMMETRICAL_COUNTERPARTS_APPROXIMATION:
        logger.debug('Phase span of the observation is not sufficient to '
                     'utilize symmetrical counterparts approximation.')
        return False, None

    # create object of separated objects and supplements to bodies
    args = (binary, radii, base_orbit_arr, orbit_supplement_arr)
    orbital_supplements = dynamic.find_apsidally_corresponding_positions(*args)

    orbital_supplements.sort(by='distance')

    # checking how much mirrors are used
    idxs_where_nan = np.argwhere(np.isnan(orbital_supplements.mirror[:, 0]))
    if idxs_where_nan.shape[0] == orbital_supplements.mirror.shape[0]:
        return False, None

    return True, orbital_supplements


def eval_similar_neighbours_approximation(binary, radii, all_orbital_pos_arr):
    """
    Test if it is possible to compute eccentric binary system with similar neighbours approximation.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param radii: numpy.array; forward radii for all_orbital_pos_arr)
    :param all_orbital_pos_arr: numpy.array; array of all orbital positions
    :return: Tuple; (bool, numpy.array), approximation test, new_geometry_test
    """
    if not settings.USE_SIMILAR_NEIGHBOURS_APPROXIMATION:
        return False, None, None

    sort_idxs = all_orbital_pos_arr[:, 1].argsort()
    sorted_all_orbital_pos_arr = all_orbital_pos_arr[sort_idxs]
    sorted_radii = radii[:, sort_idxs]

    rel_d_radii = crv_utils.compute_rel_d_geometry(binary, sorted_radii[:, 1:], sorted_radii[:, 1:])
    args = (binary.has_spots(), all_orbital_pos_arr.shape[0], rel_d_radii)
    new_geometry_mask = dynamic.resolve_object_geometry_update(*args)

    rel_irrad = crv_utils.compute_rel_d_irradiation(binary, all_orbital_pos_arr[:, 1])
    new_irrad_mask = dynamic.resolve_irrad_update(rel_irrad, all_orbital_pos_arr.shape[0])
    new_build_mask = np.logical_or(new_geometry_mask, new_irrad_mask)

    approx_test = not new_build_mask.all()
    return approx_test, new_build_mask, sorted_all_orbital_pos_arr


# *******************************************approximation curve_methods************************************************
def integrate_eccentric_curve_interp_appx(
        binary,
        radii,
        phases,
        reduced_orbit_arr,
        counterpart_postion_arr,
        potentials,
        crv_labels,
        curve_fn,
        **kwargs
):
    """
    Function calculates curves for eccentric orbits for selected filters using approximation
    where curve points on the one side of the apsidal line are calculated exactly and the second
    half of the curve points are calculated by mirroring the surface geometries of the first
    half of the points to the other side of the apsidal line. Since those mirrored points are not
    usually alligned with desired phases, the fluxes for each user defined phase is interpolated.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param radii: numpy.array; forward radii
    :param phases: numpy.array;
    :param reduced_orbit_arr: numpy.array; base orbital positions
    :param counterpart_postion_arr: numpy.array; orbital positions symmetric to the `base_orbit_arr`
    :param potentials: Dict; corrected potentials
    :param crv_labels: List; curve_labels
    :param curve_fn: curve integrator function
    :param kwargs: Dict;
    :**kwargs options**:
        * ** passband ** - Dict[str, elisa.observer.PassbandContainer]
        * ** left_bandwidth ** - float
        * ** right_bandwidth ** - float
        * ** atlas ** - str
    :return: Dict[str, numpy.array];
    """
    n = 5 if phases.shape[0] > 10 else int(phases.shape[0] / 2) - 1
    orbital_supplements = OrbitalSupplements(body=reduced_orbit_arr, mirror=counterpart_postion_arr)
    orbital_supplements = orbital_supplements.sort(by='distance')

    orbital_positions = np.stack((orbital_supplements.body, orbital_supplements.mirror), axis=1)
    fn_args = (binary, potentials, radii, crv_labels, curve_fn)
    fn = c_managed.integrate_eccentric_curve_w_orbital_symmetry
    stacked_band_curves = manage_observations(fn=fn, fn_args=fn_args, position=orbital_positions, **kwargs)

    # interpolation of the points in the second half of the light curves using splines
    x = np.concatenate((orbital_supplements.body[:, 4], orbital_supplements.mirror[:, 4]))
    not_nan_test = ~np.isnan(x)
    x = x[not_nan_test] % 1

    # checking for accidental allignement between templates and mirrors (corresponding to eclipse regions transferred
    # from mirrored orbit area)
    x, uniqe_idx = np.unique(x, return_index=True)

    sort_idx = np.argsort(x)
    x = x[sort_idx]
    x = np.concatenate((x[-n:] - 1, x, x[:n] + 1))

    band_curves = dict()
    for curve in crv_labels:
        y = np.concatenate((stacked_band_curves[curve][:, 0], stacked_band_curves[curve][:, 1]))
        y = ((y[not_nan_test])[uniqe_idx])[sort_idx]
        y = np.concatenate((y[-n:], y, y[:n]))

        i = Akima1DInterpolator(x, y)
        f = i(phases)
        band_curves[curve] = f

    return band_curves


def integrate_eccentric_curve_symmetrical_counterparts_appx(
        binary,
        radii,
        phases,
        orbital_supplements,
        potentials,
        crv_labels,
        curve_fn,
        **kwargs
):
    """
    Function calculates curve for eccentric orbit using
    approximation where to each OrbitalPosition on one side of the apsidal line,
    the closest counterpart OrbitalPosition is assigned and the same surface geometry is
    assumed for both of them.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param radii: numpy.array; forward radii
    :param phases: numpy.array;
    :param orbital_supplements: elisa.binary_system.orbit.container.OrbitalSupplements;
    :param potentials: Dict; corrected potentials
    :param crv_labels: List; curve_labels
    :param curve_fn: curve integrator function
    :param kwargs: Dict;
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
    :return: Dict[str, numpy.array];
    """
    orbital_positions = np.stack((orbital_supplements.body, orbital_supplements.mirror), axis=1)
    fn_args = (binary, potentials, radii, crv_labels, curve_fn)
    fn = c_managed.integrate_eccentric_curve_w_orbital_symmetry
    stacked_band_curves = manage_observations(fn=fn, fn_args=fn_args, position=orbital_positions, **kwargs)

    # re-arranging points to original order
    band_curves = {key: np.empty(phases.shape) for key in crv_labels}
    for lbl in crv_labels:
        base_idxs = np.array(orbital_supplements.body[:, 0], dtype=np.int32)
        band_curves[lbl][base_idxs] = stacked_band_curves[lbl][:, 0]
        not_nan_test = ~np.isnan(orbital_supplements.mirror[:, 0])
        mirror_idxs = np.array(orbital_supplements.mirror[not_nan_test, 0], dtype=np.int32)
        band_curves[lbl][mirror_idxs] = stacked_band_curves[lbl][not_nan_test, 1]

    return band_curves


def integrate_eccentric_curve_similar_neighbours_appx(
        binary,
        orbital_positions,
        new_geometry_mask,
        potentials,
        crv_labels,
        curve_fn,
        **kwargs
):
    """
    Function calculates curve for eccentric orbit using approximation where surface
    is not fully recalculated between sufficiently similar neighbouring orbital positions.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param orbital_positions: numpy.array; orbital positions sorted by components distance
    :param new_geometry_mask: bool; mask to `orbital_positions` which determines which surface
                                    geometry should be fully recalculated
    :param potentials: Dict; corrected surface potentials
    :param crv_labels: List; curve_labels
    :param curve_fn: curve integrator function
    :param kwargs: kwargs: Dict;
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
    :return: Dict[str, numpy.array];
    """
    fn_args = (binary, potentials, new_geometry_mask, crv_labels, curve_fn)
    fn = c_managed.similar_neighbour_approximation_ecc_curve_integration
    band_curves_unsorted = manage_observations(fn=fn, fn_args=fn_args, position=orbital_positions, **kwargs)

    # re-arranging points to original order
    return {key: band_curves_unsorted[key][orbital_positions[:, 0].argsort()] for key in crv_labels}


def integrate_eccentric_curve_exactly(binary, orbital_motion, potentials, crv_labels, curve_fn, **kwargs):
    """
    Function calculates curves for eccentric orbit for selected filters.
    Curve is calculated exactly for each OrbitalPosition.
    It is very slow and it should be used only as a benchmark.

    :param binary: elisa.binary_system.system.BinarySystem; instance
    :param orbital_motion: list of all OrbitalPositions at which curve will be calculated
    :param potentials: Dict; corrected potentials
    :param kwargs: kwargs taken from `produce_eccentric_curve` function
    :param crv_labels: labels of the calculated curves (passbands, components,...)
    :param curve_fn: curve function
    :return: Dict; dictionary of fluxes for each filter
    """
    # surface potentials with constant volume of components
    fn_args = (binary, potentials, None, crv_labels, curve_fn)
    fn = c_managed.integrate_eccentric_curve_exactly
    curves = manage_observations(fn=fn, fn_args=fn_args, position=orbital_motion, **kwargs)
    return curves
