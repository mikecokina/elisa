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
from ... binary_system.orbit.orbit import component_distance_from_mean_anomaly, get_approx_ecl_angular_width
from ... observer.mp_manager import manage_observations


logger = getLogger("binary_system.curves.c_appx_router")


def look_for_approximation(not_pulsations_test):
    """
    This condition checks if even to attempt to utilize apsidal line symmetry approximations.

    :param not_pulsations_test: bool;
    :return: bool;
    """
    appx_one = settings.POINTS_ON_ECC_ORBIT > 0 and settings.POINTS_ON_ECC_ORBIT is not None
    appx_two = settings.MAX_SUPPLEMENTAR_D_DISTANCE > 0 and settings.MAX_SUPPLEMENTAR_D_DISTANCE is not None
    appx_three = settings.MAX_RELATIVE_D_R_POINT > 0 and settings.MAX_RELATIVE_D_R_POINT is not None
    appx = appx_one or appx_two or appx_three
    return appx and not_pulsations_test


def resolve_ecc_approximation_method(binary, phases, position_method, try_to_find_appx, phases_span_test,
                                     approx_method_list, crv_labels, curve_fn, **kwargs):
    """
    Resolve and return approximation method to compute lightcurve in case of eccentric orbit.
    Return value is lambda function with already prepared params.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param phases: numpy.array;
    :param position_method: function;
    :param try_to_find_appx: bool;
    :param phases_span_test: bool; test if phases coverage is sufiicient for phases mirroring along apsidal line
    :param approx_method_list: List; curve generator functions [exact curve integrator,
                                                           interpolation on apsidaly symmetrical points,
                                                           copying geometry from apsidaly symmetrical points,
                                                           geometry similarity]
    :param crv_labels: labels of the calculated curves (passbands, components,...)
    :param curve_fn: curve function
    :param kwargs: Dict;
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
    :return: lambda;
    """
    params = dict(input_argument=phases, return_nparray=True, calculate_from='phase')
    all_orbital_pos_arr = position_method(**params)
    all_orbital_pos = utils.convert_binary_orbital_motion_arr_to_positions(all_orbital_pos_arr)
    potentials = binary.correct_potentials(phases, component="all", iterations=2)

    azimuths = all_orbital_pos_arr[:, 2]
    reduced_phase_ids, counterpart_postion_arr, reduced_phase_mask = \
        crv_utils.prepare_apsidaly_symmetric_orbit(binary, azimuths, phases)

    # spliting orbital motion into two separate groups on different sides of apsidal line
    reduced_orbit_arr, reduced_orbit_supplement_arr = \
        crv_utils.split_orbit_by_apse_line(all_orbital_pos_arr, reduced_phase_mask)

    # APPX ZERO ********************************************************************************************************
    if not try_to_find_appx:
        args = binary, all_orbital_pos, potentials, crv_labels, curve_fn
        return 'zero', lambda: approx_method_list[0](*args, **kwargs)

    # APPX ONE *********************************************************************************************************
    args = (binary, phases, phases_span_test, reduced_orbit_arr, counterpart_postion_arr, reduced_orbit_supplement_arr)
    appx_one, reduced_orbit_arr, counterpart_postion_arr = eval_approximation_one(*args)

    if appx_one:
        args = binary, phases, reduced_orbit_arr, counterpart_postion_arr, potentials, crv_labels, curve_fn
        return 'one', lambda: approx_method_list[1](*args, **kwargs)

    # APPX TWO *********************************************************************************************************

    appx_two, orbital_supplements = eval_approximation_two(binary, potentials, reduced_orbit_arr,
                                                           reduced_orbit_supplement_arr, phases_span_test)

    if appx_two:
        args = binary, phases, orbital_supplements, potentials, crv_labels, curve_fn
        return 'two', lambda: approx_method_list[2](*args, **kwargs)

    # APPX THREE *******************************************************************************************************
    approx_three, new_geometry_mask, sorted_positions = eval_approximation_three(binary, all_orbital_pos_arr)
    if approx_three:
        args = binary, sorted_positions, new_geometry_mask, potentials, crv_labels, curve_fn
        return 'three', lambda: approx_method_list[3](*args, **kwargs)

    args = binary, all_orbital_pos, potentials, crv_labels, curve_fn
    return 'zero', lambda: approx_method_list[0](*args, **kwargs)


# *******************************************evaluate_approximations****************************************************
def eval_approximation_one(binary, phases, phases_span_test, reduced_orbit_array, counterpart_position_array,
                           reduced_orbit_supplement_arr):
    """
    Test if it is possible to compute eccentric binary system with approximation approximation one.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param phases_span_test: bool; test for sufficient phase span of observations
    :param phases: numpy.array; photometric phases
    :param reduced_orbit_array: numpy.array;
    :return: bool;
    """
    # base test to establish, if curve contains enough points
    base_test = len(phases) > settings.POINTS_ON_ECC_ORBIT > 0 and phases_span_test
    if not base_test:
        return False, reduced_orbit_array, counterpart_position_array

    # true anomalies of orbital positions modelled by approximation 1
    true_anomalies_supplements = reduced_orbit_supplement_arr[:, 3]
    true_anomalies_counterparts = counterpart_position_array[:, 3]

    # component distance during eclipses
    ecl_true_anomalies = np.array([binary.orbit.conjunctions[f'{component}_eclipse']['true_anomaly']
                                  for component in settings.BINARY_COUNTERPARTS])
    distances_at_ecl = component_distance_from_mean_anomaly(binary.eccentricity, ecl_true_anomalies)

    # angular width of each eclipse
    angular_ecl_widths = [get_approx_ecl_angular_width(binary.primary.forward_radius, binary.secondary.forward_radius,
                                                       distance, binary.inclination)
                          for distance in distances_at_ecl]

    s_nu = np.sort(true_anomalies_supplements)
    d_nu = np.max(s_nu[1:] - s_nu[:-1])
    # counting if eclipse contains sufficient amount of points
    for ii, ecl_nu in enumerate(ecl_true_anomalies):
        if angular_ecl_widths[ii] == 0.0:
            continue
        width = angular_ecl_widths[ii] + d_nu
        points_ecl_mask_suplements = np.logical_and(true_anomalies_supplements > ecl_nu - width,
                                                    true_anomalies_supplements < ecl_nu + width)
        points_ecl_mask_counterparts = np.logical_and(true_anomalies_counterparts > ecl_nu - width,
                                                      true_anomalies_counterparts < ecl_nu + width)
        points_in_ecl_suplements = np.sum(points_ecl_mask_suplements)
        # points_in_ecl_counterparts = np.sum(points_ecl_mask_counterparts)

        if points_in_ecl_suplements < settings.MIN_POINTS_IN_ECLIPSE:
            reduced_orbit_array =\
                np.row_stack((reduced_orbit_array, reduced_orbit_supplement_arr[points_ecl_mask_suplements]))
            # counterpart_position_array[points_ecl_mask_counterparts] = np.full((points_in_ecl_counterparts, 5), np.nan)
            counterpart_position_array = np.row_stack((counterpart_position_array,
                                                       np.full((points_in_ecl_suplements, 5), np.nan)))

    return True, reduced_orbit_array, counterpart_position_array


def eval_approximation_two(binary, potentials, base_orbit_arr, orbit_supplement_arr, phases_span_test):
    """
    Test if it is possible to compute eccentric binary system with approximation approx two.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param potentials: Dict; corrected potentials
    :param base_orbit_arr: numpy.array;
    :param orbit_supplement_arr: numpy.array;
    :param phases_span_test: bool;
    :return: Tuple; approximation test, orbital supplements
    """
    if not phases_span_test:
        return False, None

    # create object of separated objects and supplements to bodies
    orbital_supplements = dynamic.find_apsidally_corresponding_positions(base_orbit_arr[:, 1],
                                                                         base_orbit_arr,
                                                                         orbit_supplement_arr[:, 1],
                                                                         orbit_supplement_arr,
                                                                         tol=settings.MAX_SUPPLEMENTAR_D_DISTANCE)

    orbital_supplements.sort(by='distance')

    base_potentials = {component: pot[np.array(orbital_supplements.body[:, 0], dtype=np.int)]
                       for component, pot in potentials.items()}

    # removing invalid supplements indices
    idxs_where_nan = np.argwhere(np.isnan(orbital_supplements.mirror[:, 0]))
    if idxs_where_nan.shape[0] == orbital_supplements.mirror.shape[0]:
        return False, None
    idxs = np.array(orbital_supplements.mirror[:, 0], dtype=np.int)
    # replace all nan values with corresponding body
    idxs[idxs_where_nan] = np.array(orbital_supplements.body[idxs_where_nan, 0], dtype=np.int)
    counterpart_potentials = {component: pot[idxs] for component, pot in potentials.items()}
    rel_d = crv_utils.compute_rel_d_radii_from_counterparts(binary, orbital_supplements.body[:, 1],
                                                            orbital_supplements.mirror[:, 1],
                                                            base_potentials=base_potentials,
                                                            counterpart_potentials=counterpart_potentials)

    # defined bodies/objects/templates in orbital supplements instance are sorted by distance,
    # That means that also radii `rel_d` computed from such values have to be already sorted by
    # their own size (forward radius changes based on components distance and it is monotonic function)

    if np.max(rel_d) < settings.MAX_SUPPLEMENTAR_D_DISTANCE and phases_span_test:
        return True, orbital_supplements
    return False, None


def eval_approximation_three(binary, all_orbital_pos_arr):
    """
    Test if it is possible to compute eccentric binary system with approximation approx three.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param all_orbital_pos_arr: numpy.array; array of all orbital positions
    :return: Tuple; (bool, numpy.array), approximation test, new_geometry_test
    """
    sorted_all_orbital_pos_arr = all_orbital_pos_arr[all_orbital_pos_arr[:, 1].argsort()]
    rel_d_radii = crv_utils.compute_rel_d_radii(binary, sorted_all_orbital_pos_arr[:, 1])
    new_geometry_mask = dynamic.resolve_object_geometry_update(binary.has_spots(),
                                                               all_orbital_pos_arr.shape[0], rel_d_radii)

    rel_irrad = crv_utils.compute_rel_d_irradiation(binary, all_orbital_pos_arr[:, 1])
    new_irrad_mask = dynamic.resolve_irrad_update(rel_irrad, all_orbital_pos_arr.shape[0])
    new_build_mask = np.logical_or(new_geometry_mask, new_irrad_mask)

    approx_test = not new_build_mask.all()
    return approx_test, new_build_mask, sorted_all_orbital_pos_arr


# *******************************************approximation curve_methods************************************************
def integrate_eccentric_curve_appx_one(binary, phases, reduced_orbit_arr, counterpart_postion_arr, potentials,
                                       crv_labels, curve_fn, **kwargs):
    """
    Function calculates curves for eccentric orbits for selected filters using approximation
    where curve points on the one side of the apsidal line are calculated exactly and the second
    half of the curve points are calculated by mirroring the surface geometries of the first
    half of the points to the other side of the apsidal line. Since those mirrored
    points are not alligned with desired phases, the fluxes for each phase is interpolated.

    :param binary: elisa.binary_system.system.BinarySystem;
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
    orbital_supplements = OrbitalSupplements(body=reduced_orbit_arr, mirror=counterpart_postion_arr)
    orbital_supplements.sort(by='distance')

    orbital_positions = np.stack((orbital_supplements.body, orbital_supplements.mirror), axis=1)
    fn_args = (binary, potentials, crv_labels, curve_fn)

    stacked_band_curves = manage_observations(fn=c_managed.integrate_eccentric_curve_w_orbital_symmetry,
                                              fn_args=fn_args,
                                              position=orbital_positions,
                                              **kwargs)

    # interpolation of the points in the second half of the light curves using splines
    x = np.concatenate((orbital_supplements.body[:, 4], orbital_supplements.mirror[:, 4]))
    not_nan_test = ~np.isnan(x)
    x = x[not_nan_test] % 1
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    x = np.concatenate((x[-5:] - 1, x, x[:5] + 1))

    band_curves = dict()
    for curve in crv_labels:
        y = np.concatenate((stacked_band_curves[curve][:, 0], stacked_band_curves[curve][:, 1]))
        y = y[not_nan_test][sort_idx]
        y = np.concatenate((y[-5:], y, y[:5]))

        i = Akima1DInterpolator(x, y)
        f = i(phases)
        band_curves[curve] = f

    return band_curves


def integrate_eccentric_curve_appx_two(binary, phases, orbital_supplements, potentials, crv_labels, curve_fn, **kwargs):
    """
    Function calculates curve for eccentric orbit using
    approximation where to each OrbitalPosition on one side of the apsidal line,
    the closest counterpart OrbitalPosition is assigned and the same surface geometry is
    assumed for both of them.

    :param binary: elisa.binary_system.system.BinarySystem;
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
    fn_args = (binary, potentials, crv_labels, curve_fn)

    stacked_band_curves = manage_observations(fn=c_managed.integrate_eccentric_curve_w_orbital_symmetry,
                                              fn_args=fn_args,
                                              position=orbital_positions,
                                              **kwargs)

    # re-arranging points to original order
    band_curves = {key: np.empty(phases.shape) for key in crv_labels}
    for lbl in crv_labels:
        base_idxs = np.array(orbital_supplements.body[:, 0], dtype=np.int)
        band_curves[lbl][base_idxs] = stacked_band_curves[lbl][:, 0]
        not_nan_test = ~np.isnan(orbital_supplements.mirror[:, 0])
        mirror_idxs = np.array(orbital_supplements.mirror[not_nan_test, 0], dtype=np.int)
        band_curves[lbl][mirror_idxs] = stacked_band_curves[lbl][not_nan_test, 1]

    return band_curves


def integrate_eccentric_curve_appx_three(binary, orbital_positions, new_geometry_mask, potentials, crv_labels, curve_fn,
                                         **kwargs):
    """
    Function calculates curve for eccentric orbit using approximation where surface is not fully recalculated between
    sufficiently similar neighbouring orbital positions.

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

    band_curves_unsorted = manage_observations(fn=c_managed.integrate_eccentric_curve_approx_three,
                                               fn_args=fn_args,
                                               position=orbital_positions,
                                               **kwargs)

    # re-arranging points to original order
    return {key: band_curves_unsorted[key][orbital_positions[:, 0].argsort()] for key in crv_labels}
