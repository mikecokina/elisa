import numpy as np
from scipy.interpolate import Akima1DInterpolator

from .. import dynamic
from .. curves import (
    utils as crv_utils,
    c_managed
)
from ... import utils
from ... import settings
from ... binary_system.orbit.container import OrbitalSupplements
from ... observer.mp_manager import manage_observations


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
    :param approx_method_list: list; curve generator functions [exact curve integrator,
                                                           interpolation on apsidaly symmetrical points,
                                                           copying geometry from apsidaly symmetrical points,
                                                           geometry similarity]
    :param crv_labels: labels of the calculated curves (passbands, components,...)
    :param curve_fn: curve function
    :param kwargs: Dict;
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** atlas ** * - str
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
        return 'zero', lambda: approx_method_list[0](binary, all_orbital_pos, potentials, crv_labels, curve_fn,
                                                     **kwargs)

    # APPX ONE *********************************************************************************************************
    appx_one = eval_approximation_one(phases, phases_span_test)

    if appx_one:
        return 'one', lambda: approx_method_list[1](binary, phases, reduced_orbit_arr, counterpart_postion_arr,
                                                    potentials, crv_labels, curve_fn, **kwargs)

    # APPX TWO *********************************************************************************************************

    appx_two, orbital_supplements = eval_approximation_two(binary, potentials, reduced_orbit_arr,
                                                           reduced_orbit_supplement_arr, phases_span_test)

    if appx_two:
        return 'two', lambda: approx_method_list[2](binary, phases, orbital_supplements, potentials, crv_labels,
                                                    curve_fn, **kwargs)

    # APPX THREE *******************************************************************************************************
    approx_three, new_geometry_mask, sorted_positions = eval_approximation_three(binary, all_orbital_pos_arr)
    if approx_three:
        return 'three', lambda: approx_method_list[3](binary, sorted_positions, new_geometry_mask, potentials,
                                                      crv_labels, curve_fn, **kwargs)

    return 'zero', lambda: approx_method_list[0](binary, all_orbital_pos, potentials, crv_labels, curve_fn, **kwargs)


# *******************************************evaluate_approximations****************************************************
def eval_approximation_one(phases, phases_span_test):
    """
    Test if it is possible to compute eccentric binary system with approximation approximation one.

    :param phases_span_test: bool; test for sufficient phase span of observations
    :param phases: numpy.array;
    :return: bool;
    """
    if len(phases) > settings.POINTS_ON_ECC_ORBIT > 0 and phases_span_test:
        return True
    return False


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
    approx_test = not new_geometry_mask.all()
    return approx_test, new_geometry_mask, sorted_all_orbital_pos_arr


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
    :param potentials: dict; corrected potentials
    :param crv_labels: list; curve_labels
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
    x = np.concatenate((orbital_supplements.body[:, 4], orbital_supplements.mirror[:, 4] % 1))
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    x = np.concatenate((x[-5:] - 1, x, x[:5] + 1))

    band_curves = dict()
    for curve in crv_labels:
        y = np.concatenate((stacked_band_curves[curve][:, 0], stacked_band_curves[curve][:, 1]))
        y = y[sort_idx]
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
    :param potentials: dict; corrected potentials
    :param crv_labels: list; curve_labels
    :param curve_fn: curve integrator function
    :param kwargs: Dict;
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** atlas ** * - str
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
        mirror_idxs = np.array(orbital_supplements.mirror[not_nan_test, 0],
                               dtype=np.int)
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
    :param potentials: dict; corrected surface potentials
    :param crv_labels: list; curve_labels
    :param curve_fn: curve integrator function
    :param kwargs: kwargs: Dict;
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** atlas ** * - str
    :return: Dict[str, numpy.array];
    """
    fn_args = (binary, potentials, new_geometry_mask, crv_labels, curve_fn)

    band_curves_unsorted = manage_observations(fn=c_managed.integrate_eccentric_curve_approx_three,
                                               fn_args=fn_args,
                                               position=orbital_positions,
                                               **kwargs)

    # re-arranging points to original order
    return {key: band_curves_unsorted[key][orbital_positions[:, 0].argsort()] for key in crv_labels}
