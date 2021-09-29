import numpy as np

from copy import copy
from . orbit.container import OrbitalSupplements
from . import utils as bsutils
from .. import settings
from .. import (
    utils,
    const,
    umpy as up
)
from .curves.utils import compute_rel_d_geometry, compute_counterparts_rel_d_irrad


def get_eclipse_boundaries(binary, components_distance):
    """
    Calculates the ranges in orbital azimuths (for phase=0 -> azimuth=pi/2)!!!  where eclipses occur.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param components_distance: float;
    :return: numpy.array;

    shape::

        [primary ecl_start, primary_ecl_stop, sec_ecl_start, sec_ecl_stop]
    """
    # check whether the inclination is high enough to enable eclipses
    if binary.morphology != 'over-contact':
        radius1 = np.mean([binary.primary.side_radius, binary.primary.forward_radius, binary.primary.backward_radius,
                           binary.primary.polar_radius])
        radius2 = np.mean([binary.secondary.side_radius, binary.secondary.forward_radius,
                           binary.secondary.backward_radius, binary.secondary.polar_radius])
        sin_i_critical = (radius1 + radius2) / components_distance
        sin_i = up.sin(binary.inclination)
        if sin_i < sin_i_critical:
            return np.array([const.HALF_PI, const.HALF_PI, const.PI, const.PI])
        radius1 = binary.primary.forward_radius
        radius2 = binary.secondary.forward_radius
        sin_i_critical = 1.01 * (radius1 + radius2) / components_distance
        square = up.power(sin_i_critical, 2) - up.power(up.cos(binary.inclination), 2)
        square = 0 if square < 0 else square
        square = 1 if square > 1 else square
        azimuth = up.arcsin(up.sqrt(square))
        azimuths = np.array([const.HALF_PI - azimuth, const.HALF_PI + azimuth, 1.5 * const.PI - azimuth,
                             1.5 * const.PI + azimuth]) % const.FULL_ARC
        return azimuths
    else:
        return np.array([0, const.PI, const.PI, const.FULL_ARC])


def find_apsidally_corresponding_positions(binary, radii, base_arr, supplement_arr, as_empty=None):
    """
    Function is intended to look for the couples of orbital positions from both sides of apsidal line that are most
    similar in terms of surface geometry (eg.: forward radius \approx forward radius).

    :param binary: elisa.binary_system.system.BinarySystem;
    :param radii: numpy.array; forward_radii
    :param base_arr: numpy.array; base orbital positions
    :param supplement_arr: numpy.array; orbital position from the opposite side
    :param as_empty: numpy.array; e.g. [np.nan, np.nan] depends on shape of base_arr item
    :return: elisa.binary_system.container.OrbitalSupplements;
    """
    if as_empty is None:
        as_empty = np.empty(5) * np.nan

    mean_r = np.mean(radii, axis=1)
    bigger_comp = np.argmax(mean_r)
    r_body = radii[:, base_arr[:, 0].astype(np.int)]
    r_supplement = radii[:, supplement_arr[:, 0].astype(np.int)]

    # finding indices of supplements_array closest to the base_array by comparing radius of larger component
    ids_of_closest_reduced_values = utils.find_idx_of_nearest(r_body[bigger_comp], r_supplement[bigger_comp])

    # making sure that found orbital positions are close enough to satisfy tolerance
    rel_geometry = compute_rel_d_geometry(binary, r_body[:, ids_of_closest_reduced_values], r_supplement)
    rel_geometry = np.max(rel_geometry, axis=0)
    is_supplement_geom = rel_geometry < settings.MAX_D_FLUX

    # making sure that found orbital positions are close enough to satisfy tolerance in relative irradiation
    args = (binary, base_arr[ids_of_closest_reduced_values, 1], supplement_arr[:, 1])
    rel_irrad = compute_counterparts_rel_d_irrad(*args)
    rel_irrad = np.max(rel_irrad, axis=0)
    is_supplement_irrad = rel_irrad < settings.MAX_D_FLUX

    is_supplement = np.logical_and(is_supplement_geom, is_supplement_irrad)

    # crating array which crates valid orbital position couples
    twin_in_reduced = np.full(ids_of_closest_reduced_values.shape, -1, dtype=np.int)
    twin_in_reduced[is_supplement] = ids_of_closest_reduced_values[is_supplement]

    supplements = OrbitalSupplements()

    for id_supplement, id_reduced in enumerate(twin_in_reduced):
        args = (supplement_arr[id_supplement], as_empty)
        if id_reduced > -1:
            args = (base_arr[id_reduced], supplement_arr[id_supplement])

        if not utils.is_empty(args):
            supplements.append(*args)

    # treating a case of unmatched item in base array
    base_all_ids = up.arange(0, len(base_arr))
    is_not_in = ~np.isin(base_all_ids, twin_in_reduced)

    for is_not_in_id in base_all_ids[is_not_in]:
        if base_arr[is_not_in_id] not in supplement_arr:
            supplements.append(*(base_arr[is_not_in_id], as_empty))

    return supplements


def resolve_object_geometry_update(has_spots, size, rel_d, max_allowed_difference=None):
    """
    Evaluate where on orbital position is necessary to fully update geometry.
    Evaluation depends on difference of relative radii between neighbouring orbital positions.

    :param has_spots: bool; define if system has spots
    :param size: int; number of orbital positions
    :param rel_d: numpy.array; parameter characterizing change in flux due to change in surface geometry
    :param max_allowed_difference: float; maximum allowed change in flux change estimation between orbital positions
    :return: numpy.array[bool];
    """
    return _resolve_geometry_update(has_spots=has_spots, size=size, rel_d=rel_d, resolve="object",
                                    max_allowed_difference=max_allowed_difference or settings.MAX_D_FLUX)


def resolve_spots_geometry_update(spots_longitudes, size, pulsations_tests,
                                  max_allowed_difference=None):
    """
    Evaluate where on orbital position is necessary to fully update geometry.
    Evaluation depends on difference of spots longitudes between neighboring orbital positions.

    :param spots_longitudes: numpy.array; array of spot longitudes of spots during orbital motion
    :param size: int; number of orbital positions
    :param pulsations_tests: Tuple[bool]; True if component contains pulsations
    :param max_allowed_difference: float; maximum allowed difference of spot position on neighbouring orbital positions
    :return: Tuple[numpy.array[bool], numpy.array[bool]]; geometry update arrays for primary and secondary array.
                                                          If value in array is True, component geometry has to be
                                                          recalculated at a given orbital position.
    """
    reducer = {}
    for component in settings.BINARY_COUNTERPARTS.keys():
        if pulsations_tests[component]:
            # in case of pulsations, the geometry is recalculated always
            reducer[component] = np.ones(size, dtype=np.bool)
            continue

        # longitude of all spots stored in array (longitudes of the first spot are enough)
        longitude_array = np.array(list(utils.nested_dict_values(spots_longitudes[component]))[0]) if \
            not utils.is_empty(spots_longitudes[component]) else np.array([])

        d_long = np.abs(longitude_array - np.roll(longitude_array, shift=1))[1:]
        # creating 2*n array due to compatibility with new geometry assessment based on change in spot longitude where
        # both components are evaluated at once
        d_long = np.row_stack((d_long, d_long))

        reducer[component] = _resolve_geometry_update(
            has_spots=True, size=size, rel_d=d_long, resolve="spot",
            max_allowed_difference=max_allowed_difference or settings.MAX_SPOT_D_LONGITUDE
        )

    return reducer['primary'], reducer['secondary']


def _resolve_geometry_update(has_spots, size, rel_d, max_allowed_difference, resolve="object"):
    """
    Evaluate where on orbital position is necessary to fully update geometry.

    :param has_spots: bool; define if system has spots
    :param size: int; number of orbital positions
    :param rel_d: numpy.array; parameter characterizing change in flux due to change in surface geometry
    :param max_allowed_difference: float; maximum allowed change in flux change estimation between orbital positions
    :param resolve: str; decision parameter whether resolved object on eccentric orbit or spots movement,
                         "object" or "spots"
    :return: numpy.array[bool];
    """
    if resolve not in ["object", "spot"]:
        raise ValueError("Invalid option for `resolve`, use `object` or `spot`")

    # in case of spots, the boundary points will cause problems if you want to use the same geometry
    if has_spots and resolve == "object":
        return np.ones(size, dtype=np.bool)
    elif utils.is_empty(rel_d) and resolve == "spot":
        # if `rel_d` is empty and resolve is equal to `spot` it means given component has no spots
        # and does require build only on first position
        arr = up.zeros(size, dtype=np.bool)
        arr[0] = True
        return arr

    require_new_geo = np.ones(size, dtype=np.bool)

    cumulative_sum = np.array([0.0, 0.0])
    for i in range(1, size):
        cumulative_sum += rel_d[:, i - 1]
        if (cumulative_sum <= max_allowed_difference).all():
            require_new_geo[i] = False
        else:
            require_new_geo[i] = True
            cumulative_sum = np.array([0.0, 0.0])

    return require_new_geo


def resolve_irrad_update(rel_d_irrad, size):
    """
    Evaluate where new temperature distribution should be calculated

    :param rel_d_irrad: numpy.array; change in flux due to the change in mutual irradiation
    :param size: int; number of orbital positions
    :return: numpy.array[bool]; if true orbital position has to be recalculated due to change in reflected flux
    """
    require_new_build = np.ones(size, dtype=np.bool)

    cumulative_sum = np.array([0.0, 0.0])
    for i in range(1, size):
        cumulative_sum += rel_d_irrad[:, i - 1]
        if (cumulative_sum <= settings.MAX_D_FLUX).all():
            require_new_build[i] = False
        else:
            require_new_build[i] = True
            cumulative_sum = np.array([0.0, 0.0])

    return require_new_build


def phase_crv_symmetry(binary_system, phase):
    """
    Utilizing symmetry of circular systems without spots and pulsations where you need to evaluate only half
    of the phases. Function finds such redundant phases and returns only unique phases.
    Expects phases from 0 to 1.0.

    :param binary_system: elisa.binary_system.system.BinarySystem;
    :param phase: numpy.array;
    :return: Tuple[numpy.array, numpy.array];
    """
    # keep those fucking methods imutable
    phase = phase.copy()
    if (not binary_system.has_pulsations()) & (not binary_system.has_spots()):
        symmetrical_counterpart = phase > 0.5
        phase[symmetrical_counterpart] = np.round(1.0 - phase[symmetrical_counterpart], 9)
        res_phases, reverse_idx = np.unique(phase, return_inverse=True)
        return res_phases, reverse_idx
    else:
        return phase, up.arange(phase.shape[0])


def in_eclipse_test(azimuths, ecl_boundaries):
    """
    Test whether in given phases eclipse occurs or not.
    !It works only for circular oribts!

    :param azimuths: Union[List, numpy.array];
    :param ecl_boundaries: Union[List, numpy.array];
    :return: numpy.array[bool];
    """

    if utils.is_empty(ecl_boundaries):
        return np.ones(len(azimuths), dtype=bool)

    if ecl_boundaries[0] < 1.5 * const.PI:
        primary_ecl_test = up.logical_and((azimuths >= ecl_boundaries[0]), (azimuths <= ecl_boundaries[1]))
    else:
        primary_ecl_test = up.logical_or((azimuths >= ecl_boundaries[0]), (azimuths < ecl_boundaries[1]))

    if ecl_boundaries[2] > const.HALF_PI:
        if ecl_boundaries[3] > const.HALF_PI:
            secondary_ecl_test = up.logical_and((azimuths >= ecl_boundaries[2]), (azimuths <= ecl_boundaries[3]))
        else:
            secondary_ecl_test = up.logical_or((azimuths >= ecl_boundaries[2]), (azimuths <= ecl_boundaries[3]))
    else:
        secondary_ecl_test = up.logical_and((azimuths >= ecl_boundaries[2]), (azimuths <= ecl_boundaries[3]))

    return up.logical_or(primary_ecl_test, secondary_ecl_test)


def correct_spot_positions_for_libration(system, phases):
    """
    Function corrects the position of spots for the libration motion caused by an eccentric orbit.

    :param system: Union[elisa.binary_system.system.BinarySystem,
                   elisa.binary_system.container.OrbitalPositionContainer];
    :param phases: numpy.array;
    :return: numpy.array; angular correction for each phase
    """
    # ensuring that phase = 0 is in dataset
    phases_p = [phases, ] if np.isscalar(phases) else copy(phases)
    phases_p = np.concatenate((phases_p, [0, ]))

    positions = system.calculate_orbital_motion(phases_p, return_nparray=True)

    ecc_anomaly = system.orbit.true_anomaly_to_eccentric_anomaly(positions[:, 3])
    mean_anomaly = system.orbit.eccentric_anomaly_to_mean_anomaly(ecc_anomaly)

    diff = mean_anomaly - positions[:, 3]
    diff = diff[:-1] - diff[-1]
    return diff if diff.shape[0] > 1 else diff[0]


def calculate_spot_longitudes(system, phases, component="all", correct_libration=True):
    """
    Function calculates the latitudes of every spot on given component(s) for every phase.

    :param system: Union[elisa.binary_system.system.BinarySystem,
                   elisa.binary_system.container.OrbitalPositionContainer];
    :param phases: numpy.array;
    :param component: str; 'primary' or 'secondary', if None both will be calculated
    :param correct_libration: bool; switch for calculation of correction for the libration motion of spots for EBs with
                                    eccentric orbit
    :return: Dict; {component: {spot_idx: np.array([....]), ...}, ...}
    """
    components = bsutils.component_to_list(component)
    components = {comp: getattr(system, comp) for comp in components}

    libration_correction = correct_spot_positions_for_libration(system, phases) if correct_libration else 0

    spots_longitudes = {
        comp: {
            spot_index: (instance.synchronicity - 1.0) * phases * const.FULL_ARC + spot.longitude + libration_correction
            for spot_index, spot in instance.spots.items()}
        for comp, instance in components.items()
    }
    return spots_longitudes


def assign_spot_longitudes(system, spots_longitudes, index=None, component="all"):
    """
    function assigns spot latitudes for each spot according to values in `spots_longitudes` in index `index`

    :param system: Union[elisa.binary_system.system.BinarySystem,
                   elisa.binary_system.container.OrbitalPositionContainer]
    :param spots_longitudes: Dict; {component: {spot_idx: np.array([....]), ...}, ...}, takes output of function
                                   `calculate_spot_latitudes`
    :param index: int; index of spot longitude values to be used, if none is given, scalar values are expected in
                      `spots_longitudes`
    :param component: str; 'primary' or 'secondary', if None both will be calculated
    """
    components = bsutils.component_to_list(component)
    components = {comp: getattr(system, comp) for comp in components}
    for comp, instance in components.items():
        for spot_index, spot in instance.spots.items():
            spot.longitude = spots_longitudes[comp][spot_index] if index is None else \
                spots_longitudes[comp][spot_index][index]
