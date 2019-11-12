import numpy as np

from elisa.binary_system.orbit.container import OrbitalSupplements
from elisa.conf import config
from elisa.binary_system import utils as bsutils
from elisa import (
    utils,
    const,
    umpy as up
)


def darkside_filter(line_of_sight, normals):
    """
    Return indices for visible faces defined by given normals.
    Function assumes that `line_of_sight` ([1, 0, 0]) and `normals` are already normalized to one.

    :param line_of_sight: numpy.array;
    :param normals: numpy.array;
    :return: numpy.array;
    """
    # todo: require to resolve self shadowing in case of W UMa
    # calculating normals utilizing the fact that normals and line of sight vector [1, 0, 0] are already normalized
    if (line_of_sight == np.array([1.0, 0.0, 0.0])).all():
        cosines = utils.calculate_cos_theta_los_x(normals=normals)
    else:
        cosines = utils.calculate_cos_theta(normals=normals, line_of_sight_vector=np.array([1, 0, 0]))
    # recovering indices of points on near-side (from the point of view of observer)
    return up.arange(np.shape(normals)[0])[cosines > 0]


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
        azimuth = up.arcsin(up.sqrt(up.power(sin_i_critical, 2) - up.power(up.cos(binary.inclination), 2)))
        azimuths = np.array([const.HALF_PI - azimuth, const.HALF_PI + azimuth, 1.5 * const.PI - azimuth,
                             1.5 * const.PI + azimuth]) % const.FULL_ARC
        return azimuths
    else:
        return np.array([0, const.PI, const.PI, const.FULL_ARC])


def find_apsidally_corresponding_positions(reduced_constraint, reduced_arr, supplement_constraint, supplement_arr,
                                           tol=1e-10, as_empty=None):
    """
    Function is inteded to look for orbital positions from reduced_arr which
    are supplementar to supplement_arr. Similarity to be a pair is based
    on constraints from input arguments, usually it is current separation of
    primary and secondary component on orbit.

    :param reduced_constraint: numpy.array;
    :param reduced_arr: numpy.array;
    :param supplement_constraint: numpy.array;
    :param supplement_arr: numpy.array;
    :param tol: float;
    :param as_empty: numpy.array; e.g. [np.nan, np.nan] depends on shape of reduced_arr item
    :return: elisa.binary_system.container.OrbitalSupplements;
    """
    if as_empty is None:
        as_empty = [np.nan] * 5

    ids_of_closest_reduced_values = utils.find_idx_of_nearest(reduced_constraint, supplement_constraint)

    matrix_mask = abs(up.abs(reduced_constraint[np.newaxis, :] - supplement_constraint[:, np.newaxis])) <= tol
    is_supplement = [matrix_mask[i][idx] for i, idx in enumerate(ids_of_closest_reduced_values)]

    twin_in_reduced = np.array([-1] * len(ids_of_closest_reduced_values))
    twin_in_reduced[is_supplement] = ids_of_closest_reduced_values[is_supplement]

    supplements = OrbitalSupplements()

    for id_supplement, id_reduced in enumerate(twin_in_reduced):
        args = (reduced_arr[id_reduced], supplement_arr[id_supplement]) \
            if id_reduced > -1 else (supplement_arr[id_supplement], as_empty)
        # if id_reduced > -1 else (as_empty, supplement_arr[id_supplement])

        if not utils.is_empty(args):
            supplements.append(*args)

    reduced_all_ids = up.arange(0, len(reduced_arr))
    is_not_in = ~np.isin(reduced_all_ids, twin_in_reduced)

    for is_not_in_id in reduced_all_ids[is_not_in]:
        if reduced_arr[is_not_in_id] not in supplement_arr:
            supplements.append(*(reduced_arr[is_not_in_id], as_empty))

    return supplements


def resolve_object_geometry_update(has_spots, size, rel_d, max_allowed_difference=None):
    """
    Evaluate where on orbital position is necessary to fully update geometry.
    Evaluation depends on difference of relative radii between upcomming orbital positions.
    """
    return _resolve_geometry_update(has_spots=has_spots, size=size, rel_d=rel_d, resolve="object",
                                    max_allowed_difference=max_allowed_difference or config.MAX_RELATIVE_D_R_POINT)


def resolve_spots_geometry_update(spots_longitudes, max_allowed_difference=None):
    """
    Evaluate where on orbital position is necessary to fully update geometry.
    Evaluation depends on difference of spots longitudes between upcomming orbital positions.
    """
    slp, sls = spots_longitudes["primary"], spots_longitudes["secondary"]

    reference_long_p = list(utils.nested_dict_values(slp))[0] if not utils.is_empty(slp) else np.array([])
    reference_long_s = list(utils.nested_dict_values(sls))[0] if not utils.is_empty(sls) else np.array([])

    # compute longitudes differences, slice according to shift (zero value is difference agains last)
    # and setup shape compatible with `_resolve_geometry_update` method
    d_long_p = abs(reference_long_p - np.roll(reference_long_p, shift=1))
    d_long_p = np.array([d_long_p[1:]] * 2)

    d_long_s = abs(reference_long_s - np.roll(reference_long_s, shift=1))
    d_long_s = np.array([d_long_s[1:]] * 2)

    # this will find an array of longitudes of any spot and len basically defines how many
    size = len(list(utils.nested_dict_values(spots_longitudes))[0])
    if size <= 0:
        raise ValueError("Unexpected value, at least single spot should be detected if this method is called")

    primary_reducer = _resolve_geometry_update(
        has_spots=True, size=size, rel_d=d_long_p, resolve="spot",
        max_allowed_difference=max_allowed_difference or config.MAX_SPOT_D_LONGITUDE
    )

    secondary_reducer = _resolve_geometry_update(
        has_spots=True, size=size, rel_d=d_long_s, resolve="spot",
        max_allowed_difference=max_allowed_difference or config.MAX_SPOT_D_LONGITUDE
    )

    return primary_reducer, secondary_reducer


def _resolve_geometry_update(has_spots, size, rel_d, max_allowed_difference, resolve="object"):
    """
    Evaluate where on orbital position is necessary to fully update geometry.

    :param max_allowed_difference: float;
    :param has_spots: bool; define if system has spots
    :param size: int;
    :param rel_d: numpy.array; array, based on geometry change is going to be evaluated
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
        # fixme: do it better, this is really ugly way
        return np.array([True] + [False] * (size - 1), dtype=np.bool)

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


def phase_crv_symmetry(self, phase):
    """
    Utilizing symmetry of circular systems without spots and pulastions where you need to evaluate only half
    of the phases. Function finds such redundant phases and returns only unique phases.
    Expects phases from 0 to 1.0.

    :param self: elisa.binary_system.system.BinarySystem;
    :param phase: numpy.array;
    :return: Tuple[numpy.array, numpy.array];
    """
    # keep those fucking methods imutable
    phase = phase.copy()
    if (not self.has_pulsations()) & (not self.has_spots()):
        symmetrical_counterpart = phase > 0.5
        # phase[symmetrical_counterpart] = 0.5 - (phase[symmetrical_counterpart] - 0.5)
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


def calculate_spot_longitudes(system, phases, component="all"):
    """
    Function calculates the latitudes of every spot on given component(s) for every phase.

    :param system: Union[elisa.binary_system.system.BinarySystem,
                   elisa.binary_system.container.OrbitalPositionContainer];
    :param phases: numpy.array;
    :param component: str; 'primary' or 'secondary', if None both will be calculated
    :return: Dict; {component: {spot_idx: np.array([....]), ...}, ...}
    """
    components = bsutils.component_to_list(component)
    components = {comp: getattr(system, comp) for comp in components}
    spots_longitudes = {
        comp: {
            spot_index: (instance.synchronicity - 1.0) * phases * const.FULL_ARC + spot.longitude
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
