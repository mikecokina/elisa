from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np

from elisa import const, settings, utils
from elisa import umpy as up
from elisa.base.types import BOOL, FLOAT, INT
from elisa.binary_system import utils as bsutils
from elisa.binary_system.curves.utils import (
    compute_counterparts_rel_d_irrad,
    compute_rel_d_geometry,
)
from elisa.binary_system.orbit.container import OrbitalSupplements

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.binary_system.container import OrbitalPositionContainer
    from elisa.binary_system.system import BinarySystem
    from elisa.types import ComponentSelection, Float, Int

MID_PHOTOMERIC_PHASE = 0.5


def get_eclipse_boundaries(
    binary: BinarySystem,
    components_distance: Float,
) -> NDArray[Float]:
    """Calculate orbital azimuth ranges where eclipses occur.

    The returned array contains the eclipse boundary azimuths in the form::

        [
            primary_eclipse_start,
            primary_eclipse_stop,
            secondary_eclipse_start,
            secondary_eclipse_stop,
        ]

    The convention assumes that for phase ``0`` the azimuth equals
    ``pi / 2``.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param components_distance: Instantaneous distance between components.
    :type components_distance: Float
    :return: Eclipse boundary azimuths.
    :rtype: NDArray[numpy.float64]
    """
    if binary.morphology == "over-contact":
        return np.array([0.0, const.PI, const.PI, const.FULL_ARC], dtype=np.float64)

    radius1 = np.mean(
        [
            binary.primary.side_radius,
            binary.primary.forward_radius,
            binary.primary.backward_radius,
            binary.primary.polar_radius,
        ],
    )
    radius2 = np.mean(
        [
            binary.secondary.side_radius,
            binary.secondary.forward_radius,
            binary.secondary.backward_radius,
            binary.secondary.polar_radius,
        ],
    )

    sin_i_critical = (radius1 + radius2) / components_distance
    sin_i = up.sin(binary.inclination)

    if sin_i < sin_i_critical:
        return np.array(
            [const.HALF_PI, const.HALF_PI, const.PI, const.PI],
            dtype=np.float64,
        )

    radius1 = binary.primary.forward_radius
    radius2 = binary.secondary.forward_radius
    sin_i_critical = 1.01 * (radius1 + radius2) / components_distance

    square = up.power(sin_i_critical, 2) - up.power(up.cos(binary.inclination), 2)
    square = 0.0 if square < 0 else square
    square = 1.0 if square > 1 else square

    azimuth = up.arcsin(up.sqrt(square))
    return (
        np.array(
            [
                const.HALF_PI - azimuth,
                const.HALF_PI + azimuth,
                1.5 * const.PI - azimuth,
                1.5 * const.PI + azimuth,
            ],
            dtype=np.float64,
        )
        % const.FULL_ARC
    )


def find_apsidally_corresponding_positions(
    binary: BinarySystem,
    radii: NDArray,
    base_arr: NDArray,
    supplement_arr: NDArray,
    as_empty: NDArray | None = None,
) -> OrbitalSupplements:
    """Find apsidally corresponding orbital positions with similar geometry.

    This function searches for pairs of orbital positions on opposite sides of
    the apsidal line that are sufficiently similar in terms of surface geometry
    and irradiation. Matching is based primarily on the radius of the larger
    component and is then filtered using tolerance criteria.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param radii: Forward radii or equivalent component radii array.
    :type radii: NDArray
    :param base_arr: Base orbital positions.
    :type base_arr: NDArray
    :param supplement_arr: Orbital positions on the opposite side of the
        apsidal line.
    :type supplement_arr: NDArray
    :param as_empty: Placeholder value used when no matching counterpart is
        found. If ``None``, an all-``NaN`` placeholder is used.
    :type as_empty: NDArray | None
    :return: Container with paired orbital supplements.
    :rtype: OrbitalSupplements
    """
    as_empty_array = np.full(5, np.nan, dtype=FLOAT) if as_empty is None else np.asarray(as_empty)

    radii_array = np.asarray(radii)
    base_array = np.asarray(base_arr)
    supplement_array = np.asarray(supplement_arr)

    mean_r = np.mean(radii_array, axis=1)
    bigger_comp = np.argmax(mean_r)

    r_body = radii_array[:, base_array[:, 0].astype(INT)]
    r_supplement = radii_array[:, supplement_array[:, 0].astype(INT)]

    ids_of_closest_reduced_values = utils.find_idx_of_nearest(
        r_body[bigger_comp],
        r_supplement[bigger_comp],
    )

    # Ensure ids_of_closest_reduced_values is explicitly cast to INT
    ids_of_closest_reduced_values = ids_of_closest_reduced_values.astype(INT)

    rel_geometry = compute_rel_d_geometry(
        binary,
        r_body[:, ids_of_closest_reduced_values],
        r_supplement,
    )
    rel_geometry = np.max(rel_geometry, axis=0)
    is_supplement_geom = rel_geometry < settings.MAX_D_FLUX

    rel_irrad = compute_counterparts_rel_d_irrad(
        binary,
        base_array[ids_of_closest_reduced_values, 1],
        supplement_array[:, 1],
    )
    rel_irrad = np.max(rel_irrad, axis=0)
    is_supplement_irrad = rel_irrad < settings.MAX_D_FLUX

    is_supplement = np.asarray(np.logical_and(is_supplement_geom, is_supplement_irrad), dtype=bool)

    # crating array which crates valid orbital position couples
    twin_in_reduced = np.full(ids_of_closest_reduced_values.shape, -1, dtype=INT)
    twin_in_reduced[is_supplement] = ids_of_closest_reduced_values[is_supplement]

    supplements = OrbitalSupplements()

    for id_supplement, id_reduced in enumerate(twin_in_reduced):
        reduced_index: INT = id_reduced

        if reduced_index > -1:
            append_args = (
                base_array[reduced_index],
                supplement_array[id_supplement],
            )
        else:
            append_args = (
                supplement_array[id_supplement],
                as_empty_array,
            )

        if not utils.is_empty(append_args):
            supplements.append(*append_args)

    base_all_ids = up.arange(0, len(base_array))
    is_not_in = ~np.isin(base_all_ids, twin_in_reduced)

    for missing_id in base_all_ids[is_not_in]:
        if base_array[missing_id] not in supplement_array:
            supplements.append(*(base_array[missing_id], as_empty_array))

    return supplements


def resolve_object_geometry_update(
    # function is used with dynamic feedeing of arguments, so we cannot apply FBT001 here, keep it as is so far
    has_spots: bool,  # noqa: FBT001
    size: int,
    rel_d: NDArray,
    max_allowed_difference: Float | None = None,
) -> NDArray[np.bool_]:
    """Evaluate where object geometry must be fully updated.

    The decision depends on the cumulative difference of relative radii between
    neighboring orbital positions.

    :param has_spots: Whether the system contains spots.
    :type has_spots: bool
    :param size: Number of orbital positions.
    :type size: int
    :param rel_d: Parameter characterizing flux change due to variation in
        surface geometry.
    :type rel_d: NDArray
    :param max_allowed_difference: Maximum allowed accumulated geometry-related
        flux change between full updates. If ``None``, the configured default is
        used.
    :type max_allowed_difference: Float | None
    :return: Boolean mask indicating where full geometry recalculation is
        required.
    :rtype: NDArray[numpy.bool_]
    """
    return _resolve_geometry_update(
        has_spots=has_spots,
        size=size,
        rel_d=rel_d,
        max_allowed_difference=(max_allowed_difference or settings.MAX_D_FLUX),
        resolve="object",
    )


def resolve_spots_geometry_update(
    spots_longitudes: dict[str, dict[int, NDArray]],
    size: int,
    pulsations_tests: dict[str, bool],
    max_allowed_difference: Float | None = None,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Evaluate where spot geometry must be fully updated.

    The decision depends on the cumulative difference of spot longitudes
    between neighboring orbital positions.

    If a component contains pulsations, its geometry is recalculated at all
    positions.

    :param spots_longitudes: Spot longitudes during orbital motion organized as
        ``{component: {spot_index: longitude_array}}``.
    :type spots_longitudes: dict[str, dict[int, NDArray]]
    :param size: Number of orbital positions.
    :type size: int
    :param pulsations_tests: Mapping indicating whether each component contains
        pulsations.
    :type pulsations_tests: dict[str, bool]
    :param max_allowed_difference: Maximum allowed change in spot longitude
        between full updates. If ``None``, the configured default is used.
    :type max_allowed_difference: Float | None
    :return: Geometry update masks for primary and secondary components.
    :rtype: tuple[NDArray[numpy.bool_], NDArray[numpy.bool_]]
    """
    reducer: dict[str, NDArray[np.bool_]] = {}

    for component in settings.BINARY_COUNTERPARTS:
        if pulsations_tests[component]:
            reducer[component] = np.ones(size, dtype=BOOL)
            continue

        if utils.is_empty(spots_longitudes[component]):
            longitude_array = np.array([], dtype=np.float64)
        else:
            longitude_array = np.asarray(
                next(iter(utils.nested_dict_values(spots_longitudes[component]))),
                dtype=np.float64,
            )

        d_long = np.abs(longitude_array - np.roll(longitude_array, shift=1))[1:]
        d_long = np.vstack((d_long, d_long))

        reducer[component] = _resolve_geometry_update(
            has_spots=True,
            size=size,
            rel_d=d_long,
            max_allowed_difference=(max_allowed_difference or settings.MAX_SPOT_D_LONGITUDE),
            resolve="spot",
        )

    return reducer["primary"], reducer["secondary"]


def _resolve_geometry_update(
    has_spots: bool,  # noqa: FBT001
    size: int,
    rel_d: NDArray,
    max_allowed_difference: Float,
    *,
    resolve: Literal["object", "spot"] = "object",
) -> NDArray[np.bool_]:
    """Evaluate where full geometry updates are required.

    The decision is made from cumulative changes between neighboring orbital
    positions, either for object geometry or for spot motion.

    :param has_spots: Whether the system contains spots.
    :type has_spots: bool
    :param size: Number of orbital positions.
    :type size: int
    :param rel_d: Parameter characterizing cumulative geometric change.
    :type rel_d: NDArray
    :param max_allowed_difference: Maximum allowed accumulated change before a
        full rebuild is required.
    :type max_allowed_difference: Float
    :param resolve: Decision mode, either ``"object"`` or ``"spot"``.
    :type resolve: Literal["object", "spot"]
    :return: Boolean mask indicating where full geometry recalculation is
        required.
    :rtype: NDArray[numpy.bool_]
    :raises ValueError: If ``resolve`` has an invalid value.
    """
    if resolve not in {"object", "spot"}:
        message = "Invalid option for `resolve`, use `object` or `spot`."
        raise ValueError(message)

    if has_spots and resolve == "object":
        return np.ones(size, dtype=BOOL)

    rel_d_array = np.asarray(rel_d)

    if utils.is_empty(rel_d_array) and resolve == "spot":
        arr = up.zeros(size, dtype=BOOL)
        arr[0] = True
        return arr

    require_new_geo = np.ones(size, dtype=BOOL)
    cumulative_sum = np.array([0.0, 0.0], dtype=np.float64)

    for index in range(1, size):
        cumulative_sum += rel_d_array[:, index - 1]
        if (cumulative_sum <= max_allowed_difference).all():
            require_new_geo[index] = False
        else:
            require_new_geo[index] = True
            cumulative_sum = np.array([0.0, 0.0], dtype=np.float64)

    return require_new_geo


def resolve_irrad_update(
    rel_d_irrad: NDArray,
    size: int,
) -> NDArray[np.bool_]:
    """Evaluate where irradiation must be recalculated.

    :param rel_d_irrad: Change in flux due to variation in mutual irradiation.
    :type rel_d_irrad: NDArray
    :param size: Number of orbital positions.
    :type size: int
    :return: Boolean mask indicating positions where reflected flux must be
        recalculated.
    :rtype: NDArray[numpy.bool_]
    """
    rel_d_irrad_array = np.asarray(rel_d_irrad)
    require_new_build = np.ones(size, dtype=BOOL)
    cumulative_sum = np.array([0.0, 0.0], dtype=np.float64)

    for index in range(1, size):
        cumulative_sum += rel_d_irrad_array[:, index - 1]
        if (cumulative_sum <= settings.MAX_D_FLUX).all():
            require_new_build[index] = False
        else:
            require_new_build[index] = True
            cumulative_sum = np.array([0.0, 0.0], dtype=np.float64)

    return require_new_build


def phase_crv_symmetry(
    binary_system: BinarySystem,
    phase: NDArray[Float],
) -> tuple[NDArray[Float], NDArray[Int]]:
    """Exploit symmetry of circular systems without spots or pulsations.

    For circular systems without spots and pulsations, only one half of the
    orbital phases needs to be evaluated. This function finds redundant phases
    and returns only the unique ones together with reverse indices.

    Phases are expected in the interval from ``0`` to ``1``.

    :param binary_system: Binary system instance.
    :type binary_system: BinarySystem
    :param phase: Orbital phases.
    :type phase: NDArray[numpy.float64]
    :return: Unique phases and reverse indices reconstructing the original
        ordering.
    :rtype: tuple[NDArray[numpy.float64], NDArray[Int]]
    """
    phase_array = phase.copy()
    if (not binary_system.has_pulsations()) and (not binary_system.has_spots()):
        symmetrical_counterpart = phase_array > MID_PHOTOMERIC_PHASE
        phase_array[symmetrical_counterpart] = np.round(
            1.0 - phase_array[symmetrical_counterpart],
            9,
        )
        res_phases, reverse_idx = np.unique(phase_array, return_inverse=True)
        return res_phases, reverse_idx

    return phase_array, up.arange(phase_array.shape[0])


def in_eclipse_test(
    azimuths: NDArray,
    ecl_boundaries: NDArray,
) -> NDArray[np.bool_]:
    """Test whether eclipse occurs at the given azimuths.

    This function works only for circular orbits.

    :param azimuths: Orbital azimuths.
    :type azimuths: NDArray
    :param ecl_boundaries: Eclipse boundary azimuths.
    :type ecl_boundaries: NDArray
    :return: Boolean mask indicating eclipse presence.
    :rtype: NDArray[numpy.bool_]
    """
    azimuths_array = np.asarray(azimuths)
    ecl_boundaries_array = np.asarray(ecl_boundaries)

    if utils.is_empty(ecl_boundaries_array):
        return np.ones(len(azimuths_array), dtype=bool)

    if ecl_boundaries_array[0] < 1.5 * const.PI:
        primary_ecl_test = up.logical_and(
            azimuths_array >= ecl_boundaries_array[0],
            azimuths_array <= ecl_boundaries_array[1],
        )
    else:
        primary_ecl_test = up.logical_or(
            azimuths_array >= ecl_boundaries_array[0],
            azimuths_array < ecl_boundaries_array[1],
        )

    if ecl_boundaries_array[2] > const.HALF_PI:
        if ecl_boundaries_array[3] > const.HALF_PI:
            secondary_ecl_test = up.logical_and(
                azimuths_array >= ecl_boundaries_array[2],
                azimuths_array <= ecl_boundaries_array[3],
            )
        else:
            secondary_ecl_test = up.logical_or(
                azimuths_array >= ecl_boundaries_array[2],
                azimuths_array <= ecl_boundaries_array[3],
            )
    else:
        secondary_ecl_test = up.logical_and(
            azimuths_array >= ecl_boundaries_array[2],
            azimuths_array <= ecl_boundaries_array[3],
        )

    return up.logical_or(primary_ecl_test, secondary_ecl_test)


def correct_spot_positions_for_libration(
    system: BinarySystem | OrbitalPositionContainer,
    phases: Float | NDArray,
) -> Float | NDArray[Float]:
    """Correct spot positions for libration caused by eccentric orbit.

    The returned angular correction is computed relative to the correction at
    phase ``0``.

    :param system: Binary system or orbital position container instance.
    :type system: BinarySystem | OrbitalPositionContainer
    :param phases: Orbital phases.
    :type phases: Float | NDArray
    :return: Angular libration correction for each phase.
    :rtype: Float | NDArray[numpy.float64]
    """
    phases_array = np.array([phases], dtype=FLOAT) if np.isscalar(phases) else copy(phases)
    phases_array = np.concatenate((phases_array, [0.0]))

    positions = system.calculate_orbital_motion(phases_array, return_nparray=True)
    ecc_anomaly = system.orbit.true_anomaly_to_eccentric_anomaly(positions[:, 3])
    mean_anomaly = system.orbit.eccentric_anomaly_to_mean_anomaly(ecc_anomaly)

    diff = mean_anomaly - positions[:, 3]
    diff = diff[:-1] - diff[-1]
    return diff if diff.shape[0] > 1 else diff[0]


def calculate_spot_longitudes(
    system: BinarySystem | OrbitalPositionContainer,
    phases: Float | NDArray,
    component: ComponentSelection | None = "all",
    *,
    correct_libration: bool = True,
) -> dict[str, dict[int, Float | NDArray[Float]]]:
    """Calculate spot longitudes for the selected component or components.

    Longitudes are evaluated for every spot and every supplied phase.

    :param system: Binary system or orbital position container instance.
    :type system: BinarySystem | OrbitalPositionContainer
    :param phases: Orbital phases.
    :type phases: Float | NDArray
    :param component: Component selector. If ``None`` or empty, no components
        are processed. If ``"all"`` or ``"both"``, both components are used.
    :type component: Literal["primary", "secondary", "all", "both"] | None
    :param correct_libration: Whether to apply libration correction for
        eccentric systems.
    :type correct_libration: bool
    :return: Nested mapping of spot longitudes in the form
        ``{component: {spot_index: longitudes}}``.
    :rtype: dict[str, dict[int, Float | NDArray[numpy.float64]]]
    """
    phases_array = np.asarray(phases, dtype=FLOAT) if not np.isscalar(phases) else phases

    components = bsutils.component_to_list(component)
    components_map = {comp: getattr(system, str(comp)) for comp in components}

    libration_correction = correct_spot_positions_for_libration(system, phases_array) if correct_libration else 0

    return {
        str(comp): {
            spot_index: (
                (instance.synchronicity - 1.0) * phases_array * const.FULL_ARC
                + float(spot.longitude)
                + libration_correction
            )
            for spot_index, spot in instance.spots.items()
        }
        for comp, instance in components_map.items()
    }


def assign_spot_longitudes(
    system: BinarySystem | OrbitalPositionContainer,
    spots_longitudes: dict[str, dict[int, Float | NDArray[Float]]],
    index: int | None = None,
    component: ComponentSelection | None = "all",
) -> None:
    """Assign spot longitudes from precomputed longitude values.

    If ``index`` is provided, indexed longitude values are used. Otherwise,
    scalar longitude values are expected.

    :param system: Binary system or orbital position container instance.
    :type system: BinarySystem | OrbitalPositionContainer
    :param spots_longitudes: Nested mapping of spot longitudes in the form
        ``{component: {spot_index: longitudes}}``.
    :type spots_longitudes: dict[str, dict[int, Float | NDArray[numpy.float64]]]
    :param index: Index of longitude values to assign. If ``None``, scalar
        values are used directly.
    :type index: int | None
    :param component: Component selector. If ``None`` or empty, no components
        are processed. If ``"all"`` or ``"both"``, both components are used.
    :type component: Literal["primary", "secondary", "all", "both"] | None
    :return: ``None``.
    :rtype: None
    """
    components = bsutils.component_to_list(component)
    components_map = {comp: getattr(system, str(comp)) for comp in components}

    for comp, instance in components_map.items():
        for spot_index, spot in instance.spots.items():
            if index is None:
                spot.longitude = spots_longitudes[str(comp)][spot_index]
            else:
                spot.longitude = spots_longitudes[str(comp)][spot_index][index]
