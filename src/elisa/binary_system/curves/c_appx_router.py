from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.interpolate import Akima1DInterpolator

from elisa import const, settings, utils
from elisa.binary_system import dynamic
from elisa.binary_system.curves import c_managed
from elisa.binary_system.curves import utils as crv_utils
from elisa.binary_system.orbit.container import OrbitalSupplements
from elisa.binary_system.orbit.orbit import (
    component_distance_from_mean_anomaly,
    get_approx_ecl_angular_width,
)
from elisa.logger import getLogger
from elisa.observer.mp_manager import manage_observations

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from elisa.binary_system.system import BinarySystem
    from elisa.types import NP_BOOL_, Float


logger = getLogger("binary_system.curves.c_appx_router")

# Constant to avoid magic number lint warning
_PHASES_COUNT_THRESHOLD = 10


def look_for_approximation(*, not_pulsations_test: bool) -> bool:
    """Determine whether any eccentric-curve approximation should be attempted.

    The decision depends on the current approximation settings and on whether
    pulsations are absent. Some approximation modes also require valid numeric
    thresholds to be configured.

    :param not_pulsations_test: Flag indicating that pulsations are not present.
    :type not_pulsations_test: bool
    :returns: ``True`` if at least one approximation mode is enabled and valid,
        and pulsations do not prevent approximation use.
    :rtype: bool
    """
    valid_nu_separation = settings.MAX_NU_SEPARATION is not None and settings.MAX_NU_SEPARATION > 0
    valid_max_d_flux = settings.MAX_D_FLUX is not None and settings.MAX_D_FLUX > 0

    interp_approx = valid_nu_separation and settings.USE_INTERPOLATION_APPROXIMATION
    symmetrical_approx = settings.USE_SYMMETRICAL_COUNTERPARTS_APPROXIMATION
    neighbour_approx = valid_max_d_flux and settings.USE_SIMILAR_NEIGHBOURS_APPROXIMATION

    approximation_enabled = interp_approx or neighbour_approx or symmetrical_approx
    return approximation_enabled and not_pulsations_test


def resolve_ecc_approximation_method(
    binary: BinarySystem,
    phases: NDArray[Float],
    position_method: Callable[..., NDArray[Float]],
    crv_labels: Sequence[str],
    curve_fn: Callable[..., dict[str, NDArray[Float]]],
    *,
    try_to_find_appx: bool,
    phases_span_test: bool,
    **kwargs: Any,
) -> tuple[str, Callable[[], dict[str, NDArray[Float]]]]:
    """Resolve the approximation method for eccentric curve computation.

    This function selects the most suitable integration strategy for an
    eccentric binary system. If no approximation can be used, it falls back
    to exact integration. The returned callable is preconfigured and can be
    executed later without additional arguments.

    Approximation methods are evaluated in the following order:

    1. Exact integration fallback
    2. Interpolation approximation
    3. Symmetrical-counterparts approximation
    4. Similar-neighbors approximation

    :param binary: Binary-system instance.
    :type binary: elisa.binary_system.system.BinarySystem
    :param phases: Observation phases for which the curve should be computed.
    :type phases: NDArray[Float]
    :param position_method: Callable returning orbital positions for the given
                            phase input.
    :type position_method: Callable[..., NDArray[Float]]
    :param try_to_find_appx: Flag controlling whether approximation methods
                             should be evaluated.
    :type try_to_find_appx: bool
    :param phases_span_test: Test indicating whether phase coverage is
                             sufficient for apsidal-line mirroring.
    :type phases_span_test: bool
    :param crv_labels: Labels of the calculated curves, such as passbands or
                       component identifiers.
    :type crv_labels: Sequence[str]
    :param curve_fn: Low-level curve integration function. Should return a
                     mapping from label to flux array when called.
    :type curve_fn: Callable[..., dict[str, numpy.ndarray]]
    :param kwargs: Additional keyword arguments forwarded to the selected
                   integration routine. Supported keys may include passband
                   containers, bandwidth limits, atlas selection, and other
                   observer options.
    :type kwargs: dict[str, object]
    :returns: Tuple containing the approximation identifier and a zero-argument
             callable that performs the selected integration.
    :rtype: tuple[str, Callable[[], dict[str, NDArray[Float]]]]
    """
    approx_method_list: list[Callable[..., dict[str, NDArray[Float]]]] = [
        integrate_eccentric_curve_exactly,
        integrate_eccentric_curve_interp_appx,
        integrate_eccentric_curve_symmetrical_counterparts_appx,
        integrate_eccentric_curve_similar_neighbours_appx,
    ]

    params = {
        "input_argument": phases,
        "return_nparray": True,
        "calculate_from": "phase",
    }
    all_orbital_pos_arr = position_method(**params)
    all_orbital_pos = utils.convert_binary_orbital_motion_arr_to_positions(
        all_orbital_pos_arr,
    )
    potentials = binary.correct_potentials(phases, component="all", iterations=2)

    # APPX ZERO
    if not try_to_find_appx:
        args = (binary, all_orbital_pos, potentials, crv_labels, curve_fn)
        return "zero", lambda: approx_method_list[0](*args, **kwargs)

    # calculating components radii for each orbital position
    radii = crv_utils.forward_radii_from_distances(
        binary,
        all_orbital_pos_arr[:, 1],
        potentials,
    )

    azimuths = all_orbital_pos_arr[:, 2]
    _, counterpart_position_arr, reduced_phase_mask = crv_utils.prepare_apsidaly_symmetric_orbit(
        binary,
        azimuths,
        phases,
    )

    # spliting orbital motion into two separate groups on different sides of apsidal line
    reduced_orbit_arr, reduced_orbit_supplement_arr = crv_utils.split_orbit_by_apse_line(
        all_orbital_pos_arr,
        reduced_phase_mask,
    )

    # APPX ONE
    # call with phases_span_test as a keyword to keep boolean args keyword-only
    interp_appx, reduced_orbit_arr, counterpart_position_arr = eval_interpolation_approximation(
        binary,
        reduced_orbit_arr,
        counterpart_position_arr,
        reduced_orbit_supplement_arr,
        phases_span_test=phases_span_test,
    )
    if interp_appx:
        args = (
            binary,
            radii,
            phases,
            reduced_orbit_arr,
            counterpart_position_arr,
            potentials,
            crv_labels,
            curve_fn,
        )
        return "one", lambda: approx_method_list[1](*args, **kwargs)

    # APPX TWO
    # evaluate symmetrical counterparts approximation with keyword-only flag
    symmetrical_counterparts_appx, orbital_supplements = eval_symmetrical_counterparts_approximation(
        binary,
        radii,
        reduced_orbit_arr,
        reduced_orbit_supplement_arr,
        phases_span_test=phases_span_test,
    )
    if symmetrical_counterparts_appx:
        args = (
            binary,
            radii,
            phases,
            orbital_supplements,
            potentials,
            crv_labels,
            curve_fn,
        )
        return "two", lambda: approx_method_list[2](*args, **kwargs)

    # APPX THREE
    neighbour_args = (binary, radii, all_orbital_pos_arr)
    similar_neighbours_appx, new_geometry_mask, sorted_positions = eval_similar_neighbours_approximation(
        *neighbour_args,
    )
    if similar_neighbours_appx:
        args = (
            binary,
            sorted_positions,
            new_geometry_mask,
            potentials,
            crv_labels,
            curve_fn,
        )
        return "three", lambda: approx_method_list[3](*args, **kwargs)

    args = (binary, all_orbital_pos, potentials, crv_labels, curve_fn)
    return "zero", lambda: approx_method_list[0](*args, **kwargs)


def eval_interpolation_approximation(
    binary: BinarySystem,
    reduced_orbit_array: NDArray[Float],
    counterpart_position_array: NDArray[Float],
    reduced_orbit_supplement_arr: NDArray[Float],
    *,
    phases_span_test: bool,
) -> tuple[bool, NDArray[Float], NDArray[Float]]:
    """Evaluate whether interpolation approximation can be used.

     The approximation computes one side of the apsidal line directly and
     interpolates values for the other side. To be valid, the orbit sampling
     must be dense enough, phase coverage must be sufficient, eclipse regions
     must be adequately populated, and interpolation must not introduce
     artifacts in flat eclipse plateaus.

    :param binary: Binary-system instance.
    :type binary: elisa.binary_system.system.BinarySystem
    :param phases_span_test: Test for sufficient phase span of observations.
    :type phases_span_test: bool
    :param reduced_orbit_array: Orbital positions defined by the user on one
                                side of the apsidal line.
    :type reduced_orbit_array: NDArray[Float]
    :param counterpart_position_array: Symmetrical counterparts to
                                       ``reduced_orbit_array``.
    :type counterpart_position_array: NDArray[Float]
    :param reduced_orbit_supplement_arr: Orbital positions defined by the user
                                         on the opposite side of the apsidal
                                         line.
    :type reduced_orbit_supplement_arr: NDArray[Float]
    :return: Tuple containing the approximation decision and the possibly
             augmented reduced orbit and counterpart arrays.
    :rtype: tuple[bool, NDArray[Float], NDArray[Float]]
    """
    true_anomalies_supplements = reduced_orbit_supplement_arr[:, 3]

    max_nu_sep = np.max(np.diff(np.sort(true_anomalies_supplements)))
    interpolation_disabled = not settings.USE_INTERPOLATION_APPROXIMATION
    invalid_nu_threshold = settings.MAX_NU_SEPARATION is None or settings.MAX_NU_SEPARATION <= 0
    insufficient_sampling = (
        invalid_nu_threshold
        or max_nu_sep > settings.MAX_NU_SEPARATION
        or not phases_span_test
        or interpolation_disabled
    )
    if insufficient_sampling:
        logger.debug(
            "Orbit is not sufficiently populated to implement interpolation approximation 1.",
        )
        return False, reduced_orbit_array, counterpart_position_array

    ecl_true_anomalies = np.array(
        [
            binary.orbit.conjunctions[f"{component}_eclipse"]["true_anomaly"]
            for component in settings.BINARY_COUNTERPARTS
        ],
    )
    distances_at_ecl = component_distance_from_mean_anomaly(
        binary.eccentricity,
        ecl_true_anomalies,
    )

    angular_ecl_widths = [
        get_approx_ecl_angular_width(
            binary.primary.forward_radius,
            binary.secondary.forward_radius,
            distance,
            binary.inclination,
        )
        for distance in distances_at_ecl
    ]

    for ii, ecl_nu in enumerate(ecl_true_anomalies):
        if angular_ecl_widths[ii][0] == 0.0:
            continue

        # including adjacent points to the eclipse to ensure smoothness
        d_nu1 = crv_utils.adjust_eclipse_width(
            true_anomalies_supplements,
            ecl_nu - angular_ecl_widths[ii][0],
        )
        d_nu2 = crv_utils.adjust_eclipse_width(
            true_anomalies_supplements,
            ecl_nu + angular_ecl_widths[ii][0],
        )

        bottom = ecl_nu - angular_ecl_widths[ii][0] - d_nu1
        top = ecl_nu + angular_ecl_widths[ii][0] + d_nu2

        points_ecl_mask_supplements = np.logical_and(
            true_anomalies_supplements > bottom,
            true_anomalies_supplements < top,
        )

        # treating eclipses on boundaries of 0, 2pi interval
        if bottom < 0.0:
            points_ecl_mask_supplements = np.logical_or(
                points_ecl_mask_supplements,
                true_anomalies_supplements > bottom + const.FULL_ARC,
            )
        elif top > const.FULL_ARC:
            points_ecl_mask_supplements = np.logical_or(
                points_ecl_mask_supplements,
                true_anomalies_supplements < top - const.FULL_ARC,
            )

        points_in_ecl_supplements = np.sum(points_ecl_mask_supplements)

        # subtraction of the central plateau
        plateau_factor = 1 - angular_ecl_widths[ii][1] / angular_ecl_widths[ii][0]

        if plateau_factor * points_in_ecl_supplements < settings.MIN_POINTS_IN_ECLIPSE:
            reduced_orbit_array = np.vstack(
                (
                    reduced_orbit_array,
                    reduced_orbit_supplement_arr[points_ecl_mask_supplements],
                ),
            )
            counterpart_position_array = np.vstack(
                (
                    counterpart_position_array,
                    np.full((points_in_ecl_supplements, 5), np.nan),
                ),
            )
        else:
            # interpolation approximation causes artifacts in case of the very
            # flat plateaus in the bottom of the eclipse
            return False, reduced_orbit_array, counterpart_position_array

    # removing duplicate entries
    _, indices = np.unique(reduced_orbit_array[:, 0], return_index=True)
    reduced_orbit_array = reduced_orbit_array[indices]
    counterpart_position_array = counterpart_position_array[indices]

    return True, reduced_orbit_array, counterpart_position_array


def eval_symmetrical_counterparts_approximation(
    binary: BinarySystem,
    radii: NDArray[Float],
    base_orbit_arr: NDArray[Float],
    orbit_supplement_arr: NDArray[Float],
    *,
    phases_span_test: bool,
) -> tuple[bool, OrbitalSupplements | None]:
    """Evaluate whether symmetrical-counterparts approximation can be used.

    This approximation pairs orbital positions across the apsidal line and
    reuses the same surface geometry for both positions when the pairing is
    acceptable.

    :param binary: Binary-system instance.
    :type binary: elisa.binary_system.system.BinarySystem
    :param radii: Forward radii associated with orbital positions.
    :type radii: NDArray[Float]
    :param base_orbit_arr: Base orbital positions.
    :type base_orbit_arr: NDArray[Float]
    :param orbit_supplement_arr: Supplementary orbital positions located on the
                                 opposite side of the apsidal line.
    :type orbit_supplement_arr: NDArray[Float]
    :param phases_span_test: Test for sufficient phase span of observations.
    :type phases_span_test: bool
    :return: Tuple containing the approximation decision and the constructed
             orbital supplements. If the approximation is not valid, the second
             item is ``None``.
    :rtype: tuple[bool, OrbitalSupplements | None]
    """
    if not phases_span_test or not settings.USE_SYMMETRICAL_COUNTERPARTS_APPROXIMATION:
        logger.debug(
            "Phase span of the observation is not sufficient to utilize symmetrical counterparts approximation.",
        )
        return False, None

    orbital_supplements = dynamic.find_apsidally_corresponding_positions(
        binary,
        radii,
        base_orbit_arr,
        orbit_supplement_arr,
    )
    orbital_supplements = orbital_supplements.sort(by="distance")

    idxs_where_nan = np.argwhere(np.isnan(orbital_supplements.mirror[:, 0]))
    if idxs_where_nan.shape[0] == orbital_supplements.mirror.shape[0]:
        return False, None

    return True, orbital_supplements


def eval_similar_neighbours_approximation(
    binary: BinarySystem,
    radii: NDArray[Float],
    all_orbital_pos_arr: NDArray[Float],
) -> tuple[bool, NDArray[NP_BOOL_] | None, NDArray[Float] | None]:
    """Evaluate whether similar-neighbours approximation can be used.

    This approximation avoids rebuilding surface geometry for neighbouring
    orbital positions whose geometry and irradiation changes are sufficiently
    small.

    :param binary: Binary-system instance.
    :type binary: elisa.binary_system.system.BinarySystem
    :param radii: Forward radii for ``all_orbital_pos_arr``.
    :type radii: NDArray[Float]
    :param all_orbital_pos_arr: Array of all orbital positions.
    :type all_orbital_pos_arr: NDArray[Float]
    :return: Tuple containing the approximation decision, the mask indicating
        where full geometry updates are required, and orbital positions
        sorted by component distance. If the approximation is unavailable,
        the last two items are ``None``.
    :rtype: tuple[bool, NDArray[bool] | None, NDArray[Float] | None]
    """
    if not settings.USE_SIMILAR_NEIGHBOURS_APPROXIMATION:
        return False, None, None

    sort_idxs = all_orbital_pos_arr[:, 1].argsort()
    sorted_all_orbital_pos_arr = all_orbital_pos_arr[sort_idxs]
    sorted_radii = radii[:, sort_idxs]

    # Compare neighbouring geometries pairwise in the sorted sequence.
    rel_d_radii = crv_utils.compute_rel_d_geometry(
        binary,
        sorted_radii[:, :-1],
        sorted_radii[:, 1:],
    )
    geometry_args = (
        binary.has_spots(),
        all_orbital_pos_arr.shape[0],
        rel_d_radii,
    )
    new_geometry_mask = dynamic.resolve_object_geometry_update(*geometry_args)

    rel_irrad = crv_utils.compute_rel_d_irradiation(
        binary,
        sorted_all_orbital_pos_arr[:, 1],
    )
    new_irrad_mask = dynamic.resolve_irrad_update(
        rel_irrad,
        all_orbital_pos_arr.shape[0],
    )
    new_build_mask = np.logical_or(new_geometry_mask, new_irrad_mask)

    approx_test = not new_build_mask.all()
    return approx_test, new_build_mask, sorted_all_orbital_pos_arr


def integrate_eccentric_curve_interp_appx(
    binary: BinarySystem,
    radii: NDArray[Float],
    phases: NDArray[Float],
    reduced_orbit_arr: NDArray[Float],
    counterpart_position_arr: NDArray[Float],
    potentials: dict[str, NDArray[Float]],
    crv_labels: Sequence[str],
    curve_fn: Callable[..., dict[str, NDArray[Float]]],
    **kwargs: Any,
) -> dict[str, NDArray[Float]]:
    """Integrate an eccentric curve using interpolation approximation.

    Curve points on one side of the apsidal line are calculated exactly.
    Points on the opposite side reuse mirrored surface geometries, and the
    resulting fluxes are interpolated onto the user-defined phase grid using
    an Akima interpolator.

    :param binary: Binary-system instance.
    :type binary: elisa.binary_system.system.BinarySystem
    :param radii: Forward radii.
    :type radii: NDArray[Float]
    :param phases: Requested phase grid.
    :type phases: NDArray[Float]
    :param reduced_orbit_arr: Base orbital positions.
    :type reduced_orbit_arr: NDArray[Float]
    :param counterpart_position_arr: Orbital positions symmetric to
                                     ``reduced_orbit_arr``.
    :type counterpart_position_arr: NDArray[Float]
    :param potentials: Corrected potentials.
    :type potentials: dict[str, ArrayLike]
    :param crv_labels: Curve labels.
    :type crv_labels: Sequence[str]
    :param curve_fn: Curve integrator function.
    :type curve_fn: Callable[..., object]
    :param kwargs: Additional integration keyword arguments such as passband,
                   left bandwidth, right bandwidth, or atlas.
    :type kwargs: dict[str, object]
    :return: Mapping of curve labels to interpolated flux arrays.
    :rtype: dict[str, NDArray[Float]]
    """
    n = 5 if phases.shape[0] > _PHASES_COUNT_THRESHOLD else int(phases.shape[0] / 2) - 1

    orbital_supplements = OrbitalSupplements(
        body=reduced_orbit_arr,
        mirror=counterpart_position_arr,
    )
    orbital_supplements = orbital_supplements.sort(by="distance")

    orbital_positions = np.stack(
        (orbital_supplements.body, orbital_supplements.mirror),
        axis=1,
    )
    fn_args = (binary, potentials, radii, crv_labels, curve_fn)
    fn = c_managed.integrate_eccentric_curve_w_orbital_symmetry
    stacked_band_curves = manage_observations(
        fn=fn,
        fn_args=fn_args,
        position=orbital_positions,
        **kwargs,
    )

    # interpolation of the points in the second half of the light curves using splines
    x = np.concatenate(
        (orbital_supplements.body[:, 4], orbital_supplements.mirror[:, 4]),
    )
    not_nan_test = ~np.isnan(x)
    x = x[not_nan_test] % 1

    # checking for accidental alignment between templates and mirrors
    x, unique_idx = np.unique(x, return_index=True)

    # np.unique already returns a sorted array, so no further sort is needed
    x = np.concatenate((x[-n:] - 1, x, x[:n] + 1))

    band_curves: dict[str, NDArray[Float]] = {}
    for curve in crv_labels:
        y = np.concatenate(
            (stacked_band_curves[curve][:, 0], stacked_band_curves[curve][:, 1]),
        )
        y = (y[not_nan_test])[unique_idx]
        y = np.concatenate((y[-n:], y, y[:n]))

        interpolator = Akima1DInterpolator(x, y)
        band_curves[curve] = interpolator(phases)

    return band_curves


def integrate_eccentric_curve_symmetrical_counterparts_appx(
    binary: BinarySystem,
    radii: NDArray[Float],
    phases: NDArray[Float],
    orbital_supplements: OrbitalSupplements,
    potentials: dict[str, NDArray[Float]],
    crv_labels: Sequence[str],
    curve_fn: Callable[..., dict[str, NDArray[Float]]],
    **kwargs: Any,
) -> dict[str, NDArray[Float]]:
    """Integrate an eccentric curve using symmetrical-counterparts approximation.

    For each orbital position on one side of the apsidal line, the closest
    apsidally symmetric counterpart is assigned and the same surface geometry
    is reused for both positions.

    :param binary: Binary-system instance.
    :type binary: elisa.binary_system.system.BinarySystem
    :param radii: Forward radii.
    :type radii: NDArray[Float]
    :param phases: Requested phase grid.
    :type phases: NDArray[Float]
    :param orbital_supplements: Paired base and mirrored orbital positions.
    :type orbital_supplements: OrbitalSupplements
    :param potentials: Corrected potentials.
    :type potentials: dict[str, ArrayLike]
    :param crv_labels: Curve labels.
    :type crv_labels: Sequence[str]
    :param curve_fn: Curve integrator function.
    :type curve_fn: Callable[..., object]
    :param kwargs: Additional integration keyword arguments such as passband,
                   left bandwidth, and right bandwidth.
    :type kwargs: dict[str, object]
    :return: Mapping of curve labels to flux arrays restored to the original
             phase ordering.
    :rtype: dict[str, NDArray[Float]]
    """
    orbital_positions = np.stack(
        (orbital_supplements.body, orbital_supplements.mirror),
        axis=1,
    )
    fn_args = (binary, potentials, radii, crv_labels, curve_fn)
    fn = c_managed.integrate_eccentric_curve_w_orbital_symmetry
    stacked_band_curves = manage_observations(
        fn=fn,
        fn_args=fn_args,
        position=orbital_positions,
        **kwargs,
    )

    # hoist loop-invariant index arrays outside the per-label loop
    base_idxs = np.array(orbital_supplements.body[:, 0], dtype=np.int32)
    not_nan_test = ~np.isnan(orbital_supplements.mirror[:, 0])
    mirror_idxs = np.array(
        orbital_supplements.mirror[not_nan_test, 0],
        dtype=np.int32,
    )

    band_curves = {key: np.empty(phases.shape) for key in crv_labels}
    for lbl in crv_labels:
        band_curves[lbl][base_idxs] = stacked_band_curves[lbl][:, 0]
        band_curves[lbl][mirror_idxs] = stacked_band_curves[lbl][not_nan_test, 1]

    return band_curves


def integrate_eccentric_curve_similar_neighbours_appx(
    binary: BinarySystem,
    orbital_positions: NDArray[Float],
    new_geometry_mask: NDArray[NP_BOOL_],
    potentials: dict[str, NDArray[Float]],
    crv_labels: Sequence[str],
    curve_fn: Callable[..., dict[str, NDArray[Float]]],
    **kwargs: Any,
) -> dict[str, NDArray[Float]]:
    """Integrate an eccentric curve using similar-neighbours approximation.

    Surface geometry is not fully recalculated between sufficiently similar
    neighbouring orbital positions. The results are later reordered back to
    the original phase ordering.

    :param binary: Binary-system instance.
    :type binary: elisa.binary_system.system.BinarySystem
    :param orbital_positions: Orbital positions sorted by component distance.
    :type orbital_positions: NDArray[Float]
    :param new_geometry_mask: Mask indicating which positions require full
                              surface-geometry recalculation.
    :type new_geometry_mask: NDArray[bool]
    :param potentials: Corrected surface potentials.
    :type potentials: dict[str, ArrayLike]
    :param crv_labels: Curve labels.
    :type crv_labels: Sequence[str]
    :param curve_fn: Curve integrator function.
    :type curve_fn: Callable[..., object]
    :param kwargs: Additional integration keyword arguments such as passband,
                   left bandwidth, and right bandwidth.
    :type kwargs: dict[str, object]
    :return: Mapping of curve labels to flux arrays in original order.
    :rtype: dict[str, NDArray[Float]]
    """
    fn_args = (binary, potentials, new_geometry_mask, crv_labels, curve_fn)
    fn = c_managed.similar_neighbour_approximation_ecc_curve_integration
    band_curves_unsorted = manage_observations(
        fn=fn,
        fn_args=fn_args,
        position=orbital_positions,
        **kwargs,
    )

    sort_idx = orbital_positions[:, 0].argsort()
    return {key: band_curves_unsorted[key][sort_idx] for key in crv_labels}


def integrate_eccentric_curve_exactly(
    binary: BinarySystem,
    orbital_motion: NDArray[Float],
    potentials: dict[str, NDArray[Float]],
    crv_labels: Sequence[str],
    curve_fn: Callable[..., dict[str, NDArray[Float]]],
    **kwargs: Any,
) -> dict[str, NDArray[Float]]:
    """Integrate an eccentric curve exactly for all orbital positions.

    Each orbital position is evaluated independently without approximation.
    This mode is slow and is mainly useful as a benchmark or fallback.

    :param binary: Binary-system instance.
    :type binary: elisa.binary_system.system.BinarySystem
    :param orbital_motion: All orbital positions at which the curve will be
                           calculated.
    :type orbital_motion: Sequence[ArrayLike]
    :param potentials: Corrected surface potentials.
    :type potentials: dict[str, ArrayLike]
    :param crv_labels: Labels of the calculated curves, such as passbands or
                       components.
    :type crv_labels: Sequence[str]
    :param curve_fn: Curve integration function.
    :type curve_fn: Callable[..., object]
    :param kwargs: Additional keyword arguments forwarded from the eccentric
                   curve production pipeline.
    :type kwargs: dict[str, object]
    :return: Dictionary of flux arrays for each curve label.
    :rtype: dict[str, NDArray[Float]]
    """
    fn_args = (binary, potentials, None, crv_labels, curve_fn)
    fn = c_managed.integrate_eccentric_curve_exactly
    return manage_observations(
        fn=fn,
        fn_args=fn_args,
        position=orbital_motion,
        **kwargs,
    )
