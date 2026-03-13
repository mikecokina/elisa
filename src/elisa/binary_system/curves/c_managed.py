from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Any

import numpy as np

from elisa import const, settings, utils
from elisa.base.types import FLOAT
from elisa.binary_system import dynamic
from elisa.binary_system import utils as bsutils
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.binary_system.curves import utils as crv_utils
from elisa.binary_system.orbit.container import OrbitalSupplements
from elisa.binary_system.surface.coverage import compute_surface_coverage
from elisa.binary_system.surface.mesh import add_spots_to_mesh

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from elisa.binary_system.system import BinarySystem
    from elisa.const import Position
    from elisa.types import Float


def produce_circ_sync_curves_mp(
    binary: BinarySystem,
    initial_system: OrbitalPositionContainer,
    phase_batch: NDArray[Float],
    crv_labels: list[str],
    curves_fn: Callable[..., dict[str, NDArray[Float]]],
    kwargs: dict[str, Any],
) -> dict[str, NDArray[Float]]:
    """Generate light/radial-velocity curves for circular synchronous systems.

    This function is designed to be used as a multiprocessing worker and is
    called via :func:`elisa.observer.mp_manager.manage_observations`.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param initial_system: Orbital position container with pre-built geometry.
    :type initial_system: OrbitalPositionContainer
    :param phase_batch: Orbital phases for this worker batch.
    :type phase_batch: NDArray[Float]
    :param crv_labels: Curve band / observable labels.
    :type crv_labels: list[str]
    :param curves_fn: Function that accumulates curve values at each orbital
        position.
    :type curves_fn: Callable[..., dict[str, NDArray[Float]]]
    :param kwargs: Extra keyword arguments forwarded to surface parameter
        helpers (passband, bandwidth, ``position_method``, etc.).
    :type kwargs: dict[str, Any]
    :return: Dictionary mapping each curve label to its computed values.
    :rtype: dict[str, NDArray[Float]]
    """
    position_method = kwargs.pop("position_method")
    orbital_motion = position_method(input_argument=phase_batch, return_nparray=False, calculate_from="phase")
    # is in eclipse test eval
    ecl_boundaries = dynamic.get_eclipse_boundaries(binary, 1.0)
    azimuths = np.asarray([position.azimuth for position in orbital_motion], dtype=FLOAT)
    in_eclipse = dynamic.in_eclipse_test(azimuths, ecl_boundaries)

    curves = {key: np.zeros(phase_batch.shape) for key in crv_labels}

    for pos_idx, position in enumerate(orbital_motion):
        on_pos = bsutils.move_sys_onpos(initial_system, position, on_copy=True)

        compute_surface_coverage(
            on_pos,
            binary.semi_major_axis,
            in_eclipse=in_eclipse[pos_idx],
            return_values=False,
            write_to_containers=True,
        )

        curves = curves_fn(curves, pos_idx, crv_labels, on_pos)

    return curves


def produce_circ_spotty_async_curves_mp(
    binary: BinarySystem,
    initial_system: OrbitalPositionContainer,
    motion_batch: list[Position],
    base_points: dict[str, NDArray[Float]],
    ecl_boundaries: NDArray[Float],
    crv_labels: list[str],
    curve_fn: Callable[..., dict[str, NDArray[Float]]],
    kwargs: dict[str, Any],
) -> dict[str, NDArray[Float]]:
    """Generate light/radial-velocity curves for circular asynchronous spotty systems.

    This function is designed to be used as a multiprocessing worker and is
    called via :func:`elisa.observer.mp_manager.manage_observations`.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param initial_system: Orbital position container with pre-built geometry
        and clean (spot-free) surface points.
    :type initial_system: OrbitalPositionContainer
    :param motion_batch: Orbital positions for this worker batch.
    :type motion_batch: list[Position]
    :param base_points: Clean surface mesh points keyed by component name,
        used as the baseline before spot geometry is added.
    :type base_points: dict[str, NDArray[Float]]
    :param ecl_boundaries: Eclipse boundary azimuths for both eclipses.
    :type ecl_boundaries: NDArray[Float]
    :param crv_labels: Curve band / observable labels.
    :type crv_labels: list[str]
    :param curve_fn: Function that accumulates curve values at each orbital
        position.
    :type curve_fn: Callable[..., dict[str, NDArray[Float]]]
    :param kwargs: Extra keyword arguments forwarded to surface parameter
        helpers.
    :type kwargs: dict[str, Any]
    :return: Dictionary mapping each curve label to its computed values.
    :rtype: dict[str, NDArray[Float]]
    """
    # pre-calculate the longitudes of each spot for each phase
    phases = np.array([val.phase for val in motion_batch])
    azimuths = np.asarray([position.azimuth for position in motion_batch], dtype=FLOAT)
    in_eclipse = dynamic.in_eclipse_test(azimuths, ecl_boundaries)
    spots_longitudes = dynamic.calculate_spot_longitudes(binary, phases, component="all", correct_libration=False)
    pulsation_tests = {"primary": binary.primary.has_pulsations(), "secondary": binary.secondary.has_pulsations()}
    primary_reducer, secondary_reducer = dynamic.resolve_spots_geometry_update(
        spots_longitudes,
        len(phases),
        pulsation_tests,
    )
    combined_reducer = primary_reducer & secondary_reducer

    # calculating lc with spots gradually shifting their positions in each phase
    curves = {key: np.empty(len(motion_batch)) for key in crv_labels}
    normal_radiance, ld_cfs = None, None
    for pos_idx, orbital_position in enumerate(motion_batch):
        initial_system.set_on_position_params(position=orbital_position)
        initial_system.time = initial_system.set_time()
        # setup component necessary to build/rebuild

        require_build = (
            "all"
            if combined_reducer[pos_idx]
            else "primary"
            if primary_reducer[pos_idx]
            else "secondary"
            if secondary_reducer[pos_idx]
            else None
        )

        # use clear system surface points as a starting place to save a time
        # if reducers for related component is set to False, previous build will be used

        if primary_reducer[pos_idx]:
            initial_system.primary.points = copy(base_points["primary"])
        if secondary_reducer[pos_idx]:
            initial_system.secondary.points = copy(base_points["secondary"])

        # assigning new longitudes for each spot
        dynamic.assign_spot_longitudes(
            initial_system,
            spots_longitudes,
            index=pos_idx,
            component=require_build,
        )

        # build the spots points
        add_spots_to_mesh(initial_system, orbital_position.distance, component=require_build)
        # build the rest of the surface based on preset surface points
        _build_args = {"components_distance": orbital_position.distance, "component": require_build}
        initial_system.build_faces_and_kinematic_quantities(**_build_args)
        initial_system.build_temperature_distribution(components_distance=orbital_position.distance, component="all")

        if initial_system.has_pulsations():
            on_pos = initial_system.copy()
            on_pos.flat_it()
            on_pos.build_pulsations(components_distance=orbital_position.distance, component="all")
            on_copy, sys_to_rotate = False, on_pos
        else:
            on_copy, sys_to_rotate = True, initial_system

        on_pos = bsutils.move_sys_onpos(sys_to_rotate, orbital_position, on_copy=on_copy)

        # if None of components has to be rebuilt, use previously computed radiances and limbdarkening when available
        require_build_test = require_build is not None
        on_pos, normal_radiance, ld_cfs = crv_utils.update_surface_params(
            on_pos,
            normal_radiance,
            ld_cfs,
            require_rebuild=require_build_test,
            **kwargs,
        )

        _kwargs = {"in_eclipse": in_eclipse[pos_idx], "return_values": False, "write_to_containers": True}
        compute_surface_coverage(on_pos, binary.semi_major_axis, **_kwargs)

        curves = curve_fn(curves, pos_idx, crv_labels, on_pos)

    return curves


def produce_circ_pulsating_curves_mp(
    binary: BinarySystem,
    initial_system: OrbitalPositionContainer,
    phase_batch: NDArray[Float],
    crv_labels: list[str],
    curves_fn: Callable[..., dict[str, NDArray[Float]]],
    kwargs: dict[str, Any],
) -> dict[str, NDArray[Float]]:
    """Generate light/radial-velocity curves for circular pulsating systems.

    This function is designed to be used as a multiprocessing worker and is
    called via :func:`elisa.observer.mp_manager.manage_observations`.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param initial_system: Orbital position container with pre-built geometry.
    :type initial_system: OrbitalPositionContainer
    :param phase_batch: Orbital phases for this worker batch.
    :type phase_batch: NDArray[Float]
    :param crv_labels: Curve band / observable labels.
    :type crv_labels: list[str]
    :param curves_fn: Function that accumulates curve values at each orbital
        position.
    :type curves_fn: Callable[..., dict[str, NDArray[Float]]]
    :param kwargs: Extra keyword arguments forwarded to surface parameter
        helpers (passband, bandwidth, ``position_method``, etc.).
    :type kwargs: dict[str, Any]
    :return: Dictionary mapping each curve label to its computed values.
    :rtype: dict[str, NDArray[Float]]
    """
    position_method = kwargs.pop("position_method")
    orbital_motion = position_method(
        input_argument=phase_batch,
        return_nparray=False,
        calculate_from="phase",
    )
    # is in eclipse test eval
    ecl_boundaries = dynamic.get_eclipse_boundaries(binary, 1.0)
    azimuths = np.asarray([position.azimuth for position in orbital_motion], dtype=FLOAT)
    in_eclipse = dynamic.in_eclipse_test(azimuths, ecl_boundaries)

    curves = {key: np.zeros(phase_batch.shape) for key in crv_labels}

    for pos_idx, position in enumerate(orbital_motion):
        on_pos = initial_system.copy()
        on_pos.set_on_position_params(position)
        on_pos.set_time()

        on_pos.build_pulsations(components_distance=position.distance)
        crv_utils.prep_surface_params(on_pos, return_values=False, write_to_containers=True, **kwargs)
        on_pos = bsutils.move_sys_onpos(on_pos, position, on_copy=False, recalculate_velocities=False)

        compute_surface_coverage(
            on_pos,
            binary.semi_major_axis,
            in_eclipse=in_eclipse[pos_idx],
            return_values=False,
            write_to_containers=True,
        )

        curves = curves_fn(curves, pos_idx, crv_labels, on_pos)

    return curves


def integrate_eccentric_curve_exactly(
    binary: BinarySystem,
    potentials: dict[str, NDArray[Float]],
    motion_batch: list[Position],
    spots_longitudes: dict[str, dict[int, Float | NDArray[Float]]] | None,
    crv_labels: list[str],
    curve_fn: Callable[..., dict[str, NDArray[Float]]],
    kwargs: dict[str, Any],
) -> dict[str, NDArray[Float]]:
    """Generate curves via exact per-position integration for eccentric orbits.

    Each orbital position is built fully from scratch; no geometry
    approximations are applied. This function is designed to be used as a
    multiprocessing worker and is called via
    :func:`elisa.observer.mp_manager.manage_observations`.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param potentials: Corrected surface potentials keyed by component name.
    :type potentials: dict[str, NDArray[Float]]
    :param motion_batch: Orbital positions for this worker batch.
    :type motion_batch: list[Position]
    :param spots_longitudes: Precomputed spot longitudes per component and spot
        index, or ``None`` when the system has no spots.
    :type spots_longitudes: dict[str, dict[int, Float | NDArray[Float]]] | None
    :param crv_labels: Curve band / observable labels.
    :type crv_labels: list[str]
    :param curve_fn: Function that accumulates curve values at each orbital
        position.
    :type curve_fn: Callable[..., dict[str, NDArray[Float]]]
    :param kwargs: Extra keyword arguments forwarded to surface parameter
        helpers.
    :type kwargs: dict[str, Any]
    :return: Dictionary mapping each curve label to its computed values.
    :rtype: dict[str, NDArray[Float]]
    """
    curves = {key: np.empty(len(motion_batch)) for key in crv_labels}
    for run_idx, position in enumerate(motion_batch):
        pos_idx = int(position.idx)
        from_this = {"binary_system": binary, "position": position}
        on_pos = OrbitalPositionContainer.from_binary_system(**from_this)
        dynamic.assign_spot_longitudes(on_pos, spots_longitudes, index=pos_idx, component="all")
        on_pos.set_on_position_params(position, potentials["primary"][pos_idx], potentials["secondary"][pos_idx])
        on_pos.build(components_distance=position.distance)
        on_pos = bsutils.move_sys_onpos(on_pos, position, on_copy=False)

        crv_utils.prep_surface_params(on_pos, return_values=False, write_to_containers=True, **kwargs)
        # TODO: properly calculate in_eclipse parameter  # noqa: FIX002, TD002, TD003
        _kwargs = {"in_eclipse": True, "return_values": False, "write_to_containers": True}
        compute_surface_coverage(on_pos, binary.semi_major_axis, **_kwargs)

        curves = curve_fn(curves, run_idx, crv_labels, on_pos)
    return curves


# managing approximations in eccentric orbits ##########################################################################


def _update_surface_in_ecc_orbits(
    system: OrbitalPositionContainer,
    orbital_position: Position,
    *,
    new_geometry_test: bool,
) -> OrbitalPositionContainer:
    """Update the surface of an eccentric-orbit container for the given position.

    Decides how to update surface properties based on how much the geometry has
    changed since the last evaluation.  When ``new_geometry_test`` is ``True``
    the surface is rebuilt completely from scratch; otherwise only the symmetric
    mesh, face orientation, and surface areas are refreshed, which is faster.

    :param system: Orbital position container whose surface is to be updated.
    :type system: OrbitalPositionContainer
    :param orbital_position: Orbital position for which the surface is updated.
    :type orbital_position: Position
    :param new_geometry_test: If ``True``, the full surface is rebuilt from
        scratch; if ``False``, only the symmetric part of the mesh is
        recomputed, saving computation time.
    :type new_geometry_test: bool
    :return: The same container instance with updated surface geometry.
    :rtype: OrbitalPositionContainer
    """
    if new_geometry_test:
        system.build(components_distance=orbital_position.distance)
    else:
        system.rebuild_symmetric_detached_mesh(component="all", components_distance=orbital_position.distance)
        system.build_velocities(components_distance=orbital_position.distance, component="all")
        system.build_faces_orientation(component="all", components_distance=orbital_position.distance)
        system.correct_mesh(component="all", components_distance=orbital_position.distance)
        system.build_surface_areas(component="all")

    return system


def _update_ldc_and_radiance_on_orb_pair(
    base_container: OrbitalPositionContainer,
    mirror_container: OrbitalPositionContainer | None,
    old_normal_radiance: dict[str, dict[str, NDArray[Float]]] | None,
    old_ld_cfs: dict[str, dict[str, NDArray[Float]]] | None,
    *,
    new_geometry_test: bool,
    **kwargs,
) -> tuple[
    dict[str, dict[str, NDArray[Float]]],
    dict[str, dict[str, NDArray[Float]]],
]:
    """Recalculate or propagate radiances and limb-darkening coefficients for an orbital pair.

    When ``new_geometry_test`` is ``True`` the normal radiances and limb-darkening
    coefficients are recomputed from the current surface geometry of
    ``base_container`` and then also written to ``mirror_container`` (if
    provided).  When ``False``, the previously computed values
    (``old_normal_radiance`` and ``old_ld_cfs``) are written directly to both
    containers without any recalculation, saving computation time.

    :param base_container: Orbital position container for the base position.
    :type base_container: OrbitalPositionContainer
    :param mirror_container: Orbital position container for the mirror position,
        or ``None`` if no mirror exists.
    :type mirror_container: OrbitalPositionContainer | None
    :param old_normal_radiance: Previously computed normal radiances per component
        and passband. Ignored when ``new_geometry_test`` is ``True``.
    :type old_normal_radiance: dict[str, dict[str, NDArray[Float]]] | None
    :param old_ld_cfs: Previously computed limb-darkening coefficients per
        component and passband. Ignored when ``new_geometry_test`` is ``True``.
    :type old_ld_cfs: dict[str, dict[str, NDArray[Float]]] | None
    :param new_geometry_test: If ``True``, radiances and limb-darkening
        coefficients are recomputed from the current geometry; if ``False``,
        the old values are reused.
    :type new_geometry_test: bool
    :param kwargs: Additional keyword arguments forwarded to
        :func:`elisa.binary_system.curves.utils.prep_surface_params`
        (e.g. passband, left_bandwidth, right_bandwidth).
    :return: Tuple of ``(normal_radiance, ld_cfs)`` dictionaries after update.
    :rtype: tuple[dict[str, dict[str, NDArray[Float]]], dict[str, dict[str, NDArray[Float]]]]
    """
    if new_geometry_test:
        normal_radiance, ld_cfs = crv_utils.prep_surface_params(
            base_container,
            return_values=True,
            write_to_containers=True,
            **kwargs,
        )
        if mirror_container is None:
            return normal_radiance, ld_cfs
        for component in settings.BINARY_COUNTERPARTS:
            star = getattr(mirror_container, component)
            star.normal_radiance = normal_radiance[component]
            star.ld_cfs = ld_cfs[component]
        return normal_radiance, ld_cfs

    for on_pos in [base_container, mirror_container]:
        if on_pos is None:
            continue
        for component in settings.BINARY_COUNTERPARTS:
            star = getattr(on_pos, component)
            star.normal_radiance = old_normal_radiance[component]
            star.ld_cfs = old_ld_cfs[component]
    return old_normal_radiance, old_ld_cfs


def integrate_eccentric_curve_w_orbital_symmetry(
    binary: BinarySystem,
    all_potentials: dict[str, NDArray[Float]],
    orbital_positions: NDArray[Float],
    radii: NDArray[Float],
    crv_labels: list[str],
    curve_fn: Callable[..., dict[str, NDArray[Float]]],
    kwargs: dict[str, Any],
) -> dict[str, NDArray[Float]]:
    """Generate curves exploiting apsidal symmetry for eccentric orbits without spots.

    Couples of orbital positions that are symmetrically positioned around the
    apsidal line share the same surface geometry, so only one full build per
    pair is required. This function is designed to be used as a multiprocessing
    worker and is called via
    :func:`elisa.observer.mp_manager.manage_observations`.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param all_potentials: Corrected surface potentials for all orbital
        positions, indexed by position index, keyed by component name.
    :type all_potentials: dict[str, NDArray[Float]]
    :param orbital_positions: Stacked ``(N, 2, 5)`` array of (body, mirror)
        orbital position pairs for this worker batch.
    :type orbital_positions: NDArray[Float]
    :param radii: Forward radii array used to detect geometry changes between
        consecutive positions.
    :type radii: NDArray[Float]
    :param crv_labels: Curve band / observable labels.
    :type crv_labels: list[str]
    :param curve_fn: Function that accumulates curve values at each orbital
        position.
    :type curve_fn: Callable[..., dict[str, NDArray[Float]]]
    :param kwargs: Extra keyword arguments forwarded to surface parameter
        helpers.
    :type kwargs: dict[str, Any]
    :return: Dictionary mapping each curve label to a ``(N, 2)`` array whose
        first column contains body values and second column mirror values.
    :rtype: dict[str, NDArray[Float]]
    """
    # surface potentials with constant volume of components
    potentials = {component: pot[orbital_positions[:, 0, 0].astype(int)] for component, pot in all_potentials.items()}

    base_radii = radii[:, orbital_positions[:, 0, 0].astype(int)]
    rel_d_radii = crv_utils.compute_rel_d_geometry(binary, base_radii[:, 1:], base_radii[:, :-1])
    _args = (binary.has_spots(), orbital_positions.shape[0], rel_d_radii)
    new_geometry_mask = dynamic.resolve_object_geometry_update(*_args)

    rel_irrad = crv_utils.compute_rel_d_irradiation(binary, orbital_positions[:, 0, 1])
    new_irrad_mask = dynamic.resolve_irrad_update(rel_irrad, orbital_positions.shape[0])

    new_build_mask = np.logical_or(new_geometry_mask, new_irrad_mask)

    curves_body = {key: np.zeros(orbital_positions.shape[0]) for key in crv_labels}
    curves_mirror = {key: np.zeros(orbital_positions.shape[0]) for key in crv_labels}

    # prepare initial orbital position container
    from_this = {"binary_system": binary, "position": const.Position(0, 1.0, 0.0, 0.0, 0.0)}
    initial_system = OrbitalPositionContainer.from_binary_system(**from_this)

    ld_cfs, normal_radiance = None, None
    for idx in range(orbital_positions.shape[0]):
        body, mirror = orbital_positions[idx, 0, :], orbital_positions[idx, 1, :]
        base_orb_pos, mirror_orb_pos = utils.convert_binary_orbital_motion_arr_to_positions(
            np.asarray([body, mirror]),
        )

        initial_system.set_on_position_params(base_orb_pos, potentials["primary"][idx], potentials["secondary"][idx])
        initial_system = _update_surface_in_ecc_orbits(
            initial_system,
            base_orb_pos,
            new_geometry_test=new_build_mask[idx],
        )

        on_pos_base = bsutils.move_sys_onpos(initial_system, base_orb_pos, on_copy=True)
        _kwargs = {"in_eclipse": True, "return_values": False, "write_to_containers": True}
        compute_surface_coverage(on_pos_base, binary.semi_major_axis, **_kwargs)

        if OrbitalSupplements.is_empty(mirror):
            on_pos_mirror = None
        else:
            # orbital velocities are not symmetrical along apsidal lines
            d_distance = mirror_orb_pos.distance - base_orb_pos.distance
            initial_system.secondary.points[:, 0] += d_distance
            _kwargs = {"recalculate_velocities": True, "on_copy": True}
            on_pos_mirror = bsutils.move_sys_onpos(initial_system, mirror_orb_pos, **_kwargs)
            _kwargs = {"in_eclipse": True, "return_values": False, "write_to_containers": True}
            compute_surface_coverage(on_pos_mirror, binary.semi_major_axis, **_kwargs)

        # normal radiances and ld coefficients will be used for both base and mirror orbital positions
        normal_radiance, ld_cfs = _update_ldc_and_radiance_on_orb_pair(
            on_pos_base,
            on_pos_mirror,
            normal_radiance,
            ld_cfs,
            new_geometry_test=new_build_mask[idx],
            **kwargs,
        )

        curves_body = curve_fn(curves_body, idx, crv_labels, on_pos_base)
        curves_mirror = (
            curves_mirror if on_pos_mirror is None else curve_fn(curves_mirror, idx, crv_labels, on_pos_mirror)
        )

    return {key: np.stack((curves_body[key], curves_mirror[key]), axis=1) for key in crv_labels}


def similar_neighbour_approximation_ecc_curve_integration(
    binary: BinarySystem,
    potentials: dict[str, NDArray[Float]],
    motion_batch: NDArray[Float],
    new_geometry_mask: NDArray[np.bool_],
    crv_labels: list[str],
    curve_fn: Callable[..., dict[str, NDArray[Float]]],
    kwargs: dict[str, Any],
) -> dict[str, NDArray[Float]]:
    """Generate curves for eccentric orbits using a similar-neighbour approximation.

    Orbital positions that are sufficiently similar to a preceding position
    reuse the previously computed surface geometry instead of rebuilding it
    from scratch, reducing the overall computation time significantly.
    This function is designed to be used as a multiprocessing worker and is
    called via :func:`elisa.observer.mp_manager.manage_observations`.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param potentials: Corrected surface potentials keyed by component name.
    :type potentials: dict[str, NDArray[Float]]
    :param motion_batch: Orbital positions for this worker batch, sorted by
        component distance.
    :type motion_batch: NDArray[Float]
    :param new_geometry_mask: Boolean mask over the full orbital position array
        indicating which positions require a full surface rebuild.
    :type new_geometry_mask: NDArray[numpy.bool_]
    :param crv_labels: Curve band / observable labels.
    :type crv_labels: list[str]
    :param curve_fn: Function that accumulates curve values at each orbital
        position.
    :type curve_fn: Callable[..., dict[str, NDArray[Float]]]
    :param kwargs: Extra keyword arguments forwarded to surface parameter
        helpers.
    :type kwargs: dict[str, Any]
    :return: Dictionary mapping each curve label to its computed values.
    :rtype: dict[str, NDArray[Float]]
    """
    curves = {key: np.empty(len(motion_batch)) for key in crv_labels}
    positions = utils.convert_binary_orbital_motion_arr_to_positions(motion_batch)

    # prepare initial orbital position container
    from_this = {"binary_system": binary, "position": const.Position(0, 1.0, 0.0, 0.0, 0.0)}
    initial_system = OrbitalPositionContainer.from_binary_system(**from_this)

    normal_radiance, ld_cfs = None, None
    for run_idx, position in enumerate(positions):
        pos_idx = int(position.idx)
        require_rebuild = new_geometry_mask[pos_idx] or run_idx == 0

        initial_system.set_on_position_params(
            position,
            potentials["primary"][pos_idx],
            potentials["secondary"][pos_idx],
        )

        _update_surface_in_ecc_orbits(initial_system, position, new_geometry_test=require_rebuild)
        on_pos = bsutils.move_sys_onpos(initial_system, position, on_copy=True, recalculate_velocities=True)
        on_pos, normal_radiance, ld_cfs = crv_utils.update_surface_params(
            on_pos, normal_radiance, ld_cfs, require_rebuild=require_rebuild, **kwargs,
        )
        # TODO: properly calculate in_eclipse parameter  # noqa: FIX002, TD002, TD003
        compute_surface_coverage(on_pos, binary.semi_major_axis, in_eclipse=True, return_values=False,
                                 write_to_containers=True)
        curves = curve_fn(curves, run_idx, crv_labels, on_pos)
    return curves
