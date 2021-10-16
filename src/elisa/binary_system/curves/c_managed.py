import numpy as np
from copy import copy

from . import utils as crv_utils
from .. surface.mesh import add_spots_to_mesh
from .. import (
    utils as bsutils,
    dynamic
)
from .. container import OrbitalPositionContainer
from .. surface.coverage import compute_surface_coverage
from .. orbit.container import OrbitalSupplements
from ... import utils, const
from ... import settings


def produce_circ_sync_curves_mp(*args):
    """
    Curve generator function for circular synchronous systems.

    :param args: Tuple;

    ::

        Tuple[
                binary: elisa.binary_system.BinarySystem,
                initial_system: elisa.binary_system.container.OrbitalPositionContainer, system container with built
                geometry
                phase_batch: numpy.array; phases at which to calculate curves,
                normal_radiance: Dict; {component: numpy.array; normal radiances for each surface element},
                ld_cfs: Dict;
                crv_labels: List;
                curves_fn: function to calculate curve points at given orbital positions,
                kwargs: Dict,
            ]

    :return: Dict;
    """
    # todo: too much replicated code from produce_circ_pulsating_curves_mp
    binary, initial_system, phase_batch, crv_labels, curves_fn, kwargs = args

    position_method = kwargs.pop("position_method")
    orbital_motion = position_method(input_argument=phase_batch, return_nparray=False, calculate_from='phase')
    # is in eclipse test eval
    ecl_boundaries = dynamic.get_eclipse_boundaries(binary, 1.0)
    azimuths = [position.azimuth for position in orbital_motion]
    in_eclipse = dynamic.in_eclipse_test(azimuths, ecl_boundaries)

    curves = {key: np.zeros(phase_batch.shape) for key in crv_labels}

    for pos_idx, position in enumerate(orbital_motion):
        on_pos = bsutils.move_sys_onpos(initial_system, position, on_copy=True)

        compute_surface_coverage(on_pos, binary.semi_major_axis, in_eclipse=in_eclipse[pos_idx],
                                 return_values=False, write_to_containers=True)

        curves = curves_fn(curves, pos_idx, crv_labels, on_pos)

    return curves


def produce_circ_spotty_async_curves_mp(*args):
    """
    Curve generator function for circular asynchronous spotty systems.

    :param args: Tuple;

    ::

        Tuple[
                binary: elisa.binary_system.BinarySystem,
                initial_system: elisa.binary_system.container.OrbitalPositionContainer, system container with built
                geometry: surface points of a clean system
                phase_batch: numpy.array; phases at which to calculate curves,
                ecl_boundaries: boundaries for both eclipses
                crv_labels: List;
                curves_fn: function to calculate curve points at given orbital positions,
                kwargs: Dict,
            ]

    :return: Dict;
    """
    binary, initial_system, motion_batch, base_points, ecl_boundaries, crv_labels, curve_fn, kwargs = args

    # pre-calculate the longitudes of each spot for each phase
    phases = np.array([val.phase for val in motion_batch])
    in_eclipse = dynamic.in_eclipse_test([position.azimuth for position in motion_batch], ecl_boundaries)
    spots_longitudes = dynamic.calculate_spot_longitudes(binary, phases, component="all", correct_libration=False)
    pulsation_tests = {'primary': binary.primary.has_pulsations(), 'secondary': binary.secondary.has_pulsations()}
    primary_reducer, secondary_reducer = \
        dynamic.resolve_spots_geometry_update(spots_longitudes, len(phases), pulsation_tests)
    combined_reducer = primary_reducer & secondary_reducer

    # calculating lc with spots gradually shifting their positions in each phase
    curves = {key: np.empty(len(motion_batch)) for key in crv_labels}
    normal_radiance, ld_cfs = None, None
    for pos_idx, orbital_position in enumerate(motion_batch):
        initial_system.set_on_position_params(position=orbital_position)
        initial_system.time = initial_system.set_time()
        # setup component necessary to build/rebuild

        # require_build = "all"
        require_build = "all" if combined_reducer[pos_idx] \
            else "primary" if primary_reducer[pos_idx] \
            else "secondary" if secondary_reducer[pos_idx] \
            else None

        # use clear system surface points as a starting place to save a time
        # if reducers for related component is set to False, previous build will be used

        if primary_reducer[pos_idx]:
            initial_system.primary.points = copy(base_points['primary'])
        if secondary_reducer[pos_idx]:
            initial_system.secondary.points = copy(base_points['secondary'])

        # assigning new longitudes for each spot
        dynamic.assign_spot_longitudes(initial_system, spots_longitudes, index=pos_idx, component=require_build)

        # build the spots points
        add_spots_to_mesh(initial_system, orbital_position.distance, component=require_build)
        # build the rest of the surface based on preset surface points
        _build_args = dict(components_distance=orbital_position.distance, component=require_build)
        initial_system.build_faces_and_kinematic_quantities(**_build_args)
        initial_system.build_temperature_distribution(components_distance=orbital_position.distance, component='all')

        if initial_system.has_pulsations():
            on_pos = initial_system.copy()
            on_pos.flat_it()
            on_pos.build_pulsations(components_distance=orbital_position.distance, component='all')
            on_copy, sys_to_rotate = False, on_pos
        else:
            on_copy, sys_to_rotate = True, initial_system

        on_pos = bsutils.move_sys_onpos(sys_to_rotate, orbital_position, on_copy=on_copy)

        # if None of components has to be rebuilt, use previously computed radiances and limbdarkening when available
        require_build_test = require_build is not None
        on_pos, normal_radiance, ld_cfs = \
            crv_utils.update_surface_params(require_build_test, on_pos, normal_radiance, ld_cfs, **kwargs)

        _kwargs = dict(in_eclipse=in_eclipse[pos_idx], return_values=False, write_to_containers=True)
        compute_surface_coverage(on_pos, binary.semi_major_axis, **_kwargs)

        curves = curve_fn(curves, pos_idx, crv_labels, on_pos)

    return curves


def produce_circ_pulsating_curves_mp(*args):
    """
    Curve generator function for circular pulsating systems.

    :param args: Tuple;

    ::

        Tuple[
                binary: elisa.binary_system.BinarySystem,
                initial_system: elisa.binary_system.container.OrbitalPositionContainer, system container with built
                phase_batch: numpy.array; phases at which to calculate curves,
                crv_labels: List;
                curves_fn: function to calculate curve points at given orbital positions,
                kwargs: Dict,
            ]

    :return:
    """
    # todo: too much replicated code from produce_circ_sync_curves_mp
    binary, initial_system, phase_batch, crv_labels, curves_fn, kwargs = args

    position_method = kwargs.pop("position_method")
    orbital_motion = position_method(input_argument=phase_batch, return_nparray=False, calculate_from='phase')
    # is in eclipse test eval
    ecl_boundaries = dynamic.get_eclipse_boundaries(binary, 1.0)
    azimuths = [position.azimuth for position in orbital_motion]
    in_eclipse = dynamic.in_eclipse_test(azimuths, ecl_boundaries)

    curves = {key: np.zeros(phase_batch.shape) for key in crv_labels}

    for pos_idx, position in enumerate(orbital_motion):
        on_pos = initial_system.copy()
        on_pos.set_on_position_params(position)
        on_pos.set_time()

        on_pos.build_pulsations(components_distance=position.distance)
        crv_utils.prep_surface_params(on_pos, return_values=False, write_to_containers=True, **kwargs)
        on_pos = bsutils.move_sys_onpos(on_pos, position, on_copy=False, recalculate_velocities=False)

        compute_surface_coverage(on_pos, binary.semi_major_axis, in_eclipse=in_eclipse[pos_idx],
                                 return_values=False, write_to_containers=True)

        curves = curves_fn(curves, pos_idx, crv_labels, on_pos)

    return curves


def integrate_eccentric_curve_exactly(*args):
    """
    Curve generator function for exact integration of eccentric orbits.

    :param args: Tuple;

    ::

        Tuple[
                binary: elisa.binary_system.BinarySystem,
                potentials: Dict; corrected surface potentials
                phase_batch: numpy.array; phases at which to calculate curves,
                spots_longitudes: longitudes of each spots for each orbital position
                curves_fn: function to calculate curve points at given orbital positions,
                kwargs: Dict,
            ]

    :return: Dict;
    """
    binary, potentials, motion_batch, spots_longitudes, crv_labels, curve_fn, kwargs = args
    curves = {key: np.empty(len(motion_batch)) for key in crv_labels}
    for run_idx, position in enumerate(motion_batch):
        pos_idx = int(position.idx)
        from_this = dict(binary_system=binary, position=position)
        on_pos = OrbitalPositionContainer.from_binary_system(**from_this)
        dynamic.assign_spot_longitudes(on_pos, spots_longitudes, index=pos_idx, component="all")
        on_pos.set_on_position_params(position, potentials["primary"][pos_idx], potentials["secondary"][pos_idx])
        on_pos.build(components_distance=position.distance)
        on_pos = bsutils.move_sys_onpos(on_pos, position, on_copy=False)

        crv_utils.prep_surface_params(on_pos, return_values=False, write_to_containers=True, **kwargs)
        # TODO: properly calculate in_eclipse parameter
        _kwargs = dict(in_eclipse=True, return_values=False, write_to_containers=True)
        compute_surface_coverage(on_pos, binary.semi_major_axis, **_kwargs)

        curves = curve_fn(curves, run_idx, crv_labels, on_pos)
    return curves


# managing approximations in eccentric orbits ##########################################################################

def _update_surface_in_ecc_orbits(system, orbital_position, new_geometry_test):
    """
    Function decides how to update surface properties with respect to the degree of change
    in surface geometry given by new_geometry test.
    If true, only points and normals are recalculated, otherwise surface is calculated from scratch.

    :param system: elisa.binary_system.container.OrbitalPositionContainer
    :param orbital_position:  OrbitalPosition list
    :param new_geometry_test: bool; test that will decide, how the following phase will be calculated
    :return: elisa.binary_system.system.BinarySystem; instance with updated geometry
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
        new_geometry_test,
        base_container,
        mirror_container,
        old_normal_radiance,
        old_ld_cfs,
        **kwargs
):
    """
    Function recalculates or assigns old values to normal radiances or limb darkening coefficients.

    :param new_geometry_test: bool; if True, parameters will be recalculated according
                                    to new geometry, otherwise they will be copied
    :param base_container: elisa.binary_system.container.OrbitalPositionContainer;
    :param mirror_container: elisa.binary_system.container.OrbitalPositionContainer;
    :param old_normal_radiance: Dict; normal radiances to be copied if `new_geometry_test` is False
    :param old_ld_cfs: Dict; normal radiances to be copied if `new_geometry_test` is False
    :param kwargs: kwargs;
    :return: Tuple;
    """
    if new_geometry_test:
        normal_radiance, ld_cfs = crv_utils.prep_surface_params(base_container, return_values=True,
                                                                write_to_containers=True, **kwargs)
        if mirror_container is None:
            return normal_radiance, ld_cfs
        for component in settings.BINARY_COUNTERPARTS:
            star = getattr(mirror_container, component)
            setattr(star, 'normal_radiance', normal_radiance[component])
            setattr(star, 'ld_cfs', ld_cfs[component])
        return normal_radiance, ld_cfs
    else:
        for on_pos in [base_container, mirror_container]:
            if on_pos is None:
                continue
            for component in settings.BINARY_COUNTERPARTS:
                star = getattr(on_pos, component)
                setattr(star, 'normal_radiance', old_normal_radiance[component])
                setattr(star, 'ld_cfs', old_ld_cfs[component])
        return old_normal_radiance, old_ld_cfs


def integrate_eccentric_curve_w_orbital_symmetry(*args):
    """
    Curve generator for eccentric curves without spots for couples of orbital positions that
    are symmetrically positioned around apsidal line and thus share the same surface geometry.

    :param args: Tuple;

    ::

        Tuple[
                binary: elisa.binary_system.BinarySystem,
                initial_system: elisa.
                orbital_positions: numpy.array; (N*2*5) stacked couples of orbital positions
                radii: numpy.array; forward radii
                crv_labels: List;
                curves_fn: function to calculate curve points at given orbital positions,
                kwargs: Dict,
            ]

    :return: Dict; curves
    """
    binary, all_potentials, orbital_positions, radii, crv_labels, curve_fn, kwargs = args

    # surface potentials with constant volume of components
    potentials = {
        component: pot[orbital_positions[:, 0, 0].astype(np.int)]
        for component, pot in all_potentials.items()
    }

    base_radii = radii[:, orbital_positions[:, 0, 0].astype(np.int)]
    rel_d_radii = crv_utils.compute_rel_d_geometry(binary, base_radii[:, 1:], base_radii[:, :-1])
    args = (binary.has_spots(), orbital_positions.shape[0], rel_d_radii)
    new_geometry_mask = dynamic.resolve_object_geometry_update(*args)

    rel_irrad = crv_utils.compute_rel_d_irradiation(binary, orbital_positions[:, 0, 1])
    new_irrad_mask = dynamic.resolve_irrad_update(rel_irrad, orbital_positions.shape[0])

    new_build_mask = np.logical_or(new_geometry_mask, new_irrad_mask)

    curves_body = {key: np.zeros(orbital_positions.shape[0]) for key in crv_labels}
    curves_mirror = {key: np.zeros(orbital_positions.shape[0]) for key in crv_labels}

    # prepare initial orbital position container
    from_this = dict(binary_system=binary, position=const.Position(0, 1.0, 0.0, 0.0, 0.0))
    initial_system = OrbitalPositionContainer.from_binary_system(**from_this)

    ld_cfs, normal_radiance = None, None
    for idx in range(orbital_positions.shape[0]):
        body, mirror = orbital_positions[idx, 0, :], orbital_positions[idx, 1, :]
        base_orb_pos, mirror_orb_pos = utils.convert_binary_orbital_motion_arr_to_positions([body, mirror])

        initial_system.set_on_position_params(base_orb_pos, potentials['primary'][idx], potentials['secondary'][idx])
        initial_system = _update_surface_in_ecc_orbits(initial_system, base_orb_pos, new_build_mask[idx])

        on_pos_base = bsutils.move_sys_onpos(initial_system, base_orb_pos, on_copy=True)
        _kwargs = dict(in_eclipse=True, return_values=False, write_to_containers=True)
        compute_surface_coverage(on_pos_base, binary.semi_major_axis, **_kwargs)

        if OrbitalSupplements.is_empty(mirror):
            on_pos_mirror = None
        else:
            # orbital velocities are not symmetrical along apsidal lines
            d_distance = mirror_orb_pos.distance - base_orb_pos.distance
            initial_system.secondary.points[:, 0] += d_distance
            _kwargs = dict(recalculate_velocities=True, on_copy=True)
            on_pos_mirror = bsutils.move_sys_onpos(initial_system, mirror_orb_pos, **_kwargs)
            _kwargs = dict(in_eclipse=True, return_values=False, write_to_containers=True)
            compute_surface_coverage(on_pos_mirror, binary.semi_major_axis, **_kwargs)

        # normal radiances and ld coefficients will be used for both base and mirror orbital positions
        args = (new_build_mask[idx], on_pos_base, on_pos_mirror, normal_radiance, ld_cfs)
        normal_radiance, ld_cfs = _update_ldc_and_radiance_on_orb_pair(*args, **kwargs)

        curves_body = curve_fn(curves_body, idx, crv_labels, on_pos_base)
        curves_mirror = curves_mirror if on_pos_mirror is None else \
            curve_fn(curves_mirror, idx, crv_labels, on_pos_mirror)

    return {key: np.stack((curves_body[key], curves_mirror[key]), axis=1) for key in crv_labels}


def similar_neighbour_approximation_ecc_curve_integration(*args):
    """
    Curve generator for eccentric curves without spots for orbital positions that are sufficiently similar that surfaces
    does not have to be fully recalculated.

    :param args: Tuple;

    ::

        Tuple[
                binary: elisa.binary_system.BinarySystem,
                potentials: Dict; corrected surface potentials
                motion_batch: numpy.array; orbital positions sorted by components distance
                new_geometry_mask: bool; mask to `orbital_positions` which determines which surface
                                         geometry should be fully recalculated
                crv_labels: List; curve_labels
                curves_fn: function to calculate curve points at given orbital positions,
                kwargs: Dict,
            ]

    :return: Dict; curves
    """
    binary, potentials, motion_batch, new_geometry_mask, crv_labels, curve_fn, kwargs = args
    curves = {key: np.empty(len(motion_batch)) for key in crv_labels}
    positions = utils.convert_binary_orbital_motion_arr_to_positions(motion_batch)

    # prepare initial orbital position container
    from_this = dict(binary_system=binary, position=const.Position(0, 1.0, 0.0, 0.0, 0.0))
    initial_system = OrbitalPositionContainer.from_binary_system(**from_this)

    normal_radiance, ld_cfs = None, None
    for run_idx, position in enumerate(positions):
        pos_idx = int(position.idx)
        require_rebuild = new_geometry_mask[pos_idx] or run_idx == 0

        initial_system.set_on_position_params(position, potentials["primary"][pos_idx],
                                              potentials["secondary"][pos_idx])

        _update_surface_in_ecc_orbits(initial_system, orbital_position=position, new_geometry_test=require_rebuild)
        on_pos = bsutils.move_sys_onpos(initial_system, position, on_copy=True, recalculate_velocities=True)
        on_pos, normal_radiance, ld_cfs = crv_utils.update_surface_params(require_rebuild, on_pos, normal_radiance,
                                                                          ld_cfs, **kwargs)
        # TODO: properly calculate in_eclipse parameter
        compute_surface_coverage(on_pos, binary.semi_major_axis, in_eclipse=True, return_values=False,
                                 write_to_containers=True)
        curves = curve_fn(curves, run_idx, crv_labels, on_pos)
    return curves
