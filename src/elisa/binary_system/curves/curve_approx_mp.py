import numpy as np
from copy import deepcopy

from elisa import utils, const
from elisa.conf import config
from elisa.binary_system import (
    utils as bsutils,
    dynamic
)
from elisa.binary_system.curves import utils as crv_utils
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.binary_system.surface.coverage import compute_surface_coverage
from elisa.binary_system.orbit.container import OrbitalSupplements


def update_surface_in_ecc_orbits(system, orbital_position, new_geometry_test):
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
        system.build_mesh(component="all", components_distance=orbital_position.distance)
        system.build_velocities(components_distance=orbital_position.distance, component="all")
        system.build_faces_orientation(component="all", components_distance=orbital_position.distance)
        system.build_surface_areas(component="all")

    return system


def update_ldc_and_radiance_on_orb_pair(new_geometry_test, base_container, mirror_container, old_normal_radiance,
                                        old_ld_cfs, **kwargs):
    """
    Function recalculates or assigns old values tp normal radiances or limb darkening coefficients.

    :param new_geometry_test: bool; if True, parameters will be recalculated according to new geometry, otherwise they
    will be copied
    :param base_container: elisa.binary_system.container.OrbitalPositionContainer;
    :param mirror_container: elisa.binary_system.container.OrbitalPositionContainer;
    :param old_normal_radiance: dict; normal radiances to be copied if `new_geometry_test` is False
    :param old_ld_cfs: dict; normal radiances to be copied if `new_geometry_test` is False
    :param kwargs: kwargs;
    :return:
    """
    if new_geometry_test:
        normal_radiance, ld_cfs = crv_utils.prep_surface_params(base_container, return_values=True,
                                                                write_to_containers=True, **kwargs)
        if mirror_container is None:
            return normal_radiance, ld_cfs
        for component in config.BINARY_COUNTERPARTS.keys():
            star = getattr(mirror_container, component)
            setattr(star, 'normal_radiance', normal_radiance[component])
            setattr(star, 'ld_cfs', ld_cfs[component])
        return normal_radiance, ld_cfs
    else:
        for on_pos in [base_container, mirror_container]:
            if on_pos is None:
                continue
            for component in config.BINARY_COUNTERPARTS.keys():
                star = getattr(on_pos, component)
                setattr(star, 'normal_radiance', old_normal_radiance[component])
                setattr(star, 'ld_cfs', old_ld_cfs[component])
        return old_normal_radiance, old_ld_cfs


def integrate_eccentric_curve_w_orbital_symmetry(*args):
    """
    Curve generator for eccentric curves without spots for couples of orbital positions that are symmetrically
    positioned around apsidal line and thus share the same surface geometry.

    :param args: Tuple;

    ::

        Tuple[
                binary: elisa.binary_system.BinarySystem,
                crv_labels: List;
                initial_system: elisa.
                orbital_positions: numpy.array; (N*2*5) stacked couples of orbital positions
                curves_fn: function to calculate curve points at given orbital positions,
                kwargs: Dict,
            ]
    :return: Dict; curves
    """
    binary, all_potentials, orbital_positions, crv_labels, curve_fn, kwargs = args

    # # surface potentials with constant volume of components
    # potentials = binary.correct_potentials(orbital_positions[:, 0, 4], component="all", iterations=2)
    potentials = {component: pot[np.array(orbital_positions[:, 0, 0], dtype=np.int)] for component, pot in
                  all_potentials.items()}

    rel_d_radii = crv_utils.compute_rel_d_radii(binary, orbital_positions[:, 0, 1], potentials=potentials)
    new_geometry_mask = dynamic.resolve_object_geometry_update(binary.has_spots(), orbital_positions.shape[0],
                                                               rel_d_radii)

    curves_body = {key: np.empty(orbital_positions.shape[0]) for key in crv_labels}
    curves_mirror = {key: np.empty(orbital_positions.shape[0]) for key in crv_labels}

    # prepare initial orbital position container
    from_this = dict(binary_system=binary, position=const.Position(0, 1.0, 0.0, 0.0, 0.0))
    initial_system = OrbitalPositionContainer.from_binary_system(**from_this)

    ld_cfs, normal_radiance = None, None
    for idx in range(orbital_positions.shape[0]):
        body, mirror = orbital_positions[idx, 0, :], orbital_positions[idx, 1, :]
        base_orb_pos, mirror_orb_pos = utils.convert_binary_orbital_motion_arr_to_positions([body, mirror])

        initial_system.set_on_position_params(base_orb_pos, potentials['primary'][idx],
                                              potentials['secondary'][idx])
        initial_system = update_surface_in_ecc_orbits(initial_system, base_orb_pos, new_geometry_mask[idx])

        on_pos_base = bsutils.move_sys_onpos(initial_system, base_orb_pos)
        compute_surface_coverage(on_pos_base, binary.semi_major_axis, in_eclipse=True,
                                 return_values=False, write_to_containers=True)

        if not OrbitalSupplements.is_empty(mirror):
            # orbital velocities are not symmetrical along apsidal lines
            on_pos_mirror = bsutils.move_sys_onpos(initial_system, mirror_orb_pos, recalculate_velocities=True)
            compute_surface_coverage(on_pos_mirror, binary.semi_major_axis, in_eclipse=True,
                                     return_values=False, write_to_containers=True)
        else:
            on_pos_mirror = None

        # normal radiances and ld coefficients will be used for both base and mirror orbital positions
        normal_radiance, ld_cfs = update_ldc_and_radiance_on_orb_pair(new_geometry_mask[idx],
                                                                      on_pos_base, on_pos_mirror, normal_radiance,
                                                                      ld_cfs, **kwargs)

        curves_body = curve_fn(curves_body, idx, crv_labels, on_pos_base)
        curves_mirror = curves_mirror if on_pos_mirror is None else \
            curve_fn(curves_mirror, idx, crv_labels, on_pos_mirror)

    return {key: np.stack((curves_body[key], curves_mirror[key]), axis=1) for key in crv_labels}


def integrate_eccentric_curve_approx_three(*args):
    """
    Curve generator for eccentric curves without spots for orbital positions that are sufficiently similar that surfaces
    does not have to be fully recalculated.

    :param args: Tuple;

    ::

        Tuple[
                binary: elisa.binary_system.BinarySystem,
                potentials: dict; corrected surface potentials
                motion_batch: numpy.array; orbital positions sorted by components distance
                new_geometry_mask: bool; mask to `orbital_positions` which determines which surface geometry should be
                fully recalculated
                crv_labels: list; curve_labels
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

        update_surface_in_ecc_orbits(initial_system, orbital_position=position, new_geometry_test=require_rebuild)

        on_pos = bsutils.move_sys_onpos(initial_system, position, on_copy=False, recalculate_velocities=True)

        on_pos, normal_radiance, ld_cfs = crv_utils.update_surface_params(require_rebuild, on_pos, normal_radiance,
                                                                          ld_cfs, **kwargs)

        # TODO: properly calculate in_eclipse parameter
        compute_surface_coverage(on_pos, binary.semi_major_axis, in_eclipse=True, return_values=False,
                                 write_to_containers=True)

        curves = curve_fn(curves, run_idx, crv_labels, on_pos)
    return curves
