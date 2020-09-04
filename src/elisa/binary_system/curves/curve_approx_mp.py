import numpy as np

from elisa import utils, const
from elisa.conf import config
from ...binary_system import (
    utils as bsutils,
    dynamic
)
from elisa.binary_system.curves import utils as crv_utils
from ...binary_system.container import OrbitalPositionContainer
from elisa.binary_system.surface.coverage import compute_surface_coverage
from elisa.binary_system.orbit import orbit


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


def integrate_eccentric_curve_appx_one(*args):
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
    binary, crv_labels, orbital_positions, curve_fn, kwargs = args

    # surface potentials with constant volume of components
    potentials = binary.correct_potentials(orbital_positions[:, 0, 4], component="all", iterations=2)

    rel_d_radii = crv_utils.compute_rel_d_radii(binary, orbital_positions[:, 0, 1], potentials=potentials)
    new_geometry_mask = dynamic.resolve_object_geometry_update(binary.has_spots(), orbital_positions.shape[0],
                                                               rel_d_radii)

    curves_body = {key: np.empty(orbital_positions.shape[0]) for key in crv_labels}
    curves_mirror = {key: np.empty(orbital_positions.shape[0]) for key in crv_labels}

    # prepare initial orbital position container
    from_this = dict(binary_system=binary, position=const.Position(0, 1.0, 0.0, 0.0, 0.0))
    initial_system = OrbitalPositionContainer.from_binary_system(**from_this)

    # both, body and mirror should be defined in this approximation (those objects are created to be mirrored
    # one to another)
    ld_cfs, normal_radiance = None, None
    for idx in range(orbital_positions.shape[0]):
        body, mirror = orbital_positions[idx, 0, :], orbital_positions[idx, 1, :]
        body_orb_pos, mirror_orb_pos = utils.convert_binary_orbital_motion_arr_to_positions([body, mirror])

        initial_system.set_on_position_params(body_orb_pos, potentials['primary'][idx],
                                              potentials['secondary'][idx])
        initial_system = update_surface_in_ecc_orbits(initial_system, body_orb_pos, new_geometry_mask[idx])

        on_pos_base = bsutils.move_sys_onpos(initial_system, body_orb_pos)
        # orbital velocities are not symmetrical along apsidal lines
        on_pos_mirror = bsutils.move_sys_onpos(initial_system, mirror_orb_pos, recalculate_velocities=True)

        # normal radiances and ld coefficients will be used for both base and mirror orbital positions
        if new_geometry_mask[idx]:
            normal_radiance, ld_cfs = crv_utils.prep_surface_params(on_pos_base, return_values=True,
                                                                    write_to_containers=True, **kwargs)
            for component in config.BINARY_COUNTERPARTS.keys():
                star = getattr(on_pos_mirror, component)
                setattr(star, 'normal_radiance', normal_radiance[component])
                setattr(star, 'ld_cfs', ld_cfs[component])
        else:
            for on_pos in [on_pos_base, on_pos_mirror]:
                for component in config.BINARY_COUNTERPARTS.keys():
                    star = getattr(on_pos, component)
                    setattr(star, 'normal_radiance', normal_radiance[component])
                    setattr(star, 'ld_cfs', ld_cfs[component])

        compute_surface_coverage(on_pos_base, binary.semi_major_axis, in_eclipse=True,
                                 return_values=False, write_to_containers=True)
        compute_surface_coverage(on_pos_mirror, binary.semi_major_axis, in_eclipse=True,
                                 return_values=False, write_to_containers=True)

        curves_body = curve_fn(curves_body, idx, crv_labels, on_pos_base)
        curves_mirror = curve_fn(curves_mirror, idx, crv_labels, on_pos_mirror)

    return {key: np.stack((curves_body[key], curves_mirror[key]), axis=1) for key in crv_labels}