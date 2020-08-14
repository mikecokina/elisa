import numpy as np
from copy import copy

from elisa import ld, const
from elisa.conf import config
from elisa.binary_system import (
    dynamic,
    utils as bsutils,
    surface
)


def compute_circ_sync_rv_at_pos(velocities, pos_idx, crv_labels, stars, ld_cfs, ld_law_cfs_column, normal_radiance,
                                coverage):
    """
    Calculates rv points for given orbital position in case of circular orbit and synchronous rotation.

    :param velocities: Dict; {str; component : numpy.array; rvs, ...}
    :param pos_idx: int; position in `band_curves` to which calculated lc points will be assigned
    :param crv_labels: list; list of components for which to calculate rvs
    :param stars: Dict; {str; component: base.container.StarContainer, ...}
    :param ld_cfs: Dict; {str; component: {passband: np.array; ld_coefficients}}
    :param ld_law_cfs_column:
    :param normal_radiance: Dict; {str; component: numpy.array; normal radiances for each surface element}
    :param coverage: Dict; {str; component: numpy.array; visible areas for each surface element}
    :return: Dict; updated {str; passband : numpy.array; rvs, ...}
    """
    # calculating cosines between face normals and line of sight
    for component in crv_labels:
        cosines = stars[component].los_cosines
        visibility_indices = stars[component].indices
        cosines = cosines[visibility_indices]

        ld_cors = \
            ld.limb_darkening_factor(
                coefficients=ld_cfs[component]['rv_band'][ld_law_cfs_column].values[visibility_indices],
                limb_darkening_law=config.LIMB_DARKENING_LAW,
                cos_theta=cosines)

        flux = normal_radiance[component]['rv_band'][visibility_indices] * cosines * \
               coverage[component][visibility_indices] * ld_cors

        velocities[component][pos_idx] = np.sum(stars[component].velocities[visibility_indices][:, 0] * flux) / \
                                         np.sum(flux) if np.sum(flux) != 0 else np.NaN

    return velocities


def compute_eccentric_spotty_asynchronous_rv(*args):
    binary, initial_system, motion_batch, base_points, ecl_boundaries, kwargs = args

    # pre-calculate the longitudes of each spot for each phase
    phases = np.array([val.phase for val in motion_batch])
    in_eclipse = dynamic.in_eclipse_test([position.azimuth for position in motion_batch], ecl_boundaries)
    spots_longitudes = dynamic.calculate_spot_longitudes(binary, phases, component="all", correct_libration=False)
    pulsation_tests = {'primary': binary.primary.has_pulsations(),
                       'secondary': binary.secondary.has_pulsations()}
    primary_reducer, secondary_reducer = \
        dynamic.resolve_spots_geometry_update(spots_longitudes, len(phases), pulsation_tests)
    combined_reducer = primary_reducer & secondary_reducer

    normal_radiance, ld_cfs = dict(), dict()
    # calculating rv with spots gradually shifting their positions in each phase
    velocities = {component: np.zeros(phases.shape) for component in config.BINARY_COUNTERPARTS.keys()}

    for pos_idx, orbital_position in enumerate(motion_batch):
        initial_system.set_on_position_params(position=orbital_position)
        initial_system.time = initial_system.set_time()
        # setup component necessary to build/rebuild

        require_build = "all" if combined_reducer[pos_idx] \
            else "primary" if primary_reducer[pos_idx] \
            else "secondary" if secondary_reducer[pos_idx] \
            else None

        # use clear system surface points as a starting place to save a time
        # if reducers for related component is set to False, previous build will be used

        # todo/fixme: we can remove `reset_spots_properties` when methods will work as expected
        if primary_reducer[pos_idx]:
            initial_system.primary.points = copy(base_points['primary'])
            initial_system.primary.reset_spots_properties()
        if secondary_reducer[pos_idx]:
            initial_system.secondary.points = copy(base_points['secondary'])
            initial_system.secondary.reset_spots_properties()

        # assigning new longitudes for each spot
        dynamic.assign_spot_longitudes(initial_system, spots_longitudes, index=pos_idx, component="all")

        # build the spots points
        surface.mesh.add_spots_to_mesh(initial_system, orbital_position.distance, component=require_build)
        # build the rest of the surface based on preset surface points
        initial_system.build_from_points(components_distance=orbital_position.distance, component=require_build)

        on_pos = bsutils.move_sys_onpos(initial_system, orbital_position, on_copy=True)
