import numpy as np

from elisa import ld, const
from elisa.conf import config
from elisa.binary_system import (
    dynamic,
    utils as bsutils,
    surface
)


def compute_circular_synchronous_rv(*args):
    binary, initial_system, phase_batch, normal_radiance, ld_cfs, kwargs = args

    position_method = kwargs.pop("position_method")
    orbital_motion = position_method(input_argument=phase_batch, return_nparray=False, calculate_from='phase')
    # is in eclipse test eval
    ecl_boundaries = dynamic.get_eclipse_boundaries(binary, 1.0)
    azimuths = [position.azimuth for position in orbital_motion]
    in_eclipse = dynamic.in_eclipse_test(azimuths, ecl_boundaries)

    velocities = {component: np.zeros(phase_batch.shape) for component in config.BINARY_COUNTERPARTS.keys()}

    # integrating resulting flux
    ld_law_cfs_column = config.LD_LAW_CFS_COLUMNS[config.LIMB_DARKENING_LAW]
    for pos_idx, position in enumerate(orbital_motion):
        on_pos = bsutils.move_sys_onpos(initial_system, position)
        # dict of components
        stars = {component: getattr(on_pos, component) for component in config.BINARY_COUNTERPARTS}

        coverage = surface.coverage.compute_surface_coverage(on_pos, binary.semi_major_axis,
                                                             in_eclipse=in_eclipse[pos_idx])

        # calculating cosines between face normals and line of sight
        for component, star in stars.items():
            cosines = star.los_cosines
            visibility_indices = star.indices
            cosines = cosines[visibility_indices]

            ld_cors = \
                ld.limb_darkening_factor(
                    coefficients=ld_cfs[component]['rv_band'][ld_law_cfs_column].values[visibility_indices],
                    limb_darkening_law=config.LIMB_DARKENING_LAW,
                    cos_theta=cosines)

            flux = normal_radiance[component]['rv_band'][visibility_indices] * \
                   cosines * coverage[component][visibility_indices] * \
                   ld_cors

            velocities[component][pos_idx] = np.sum(star.velocities[visibility_indices][:, 0] * flux) / \
                                             np.sum(flux) if np.sum(flux) != 0 else np.NaN

    return velocities
