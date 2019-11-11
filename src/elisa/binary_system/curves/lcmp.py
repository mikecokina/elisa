import numpy as np

from elisa import (
    umpy as up,
    utils,
    ld
)
from elisa.binary_system import (
    utils as bsutils,
    dynamic,
    surface
)
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.binary_system.curves import shared
from elisa.conf import config

LD_LAW_CFS_COLUMNS = config.LD_LAW_CFS_COLUMNS[config.LIMB_DARKENING_LAW]


def compute_circular_synchronous_lightcurve(*args):
    binary, initial_system, phase_batch, normal_radiance, ld_cfs, kwargs = args

    position_method = kwargs.pop("position_method")
    orbital_motion = position_method(input_argument=phase_batch, return_nparray=False, calculate_from='phase')
    # is in eclipse test eval
    ecl_boundaries = dynamic.get_eclipse_boundaries(binary, 1.0)
    azimuths = [position.azimuth for position in orbital_motion]
    in_eclipse = dynamic.in_eclipse_test(azimuths, ecl_boundaries)

    band_curves = {key: up.zeros(phase_batch.shape) for key in kwargs["passband"].keys()}

    for pos_idx, position in enumerate(orbital_motion):
        on_pos = bsutils.move_sys_onpos(initial_system, position)
        # dict of components
        stars = {component: getattr(on_pos, component) for component in config.BINARY_COUNTERPARTS}

        coverage = surface.coverage.compute_surface_coverage(on_pos, binary.semi_major_axis,
                                                             in_eclipse=in_eclipse[pos_idx])

        # calculating cosines between face normals and line of sight
        cosines, visibility_test = dict(), dict()
        for component, star in stars.items():
            cosines[component] = utils.calculate_cos_theta_los_x(star.normals)
            visibility_test[component] = cosines[component] > 0
            cosines[component] = cosines[component][visibility_test[component]]

            # todo: pulsations adjustment should come here

        # integrating resulting flux
        for band in kwargs["passband"].keys():
            flux, ld_cors = np.empty(2), dict()

            for component_idx, component in enumerate(config.BINARY_COUNTERPARTS.keys()):
                ld_cors[component] = \
                    ld.limb_darkening_factor(
                        coefficients=ld_cfs[component][band][LD_LAW_CFS_COLUMNS].values[visibility_test[component]],
                        limb_darkening_law=config.LIMB_DARKENING_LAW,
                        cos_theta=cosines[component])

                flux[component_idx] = np.sum(normal_radiance[component][band][visibility_test[component]] *
                                             cosines[component] *
                                             coverage[component][visibility_test[component]] *
                                             ld_cors[component])
            band_curves[band][pos_idx] = np.sum(flux)

    return band_curves


def integrate_eccentric_lc_exactly(*args):
    binary, motion_batch, potentials, kwargs = args
    band_curves = {key: np.empty(len(motion_batch)) for key in kwargs["passband"]}

    for run_idx, position in enumerate(motion_batch):
        pos_idx = int(position.idx)
        from_this = dict(binary_system=binary, position=position)
        on_pos = OrbitalPositionContainer.from_binary_system(**from_this)
        on_pos.primary.surface_potential = potentials['primary'][pos_idx]
        on_pos.secondary.surface_potential = potentials['secondary'][pos_idx]
        on_pos.build(components_distance=position.distance)

        # todo: pulsations adjustment should come here
        normal_radiance, ld_cfs = shared.prep_surface_params(on_pos, **kwargs)
        on_pos = bsutils.move_sys_onpos(on_pos, position, on_copy=False)
        coverage, cosines = surface.coverage.calculate_coverage_with_cosines(on_pos, binary.semi_major_axis,
                                                                             in_eclipse=True)

        for band in kwargs["passband"]:
            band_curves[band][run_idx] = shared.calculate_lc_point(band, ld_cfs, normal_radiance, coverage, cosines)
    return band_curves
