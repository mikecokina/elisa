import numpy as np

from copy import copy
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.binary_system.curves import shared
from elisa.conf import config

from elisa import (
    umpy as up,
    utils,
    ld,
    const as c,
)
from elisa.binary_system import (
    utils as bsutils,
    dynamic,
    surface
)


def compute_circular_synchronous_lightcurve(*args):
    # fixme: kwargs has to be passed as kwargs not in args, it makes no sense
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

        # obtaining cosines between face normals and line of sight
        cosines, visibility_indices = dict(), dict()
        for component, star in stars.items():
            cosines[component] = star.los_cosines
            visibility_indices[component] = star.indices
            cosines[component] = cosines[component][visibility_indices[component]]

        # integrating resulting flux
        for band in kwargs["passband"].keys():
            ld_law_cfs_column = config.LD_LAW_CFS_COLUMNS[config.LIMB_DARKENING_LAW]
            flux, ld_cors = np.empty(2), dict()

            for component_idx, component in enumerate(config.BINARY_COUNTERPARTS.keys()):
                ld_cors[component] = \
                    ld.limb_darkening_factor(
                        coefficients=ld_cfs[component][band][ld_law_cfs_column].values[visibility_indices[component]],
                        limb_darkening_law=config.LIMB_DARKENING_LAW,
                        cos_theta=cosines[component])

                flux[component_idx] = np.sum(normal_radiance[component][band][visibility_indices[component]] *
                                             cosines[component] *
                                             coverage[component][visibility_indices[component]] *
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
        on_pos.set_on_position_params(position, potentials["primary"][pos_idx],
                                      potentials["secondary"][pos_idx])
        on_pos.build(components_distance=position.distance)

        normal_radiance, ld_cfs = shared.prep_surface_params(on_pos, **kwargs)
        on_pos = bsutils.move_sys_onpos(on_pos, position, on_copy=False)
        coverage, cosines = surface.coverage.calculate_coverage_with_cosines(on_pos, binary.semi_major_axis,
                                                                             in_eclipse=True)

        for band in kwargs["passband"]:
            band_curves[band][run_idx] = shared.calculate_lc_point(band, ld_cfs, normal_radiance, coverage, cosines)
    return band_curves


def compute_circular_spotty_asynchronous_lightcurve(*args):
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
    # calculating lc with spots gradually shifting their positions in each phase
    band_curves = {key: np.empty(len(motion_batch)) for key in kwargs["passband"]}
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

        # if None of components has to be rebuilded, use previously computed radiances and limbdarkening when available
        if utils.is_empty(normal_radiance) or not utils.is_empty(require_build):
            normal_radiance, ld_cfs = shared.prep_surface_params(on_pos, **kwargs)

        coverage, cosines = surface.coverage.calculate_coverage_with_cosines(
            on_pos, on_pos.semi_major_axis, in_eclipse=in_eclipse[pos_idx])

        for band in kwargs["passband"]:
            band_curves[band][pos_idx] = shared.calculate_lc_point(band, ld_cfs, normal_radiance, coverage, cosines)

    return band_curves
