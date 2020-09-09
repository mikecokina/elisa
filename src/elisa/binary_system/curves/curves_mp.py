import numpy as np
from copy import copy

from elisa.conf import config
from elisa.binary_system import (
    utils as butils,
    dynamic,
)
from elisa.binary_system.surface.mesh import add_spots_to_mesh
from elisa.binary_system.surface.coverage import compute_surface_coverage
from elisa.binary_system.curves import utils as crv_utils
from elisa.binary_system.container import OrbitalPositionContainer


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
    :return:
    """
    binary, initial_system, phase_batch, crv_labels, curves_fn, kwargs = args

    position_method = kwargs.pop("position_method")
    orbital_motion = position_method(input_argument=phase_batch, return_nparray=False, calculate_from='phase')
    # is in eclipse test eval
    ecl_boundaries = dynamic.get_eclipse_boundaries(binary, 1.0)
    azimuths = [position.azimuth for position in orbital_motion]
    in_eclipse = dynamic.in_eclipse_test(azimuths, ecl_boundaries)

    curves = {key: np.zeros(phase_batch.shape) for key in crv_labels}

    for pos_idx, position in enumerate(orbital_motion):
        on_pos = butils.move_sys_onpos(initial_system, position)

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

    :return:
    """
    binary, initial_system, motion_batch, base_points, ecl_boundaries, crv_labels, curve_fn, kwargs = args

    # pre-calculate the longitudes of each spot for each phase
    phases = np.array([val.phase for val in motion_batch])
    in_eclipse = dynamic.in_eclipse_test([position.azimuth for position in motion_batch], ecl_boundaries)
    spots_longitudes = dynamic.calculate_spot_longitudes(binary, phases, component="all", correct_libration=False)
    pulsation_tests = {'primary': binary.primary.has_pulsations(),
                       'secondary': binary.secondary.has_pulsations()}
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
        initial_system.build_from_points_to_temperatures(components_distance=orbital_position.distance,
                                                         component=require_build)
        initial_system.build_full_temperature_distribution(components_distance=orbital_position.distance,
                                                           component='all')

        on_pos = butils.move_sys_onpos(initial_system, orbital_position, on_copy=True)

        # if None of components has to be rebuilt, use previously computed radiances and limbdarkening when available
        require_build_test = require_build is not None
        on_pos, normal_radiance, ld_cfs = \
            crv_utils.update_surface_params(require_build_test, on_pos, normal_radiance, ld_cfs, **kwargs)

        compute_surface_coverage(on_pos, binary.semi_major_axis, in_eclipse=in_eclipse[pos_idx],
                                 return_values=False, write_to_containers=True)

        curves = curve_fn(curves, pos_idx, crv_labels, on_pos)

    return curves


def integrate_eccentric_curve_exactly(*args):
    """
    Curve generator function for exact integration of eccentric orbits.

    :param args: Tuple;

    ::

        Tuple[
                binary: elisa.binary_system.BinarySystem,
                potentials: dict; corrected surface potentials
                phase_batch: numpy.array; phases at which to calculate curves,
                spots_longitudes: longitudes of each spots for each orbital position
                crv_labels: List;
                curves_fn: function to calculate curve points at given orbital positions,
                kwargs: Dict,
            ]

    :return:
    """
    binary, potentials, motion_batch, spots_longitudes, crv_labels, curve_fn, kwargs = args
    curves = {key: np.empty(len(motion_batch)) for key in crv_labels}
    for run_idx, position in enumerate(motion_batch):
        pos_idx = int(position.idx)
        from_this = dict(binary_system=binary, position=position)
        on_pos = OrbitalPositionContainer.from_binary_system(**from_this)
        dynamic.assign_spot_longitudes(on_pos, spots_longitudes, index=pos_idx, component="all")
        on_pos.set_on_position_params(position, potentials["primary"][pos_idx],
                                      potentials["secondary"][pos_idx])
        on_pos.build(components_distance=position.distance)
        on_pos = butils.move_sys_onpos(on_pos, position, on_copy=False)

        crv_utils.prep_surface_params(on_pos, return_values=False, write_to_containers=True, **kwargs)
        # TODO: properly calculate in_eclipse parameter
        compute_surface_coverage(on_pos, binary.semi_major_axis, in_eclipse=True, return_values=False,
                                 write_to_containers=True)

        curves = curve_fn(curves, run_idx, crv_labels, on_pos)
    return curves
