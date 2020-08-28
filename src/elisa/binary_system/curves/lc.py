from multiprocessing.pool import Pool

import numpy as np

from copy import (
    deepcopy,
    copy
)
from scipy.interpolate import Akima1DInterpolator

from ...logger import getLogger
from ...conf import config
from ...binary_system.container import OrbitalPositionContainer
from ...binary_system.orbit.container import OrbitalSupplements
from ...binary_system.surface.coverage import calculate_coverage_with_cosines
from ...binary_system.curves import (
    lcmp,
    curves,
    utils as crv_utils,
    curve_approx
)
from elisa.observer.mp import manage_observations

from ... import (
    umpy as up,
    const,
    utils
)
from ...binary_system import (
    utils as bsutils,
    dynamic,
    surface
)


logger = getLogger('binary_system.curves.lc')


def _onpos_params(on_pos, **kwargs):
    """
    Helper function.

    :param on_pos: elisa.binary_system.container.OrbitalPositionContainer;
    :return: Tuple;
    """
    _normal_radiance, _ld_cfs = crv_utils.prep_surface_params(on_pos, **kwargs)

    _coverage, _cosines = calculate_coverage_with_cosines(on_pos, on_pos.semi_major_axis, in_eclipse=True)
    return _normal_radiance, _ld_cfs, _coverage, _cosines


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
        system.build_mesh(component="all", components_distance=orbital_position.distance)
        system.build_surface_areas(component="all")
        system.build_faces_orientation(component="all", components_distance=orbital_position.distance)

    return system


def compute_circular_synchronous_lightcurve(binary, **kwargs):
    """
    Compute light curve for synchronous circular binary system.


    :param binary: elisa.binary_system.system.BinarySystem;
    :param kwargs: Dict;
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** atlas ** * - str
            * ** position_method** * - function definition; to evaluate orbital positions
            * ** phases ** * - numpy.array
    :return: Dict[str, numpy.array];
    """

    initial_system = curves.prep_initial_system(binary)

    phases = kwargs.pop("phases")
    unique_phase_interval, reverse_phase_map = dynamic.phase_crv_symmetry(initial_system, phases)

    lc_labels = list(kwargs["passband"].keys())

    band_curves = curves.produce_circ_sync_curves(binary, initial_system, unique_phase_interval,
                                                  lcmp.compute_lc_on_pos, lc_labels, **kwargs)

    band_curves = {band: band_curves[band][reverse_phase_map] for band in band_curves}
    return band_curves


def compute_circular_spotty_asynchronous_lightcurve(binary, **kwargs):
    """
    Function returns light curve of asynchronous systems with circular orbits and spots.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param kwargs: Dict;
    :**kwargs options**:
        * ** passband ** - Dict[str, elisa.observer.PassbandContainer]
        * ** left_bandwidth ** - float
        * ** right_bandwidth ** - float
        * ** atlas ** - str
    :return: Dict; fluxes for each filter
    """
    lc_labels = list(kwargs["passband"].keys())

    return curves.produce_circ_spotty_async_curves(binary, lcmp.compute_lc_on_pos,
                                                   lc_labels, **kwargs)


def compute_eccentric_lightcurve_no_spots(binary, **kwargs):
    """
    General function for generating light curves of binaries with eccentric orbit and no spots.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param kwargs:  Dict;
    :**kwargs options**:
        * ** passband ** - Dict[str, elisa.observer.PassbandContainer]
        * ** left_bandwidth ** - float
        * ** right_bandwidth ** - float
        * ** atlas ** - str
    :return: Dict; fluxes for each filter
    """
    lc_labels = list(kwargs["passband"].keys())

    return curves.produce_ecc_curves_no_spots(binary, lcmp.compute_lc_on_pos, lc_labels, **kwargs)


def _integrate_eccentric_lc_appx_one(binary, phases, orbital_supplements, new_geometry_mask, **kwargs):
    """
    Function calculates light curves for eccentric orbits for selected filters using approximation
    where light curve points on the one side of the apsidal line are calculated exactly and the second
    half of the light curve points are calculated by mirroring the surface geometries of the first
    half of the points to the other side of the apsidal line. Since those mirrored
    points are not alligned with desired phases, the fluxes for each phase is interpolated if missing.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param phases: numpy.array;
    :param orbital_supplements: elisa.binary_system.orbit.container.OrbitalSupplements;
    :param new_geometry_mask: numpy.array;
    :param kwargs: Dict;
    :**kwargs options**:
        * ** passband ** - Dict[str, elisa.observer.PassbandContainer]
        * ** left_bandwidth ** - float
        * ** right_bandwidth ** - float
        * ** atlas ** - str
    :return: Dict[str, numpy.array];
    """

    band_curves = {key: list() for key in kwargs["passband"]}
    band_curves_body, band_curves_mirror = deepcopy(band_curves), deepcopy(band_curves)

    # surface potentials with constant volume of components
    potentials = binary.correct_potentials(orbital_supplements.body[:, 4], component="all", iterations=2)

    # prepare initial orbital position container
    from_this = dict(binary_system=binary, position=const.Position(0, 1.0, 0.0, 0.0, 0.0))
    initial_system = OrbitalPositionContainer.from_binary_system(**from_this)

    # both, body and mirror should be defined in this approximation (those points are created in way to be mirrored
    # one to another), if it is not defined, there is most likely issue with method `_prepare_geosymmetric_orbit`
    ld_cfs, normal_radiance = None, None
    for idx, position_pair in enumerate(orbital_supplements):
        body, mirror = position_pair
        body_orb_pos, mirror_orb_pos = utils.convert_binary_orbital_motion_arr_to_positions([body, mirror])
        require_geometry_rebuild = new_geometry_mask[idx]

        initial_system.set_on_position_params(body_orb_pos, potentials['primary'][idx], potentials['secondary'][idx])
        initial_system = _update_surface_in_ecc_orbits(initial_system, body_orb_pos, require_geometry_rebuild)

        on_pos_body = bsutils.move_sys_onpos(initial_system, body_orb_pos)
        on_pos_mirror = bsutils.move_sys_onpos(initial_system, mirror_orb_pos)

        if require_geometry_rebuild:
            normal_radiance, ld_cfs = crv_utils.prep_surface_params(on_pos_body, **kwargs)

        coverage_b, cosines_b = calculate_coverage_with_cosines(on_pos_body, binary.semi_major_axis, in_eclipse=True)
        coverage_m, cosines_m = calculate_coverage_with_cosines(on_pos_mirror, binary.semi_major_axis, in_eclipse=True)

        for band in kwargs["passband"].keys():
            band_curves_body[band].append(curves._calculate_lc_point(band, ld_cfs, normal_radiance,
                                                                     coverage_b, cosines_b))
            band_curves_mirror[band].append(curves._calculate_lc_point(band, ld_cfs, normal_radiance,
                                                                       coverage_m, cosines_m))

    # interpolation of the points in the second half of the light curves using splines
    x = up.concatenate((orbital_supplements.body[:, 4], orbital_supplements.mirror[:, 4] % 1))
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    x = up.concatenate(([x[-1] - 1], x, [x[0] + 1]))

    for band in kwargs["passband"]:
        y = up.concatenate((band_curves_body[band], band_curves_mirror[band]))
        y = y[sort_idx]
        y = up.concatenate(([y[-1]], y, [y[0]]))

        i = Akima1DInterpolator(x, y)
        f = i(phases)
        band_curves[band] = f

    return band_curves


def _integrate_eccentric_lc_appx_two(binary, phases, orbital_supplements, new_geometry_mask, **kwargs):
    """
    Function calculates light curve for eccentric orbit for selected filters using
    approximation where to each OrbitalPosition on one side of the apsidal line,
    the closest counterpart OrbitalPosition is assigned and the same surface geometry is
    assumed for both of them.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param phases: numpy.array;
    :param orbital_supplements: elisa.binary_system.orbit.container.OrbitalSupplements;
    :param new_geometry_mask: numpy.array;
    :param kwargs: Dict;
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** atlas ** * - str
    :return: Dict[str, numpy.array];
    """
    def _produce_lc_point(orbital_position, n_radiance, ldc, cvrg, csns):
        """
        Returns lightcurve point for each passband on given orbital position.

        :param orbital_position: collections.tamedtuple; elisa.const.Position;
        :param ldc: Dict[str, Dict[str, pandas.DataFrame]];
        :param n_radiance: Dict[str, Dict[str, pandas.DataFrame]];
        :param cvrg: numpy.array;
        :param csns: numpy.array;
        :return:
        """
        for band in kwargs["passband"]:
            band_curves[band][int(orbital_position.idx)] = curves._calculate_lc_point(band, ldc, n_radiance, cvrg, csns)

    # array `used_phases` is used to check, whether flux on given phase was already computed
    # orbital supplementes tolarance test can lead
    # to same phases in templates or mirrors
    used_phases = []
    band_curves = {key: up.zeros(phases.shape) for key in kwargs["passband"]}

    # surface potentials with constant volume of components
    phases_to_correct = orbital_supplements.body[:, 4]
    potentials = binary.correct_potentials(phases_to_correct, component="all", iterations=2)

    # prepare initial orbital position container
    from_this = dict(binary_system=binary, position=const.Position(0, 1.0, 0.0, 0.0, 0.0))
    initial_system = OrbitalPositionContainer.from_binary_system(**from_this)

    for idx, position_pair in enumerate(orbital_supplements):
        body, mirror = position_pair
        body_orb_pos, mirror_orb_pos = utils.convert_binary_orbital_motion_arr_to_positions([body, mirror])
        require_geometry_rebuild = new_geometry_mask[idx]

        initial_system.set_on_position_params(body_orb_pos, potentials['primary'][idx], potentials['secondary'][idx])
        initial_system = _update_surface_in_ecc_orbits(initial_system, body_orb_pos, require_geometry_rebuild)

        if body_orb_pos.phase not in used_phases:
            on_pos_body = bsutils.move_sys_onpos(initial_system, body_orb_pos, on_copy=True)

            # recalculating normal radiances only for new geometry
            _args = _onpos_params(on_pos_body, **kwargs) if require_geometry_rebuild else \
                _args[:2] + calculate_coverage_with_cosines(on_pos_body, on_pos_body.semi_major_axis, in_eclipse=True)
            _produce_lc_point(body_orb_pos, *_args)
            used_phases += [body_orb_pos.phase]

        if (not OrbitalSupplements.is_empty(mirror)) and (mirror_orb_pos.phase not in used_phases):
            on_pos_mirror = bsutils.move_sys_onpos(initial_system, mirror_orb_pos, on_copy=True)

            _args = _args[:2] + calculate_coverage_with_cosines(on_pos_mirror, on_pos_mirror.semi_major_axis,
                                                                 in_eclipse=True)
            _produce_lc_point(mirror_orb_pos, *_args)
            used_phases += [mirror_orb_pos.phase]

    return band_curves


def _integrate_eccentric_lc_appx_three(binary, phases, orbital_positions, new_geometry_mask, **kwargs):
    """
    Function calculates light curves for eccentric binary orbits where phase span condition was not met and approx two
    could not be used. Usefull when calculating light curve using multiprocessing.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param phases: numpy.array;
    :param orbital_positions: list; list of OrbitalPositions
    :param new_geometry_mask: numpy.array;
    :param kwargs: Dict;
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** atlas ** * - str
    :return: Dict[str, numpy.array];
    """
    band_curves = {key: up.zeros(phases.shape) for key in kwargs["passband"]}

    # surface potentials with constant volume of components
    potentials = binary.correct_potentials(phases, component="all", iterations=2)

    # prepare initial orbital position container
    from_this = dict(binary_system=binary, position=const.Position(0, 1.0, 0.0, 0.0, 0.0))
    initial_system = OrbitalPositionContainer.from_binary_system(**from_this)

    for idx, orbital_position in enumerate(orbital_positions):
        require_geometry_rebuild = new_geometry_mask[idx]

        initial_system.set_on_position_params(orbital_position, potentials['primary'][idx],
                                              potentials['secondary'][idx])
        initial_system = _update_surface_in_ecc_orbits(initial_system, orbital_position, require_geometry_rebuild)

        on_pos_body = bsutils.move_sys_onpos(initial_system, orbital_position, on_copy=True)

        # recalculating normal radiances only for new geometry
        if require_geometry_rebuild:
            n_radiance, ldc = crv_utils.prep_surface_params(on_pos_body, **kwargs)
        cvrg, csns = calculate_coverage_with_cosines(on_pos_body, on_pos_body.semi_major_axis, in_eclipse=True)

        for band in kwargs["passband"]:
            band_curves[band][int(orbital_position.idx)] = curves._calculate_lc_point(band, ldc, n_radiance, cvrg, csns)

    return band_curves


def compute_eccentric_spotty_lightcurve(binary, **kwargs):
    """
    Function returns light curve of assynchronous systems with eccentric orbits and spots.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param kwargs: Dict; kwargs taken from BinarySystem `compute_lightcurve` function
    :return: Dict; dictionary of fluxes for each filter
    """
    phases = kwargs.pop("phases")
    position_method = kwargs.pop("position_method")
    orbital_motion = position_method(input_argument=phases, return_nparray=False, calculate_from='phase')

    # pre-calculate the longitudes of each spot for each phase
    spots_longitudes = dynamic.calculate_spot_longitudes(binary, phases, component="all")

    # calculating lc with spots gradually shifting their positions in each phase
    band_curves = {key: up.zeros(phases.shape) for key in kwargs["passband"]}

    # surface potentials with constant volume of components
    potentials = binary.correct_potentials(phases, component="all", iterations=2)

    for pos_idx, position in enumerate(orbital_motion):
        from_this = dict(binary_system=binary, position=position)
        on_pos = OrbitalPositionContainer.from_binary_system(**from_this)
        # assigning new longitudes for each spot
        dynamic.assign_spot_longitudes(on_pos, spots_longitudes, index=pos_idx, component="all")
        on_pos.set_on_position_params(position, potentials['primary'][pos_idx], potentials['secondary'][pos_idx])
        on_pos.build(components_distance=position.distance)
        on_pos = bsutils.move_sys_onpos(on_pos, position, on_copy=False)
        normal_radiance, ld_cfs = crv_utils.prep_surface_params(on_pos, **kwargs)

        coverage, cosines = calculate_coverage_with_cosines(on_pos, binary.semi_major_axis, in_eclipse=True)

        for band in kwargs["passband"]:
            band_curves[band][pos_idx] = curves._calculate_lc_point(band, ld_cfs, normal_radiance, coverage, cosines)

    return band_curves
