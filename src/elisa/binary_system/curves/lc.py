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
from ...binary_system import radius as bsradius
from ...binary_system.orbit.container import OrbitalSupplements
from ...binary_system.surface.coverage import calculate_coverage_with_cosines
from ...binary_system.curves import lcmp, shared
from elisa.observer.mp import manage_observations
from elisa import atm

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
    _normal_radiance, _ld_cfs = shared.prep_surface_params(on_pos, **kwargs)

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


def _compute_rel_d_radii(binary, distances):
    """
    Requires `orbital_supplements` sorted by distance.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param distances: array; component distances of templates
    :return: numpy.array;
    """
    # note: defined bodies/objects/templates in orbital supplements instance are sorted by distance (line above),
    # what means that also radii computed from such values have to be already sorted by their own size (radius changes
    # based on components distance and it is, on the half of orbit defined by apsidal line, monotonic function)

    q, d = binary.mass_ratio, distances
    pargs = (d, binary.primary.surface_potential, q, binary.primary.synchronicity, "primary")
    sargs = (d, binary.secondary.surface_potential, q, binary.secondary.synchronicity, "secondary")

    fwd_radii = {
        "primary": bsradius.calculate_forward_radii(*pargs),
        "secondary": bsradius.calculate_forward_radii(*sargs)
    }
    fwd_radii = np.array(list(fwd_radii.values()))
    return up.abs(fwd_radii[:, 1:] - fwd_radii[:, :-1]) / fwd_radii[:, 1:]


def _look_for_approximation(not_pulsations_test):
    """
    This condition checks if even to attempt to utilize apsidal line symmetry approximations.

    :param not_pulsations_test: bool;
    :return: bool;
    """

    return config.POINTS_ON_ECC_ORBIT > 0 and config.POINTS_ON_ECC_ORBIT is not None \
        and not_pulsations_test


def _eval_approximation_one(binary, phases, phases_span_test):
    """
    Test if it is appropriate to compute eccentric binary system with approximation approximation one.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param phases: numpy.array;
    :return: bool;
    """
    if len(phases) > config.POINTS_ON_ECC_ORBIT and phases_span_test:
        return True
    return False


def _eval_approximation_two(rel_d, phases_span_test):
    """
    Test if it is appropriate to compute eccentric binary system with approximation approax two.

    :param rel_d: numpy.array;
    :return: bool;
    """
    # defined bodies/objects/templates in orbital supplements instance are sorted by distance,
    # That means that also radii `rel_d` computed from such values have to be already sorted by
    # their own size (forward radius changes based on components distance and it is monotonic function)

    if np.max(rel_d[:, 1:]) < config.MAX_RELATIVE_D_R_POINT and phases_span_test:
        return True
    return False


def _split_orbit_by_apse_line(orbital_motion, orbital_mask):
    """
    Split orbital positions represented by `orbital_motion` array on two groups separated by line of apsides.
    Separation is defined by `orbital_mask`

    :param orbital_motion: numpy.array; arraywhcih represents orbital positions
    :param orbital_mask: numpy.array[bool]; mask which defines separation (True is one side and False is other side)
    :return: Tuple[numpy.array, numpy.array];
    """
    reduced_orbit_arr = orbital_motion[orbital_mask]
    supplement_to_reduced_arr = orbital_motion[~orbital_mask]
    return reduced_orbit_arr, supplement_to_reduced_arr


def _prepare_geosymmetric_orbit(binary, azimuths, phases):
    """
    Prepare set of orbital positions that are symmetrical in therms of surface geometry, where orbital position is
    mirrored via apsidal line in order to reduce time for generating the light curve.

    :param binary: elisa.binary_star.system.BinarySystem;
    :param azimuths: numpy.array; orbital azimuths of positions in which LC will be calculated
    :param phases: numpy.array; orbital phase of positions in which LC will be calculated
    :return: Tuple;


    shape ::

        (numpy.array, list, numpy.array)

    - unique_phase_indices - numpy.array : indices that points to the orbital positions from one half of the
    orbital motion divided by apsidal line
    - orbital_motion_counterpart - list - Positions produced by mirroring orbital positions given by
    indices `unique_phase_indices`
    - orbital_motion_array_counterpart - numpy.array - sa as `orbital_motion_counterpart` but in numpy.array form
    """
    azimuth_boundaries = [binary.argument_of_periastron, (binary.argument_of_periastron + const.PI) % const.FULL_ARC]
    unique_geometry = up.logical_and(azimuths > azimuth_boundaries[0],
                                     azimuths < azimuth_boundaries[1]) \
        if azimuth_boundaries[0] < azimuth_boundaries[1] else up.logical_xor(azimuths < azimuth_boundaries[0],
                                                                             azimuths > azimuth_boundaries[1])
    unique_phase_indices = up.arange(phases.shape[0])[unique_geometry]
    unique_geometry_azimuths = azimuths[unique_geometry]
    unique_geometry_counterazimuths = (2 * binary.argument_of_periastron - unique_geometry_azimuths) % const.FULL_ARC

    orbital_motion_array_counterpart = \
        binary.calculate_orbital_motion(input_argument=unique_geometry_counterazimuths,
                                        return_nparray=True,
                                        calculate_from='azimuth')

    return unique_phase_indices, orbital_motion_array_counterpart, unique_geometry


def _resolve_ecc_approximation_method(binary, phases, position_method, try_to_find_appx, phases_span_test, **kwargs):
    """
    Resolve and return approximation method to compute lightcurve in case of eccentric orbit.
    Return value is lambda function with already prepared params.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param phases: numpy.array;
    :param position_method: function;
    :param try_to_find_appx: bool;
    :param phases_span_test: bool; test if phases coverage is sufiicient for phases mirroring along apsidal line
    :param kwargs: Dict;
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** atlas ** * - str
    :return: lambda;
    """
    params = dict(input_argument=phases, return_nparray=True, calculate_from='phase')
    all_orbital_pos_arr = position_method(**params)
    all_orbital_pos = utils.convert_binary_orbital_motion_arr_to_positions(all_orbital_pos_arr)

    azimuths = all_orbital_pos_arr[:, 2]
    reduced_phase_ids, counterpart_postion_arr, reduced_phase_mask = \
        _prepare_geosymmetric_orbit(binary, azimuths, phases)

    # spliting orbital motion into two separate groups on different sides of apsidal line
    reduced_orbit_arr, reduced_orbit_supplement_arr = _split_orbit_by_apse_line(all_orbital_pos_arr, reduced_phase_mask)

    # APPX ZERO ********************************************************************************************************
    if not try_to_find_appx:
        return 'zero', lambda: _integrate_eccentric_lc_exactly(binary, all_orbital_pos, phases, **kwargs)

    # APPX THREE *******************************************************************************************************
    if not phases_span_test:
        sorted_all_orbital_pos_arr = all_orbital_pos_arr[all_orbital_pos_arr[:, 1].argsort()]
        rel_d_radii = _compute_rel_d_radii(binary, sorted_all_orbital_pos_arr[:, 1])
        new_geometry_mask = dynamic.resolve_object_geometry_update(binary.has_spots(),
                                                                   all_orbital_pos_arr.shape[0], rel_d_radii)
        approx_three = not (~new_geometry_mask).all()
        if approx_three:
            return 'three', lambda: _integrate_eccentric_lc_appx_three(binary, phases, all_orbital_pos,
                                                                       new_geometry_mask, **kwargs)

    # APPX ONE *********************************************************************************************************
    appx_one = _eval_approximation_one(binary, phases, phases_span_test)

    if appx_one:
        orbital_supplements = OrbitalSupplements(body=reduced_orbit_arr, mirror=counterpart_postion_arr)
        orbital_supplements.sort(by='distance')
        rel_d_radii = _compute_rel_d_radii(binary, orbital_supplements.body[:, 1])
        new_geometry_mask = dynamic.resolve_object_geometry_update(binary.has_spots(),
                                                                   orbital_supplements.size(), rel_d_radii)

        return 'one', lambda: _integrate_eccentric_lc_appx_one(binary, phases, orbital_supplements,
                                                               new_geometry_mask, **kwargs)

    # APPX TWO *********************************************************************************************************
    # create object of separated objects and supplements to bodies
    orbital_supplements = dynamic.find_apsidally_corresponding_positions(reduced_orbit_arr[:, 1],
                                                                         reduced_orbit_arr,
                                                                         reduced_orbit_supplement_arr[:, 1],
                                                                         reduced_orbit_supplement_arr,
                                                                         tol=config.MAX_SUPPLEMENTAR_D_DISTANCE)

    orbital_supplements.sort(by='distance')
    rel_d_radii = _compute_rel_d_radii(binary, orbital_supplements.body[:, 1])
    appx_two = _eval_approximation_two(rel_d_radii, phases_span_test)
    new_geometry_mask = dynamic.resolve_object_geometry_update(binary.has_spots(),
                                                               orbital_supplements.size(), rel_d_radii)

    if appx_two:
        return 'two', lambda: _integrate_eccentric_lc_appx_two(binary, phases, orbital_supplements,
                                                               new_geometry_mask, **kwargs)

    return 'zero', lambda: _integrate_eccentric_lc_exactly(binary, all_orbital_pos, phases, **kwargs)

    # # attempt APPX_THREE if some phases allow else APPX ZERO once again *********************************************
    # sorted_all_orbital_pos_arr = all_orbital_pos_arr[all_orbital_pos_arr[:, 1].argsort()]
    # rel_d_radii = _compute_rel_d_radii(binary, sorted_all_orbital_pos_arr[:, 1])
    # new_geometry_mask = \
    #     dynamic.resolve_object_geometry_update(binary.has_spots(),
    #                                            all_orbital_pos_arr.shape[0], rel_d_radii,
    #                                            max_allowed_difference=config.MAX_RELATIVE_D_R_POINT/10.0)
    # approx_three = not (~new_geometry_mask).all()
    # if approx_three:
    #     return 'three', lambda: _integrate_eccentric_lc_appx_three(binary, phases, all_orbital_pos,
    #                                                                new_geometry_mask, **kwargs)
    # else:
    #     return 'zero', lambda: _integrate_eccentric_lc_exactly(binary, all_orbital_pos, phases, **kwargs)


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

    initial_system = shared.prep_initial_system(binary)

    phases = kwargs.pop("phases")
    unique_phase_interval, reverse_phase_map = dynamic.phase_crv_symmetry(initial_system, phases)

    band_curves = shared.produce_circ_sync_curves(binary, initial_system, unique_phase_interval,
                                                  lcmp.compute_circular_synchronous_lightcurve, **kwargs)

    band_curves = {band: band_curves[band][reverse_phase_map] for band in band_curves}
    return band_curves


def compute_eccentric_lightcurve(binary, **kwargs):
    """
    Top-level helper method to compute eccentric lightcurve.

    :param binary: elisa.binary_star.system.BinarySystem;
    :param kwargs: Dict;
    :return: Dict[str, numpy.array];
    """
    phases = kwargs.pop("phases")
    phases_span_test = np.max(phases) - np.min(phases) >= 0.8
    position_method = kwargs.pop("position_method")

    # this condition checks if even to attempt to utilize apsidal line symmetry approximations
    # curve has to have enough point on orbit and have to span at least in 0.8 phase

    try_to_find_appx = _look_for_approximation(not binary.has_pulsations())

    appx_uid, run = _resolve_ecc_approximation_method(binary, phases, position_method, try_to_find_appx,
                                                      phases_span_test, **kwargs)

    logger_messages = {
        'zero': 'lc will be calculated in a rigorous `phase to phase manner` without approximations',
        'one': 'one half of the points on LC on the one side of the apsidal line will be interpolated',
        'two': 'geometry of the stellar surface on one half of the apsidal '
               'line will be copied from their symmetrical counterparts',
        'three': 'surface geometry at some orbital positions will not be recalculated due to similarities to previous '
                 'orbital positions'
    }
    logger.info(logger_messages.get(appx_uid))
    return run()


def _integrate_eccentric_lc_exactly(binary, orbital_motion, phases, **kwargs):
    """
    Function calculates LC for eccentric orbit for selected filters.
    LC is calculated exactly for each OrbitalPosition.
    It is very slow and it should be used only as a benchmark.

    :param binary: elisa.binary_system.system.BinarySystem; instance
    :param orbital_motion: list of all OrbitalPositions at which LC will be calculated
    :param phases: phases in which the phase curve will be calculated
    :param kwargs: kwargs taken from `compute_eccentric_lightcurve` function
    :return: Dict; dictionary of fluxes for each filter
    """
    # surface potentials with constant volume of components
    potentials = binary.correct_potentials(phases, component="all", iterations=2)
    args = (binary, orbital_motion, potentials, kwargs)
    band_curves = lcmp.integrate_eccentric_lc_exactly(*args)
    return band_curves


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
        * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
        * ** left_bandwidth ** * - float
        * ** right_bandwidth ** * - float
        * ** atlas ** * - str
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
    for idx, position_pair in enumerate(orbital_supplements):
        body, mirror = position_pair
        body_orb_pos, mirror_orb_pos = utils.convert_binary_orbital_motion_arr_to_positions([body, mirror])
        require_geometry_rebuild = new_geometry_mask[idx]

        initial_system.set_on_position_params(body_orb_pos, potentials['primary'][idx], potentials['secondary'][idx])
        initial_system = _update_surface_in_ecc_orbits(initial_system, body_orb_pos, require_geometry_rebuild)

        on_pos_body = bsutils.move_sys_onpos(initial_system, body_orb_pos)
        on_pos_mirror = bsutils.move_sys_onpos(initial_system, mirror_orb_pos)

        if require_geometry_rebuild:
            normal_radiance, ld_cfs = shared.prep_surface_params(on_pos_body, **kwargs)

        coverage_b, cosines_b = calculate_coverage_with_cosines(on_pos_body, binary.semi_major_axis, in_eclipse=True)
        coverage_m, cosines_m = calculate_coverage_with_cosines(on_pos_mirror, binary.semi_major_axis, in_eclipse=True)

        for band in kwargs["passband"].keys():
            band_curves_body[band].append(shared.calculate_lc_point(band, ld_cfs, normal_radiance,
                                                                    coverage_b, cosines_b))
            band_curves_mirror[band].append(shared.calculate_lc_point(band, ld_cfs, normal_radiance,
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
            band_curves[band][int(orbital_position.idx)] = shared.calculate_lc_point(band, ldc, n_radiance, cvrg, csns)

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
            n_radiance, ldc = shared.prep_surface_params(on_pos_body, **kwargs)
        cvrg, csns = calculate_coverage_with_cosines(on_pos_body, on_pos_body.semi_major_axis, in_eclipse=True)

        for band in kwargs["passband"]:
            band_curves[band][int(orbital_position.idx)] = shared.calculate_lc_point(band, ldc, n_radiance, cvrg, csns)

    return band_curves


def compute_circular_spotty_asynchronous_lightcurve(binary, **kwargs):
    """
    Function returns light curve of assynchronous systems with circular orbits and spots.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param kwargs: Dict;
    :**kwargs options**:
        * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
        * ** left_bandwidth ** * - float
        * ** right_bandwidth ** * - float
        * ** atlas ** * - str
    :return: Dict; fluxes for each filter
    """
    phases = kwargs.pop("phases")
    position_method = kwargs.pop("position_method")
    orbital_motion = position_method(input_argument=phases, return_nparray=False, calculate_from='phase')
    ecl_boundaries = dynamic.get_eclipse_boundaries(binary, 1.0)

    from_this = dict(binary_system=binary, position=const.Position(0, 1.0, 0.0, 0.0, 0.0))
    initial_system = OrbitalPositionContainer.from_binary_system(**from_this)

    points = dict()
    for component in config.BINARY_COUNTERPARTS:
        star = getattr(initial_system, component)
        _a, _b, _c, _d = surface.mesh.mesh_detached(initial_system, 1.0, component, symmetry_output=True)
        points[component] = _a
        setattr(star, "points", copy(_a))
        setattr(star, "point_symmetry_vector", _b)
        setattr(star, "base_symmetry_points_number", _c)
        setattr(star, "inverse_point_symmetry_matrix", _d)

    fn_args = binary, initial_system, points, ecl_boundaries
    band_curves = manage_observations(fn=lcmp.compute_circular_spotty_asynchronous_lightcurve,
                                      fn_args=fn_args,
                                      position=orbital_motion,
                                      **kwargs)

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
        on_pos.build(components_distance=position.distance)
        on_pos = bsutils.move_sys_onpos(on_pos, position, potentials["primary"][pos_idx],
                                        potentials["secondary"][pos_idx], on_copy=False)
        normal_radiance, ld_cfs = shared.prep_surface_params(on_pos, **kwargs)

        coverage, cosines = calculate_coverage_with_cosines(on_pos, binary.semi_major_axis, in_eclipse=True)

        for band in kwargs["passband"]:
            band_curves[band][pos_idx] = shared.calculate_lc_point(band, ld_cfs, normal_radiance, coverage, cosines)

    return band_curves
