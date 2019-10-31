import matplotlib.path as mpltpath
import numpy as np

from scipy.spatial.qhull import ConvexHull
from elisa.conf import config
from elisa.pulse import pulsations
from elisa.binary_system.container import OrbitalPositionContainer
from elisa import (
    umpy as up,
    const,
    atm,
    ld,
    logger,
    utils
)
from elisa.binary_system import (
    utils as bsutils,
    dynamic
)


config.set_up_logging()
__logger__ = logger.getLogger(__name__)


def surface_area_coverage(size, visible, visible_coverage, partial=None, partial_coverage=None):
    """
    Prepare array with coverage os surface areas.

    :param size: int; size of array
    :param visible: numpy.array; full visible areas (numpy fancy indexing), array like [False, True, True, False]
    :param visible_coverage: numpy.array; defines coverage of visible (coverage onTrue positions)
    :param partial: numpy.array; partial visible areas (numpy fancy indexing)
    :param partial_coverage: numpy.array; defines coverage of partial visible
    :return: numpy.array
    """
    # initialize zeros, since there is no input for invisible (it means everything what left after is invisible)
    coverage = up.zeros(size)
    coverage[visible] = visible_coverage
    if partial is not None:
        coverage[partial] = partial_coverage
    return coverage


def partial_visible_faces_surface_coverage(points, faces, normals, hull):
    """
    Compute surface coverage of partialy visible faces.

    :param points: numpy.array
    :param faces: numpy.array
    :param normals: numpy.array
    :param hull: numpy.array; sorted clockwise to create
    matplotlib.path.Path; path of points boundary of infront component projection
    :return: numpy.array
    """
    pypex_hull = bsutils.hull_to_pypex_poly(hull)
    pypex_faces = bsutils.faces_to_pypex_poly(points[faces])
    # it is possible to None happens in intersection, tkae care about it latter
    pypex_intersection = bsutils.pypex_poly_hull_intersection(pypex_faces, pypex_hull)

    # think about surface normalisation like and avoid surface areas like 1e-6 which lead to loss in precission
    pypex_polys_surface_area = np.array(bsutils.pypex_poly_surface_area(pypex_intersection), dtype=np.float)

    inplane_points_3d = up.concatenate((points.T, [[0.0] * len(points)])).T
    inplane_surface_area = utils.triangle_areas(triangles=faces, points=inplane_points_3d)
    correction_cosine = utils.calculate_cos_theta_los_x(normals)
    retval = (inplane_surface_area - pypex_polys_surface_area) / correction_cosine
    return retval


def compute_surface_coverage(system, semi_major_axis, in_eclipse=True):
    # todo: add unittests
    """
    Compute surface coverage of faces for given orbital position
    defined by container/SingleOrbitalPositionContainer.

    :param semi_major_axis: float
    :param system: elisa.binary_system.geo.SingleOrbitalPositionContainer
    :param in_eclipse: bool
    :return: Dict
    """
    __logger__.debug(f"computing surface coverage for {system.position}")
    cover_component = 'secondary' if 0.0 < system.position.azimuth < const.PI else 'primary'
    cover_object = getattr(system, cover_component)
    undercover_object = getattr(system, config.BINARY_COUNTERPARTS[cover_component])
    undercover_visible_point_indices = np.unique(undercover_object.faces[undercover_object.indices])

    cover_object_obs_visible_projection = bsutils.get_visible_projection(cover_object)
    undercover_object_obs_visible_projection = bsutils.get_visible_projection(undercover_object)
    # get matplotlib boudary path defined by hull of projection
    if in_eclipse:
        bb_path = get_eclipse_boundary_path(cover_object_obs_visible_projection)
        # obtain points out of eclipse (out of boundary defined by hull of 'infront' object)
        out_of_bound = up.invert(bb_path.contains_points(undercover_object_obs_visible_projection))
        # undercover_visible_point_indices = undercover_visible_point_indices[out_of_bound]
    else:
        out_of_bound = np.ones(undercover_object_obs_visible_projection.shape[0], dtype=np.bool)

    undercover_visible_point_indices = undercover_visible_point_indices[out_of_bound]
    undercover_faces = np.array([const.FALSE_FACE_PLACEHOLDER] * np.shape(undercover_object.normals)[0])
    undercover_faces[undercover_object.indices] = undercover_object.faces[undercover_object.indices]
    eclipse_faces_visibility = np.isin(undercover_faces, undercover_visible_point_indices)

    # get indices of full visible, invisible and partial visible faces
    full_visible = np.all(eclipse_faces_visibility, axis=1)
    invisible = np.all(up.invert(eclipse_faces_visibility), axis=1)
    partial_visible = up.invert(full_visible | invisible)

    # process partial and full visible faces (get surface area of 3d polygon) of undercover object
    partial_visible_faces = undercover_object.faces[partial_visible]
    partial_visible_normals = undercover_object.normals[partial_visible]
    undercover_object_pts_projection = utils.plane_projection(undercover_object.points, "yz", keep_3d=False)
    if in_eclipse:
        partial_coverage = partial_visible_faces_surface_coverage(
            points=undercover_object_pts_projection,
            faces=partial_visible_faces,
            normals=partial_visible_normals,
            hull=bb_path.vertices
        )
    else:
        partial_coverage = None

    visible_coverage = utils.poly_areas(undercover_object.points[undercover_object.faces[full_visible]])

    undercover_obj_coverage = surface_area_coverage(
        size=np.shape(undercover_object.normals)[0],
        visible=full_visible, visible_coverage=visible_coverage,
        partial=partial_visible, partial_coverage=partial_coverage
    )

    visible_coverage = utils.poly_areas(cover_object.points[cover_object.faces[cover_object.indices]])
    cover_obj_coverage = surface_area_coverage(len(cover_object.faces), cover_object.indices, visible_coverage)

    # areas are now in SMA^2, converting to SI
    cover_obj_coverage *= up.power(semi_major_axis, 2)
    undercover_obj_coverage *= up.power(semi_major_axis, 2)

    return {
        cover_component: cover_obj_coverage,
        config.BINARY_COUNTERPARTS[cover_component]: undercover_obj_coverage
    }


def get_eclipse_boundary_path(hull):
    """
    Return `matplotlib.path.Path` object which represents boundary of component projection
    to plane `yz`.

    :param hull: numpy.array
    :return: matplotlib.path.Path
    """
    cover_bound = ConvexHull(hull)
    hull_points = hull[cover_bound.vertices]
    bb_path = mpltpath.Path(hull_points)
    return bb_path


def get_limbdarkening_cfs(system, component="all", **kwargs):
    """
    Returns limb darkening coefficients for each face of each component.

    :param system:
    :param component: str
    :param kwargs: Dict;
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** atlas ** * - str
    :return: Dict[str, numpy.array]
    """
    if component in ["all", "both"]:
        return {
            component:
                ld.interpolate_on_ld_grid(
                    temperature=getattr(system, component).temperatures,
                    log_g=getattr(system, component).log_g,
                    metallicity=getattr(system, component).metallicity,
                    passband=kwargs["passband"]
                ) for component in config.BINARY_COUNTERPARTS.keys()
        }
    elif component in config.BINARY_COUNTERPARTS.keys():
        return ld.interpolate_on_ld_grid(
            temperature=getattr(system, component).temperatures,
            log_g=getattr(system, component).log_g,
            metallicity=getattr(system, component).metallicity,
            passband=kwargs["passband"]
        )
    else:
        raise ValueError('Invalid value of `component` argument. '
                         'Available parameters are `primary`, `secondary` or `all`.')


def get_normal_radiance(system, component="all", **kwargs):
    """
    Compute normal radiance for all faces and all components in SingleOrbitalPositionContainer.

    :param component: str
    :param system: elisa.binary_system.container.OrbitalPositionContainer
    :param kwargs: Dict; arguments to be passed into light curve generator functions
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** atlas ** * - str
    :return: Dict[String, numpy.array]
    """
    if component in ["all", "both"]:
        return {
            component:
                atm.NaiveInterpolatedAtm.radiance(
                    **dict(
                        temperature=getattr(system, component).temperatures,
                        log_g=getattr(system, component).log_g,
                        metallicity=getattr(system, component).metallicity,
                        **kwargs
                    )
                ) for component in config.BINARY_COUNTERPARTS
        }
    elif component in config.BINARY_COUNTERPARTS:
        return atm.NaiveInterpolatedAtm.radiance(
            **dict(
                temperature=getattr(system, component).temperatures,
                log_g=getattr(system, component).log_g,
                metallicity=getattr(system, component).metallicity,
                **kwargs
            )
        )
    else:
        raise ValueError('Invalid value of `component` argument.\n'
                         'Available parameters are `primary`, `secondary` or `all`.')


def prep_surface_params(system, pulsations_test, **kwargs):
    """
    Prepares normal radiances and limb darkening coefficients variables.

    :param system: elisa.binary_system.container.OrbitalPositionContainer
    :param pulsations_test: Dict; {component: bool - has_pulsations, ...}
    :param kwargs: Dict;
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** atlas ** * - str
    :return:
    """
    has_pulsations = pulsations_test['primary'] or pulsations_test['secondary']
    if not has_pulsations:
        # compute normal radiance for each face and each component
        normal_radiance = get_normal_radiance(system, **kwargs)
        # obtain limb darkening factor for each face
        ld_cfs = get_limbdarkening_cfs(system, **kwargs)
    elif not pulsations_test['primary']:
        normal_radiance = {'primary': get_normal_radiance(system, 'primary', **kwargs)}
        ld_cfs = {'primary': get_limbdarkening_cfs(system, 'primary', **kwargs)}
    elif not pulsations_test['secondary']:
        normal_radiance = {'secondary': get_normal_radiance(system, 'secondary', **kwargs)}
        ld_cfs = {'secondary': get_limbdarkening_cfs(system, 'secondary', **kwargs)}
    else:
        normal_radiance, ld_cfs = dict(), dict()
    return normal_radiance, ld_cfs


def compute_circular_synchronous_lightcurve(binary, **kwargs):
    """
    Compute light curve, exactly, from position to position, for synchronous circular
    binary system.

    :param binary: elisa.binary_system.system.BinarySystem
    :param kwargs: Dict;
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** atlas ** * - str
            * ** position_method** * - function definition; to evaluate orbital positions
    :return: Dict[str, numpy.array]
    """
    ld_law_cfs_columns = config.LD_LAW_CFS_COLUMNS[config.LIMB_DARKENING_LAW]

    from_this = dict(binary_system=binary, position=const.BINARY_POSITION_PLACEHOLDER(0, 1.0, 0.0, 0.0, 0.0))
    initial_system = OrbitalPositionContainer.from_binary_system(**from_this)
    initial_system.build(components_distance=1.0)

    phases = kwargs.pop("phases")
    unique_phase_interval, reverse_phase_map = dynamic.phase_crv_symmetry(initial_system, phases)

    position_method = kwargs.pop("position_method")
    orbital_motion = position_method(input_argument=unique_phase_interval, return_nparray=False, calculate_from='phase')

    # is in eclipse test eval
    ecl_boundaries = dynamic.get_eclipse_boundaries(binary, 1.0)
    azimuths = [position.azimuth for position in orbital_motion]
    in_eclipse = dynamic.in_eclipse_test(azimuths, ecl_boundaries)

    pulsations_test = {'primary': binary.primary.has_pulsations(), 'secondary': binary.secondary.has_pulsations()}

    normal_radiance, ld_cfs = prep_surface_params(initial_system, pulsations_test, **kwargs)
    band_curves = {key: up.zeros(unique_phase_interval.shape) for key in kwargs["passband"].keys()}

    for pos_idx, position in enumerate(orbital_motion):
        on_pos = initial_system.copy()
        on_pos.position = position
        on_pos.flatt_it()
        on_pos.apply_rotation()
        on_pos.apply_darkside_filter()

        # dict of components
        stars = {component: getattr(on_pos, component) for component in config.BINARY_COUNTERPARTS}

        coverage = compute_surface_coverage(on_pos, binary.semi_major_axis, in_eclipse=in_eclipse[pos_idx])

        # calculating cosines between face normals and line of sight
        cosines, visibility_test = dict(), dict()
        for component, star in stars.items():
            cosines[component] = utils.calculate_cos_theta_los_x(star.normals)
            visibility_test[component] = cosines[component] > 0
            cosines[component] = cosines[component][visibility_test[component]]

        # integrating resulting flux
        for band in kwargs["passband"].keys():
            flux, ld_cors = np.empty(2), dict()

            for component_idx, component in enumerate(config.BINARY_COUNTERPARTS.keys()):
                ld_cors[component] = \
                    ld.limb_darkening_factor(
                        coefficients=ld_cfs[component][band][ld_law_cfs_columns].values[visibility_test[component]],
                        limb_darkening_law=config.LIMB_DARKENING_LAW,
                        cos_theta=cosines[component])

                flux[component_idx] = np.sum(normal_radiance[component][band][visibility_test[component]] *
                                             cosines[component] *
                                             coverage[component][visibility_test[component]] *
                                             ld_cors[component])
            band_curves[band][pos_idx] = np.sum(flux)

    band_curves = {band: band_curves[band][reverse_phase_map] for band in band_curves}
    return band_curves
