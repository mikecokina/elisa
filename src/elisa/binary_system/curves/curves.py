import numpy as np
from copy import copy

from elisa.conf import config
from elisa import ld, const
from elisa.logger import getLogger
from elisa.binary_system import (
    dynamic,
    surface
)
from elisa.binary_system.curves import (
    utils as crv_utils,
    curves_mp
)
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.observer.mp import manage_observations


logger = getLogger('binary_system.curves.shared')


def calculate_lc_point(band, system):
    """
    Calculates point on the light curve for given band.

    :param band: str; name of the photometric band compatibile with supported names in config
    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :return: float;
    """
    flux = 0.0
    for component in config.BINARY_COUNTERPARTS.keys():
        star = getattr(system, component)
        flux += crv_utils.flux_from_star_container(band, star)

    return flux


def _calculate_lc_point(band, ld_cfs, normal_radiance, coverage, cosines):
    """
    Calculates point on the light curve for given band.

    :param band: str; name of the photometric band
    :param ld_cfs: Dict[str, Dict[str, pandas.DataFrame]];
    :param normal_radiance: Dict[str, Dict[str, numpy.array]];
    :param coverage: Dict[str, Dict[str, numpy.array]];
    :param cosines: Dict[str, Dict[str, numpy.array]];
    :return: float;
    """
    ld_law_cfs_columns = config.LD_LAW_CFS_COLUMNS[config.LIMB_DARKENING_LAW]
    ld_cors = {
        component: ld.limb_darkening_factor(coefficients=ld_cfs[component][band][ld_law_cfs_columns].values,
                                            limb_darkening_law=config.LIMB_DARKENING_LAW,
                                            cos_theta=cosines[component])
        for component in config.BINARY_COUNTERPARTS
    }
    flux = {
        component:
            np.sum(normal_radiance[component][band] * cosines[component] * coverage[component] * ld_cors[component])
        for component in config.BINARY_COUNTERPARTS
    }

    flux = (flux['primary'] + flux['secondary'])
    return flux


def calculate_rv_point(star):
    """
    Calculates point on the rv curve for given component.

    :param star: elisa.base.container.StarContainer; star container with all necessary parameters pre-calculated
    :return: Union[numpy.float, numpy.nan];
    """
    indices = getattr(star, 'indices')
    velocities = getattr(star, 'velocities')[indices]

    fluxes = crv_utils.calculate_surface_element_fluxes('rv_band', star)

    return np.sum(velocities[:, 0] * fluxes) / np.sum(fluxes) if np.sum(fluxes) != 0 else np.NaN


def resolve_curve_method(system, fn_array):
    """
    Resolves which curve calculating method to use based on the type of the system.

    :param system: elisa.binary_system.BinarySystem;
    :param fn_array: tuple; list of curve calculating functions in specific order
    (circular synchronous or circular assynchronous without spots,
     circular assynchronous with spots,
     eccentric synchronous or eccentric assynchronous without spots,
     eccentric assynchronous with spots)
    :return: curve calculating method chosen from `fn_array`
    """
    is_circular = system.eccentricity == 0
    is_eccentric = 1 > system.eccentricity > 0
    assynchronous_spotty_p = system.primary.synchronicity != 1 and system.primary.has_spots()
    assynchronous_spotty_s = system.secondary.synchronicity != 1 and system.secondary.has_spots()
    assynchronous_spotty_test = assynchronous_spotty_p or assynchronous_spotty_s

    spotty_test_eccentric = system.primary.has_spots() or system.secondary.has_spots()

    if is_circular:
        if not assynchronous_spotty_test and not system.has_pulsations():
            logger.debug('Calculating curve for circular binary system without pulsations and without '
                         'asynchronous spotty components.')
            return fn_array[0]
        else:
            logger.debug('Calculating curve for circular binary system with pulsations or with asynchronous '
                         'spotty components.')
            return fn_array[1]
    elif is_eccentric:
        if spotty_test_eccentric:
            logger.debug('Calculating curve for eccentric binary system with spotty components.')
            return fn_array[2]
        else:
            logger.debug('Calculating curve for eccentric binary system without spotty '
                         'components.')
            return fn_array[3]

    raise NotImplementedError("Orbit type not implemented or invalid")


def prep_initial_system(binary):
    """
    Prepares base binary system from which curves will be calculated in case of circular synchronous binaries.

    :param binary: elisa.binary_system.system.BinarySystem
    :return: elisa.binary_system.container.OrbitalPositionContainer
    """
    from_this = dict(binary_system=binary, position=const.Position(0, 1.0, 0.0, 0.0, 0.0))
    initial_system = OrbitalPositionContainer.from_binary_system(**from_this)
    initial_system.build(components_distance=1.0)

    return initial_system


def produce_circ_sync_curves(binary, initial_system, phases, curve_fn, crv_labels, **kwargs):
    """
    Auxiliary function to produce curve from circular synchronous binary system

    :param binary: elisa.binary_system.system.BinarySystem;
    :param initial_system: elisa.binary_system.container.OrbitalPositionContainer
    :param phases: numpy.array
    :param curve_fn: function to calculate given type of the curve
    :param crv_labels: labels of the calculated curves (passbands, components,...)
    :param kwargs: Dict;
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** atlas ** * - str
            * ** position_method** * - function definition; to evaluate orbital positions
            * ** phases ** * - numpy.array
    :return: dict; calculated curves
    """

    crv_utils.prep_surface_params(initial_system.flatt_it(), return_values=False, write_to_containers=True, **kwargs)

    fn_args = (binary, initial_system, crv_labels, curve_fn)

    curves = manage_observations(fn=curves_mp.produce_circ_sync_curves_mp,
                                 fn_args=fn_args,
                                 position=phases,
                                 **kwargs)

    return curves


def produce_circ_spotty_async_curves(binary, curve_fn, crv_labels, **kwargs):
    """
    Function returns curve of assynchronous systems with circular orbits and spots.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param curve_fn: curve function
    :param crv_labels: labels of the calculated curves (passbands, components,...)
    :param kwargs: Dict;
    :**kwargs options**:
        * ** passband ** * - Dict[str, elisa .observer.PassbandContainer]
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

    fn_args = binary, initial_system, points, ecl_boundaries, crv_labels, curve_fn

    band_curves = manage_observations(fn=curves_mp.produce_circ_spotty_async_curves_mp,
                                      fn_args=fn_args,
                                      position=orbital_motion,
                                      **kwargs)

    return band_curves