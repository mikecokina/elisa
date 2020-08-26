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


def compute_circ_sync_lc_on_pos(band_curves, pos_idx, crv_labels, system):
    """
    Calculates lc points for given orbital position in case of circular orbit and synchronous rotation.

    :param band_curves: Dict; {str; passband : numpy.array; light curve, ...} result will be written to the
                              corresponding `pos_idx` position
    :param pos_idx: int; position in `band_curves` to which calculated lc points will be assigned
    :param crv_labels: list; list of passbands
    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :return: Dict; updated {str; passband : numpy.array; light curve, ...}
    """
    # integrating resulting flux
    for band in crv_labels:
        band_curves[band][pos_idx] = shared.calculate_lc_point(band, system)

    return band_curves


def compute_circ_spotty_async_lc_at_pos(band_curves, pos_idx, crv_labels, system):
    """
    Calculates lc points for given orbital position in case of circular orbit and asynchronous rotation with spots.

    :param band_curves: Dict; {str; passband : numpy.array; light curve, ...}
    :param pos_idx: int; position in `band_curves` to which calculated lc points will be assigned
    :param crv_labels: list; list of passbands
    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :return: Dict; updated {str; passband : numpy.array; light curve, ...}
    """
    for band in crv_labels:
        band_curves[band][pos_idx] = shared.calculate_lc_point(band, system)

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
            band_curves[band][run_idx] = shared._calculate_lc_point(band, ld_cfs, normal_radiance, coverage, cosines)
    return band_curves