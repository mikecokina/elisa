import numpy as np

from elisa.conf import config
from elisa.single_system import (
    utils as ssutils,
    surface
)
from elisa import (
    ld
)
from elisa.single_system.curves import shared

LD_LAW_CFS_COLUMNS = config.LD_LAW_CFS_COLUMNS[config.LIMB_DARKENING_LAW]


def compute_non_pulsating_lightcurve(*args):
    """
    calculates LC for single system without pulsations (surface geometry is not changed)

    :param args: tuple;
    :**kwargs options**:
        * ** single_system ** * - elisa.single_system.system.SingleSystem
        * ** system_container ** * - elisa.single_system.container.SystemContainer
        * ** phases ** * - array; phases in which to calculate LC
        * ** normal_radiance ** * - array;
        * ** limb_darkening_coefficients ** * - array;
        * ** kwargs ** * - Dict;
    :return:
    """
    single, initial_system, phase_batch, normal_radiance, ld_cfs, kwargs = args
    position_method = kwargs.pop("position_method")

    rotational_motion = position_method(input_argument=phase_batch, return_nparray=False, calculate_from='phase')
    band_curves = {key: np.zeros(phase_batch.shape) for key in kwargs["passband"].keys()}

    for pos_idx, position in enumerate(rotational_motion):
        on_pos = ssutils.move_sys_onpos(initial_system, position)
        star = on_pos.star

        # area of visible faces
        coverage = surface.coverage.compute_surface_coverage(on_pos)

        cosines = star.los_cosines
        visibility_indices = star.indices
        cosines = cosines[visibility_indices]

        # integrating resulting flux
        for band in kwargs["passband"].keys():
            ld_cors = ld.limb_darkening_factor(
                coefficients=ld_cfs['star'][band][LD_LAW_CFS_COLUMNS].values[visibility_indices],
                limb_darkening_law=config.LIMB_DARKENING_LAW,
                cos_theta=cosines)

            band_curves[band][pos_idx] = np.sum(normal_radiance['star'][band][visibility_indices] * cosines *
                                                coverage['star'][visibility_indices] * ld_cors)

    return band_curves


def compute_pulsating_light_curve(*args):
    single, system_container, phase_batch, kwargs = args

    position_method = kwargs.pop("position_method")

    rotational_motion = position_method(input_argument=phase_batch, return_nparray=False, calculate_from='phase')
    band_curves = {key: np.zeros(phase_batch.shape) for key in kwargs["passband"].keys()}

    for pos_idx, position in enumerate(rotational_motion):
        initial_system = system_container.copy()
        initial_system.set_on_position_params(position=position)
        initial_system.time = initial_system.set_time()
        initial_system.build_from_points()

        on_pos = ssutils.move_sys_onpos(initial_system, position)
        normal_radiance, ld_cfs = shared.prep_surface_params(initial_system.copy().flatt_it(), **kwargs)

        star = on_pos.star

        # area of visible faces
        coverage = surface.coverage.compute_surface_coverage(on_pos)

        cosines = star.los_cosines
        visibility_indices = star.indices
        cosines = cosines[visibility_indices]

        # integrating resulting flux
        for band in kwargs["passband"].keys():
            ld_cors = ld.limb_darkening_factor(
                coefficients=ld_cfs['star'][band][LD_LAW_CFS_COLUMNS].values[visibility_indices],
                limb_darkening_law=config.LIMB_DARKENING_LAW,
                cos_theta=cosines)

            band_curves[band][pos_idx] = np.sum(normal_radiance['star'][band][visibility_indices] * cosines *
                                                coverage['star'][visibility_indices] * ld_cors)

    return band_curves
