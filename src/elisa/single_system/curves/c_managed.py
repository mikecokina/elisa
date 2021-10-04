import numpy as np

from . import utils
from ... single_system import (
    utils as ssutils,
)


def produce_curves_wo_pulsations_mp(*args):
    """
    Calculates curve for single system without pulsations (surface geometry is not changed).

    :param args: Tuple;
    :**args options**:
        * ** single_system ** * - elisa.single_system.system.SingleSystem
        * ** system_container ** * - elisa.single_system.container.SystemContainer
        * ** phases ** * - array; phases in which to calculate curves
        * ** crv_labels ** * - List;
        * ** curves_fn ** * - function to calculate curve points at given orbital positions
        * ** kwargs ** * - Dict;

    :return:
    """
    single, initial_system, phase_batch, crv_labels, curves_fn, kwargs = args
    position_method = kwargs.pop("position_method")

    rotational_motion = position_method(input_argument=phase_batch, return_nparray=False, calculate_from='phase')
    curves = {key: np.zeros(phase_batch.shape) for key in crv_labels}

    for pos_idx, position in enumerate(rotational_motion):
        on_pos = ssutils.move_sys_onpos(initial_system, position)
        star = on_pos.star

        setattr(star, 'coverage', star.areas)

        curves = curves_fn(curves, pos_idx, crv_labels, on_pos)

    return curves


def produce_curves_with_pulsations_mp(*args):
    """
    Calculates curve for single system with pulsations.

    :param args:
    :**args options**:
        * ** single_system ** * - elisa.single_system.system.SingleSystem
        * ** system_container ** * - elisa.single_system.container.SystemContainer
        * ** phases ** * - array; phases in which to calculate curves
        * ** crv_labels ** * - List;
        * ** curves_fn ** * - function to calculate curve points at given orbital positions
        * ** kwargs ** * - Dict;

    :return:
    """
    single, initial_system, phase_batch, crv_labels, curves_fn, kwargs = args
    position_method = kwargs.pop("position_method")

    rotational_motion = position_method(input_argument=phase_batch, return_nparray=False, calculate_from='phase')
    curves = {key: np.zeros(phase_batch.shape) for key in crv_labels}

    for pos_idx, position in enumerate(rotational_motion):
        sys_on_pos = initial_system.copy()
        sys_on_pos.set_on_position_params(position)
        sys_on_pos.set_time()

        sys_on_pos.build_pulsations()
        utils.prep_surface_params(sys_on_pos, return_values=False, write_to_containers=True, **kwargs)

        sys_on_pos = ssutils.move_sys_onpos(sys_on_pos, position, on_copy=False)
        star = sys_on_pos.star
        setattr(star, 'coverage', star.areas)

        curves = curves_fn(curves, pos_idx, crv_labels, sys_on_pos)

    return curves
