import numpy as np

from ... logger import getLogger
from ... import const
from .. container import SinglePositionContainer
from . import utils as crv_utils, c_managed
from ... observer.mp_manager import manage_observations

logger = getLogger('single_system.curves.curves')


def resolve_curve_method(system, fn_array):
    """
    Resolves which curve calculating method to use based on the properties of the SingleSystem.

    :param system: elisa.single_system.SingleSystem;
    :param fn_array: Tuple; list of curve calculating functions in specific order
                            (system with pulsations, system without pulsations)
    :return: curve calculating method chosen from `fn_array`
    """
    if system.star.has_pulsations():
        logger.debug('Calculating light curve for star system with pulsation')
        return fn_array[1]
    else:
        logger.debug('Calculating light curve for a non pulsating single star system')
        return fn_array[0]

    # raise NotImplementedError("System type not implemented or invalid.")


def prep_initial_system(single, **kwargs):
    """
    Prepares base single system from which curves will be calculated in case of single system without pulsations.

    :param single: elisa.single_system.system.SingleSystem;
    :return: elisa.single_system.container.SystemContainer;
    """
    from_this = dict(single_system=single, position=const.Position(0, np.nan, 0.0, np.nan, 0.0))
    initial_system = SinglePositionContainer.from_single_system(**from_this)
    do_pulsations = kwargs.get('build_pulsations', True)
    initial_system.build(do_pulsations)
    return initial_system


def produce_curves_wo_pulsations(single, initial_system, phases, curve_fn, crv_labels, **kwargs):
    """
    General function for creation of single system light curve without pulsations.

    :param single: elisa.single_system.system.SingleSystem;
    :param initial_system: elisa.single_system.container.SystemContainer
    :param phases: numpy.array;
    :param curve_fn: callable; function to calculate given type of the curve
    :param crv_labels: List; labels of the calculated curves (passbands, components,...)
    :param kwargs: Dict;
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** position_method** * - function definition; to evaluate orbital positions
            * ** phases ** * - numpy.array

    :return: Dict; calculated curves
    """
    crv_utils.prep_surface_params(initial_system, return_values=False, write_to_containers=True, **kwargs)
    fn_args = (single, initial_system, crv_labels, curve_fn)
    return manage_observations(fn=c_managed.produce_curves_wo_pulsations_mp, fn_args=fn_args, position=phases, **kwargs)


def produce_curves_with_pulsations(single, initial_system, phases, curve_fn, crv_labels, **kwargs):
    """
    General function for creation of single system light curve with pulsations.

    :param single: elisa.single_system.system.SingleSystem;
    :param initial_system: elisa.single_system.container.SystemContainer;
    :param phases: numpy.array;
    :param curve_fn: callable; function to calculate given type of the curve
    :param crv_labels: List; labels of the calculated curves (passbands, components,...)
    :param kwargs: Dict;
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** position_method** * - function definition; to evaluate orbital positions
            * ** phases ** * - numpy.array

    :return: Dict; calculated curves
    """
    fn_args = (single, initial_system, crv_labels, curve_fn)
    return manage_observations(fn=c_managed.produce_curves_with_pulsations_mp, fn_args=fn_args, position=phases, **kwargs)
