from ... logger import getLogger
from . import c_router, lc_point

logger = getLogger('single_system.curves.lc')


def compute_light_curve_without_pulsations(single, **kwargs):
    """
    Compute light curve for single star objects without pulsations.

    :param single: elisa.single_system.system.SinarySystem;
    :param kwargs: Dict;
    :**kwargs options**:
        * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
        * ** left_bandwidth ** * - float
        * ** right_bandwidth ** * - float
        * ** position_method ** * - function definition; to evaluate orbital positions

    :return: Dict[str, numpy.array];
    """
    initial_system = c_router.prep_initial_system(single)

    lc_labels = list(kwargs["passband"].keys())
    phases = kwargs.pop("phases")

    args = single, initial_system, phases, lc_point.compute_lc_on_pos, lc_labels
    return c_router.produce_curves_wo_pulsations(*args, **kwargs)


def compute_light_curve_with_pulsations(single, **kwargs):
    """
    Compute light curve for single star objects without pulsations.

    :param single: elisa.single_system.system.SingleSystem;
    :param kwargs: Dict;
    :**kwargs options**:
        * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
        * ** left_bandwidth ** * - float
        * ** right_bandwidth ** * - float
        * ** position_method ** * - function definition; to evaluate orbital positions

    :return: Dict[str, numpy.array];
    """
    initial_system = c_router.prep_initial_system(single, **dict(build_pulsations=False))

    lc_labels = list(kwargs["passband"].keys())
    phases = kwargs.pop("phases")

    args = single, initial_system, phases, lc_point.compute_lc_on_pos, lc_labels
    return c_router.produce_curves_with_pulsations(*args, **kwargs)
