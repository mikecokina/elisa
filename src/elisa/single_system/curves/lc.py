import numpy as np

from . import c_managed
from .. container import SystemContainer
from ... import const
from ... logger import getLogger
from ... observer.mp_manager import manage_observations
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
        * ** position_method** * - function definition; to evaluate orbital positions
    :return: Dict[str, numpy.array];
    """
    initial_system = c_router.prep_initial_system(single)

    lc_labels = list(kwargs["passband"].keys())
    phases = kwargs.pop("phases")

    args = single, initial_system, phases, lc_point.compute_lc_on_pos, lc_labels
    return c_router.produce_curves_wo_pulsations(*args, **kwargs)


def compute_light_curve_with_pulsations(single, **kwargs):
    from_this = dict(single_system=single, position=const.Position(0, np.nan, 0.0, np.nan, 0.0))
    initial_system = SystemContainer.from_single_system(**from_this)
    initial_system.build_surface()

    phases = kwargs.pop("phases")

    fn_args = single, initial_system
    band_curves = manage_observations(fn=c_managed.compute_pulsating_light_curve,
                                      fn_args=fn_args,
                                      position=phases,
                                      **kwargs)

    return band_curves
