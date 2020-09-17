from . import shared, lcmp
from .. container import SystemContainer
from ... import const
from ... logger import getLogger
from ... observer.mp_manager import manage_observations

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
        * ** atlas ** * - str
        * ** position_method** * - function definition; to evaluate orbital positions
    :return: Dict[str, numpy.array];
    """
    from_this = dict(single_system=single, position=const.SinglePosition(0, 0.0, 0.0))
    initial_system = SystemContainer.from_single_system(**from_this)
    initial_system.build()

    phases = kwargs.pop("phases")
    normal_radiance, ld_cfs = shared.prep_surface_params(initial_system.copy().flatt_it(), **kwargs)

    fn_args = (single, initial_system, normal_radiance, ld_cfs)
    band_curves = manage_observations(fn=lcmp.compute_non_pulsating_lightcurve,
                                      fn_args=fn_args,
                                      position=phases,
                                      **kwargs)
    return band_curves


def compute_light_curve_with_pulsations(single, **kwargs):
    from_this = dict(single_system=single, position=const.SinglePosition(0, 0.0, 0.0))
    initial_system = SystemContainer.from_single_system(**from_this)
    initial_system.build_surface()

    phases = kwargs.pop("phases")

    fn_args = single, initial_system
    band_curves = manage_observations(fn=lcmp.compute_pulsating_light_curve,
                                      fn_args=fn_args,
                                      position=phases,
                                      **kwargs)

    return band_curves
