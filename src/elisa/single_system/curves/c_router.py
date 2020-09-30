from ... logger import getLogger


logger = getLogger('single_system.curves.curves')


def resolve_curve_method(system, fn_array):
    """
    Resolves which curve calculating method to use based on the properties of the SingleSystem.

    :param system: elisa.single_system.SingleSystem;
    :param fn_array: fn_array: tuple; list of curve calculating functions in specific order
    (system with pulsations,
     system without pulsations)
    :return: curve calculating method chosen from `fn_array`
    """
    if system.star.has_pulsations():
        logger.debug('Calculating light curve for star system with pulsation')
        return fn_array[1]
    else:
        logger.debug('Calculating light curve for a non pulsating single star system')
        return fn_array[0]

    # raise NotImplementedError("System type not implemented or invalid.")


def produce_curves_wo_pulsations(single, curve_fn, crv_labels, **kwargs):
    pass