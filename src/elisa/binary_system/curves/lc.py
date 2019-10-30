from elisa.binary_system import dynamic
from elisa.binary_system.container import OrbitalPositionContainer
from elisa import const


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
    ecl_boundaries = dynamic.get_eclipse_boundaries(binary, 1.0)
    system_container = OrbitalPositionContainer.from_binary_system(
        binary_system=binary, position=const.BINARY_POSITION_PLACEHOLDER(0, 1.0, 0.0, 0.0, 0.0)
    )


    return list()
