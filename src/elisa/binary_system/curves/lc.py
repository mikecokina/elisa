from . import (
    lc_point,
    c_router
)
from ... binary_system import dynamic

# main wrapper over lc computation


def compute_circular_synchronous_lightcurve(binary, **kwargs):
    """
    Compute light curve for synchronous circular binary system.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param kwargs: Dict;
    :**kwargs options**:
        * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
        * ** left_bandwidth ** * - float
        * ** right_bandwidth ** * - float
        * ** position_method** * - function definition; to evaluate orbital positions
        * ** phases ** * - numpy.array
    :return: Dict[str, numpy.array];
    """
    initial_system = c_router.prep_initial_system(binary)

    band_labels = [*kwargs["passband"].keys()]
    phases = kwargs.pop("phases")
    unique_phase_interval, reverse_phase_map = dynamic.phase_crv_symmetry(initial_system, phases)

    _args = (binary, initial_system, unique_phase_interval, lc_point.compute_lc_on_pos, band_labels)
    band_curves = c_router.produce_circular_sync_curves(*_args, **kwargs)
    band_curves = {band: band_curves[band][reverse_phase_map] for band in band_curves}

    return band_curves


def compute_circular_spotty_asynchronous_lightcurve(binary, **kwargs):
    """
    Function returns light curve of asynchronous systems with circular orbits and spots.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param kwargs: Dict;
    :**kwargs options**:
        * ** passband ** - Dict[str, elisa.observer.PassbandContainer]
        * ** left_bandwidth ** - float
        * ** right_bandwidth ** - float
        * ** atlas ** - str

    :return: Dict; fluxes for each filter
    """
    lc_labels = [*kwargs["passband"].keys()]
    return c_router.produce_circular_spotty_async_curves(binary, lc_point.compute_lc_on_pos, lc_labels, **kwargs)


def compute_circular_pulsating_lightcurve(binary, **kwargs):
    """
    Function returns light curve of pulsating binary systems with circular orbits.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param kwargs: Dict;
    :**kwargs options**:
        * ** passband ** - Dict[str, elisa.observer.PassbandContainer]
        * ** left_bandwidth ** - float
        * ** right_bandwidth ** - float
        * ** atlas ** - str
        * ** phases ** * - numpy.array

    :return: Dict; fluxes for each filter
    """
    initial_system = c_router.prep_initial_system(binary, **dict(build_pulsations=False))
    band_labels = list(kwargs["passband"].keys())
    _args = (binary, initial_system, kwargs.pop("phases"), lc_point.compute_lc_on_pos, band_labels)
    return c_router.produce_circular_pulsating_curves(*_args, **kwargs)


def compute_eccentric_lightcurve_no_spots(binary, **kwargs):
    """
    General function for generating light curves of binaries with eccentric orbit and no spots.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param kwargs:  Dict;
    :**kwargs options**:
        * ** passband ** - Dict[str, elisa.observer.PassbandContainer]
        * ** left_bandwidth ** - float
        * ** right_bandwidth ** - float
        * ** atlas ** - str

    :return: Dict; fluxes for each filter
    """
    lc_labels = [*kwargs["passband"].keys()]
    return c_router.produce_ecc_curves_no_spots(binary, lc_point.compute_lc_on_pos, lc_labels, **kwargs)


def compute_eccentric_spotty_lightcurve(binary, **kwargs):
    """
    Function returns light curve of systems with eccentric orbits and spots.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param kwargs: Dict;
    :**kwargs options**:
        * ** passband ** - Dict[str, elisa.observer.PassbandContainer]
        * ** left_bandwidth ** - float
        * ** right_bandwidth ** - float
        * ** atlas ** - str

    :return: Dict; dictionary of fluxes for each filter
    """
    lc_labels = list(kwargs["passband"].keys())
    return c_router.produce_ecc_curves_with_spots(binary, lc_point.compute_lc_on_pos, lc_labels, **kwargs)
