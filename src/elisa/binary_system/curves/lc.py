from __future__ import annotations

from typing import TYPE_CHECKING, Any

from elisa.binary_system import dynamic
from elisa.binary_system.curves import c_router, lc_point

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.binary_system.container import OrbitalPositionContainer
    from elisa.binary_system.system import BinarySystem
    from elisa.types import Float


def compute_circular_synchronous_lightcurve(
    binary: BinarySystem,
    **kwargs: Any,
) -> dict[str, NDArray[Float]]:
    """Compute light curve for a synchronous circular binary system.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param kwargs: Additional keyword arguments.

        Supported options include:

        - ``passband`` - ``dict[str, elisa.observer.PassbandContainer]``
        - ``left_bandwidth`` - ``Float``
        - ``right_bandwidth`` - ``Float``
        - ``position_method`` - function definition used to evaluate orbital
          positions
        - ``phases`` - ``numpy.array``
    :type kwargs: Any
    :return: Flux curves for each passband.
    :rtype: dict[str, NDArray[Float]]
    """
    initial_system: OrbitalPositionContainer = c_router.prep_initial_system(binary)

    band_labels = [*kwargs["passband"].keys()]
    phases = kwargs.pop("phases")
    unique_phase_interval, reverse_phase_map = dynamic.phase_crv_symmetry(
        initial_system,
        phases,
    )

    args = (
        binary,
        initial_system,
        unique_phase_interval,
        lc_point.compute_lc_on_pos,
        band_labels,
    )
    band_curves = c_router.produce_circular_sync_curves(*args, **kwargs)
    band_curves = {band: band_curves[band][reverse_phase_map] for band in band_curves}

    return band_curves


def compute_circular_spotty_asynchronous_lightcurve(
    binary: BinarySystem,
    **kwargs: Any,
) -> dict[str, NDArray[Float]]:
    """Return light curve of asynchronous systems with circular orbits and spots.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param kwargs: Additional keyword arguments.

        Supported options include:

        - ``passband`` - ``dict[str, elisa.observer.PassbandContainer]``
        - ``left_bandwidth`` - ``Float``
        - ``right_bandwidth`` - ``Float``
        - ``atlas`` - ``str``
    :type kwargs: Any
    :return: Flux curves for each filter.
    :rtype: dict[str, NDArray[Float]]
    """
    lc_labels = [*kwargs["passband"].keys()]
    return c_router.produce_circular_spotty_async_curves(
        binary,
        lc_point.compute_lc_on_pos,
        lc_labels,
        **kwargs,
    )


def compute_circular_pulsating_lightcurve(
    binary: BinarySystem,
    **kwargs: Any,
) -> dict[str, NDArray[Float]]:
    """Return light curve of pulsating binary systems with circular orbits.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param kwargs: Additional keyword arguments.

        Supported options include:

        - ``passband`` - ``dict[str, elisa.observer.PassbandContainer]``
        - ``left_bandwidth`` - ``Float``
        - ``right_bandwidth`` - ``Float``
        - ``atlas`` - ``str``
        - ``phases`` - ``numpy.array``
    :type kwargs: Any
    :return: Flux curves for each filter.
    :rtype: dict[str, NDArray[Float]]
    """
    initial_system = c_router.prep_initial_system(
        binary,
        build_pulsations=False,
    )
    band_labels = list(kwargs["passband"].keys())
    args = (
        binary,
        initial_system,
        kwargs.pop("phases"),
        lc_point.compute_lc_on_pos,
        band_labels,
    )
    return c_router.produce_circular_pulsating_curves(*args, **kwargs)


def compute_eccentric_lightcurve_no_spots(
    binary: BinarySystem,
    **kwargs: Any,
) -> dict[str, NDArray[Float]]:
    """Generate light curves of binaries with eccentric orbit and no spots.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param kwargs: Additional keyword arguments.

        Supported options include:

        - ``passband`` - ``dict[str, elisa.observer.PassbandContainer]``
        - ``left_bandwidth`` - ``Float``
        - ``right_bandwidth`` - ``Float``
        - ``atlas`` - ``str``
    :type kwargs: Any
    :return: Flux curves for each filter.
    :rtype: dict[str, NDArray[Float]]
    """
    lc_labels = [*kwargs["passband"].keys()]
    return c_router.produce_ecc_curves_no_spots(
        binary,
        lc_point.compute_lc_on_pos,
        lc_labels,
        **kwargs,
    )


def compute_eccentric_spotty_lightcurve(
    binary: BinarySystem,
    **kwargs: Any,
) -> dict[str, NDArray[Float]]:
    """Return light curve of systems with eccentric orbits and spots.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param kwargs: Additional keyword arguments.

        Supported options include:

        - ``passband`` - ``dict[str, elisa.observer.PassbandContainer]``
        - ``left_bandwidth`` - ``Float``
        - ``right_bandwidth`` - ``Float``
        - ``atlas`` - ``str``
    :type kwargs: Any
    :return: Dictionary of flux curves for each filter.
    :rtype: dict[str, NDArray[Float]]
    """
    lc_labels = list(kwargs["passband"].keys())
    return c_router.produce_ecc_curves_with_spots(
        binary,
        lc_point.compute_lc_on_pos,
        lc_labels,
        **kwargs,
    )
