from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Any

import numpy as np

from elisa import const, settings
from elisa.binary_system import dynamic, surface
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.binary_system.curves import c_appx_router, c_managed
from elisa.binary_system.curves import utils as crv_utils
from elisa.logger import getLogger
from elisa.observer.mp_manager import manage_observations

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from elisa.binary_system.system import BinarySystem
    from elisa.types import Float


logger = getLogger("binary_system.curves.c_router")


PHASE_SPAN_TEST_THRESHOLD = 0.79


def resolve_curve_method(
    system: BinarySystem,
    curve: str,
) -> Callable[..., dict[str, NDArray[Float]]]:
    """Resolve which curve-calculation method to use.

    The selection depends on the requested curve type and on the properties of
    the binary system.

    :param system: Binary system instance.
    :type system: BinarySystem
    :param curve: Curve selector. Allowed values are ``"lc"`` and ``"rv"``.
    :type curve: str
    :return: Callable curve-calculation method.
    :rtype: Callable[..., dict[str, NDArray[Float]]]
    """
    if curve == "lc":
        # noinspection PyProtectedMember
        fn_array = (
            system._compute_circular_synchronous_lightcurve,  # noqa: SLF001
            system._compute_circular_spotty_asynchronous_lightcurve,  # noqa: SLF001
            system._compute_circular_pulsating_lightcurve,  # noqa: SLF001
            system._compute_eccentric_spotty_lightcurve,  # noqa: SLF001
            system._compute_eccentric_lightcurve,  # noqa: SLF001
        )
    elif curve == "rv":
        # noinspection PyProtectedMember
        fn_array = (
            system._compute_circular_synchronous_rv_curve,  # noqa: SLF001
            system._compute_circular_spotty_asynchronous_rv_curve,  # noqa: SLF001
            system._compute_circular_pulsating_rv_curve,  # noqa: SLF001
            system._compute_eccentric_spotty_rv_curve,  # noqa: SLF001
            system._compute_eccentric_rv_curve_no_spots,  # noqa: SLF001
        )
    else:
        message = "Invalid value of argument `curve`. Only `lc` and `rv` are allowed."
        raise ValueError(message)

    is_circular = system.eccentricity == 0
    is_eccentric = 1 > system.eccentricity > 0
    asynchronous_spotty_primary = system.primary.synchronicity != 1 and system.primary.has_spots()
    asynchronous_spotty_secondary = system.secondary.synchronicity != 1 and system.secondary.has_spots()
    asynchronous_spotty_test = asynchronous_spotty_primary or asynchronous_spotty_secondary

    spotty_test_eccentric = system.primary.has_spots() or system.secondary.has_spots()

    if is_circular:
        if asynchronous_spotty_test:
            logger.debug(
                "calculating curve for circular binary system with asynchronous spotty components",
            )
            return fn_array[1]
        if system.has_pulsations():
            logger.debug(
                "calculating curve for circular binary system with pulsations without asynchronous spots",
            )
            return fn_array[2]

        logger.debug(
            "calculating curve for circular binary system without pulsations "
            "and without asynchronous spotty components",
        )
        return fn_array[0]

    if is_eccentric:
        if spotty_test_eccentric:
            logger.debug("calculating curve for eccentric binary system with spotty components")
            return fn_array[3]

        logger.debug("calculating curve for eccentric binary system without spotty components")
        return fn_array[4]

    message = "Orbit type not implemented or invalid."
    raise NotImplementedError(message)


def prep_initial_system(
    binary: BinarySystem,
    **kwargs: Any,
) -> OrbitalPositionContainer:
    """Prepare the base binary system for circular synchronous curves.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param kwargs: Additional keyword arguments.

        Supported options include:

        - ``build_pulsations`` - ``bool``
    :type kwargs: Any
    :return: Prepared orbital-position container.
    :rtype: OrbitalPositionContainer
    """
    from_this = {
        "binary_system": binary,
        "position": const.Position(0, 1.0, 0.0, 0.0, 0.0),
    }
    initial_system = OrbitalPositionContainer.from_binary_system(**from_this)
    do_pulsations = kwargs.get("build_pulsations", True)
    initial_system.build(
        components_distance=1.0,
        build_pulsations=do_pulsations,
    )
    return initial_system


def produce_circular_sync_curves(
    binary: BinarySystem,
    initial_system: OrbitalPositionContainer,
    phases: NDArray[Float],
    curve_fn: Callable[..., dict[str, NDArray[Float]]],
    crv_labels: list[str],
    **kwargs: Any,
) -> dict[str, NDArray[Float]]:
    """Produce curves for a circular synchronous binary system.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param initial_system: Initial orbital-position container.
    :type initial_system: OrbitalPositionContainer
    :param phases: Orbital phases.
    :type phases: NDArray[Float]
    :param curve_fn: Function used to calculate the given type of curve.
    :type curve_fn: Callable[..., dict[str, NDArray[Float]]]
    :param crv_labels: Labels of the calculated curves.
    :type crv_labels: list[str]
    :param kwargs: Additional keyword arguments.

        Supported options include:

        - ``passband`` - ``dict[str, elisa.observer.PassbandContainer]``
        - ``left_bandwidth`` - ``Float``
        - ``right_bandwidth`` - ``Float``
        - ``position_method`` - function definition to evaluate orbital positions
        - ``phases`` - ``numpy.array``
    :type kwargs: Any
    :return: Calculated curves.
    :rtype: dict[str, NDArray[Float]]
    """
    crv_utils.prep_surface_params(
        initial_system,
        return_values=False,
        write_to_containers=True,
        **kwargs,
    )
    fn = c_managed.produce_circ_sync_curves_mp
    fn_args = (binary, initial_system, crv_labels, curve_fn)
    return manage_observations(
        fn=fn,
        fn_args=fn_args,
        position=phases,
        **kwargs,
    )


def produce_circular_spotty_async_curves(
    binary: BinarySystem,
    curve_fn: Callable[..., dict[str, NDArray[Float]]],
    crv_labels: list[str],
    **kwargs: Any,
) -> dict[str, NDArray[Float]]:
    """Return curves of asynchronous systems with circular orbits and spots.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param curve_fn: Curve function.
    :type curve_fn: Callable[..., dict[str, NDArray[Float]]]
    :param crv_labels: Labels of the calculated curves.
    :type crv_labels: list[str]
    :param kwargs: Additional keyword arguments.

        Supported options include:

        - ``passband`` - ``dict[str, elisa.observer.PassbandContainer]``
        - ``left_bandwidth`` - ``Float``
        - ``right_bandwidth`` - ``Float``
    :type kwargs: Any
    :return: Calculated curves.
    :rtype: dict[str, NDArray[Float]]
    """
    phases = kwargs.pop("phases")
    position_method = kwargs.pop("position_method")
    orbital_motion = position_method(
        input_argument=phases,
        return_nparray=False,
        calculate_from="phase",
    )
    ecl_boundaries = dynamic.get_eclipse_boundaries(binary, 1.0)

    from_this = {
        "binary_system": binary,
        "position": const.Position(0, 1.0, 0.0, 0.0, 0.0),
    }
    initial_system = OrbitalPositionContainer.from_binary_system(**from_this)

    points: dict[str, NDArray[Float]] = {}
    for component in settings.BINARY_COUNTERPARTS:
        star = getattr(initial_system, component)
        pts, symmetry_count, inverse_matrix = surface.mesh.mesh_detached(
            initial_system,
            1.0,
            component,
            symmetry_output=True,
        )
        points[component] = pts
        star.points = copy(pts)
        star.base_symmetry_points_number = symmetry_count
        star.inverse_point_symmetry_matrix = inverse_matrix

    fn_args = (
        binary,
        initial_system,
        points,
        ecl_boundaries,
        crv_labels,
        curve_fn,
    )
    fn = c_managed.produce_circ_spotty_async_curves_mp
    return manage_observations(
        fn=fn,
        fn_args=fn_args,
        position=orbital_motion,
        **kwargs,
    )


def produce_circular_pulsating_curves(
    binary: BinarySystem,
    initial_system: OrbitalPositionContainer,
    phases: NDArray[Float],
    curve_fn: Callable[..., dict[str, NDArray[Float]]],
    crv_labels: list[str],
    **kwargs: Any,
) -> dict[str, NDArray[Float]]:
    """Return curves of pulsating systems with circular orbits.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param initial_system: Initial orbital-position container.
    :type initial_system: OrbitalPositionContainer
    :param phases: Orbital phases.
    :type phases: NDArray[Float]
    :param curve_fn: Curve function.
    :type curve_fn: Callable[..., dict[str, NDArray[Float]]]
    :param crv_labels: Labels of the calculated curves.
    :type crv_labels: list[str]
    :param kwargs: Additional keyword arguments.

        Supported options include:

        - ``passband`` - ``dict[str, elisa.observer.PassbandContainer]``
        - ``left_bandwidth`` - ``Float``
        - ``right_bandwidth`` - ``Float``
    :type kwargs: Any
    :return: Calculated curves.
    :rtype: dict[str, NDArray[Float]]
    """
    fn = c_managed.produce_circ_pulsating_curves_mp
    fn_args = (binary, initial_system, crv_labels, curve_fn)
    return manage_observations(
        fn=fn,
        fn_args=fn_args,
        position=phases,
        **kwargs,
    )


def produce_ecc_curves_no_spots(
    binary: BinarySystem,
    curve_fn: Callable[..., dict[str, NDArray[Float]]],
    crv_labels: list[str],
    **kwargs: Any,
) -> dict[str, NDArray[Float]]:
    """Generate curves of binaries with eccentric orbit and no spots.

    Different curve-integration approximations are evaluated and used when
    appropriate.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param curve_fn: Generator function of curve points.
    :type curve_fn: Callable[..., dict[str, NDArray[Float]]]
    :param crv_labels: Labels of the calculated curves.
    :type crv_labels: list[str]
    :param kwargs: Additional keyword arguments.

        Supported options include:

        - ``passband`` - ``dict[str, elisa.observer.PassbandContainer]``
        - ``left_bandwidth`` - ``Float``
        - ``right_bandwidth`` - ``Float``
    :type kwargs: Any
    :return: Calculated curves.
    :rtype: dict[str, NDArray[Float]]
    """
    phases = kwargs.pop("phases")

    # this condition checks if even to attempt to utilize apsidal line symmetry approximations
    # curve has to have enough point on orbit and have to span at least in 0.8 phase

    # this will remove large gap in phases
    max_diff = np.max(np.diff(np.sort(phases), n=1))
    phases_span_test = np.max(phases) - np.min(phases) - max_diff >= PHASE_SPAN_TEST_THRESHOLD

    position_method = kwargs.pop("position_method")
    try_to_find_appx = c_appx_router.look_for_approximation(not_pulsations_test=not binary.has_pulsations())

    args = (
        binary,
        phases,
        position_method,
        crv_labels,
        curve_fn,
    )
    appx_uid, run = c_appx_router.resolve_ecc_approximation_method(
        *args, try_to_find_appx=try_to_find_appx, phases_span_test=phases_span_test, **kwargs,
    )

    logger_messages = {
        "zero": "curve will be calculated in a rigorous `phase to phase manner` without approximations",
        "one": "one half of the curve points on the one side of the apsidal line will be interpolated",
        "two": (
            "geometry of the stellar surface on one half of the apsidal line "
            "will be copied from their closest symmetrical counterparts"
        ),
        "three": (
            "surface geometry at some orbital positions will not be recalculated "
            "from scratch due to similarities to previous orbital positions "
            "instead the previous most similar shape will be used"
        ),
    }
    logger.info(logger_messages.get(appx_uid))
    return run()


def produce_ecc_curves_with_spots(
    binary: BinarySystem,
    curve_fn: Callable[..., dict[str, NDArray[Float]]],
    crv_labels: list[str],
    **kwargs: Any,
) -> dict[str, NDArray[Float]]:
    """Generate curves of binaries with eccentric orbit and spots.

    .. note::
        Spotty eccentric systems must always use the exact integration method
        (no symmetry-based or similar-neighbours approximations are possible)
        due to the lack of symmetry introduced by spots.

    :param binary: Binary system instance.
    :type binary: BinarySystem
    :param curve_fn: Curve generator function.
    :type curve_fn: Callable[..., dict[str, NDArray[Float]]]
    :param crv_labels: Labels of the calculated curves.
    :type crv_labels: list[str]
    :param kwargs: Additional keyword arguments.

        Supported options include:

        - ``passband`` - ``dict[str, elisa.observer.PassbandContainer]``
        - ``left_bandwidth`` - ``Float``
        - ``right_bandwidth`` - ``Float``
    :type kwargs: Any
    :return: Calculated curves.
    :rtype: dict[str, NDArray[Float]]
    """
    phases = kwargs.pop("phases")
    position_method = kwargs.pop("position_method")
    orbital_motion = position_method(
        input_argument=phases,
        return_nparray=False,
        calculate_from="phase",
    )

    potentials = binary.correct_potentials(phases, component="all", iterations=2)

    # pre-calculate the longitudes of each spot for each phase
    spots_longitudes = dynamic.calculate_spot_longitudes(
        binary,
        phases,
        component="all",
    )
    fn_args = (binary, potentials, spots_longitudes, crv_labels, curve_fn)
    fn = c_managed.integrate_eccentric_curve_exactly
    return manage_observations(
        fn=fn,
        fn_args=fn_args,
        position=orbital_motion,
        **kwargs,
    )
