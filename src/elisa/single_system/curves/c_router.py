from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from elisa import const
from elisa.logger import getLogger
from elisa.observer.mp_manager import manage_observations
from elisa.single_system.container import SinglePositionContainer
from elisa.single_system.curves import c_managed
from elisa.single_system.curves import utils as crv_utils

# TYPE_CHECKING block at the end of import header
if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from elisa.single_system.system import SingleSystem  # pragma: no cover
    from elisa.types import Float  # pragma: no cover


logger = getLogger("single_system.curves.curves")


def resolve_curve_method(system: SingleSystem, fn_array: Sequence[Callable[..., Any]]) -> Callable[..., Any]:
    """Resolve which curve-calculating method to use for the provided system.

    Choose the appropriate curve production function depending on whether the
    star contains pulsations.

    :param system: Single system instance to inspect.
    :type system: elisa.single_system.system.SingleSystem
    :param fn_array: Ordered sequence of callables. Expected order matches
        callers' convention (non-pulsating function first, pulsating second).
    :type fn_array: collections.abc.Sequence[collections.abc.Callable]

    :returns: Selected callable responsible for producing the requested curves.
    :rtype: collections.abc.Callable
    """
    if system.star.has_pulsations():
        logger.debug("Calculating light curve for star system with pulsation")
        return fn_array[1]

    logger.debug("Calculating light curve for a non pulsating single star system")
    return fn_array[0]


def prep_initial_system(single: SingleSystem, **kwargs: Any) -> SinglePositionContainer:
    """Prepare a base single-system container for curve production.

    The returned container is constructed from the provided ``SingleSystem``
    instance and initialized for the nominal observing position.

    :param single: Source SingleSystem object.
    :type single: elisa.single_system.system.SingleSystem
    :param kwargs: Forwarded keyword arguments. Recognized key:

        - ``build_pulsations`` (bool) -- whether to build pulsations on the
          placeholder container (default: True).

    :type kwargs: dict

    :returns: Initialized SinglePositionContainer ready for curve production.
    :rtype: elisa.single_system.container.SinglePositionContainer
    """
    from_this = {"single_system": single, "position": const.Position(0, np.nan, 0.0, np.nan, 0.0)}
    initial_system = SinglePositionContainer.from_single_system(**from_this)
    do_pulsations = kwargs.get("build_pulsations", True)
    initial_system.build(build_pulsations=do_pulsations)
    return initial_system


def produce_curves_wo_pulsations(
    single: SingleSystem,
    initial_system: SinglePositionContainer,
    phases: NDArray[Float],
    curve_fn: Callable[..., Any],
    crv_labels: Sequence[str],
    **kwargs: Any,
) -> dict[str, NDArray[Float]]:
    """Produce curves for a single system without pulsations.

    This routine prepares surface parameters and delegates the parallel
    production of curves to the observation manager.

    :param single: Single system instance.
    :type single: elisa.single_system.system.SingleSystem
    :param initial_system: Prepared SinglePositionContainer used as a template.
    :type initial_system: elisa.single_system.container.SinglePositionContainer
    :param phases: Array of photometric phases to produce the curves for.
    :type phases: numpy.ndarray
    :param curve_fn: Function used to compute a per-position point.
    :type curve_fn: callable
    :param crv_labels: Labels of the calculated curves (passbands, components,...)
    :type crv_labels: sequence[str]
    :param kwargs: Forwarded keyword arguments. Expected keys include passband
        and optional bandwidth parameters; they are forwarded unchanged.
    :type kwargs: dict

    :returns: Calculated curves mapping passband -> array.
    :rtype: dict[str, numpy.ndarray]
    """
    crv_utils.prep_surface_params(initial_system, return_values=False, write_to_containers=True, **kwargs)
    fn_args = (single, initial_system, crv_labels, curve_fn)
    return manage_observations(
        fn=c_managed.produce_curves_wo_pulsations_mp,
        fn_args=fn_args,
        position=phases,
        **kwargs,
    )


def produce_curves_with_pulsations(
    single: SingleSystem,
    initial_system: SinglePositionContainer,
    phases: NDArray[Float],
    curve_fn: Callable[..., Any],
    crv_labels: Sequence[str],
    **kwargs: Any,
) -> dict[str, NDArray[Float]]:
    """Produce curves for a single system with pulsations.

    Delegate curve production to the observation manager that handles
    pulsation-enabled curve generation.

    :param single: Single system instance.
    :type single: elisa.single_system.system.SingleSystem
    :param initial_system: Prepared SinglePositionContainer used as a template.
    :type initial_system: elisa.single_system.container.SinglePositionContainer
    :param phases: Array of photometric phases to produce the curves for.
    :type phases: numpy.ndarray
    :param curve_fn: Function used to compute a per-position point.
    :type curve_fn: callable
    :param crv_labels: Labels of the calculated curves (passbands, components,...)
    :type crv_labels: sequence[str]
    :param kwargs: Forwarded keyword arguments. Expected keys include passband
        and optional bandwidth parameters; they are forwarded unchanged.
    :type kwargs: dict

    :returns: Calculated curves mapping passband -> array.
    :rtype: dict[str, numpy.ndarray]
    """
    fn_args = (single, initial_system, crv_labels, curve_fn)
    return manage_observations(
        fn=c_managed.produce_curves_with_pulsations_mp,
        fn_args=fn_args,
        position=phases,
        **kwargs,
    )
