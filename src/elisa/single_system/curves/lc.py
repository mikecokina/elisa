from __future__ import annotations

from typing import TYPE_CHECKING, Any

from elisa.logger import getLogger
from elisa.single_system.curves import c_router, lc_point

logger = getLogger("single_system.curves.lc")

# TYPE_CHECKING block at the end of import header
if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.single_system.system import SingleSystem  # pragma: no cover
    from elisa.types import Float  # pragma: no cover


def compute_light_curve_without_pulsations(single: SingleSystem, **kwargs: Any) -> dict[str, NDArray[Float]]:
    """Compute light curve for single star objects without pulsations.

    Prepare an initial system container and produce light curve arrays for the
    requested passbands. The function expects a ``passband`` mapping and a
    ``phases`` array inside ``kwargs``.

    :param single: Single star system instance.
    :type single: elisa.single_system.system.SingleSystem
    :param kwargs: Forwarded keyword arguments. Expected keys:

        - ``passband`` (dict[str, elisa.observer.PassbandContainer]) -- mapping
          of passband name to container
        - ``phases`` (numpy.ndarray) -- photometric phases array
        - other keys are forwarded unchanged to the curve producer

    :type kwargs: dict

    :returns: Mapping from passband name to light curve numpy arrays.
    :rtype: dict[str, numpy.ndarray]
    """
    initial_system = c_router.prep_initial_system(single)

    lc_labels = list(kwargs["passband"].keys())
    phases = kwargs.pop("phases")

    args = single, initial_system, phases, lc_point.compute_lc_on_pos, lc_labels
    return c_router.produce_curves_wo_pulsations(*args, **kwargs)


def compute_light_curve_with_pulsations(single: SingleSystem, **kwargs: Any) -> dict[str, NDArray[Float]]:
    """Compute light curve for single star objects with pulsations.

    Prepare an initial system container without prebuilt pulsations and
    produce light curves while allowing the router to add pulsations. The
    function expects a ``passband`` mapping and a ``phases`` array inside
    ``kwargs``.

    :param single: Single star system instance.
    :type single: elisa.single_system.system.SingleSystem
    :param kwargs: Forwarded keyword arguments. Expected keys:

        - ``passband`` (dict[str, elisa.observer.PassbandContainer]) -- mapping
          of passband name to container
        - ``phases`` (numpy.ndarray) -- photometric phases array
        - other keys are forwarded unchanged to the curve producer

    :type kwargs: dict

    :returns: Mapping from passband name to light curve numpy arrays.
    :rtype: dict[str, numpy.ndarray]
    """
    initial_system = c_router.prep_initial_system(single, build_pulsations=False)

    lc_labels = list(kwargs["passband"].keys())
    phases = kwargs.pop("phases")

    args = single, initial_system, phases, lc_point.compute_lc_on_pos, lc_labels
    return c_router.produce_curves_with_pulsations(*args, **kwargs)
