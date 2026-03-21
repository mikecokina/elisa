from __future__ import annotations

from typing import TYPE_CHECKING, Any

from elisa.logger import getLogger

logger = getLogger(__name__)


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.single_system.system import SingleSystem
    from elisa.types import Float


def compute_general_lightcurve(system: SingleSystem, **kwargs: Any) -> dict[str, NDArray[Float]]:
    r"""Compute a general light curve for a single-star system.

    This is a placeholder entry point used by higher-level routing
    code. Concrete light-curve generators for specific scenarios (with
    or without pulsations, different achromatic/passband options) should
    be used instead. The function currently raises
    :class:`NotImplementedError`.

    :param system: Single-star system instance to evaluate.
    :type system: elisa.single_system.system.SingleSystem
    :param kwargs: Additional keyword arguments forwarded to generator
        implementations (passbands, phases, etc.).
    :type kwargs: Any
    :returns: Mapping of passband labels to light-curve arrays.
    :rtype: dict[str, numpy.typing.NDArray[elisa.types.Float]]
    :raises NotImplementedError: Always raised for the generic stub.
    """
    logger.debug(
        "compute_general_lightcurve called for system %s with kwargs: %s",
        getattr(system, "name", None),
        kwargs,
    )
    msg = "Generic light-curve generator is not implemented; use concrete generators from curves submodule."
    raise NotImplementedError(msg)
