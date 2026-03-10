from __future__ import annotations

from typing import TYPE_CHECKING, Any

from elisa import umpy as up
from elisa.base.curves import rv_point
from elisa.single_system.curves import c_router

# TYPE_CHECKING block at the end of import header
if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.single_system.system import SingleSystem  # pragma: no cover
    from elisa.types import Float  # pragma: no cover


def com_radial_velocity(single: SingleSystem, **kwargs: Any) -> dict[str, NDArray[Float]]:
    """Calculate centre-of-mass radial velocity for a single system.

    Compute the radial velocity of the system centre-of-mass for every
    requested phase. The function expects a ``phases`` keyword argument in
    ``kwargs`` and forwards no other keys.

    :param single: Instance of the single-star system to process.
    :type single: elisa.single_system.system.SingleSystem
    :param kwargs: Keyword arguments. Expected key:
        * ``phases`` (numpy.ndarray) -- array of photometric phases for which
          the radial velocity should be computed.
    :type kwargs: dict

    :returns: Mapping with key ``'star'`` containing an array of radial velocities
              (same length as the provided phases).
    :rtype: dict[str, numpy.ndarray]

    """
    phases = kwargs.pop("phases")
    return {"star": single.gamma * up.ones(phases.shape[0])}


def compute_rv_curve_without_pulsations(single: SingleSystem, **kwargs: Any) -> dict[str, NDArray[Float]]:
    """Compute radial velocity curve for single systems without pulsations.

    Prepare an initial system container and delegate the actual curve
    production to the curve router. The function expects a ``phases`` key
    inside ``kwargs`` which will be popped and forwarded appropriately.

    :param single: The single-star system to compute radial velocities for.
    :type single: elisa.single_system.system.SingleSystem
    :param kwargs: Forwarded keyword arguments. Required key: ``phases``
        (numpy.ndarray).
    :type kwargs: dict

    :returns: Per-component radial velocity arrays.
    :rtype: dict[str, numpy.ndarray]

    """
    initial_system = c_router.prep_initial_system(single)
    rv_labels = ["star"]
    args = (single, initial_system, kwargs.pop("phases"), rv_point.compute_rv_at_pos, rv_labels)
    return c_router.produce_curves_wo_pulsations(*args, **kwargs)


def compute_rv_curve_with_pulsations(single: SingleSystem, **kwargs: Any) -> dict[str, NDArray[Float]]:
    """Compute radial velocity curve for single systems with pulsations.

    Prepare an initial system without pulsations (so the router can add
    pulsations during production) and delegate to the router's
    pulsation-capable producer. The function keeps ``kwargs`` intact except
    for the required ``phases`` key which is popped and forwarded.

    :param single: The single-star system to compute radial velocities for.
    :type single: elisa.single_system.system.SingleSystem
    :param kwargs: Forwarded keyword arguments. Required key: ``phases``
        (numpy.ndarray).
    :type kwargs: dict

    :returns: Per-component radial velocity arrays.
    :rtype: dict[str, numpy.ndarray]

    """
    initial_system = c_router.prep_initial_system(single, build_pulsations=False)
    rv_labels = ["star"]
    args = (single, initial_system, kwargs.pop("phases"), rv_point.compute_rv_at_pos, rv_labels)
    return c_router.produce_curves_with_pulsations(*args, **kwargs)
