from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from elisa.single_system import utils as ssutils
from elisa.single_system.curves import utils as crv_utils

# TYPE_CHECKING block at the end of import header
if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.types import Float  # pragma: no cover


def produce_curves_wo_pulsations_mp(*args: Any) -> dict[str, NDArray[Float]]:
    """Calculate curve for single systems without pulsations (parallel worker).

    The worker expects a single tuple of positional arguments where the last
    element is a keyword-args mapping. The expected structure of ``args`` is
    documented below.

    :param args: Tuple containing the worker inputs
    :type args: tuple

    :**args options**:
        * **single** (elisa.single_system.system.SingleSystem) - system instance
        * **initial_system** (elisa.single_system.container.SinglePositionContainer) - template container
        * **phase_batch** (numpy.ndarray) - phases for this worker
        * **crv_labels** (list[str]) - curve labels / passbands
        * **curves_fn** (callable) - function to compute point values on position
        * **kwargs** (dict) - mapping of additional keyword options; must contain
          ``position_method`` callable used to obtain positions for the provided phases

    :returns: Mapping from curve label to numpy array of calculated values
    :rtype: dict[str, numpy.ndarray]
    """
    _single, initial_system, phase_batch, crv_labels, curves_fn, kwargs = args
    position_method = kwargs.pop("position_method")

    rotational_motion = position_method(input_argument=phase_batch, return_nparray=False, calculate_from="phase")
    curves: dict[str, np.ndarray] = {key: np.zeros(phase_batch.shape) for key in crv_labels}

    for pos_idx, position in enumerate(rotational_motion):
        on_pos = ssutils.move_sys_onpos(initial_system, position)
        star = on_pos.star

        star.coverage = star.areas

        curves = curves_fn(curves, pos_idx, crv_labels, on_pos)

    return curves


def produce_curves_with_pulsations_mp(*args: Any) -> dict[str, NDArray[Float]]:
    """Calculate curve for single systems with pulsations (parallel worker).

    The worker expects the same ``args`` structure as
    :func:`produce_curves_wo_pulsations_mp`.

    :param args: Tuple containing the worker inputs
    :type args: tuple

    :returns: Mapping from curve label to numpy array of calculated values
    :rtype: dict[str, numpy.ndarray]
    """
    _single, initial_system, phase_batch, crv_labels, curves_fn, kwargs = args
    position_method = kwargs.pop("position_method")

    rotational_motion = position_method(input_argument=phase_batch, return_nparray=False, calculate_from="phase")
    curves: dict[str, np.ndarray] = {key: np.zeros(phase_batch.shape) for key in crv_labels}

    for pos_idx, position in enumerate(rotational_motion):
        sys_on_pos = initial_system.copy()
        sys_on_pos.set_on_position_params(position)
        sys_on_pos.set_time()

        sys_on_pos.build_pulsations()
        crv_utils.prep_surface_params(sys_on_pos, return_values=False, write_to_containers=True, **kwargs)

        sys_on_pos = ssutils.move_sys_onpos(sys_on_pos, position, on_copy=False)
        star = sys_on_pos.star
        star.coverage = star.areas

        curves = curves_fn(curves, pos_idx, crv_labels, sys_on_pos)

    return curves
