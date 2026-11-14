"""Public operation arguments are distinct from model and numerical failures."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from elisa.analytics.binary_fit.failure_policy import SYNTHESIS_FAILURES
from elisa.base.error import (
    ElisaError,
    ElisaOperationInputError,
    raise_elisa_operation_input_error,
)
from elisa.binary_system.system import BinarySystem
from elisa.single_system.system import SingleSystem

if TYPE_CHECKING:
    import elisa.binary_system.system
    import elisa.single_system.system


def test_operation_input_helper_preserves_cause_and_is_not_a_penalty_category() -> None:
    cause = ValueError("original argument error")
    with pytest.raises(ElisaOperationInputError, match="invalid operation") as caught:
        raise_elisa_operation_input_error("invalid operation", cause)
    assert isinstance(caught.value, ElisaError)
    assert caught.value.__cause__ is cause
    assert not isinstance(caught.value, SYNTHESIS_FAILURES)


@pytest.mark.parametrize(
    ("system_cls", "position_method"),
    [(BinarySystem, "calculate_orbital_motion"), (SingleSystem, "calculate_lines_of_sight")],
)
def test_build_container_rejects_phase_and_time_before_building(
    system_cls: type[elisa.binary_system.system.BinarySystem | elisa.single_system.system.SingleSystem],
    position_method: str,
) -> None:
    system = object.__new__(system_cls)
    with (
        patch.object(system_cls, position_method) as positions,
        pytest.raises(ElisaOperationInputError, match=r"phase.*time"),
    ):
        system.build_container(phase=0.1, time=2450000.0)
    positions.assert_not_called()


def test_equipotential_rejects_invalid_plane_before_calling_solver() -> None:
    component = SimpleNamespace(synchronicity=1.0)
    system = SimpleNamespace(primary=component, secondary=component)
    with (
        patch("elisa.binary_system.system.fsolve") as solver,
        pytest.raises(ElisaOperationInputError, match="plane"),
    ):
        BinarySystem.compute_equipotential_boundary(system, 1.0, "invalid")
    solver.assert_not_called()
