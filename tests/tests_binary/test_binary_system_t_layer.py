from __future__ import annotations

from typing import TYPE_CHECKING

# The tests package selects Astropy constants before test-module imports.
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from elisa.analytics.binary_fit.failure_policy import SYNTHESIS_FAILURES
from elisa.base.error import ElisaParametersError
from elisa.binary_system import t_layer
from tests import set_astropy_units
from tests.utils import ElisaTestCase

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


set_astropy_units()


class SupportMethodTestCase(ElisaTestCase):
    @staticmethod
    def test_jd_to_phase() -> None:
        p, t0 = 5.0, 3.0
        jd = np.array([1, 2, 3.1, 4, 5, 6, 7, 8, 9, 10, 11])
        expected = np.round([0.6, 0.8, 0.02, 0.2, 0.4, 0.6, 0.8, 0.0, 0.2, 0.4, 0.6], 5)
        obtained = np.round(t_layer.jd_to_phase(t0, p, jd), 5)
        assert_array_equal(expected, obtained)

    @staticmethod
    def test_phase_to_jd() -> None:
        p, t0 = 8.5, 3.0
        jd = np.array([3.1, 4, 5, 6, 7, 8, 9, 10, 11])
        phases = t_layer.jd_to_phase(t0, p, jd)
        obtained = np.round(t_layer.phase_to_jd(t0, p, phases), 5)
        assert_array_equal(obtained, jd)


@pytest.mark.parametrize("t0", [0.0, -1.0])
@pytest.mark.parametrize("conversion", [t_layer.jd_to_phase, t_layer.phase_to_jd])
def test_invalid_reference_time_is_a_penalizable_parameter_error(
    t0: float,
    conversion: Callable[..., float | NDArray[np.float64] | None],
) -> None:
    with pytest.raises(ElisaParametersError, match="Primary minimum time") as caught:
        conversion(t0, 1.0, np.array([0.1, 0.2]))
    assert isinstance(caught.value, SYNTHESIS_FAILURES)
