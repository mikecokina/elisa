"""Surface builders distinguish missing geometry contracts from bad candidates."""

import numpy as np
import pytest

from elisa.analytics.binary_fit.failure_policy import SYNTHESIS_FAILURES
from elisa.base.error import ElisaNumericError, ElisaUnexpectedError
from elisa.binary_system.surface.faces import build_faces
from elisa.binary_system.surface.gravity import build_surface_gravity
from elisa.binary_system.surface.validation import validate_component_distance


@pytest.mark.parametrize("builder", [build_faces, build_surface_gravity])
@pytest.mark.parametrize("distance", [np.nan, np.inf, -np.inf, 0.0, -1.0, 10**400])
def test_bad_numeric_distances_reject_candidate_before_surface_work(builder, distance):
    with pytest.raises(ElisaNumericError) as caught:
        builder(object(), distance)
    assert isinstance(caught.value, SYNTHESIS_FAILURES)


@pytest.mark.parametrize("builder", [build_faces, build_surface_gravity])
@pytest.mark.parametrize("distance", [None, [], [1.0], np.array(1.0), "1.0", True, np.bool_(True), 1 + 0j])
def test_malformed_distance_is_a_fatal_contract_before_surface_work(builder, distance):
    with pytest.raises(ElisaUnexpectedError) as caught:
        builder(object(), distance)
    assert not isinstance(caught.value, SYNTHESIS_FAILURES)


@pytest.mark.parametrize("distance", [1, 1.0, np.float64(0.1), np.float32(0.1), np.int64(2), np.nextafter(0.0, 1.0)])
def test_finite_positive_scalar_distance_is_accepted(distance):
    assert validate_component_distance(distance) is None


def test_empty_component_selection_preserves_noop_behavior():
    system = object()
    assert build_faces(system, None, component=None) is system
    assert build_surface_gravity(system, None, component=None) is None
