"""Production fitting setup, normalization and atmosphere propagation coverage."""

from copy import deepcopy

import numpy as np
import pytest

from elisa.analytics.binary_fit import least_squares, mcmc
from elisa.analytics.params import parameters
from elisa.binary_system.system import BinarySystem
from tests.tests_analytics.fitting_helpers import V, lc_parameters, make_lc


@pytest.fixture(params=[least_squares.DetachedLightCurveFit, mcmc.DetachedLightCurveFit])
def solver(request: pytest.FixtureRequest) -> object:
    return request.param()


def initialize(solver: object, cfg: dict, atmosphere: dict | None) -> parameters.BinaryInitialParameters:
    initial = parameters.BinaryInitialParameters(**cfg)
    solver.atmosphere_model = atmosphere
    solver.limb_darkening_coefficients = None
    solver.set_up(
        initial, {V: make_lc(errors=[0.1] * 3)}, passband=[V],
        discretization=10.0, interp_treshold=100, samples="uniform",
        observer_system_cls=BinarySystem,
    )
    return initial


@pytest.mark.parametrize(
    ("atmosphere", "expected"),
    [
        (None, [(-2.5, 0.5), (-2.5, 0.5)]),
    ],
)
def test_component_atmosphere_bounds_in_solver(solver: object, atmosphere: dict | None, expected: list) -> None:
    cfg, _ = lc_parameters()
    for component in ("primary", "secondary"):
        cfg[component]["metallicity"] = {"value": 0.0, "fixed": False}
        cfg[component]["t_eff"] = {"value": 6000.0, "fixed": False}
    original = deepcopy(cfg)
    initial = initialize(solver, cfg, atmosphere)
    assert cfg == original
    assert solver.atmosphere_model == atmosphere
    assert solver.normalization == initial.get_normalization_map()
    labels = solver.fitable.keys()
    normalized = parameters.vector_normalizer(solver.initial_vector, labels, solver.normalization)
    np.testing.assert_allclose(
        parameters.vector_renormalizer(normalized, labels, solver.normalization), solver.initial_vector,
    )
    for component, bounds in zip(("primary", "secondary"), expected, strict=True):
        assert solver.normalization[f"{component}@t_eff"] == (3500.0, 50000.0)
        assert solver.normalization[f"{component}@metallicity"] == bounds


def test_initialized_explicit_limits_are_not_rewritten(solver: object) -> None:
    cfg, _ = lc_parameters()
    cfg["primary"]["metallicity"] = {"value": 0.0, "fixed": False, "min": -2.5, "max": 0.5}
    cfg["secondary"]["metallicity"] = {"value": 0.0, "fixed": False, "min": -8.0, "max": 2.0}
    original = deepcopy(cfg)
    initial = initialize(solver, cfg, {"primary": "k93", "secondary": "bb"})
    assert cfg == original
    assert solver.normalization == initial.get_normalization_map()
    assert solver.normalization["primary@metallicity"] == (-2.5, 0.5)
    assert solver.normalization["secondary@metallicity"] == (-8.0, 2.0)
    assert "secondary@t_eff" in solver.constrained
    assert "primary@surface_potential" in solver.fixed
    assert not any(key.endswith("@atmosphere") for key in initial.serialize_flat_set())


@pytest.mark.parametrize(
    "atmosphere",
    [None, {"primary": "k93", "secondary": "ck04"}, {"primary": "bb", "secondary": "k93"}, {"primary": "k93"}],
)
def test_requested_atmospheres_reach_physical_stars(solver: object, atmosphere: dict | None) -> None:
    from elisa.analytics.models import lc

    cfg, flat = lc_parameters()
    initialize(solver, cfg, atmosphere)
    captured = {}

    def synthesize(_phases: object, discretization: float, _observer: object, **kwargs: object) -> dict:
        binary = lc.prepare_binary(discretization=discretization, **kwargs)
        captured.update({component: getattr(binary, component).atmosphere for component in ("primary", "secondary")})
        return {V: np.ones(len(_phases))}

    solver.predict_light_curves(flat, synthesis=synthesize)
    assert captured == {component: (atmosphere or {}).get(component, "ck04") for component in ("primary", "secondary")}


def test_rv_setup_needs_no_atmosphere() -> None:
    from elisa.binary_system.curves.community import RadialVelocitySystem
    from tests.tests_analytics.fitting_helpers import rv_problem

    cfg, data = rv_problem()
    initial = parameters.BinaryInitialParameters(**cfg)
    for cls in (least_squares.CentralRadialVelocity, mcmc.CentralRadialVelocity):
        fit = cls()
        fit.set_up(initial, data, observer_system_cls=RadialVelocitySystem)
        assert fit.atmosphere_model is None
        assert fit.normalization == initial.get_normalization_map()
