"""Real proposal construction and synthesis through LC/RV fitting boundaries."""

from __future__ import annotations

import multiprocessing
from typing import TYPE_CHECKING

import numpy as np
import pytest

from elisa import const, settings
from elisa import units as u
from elisa.analytics.binary_fit import least_squares, mcmc
from elisa.analytics.models import lc
from elisa.analytics.params import parameters
from elisa.base.error import ElisaParametersError
from elisa.binary_system.system import BinarySystem
from tests.tests_analytics.fitting_helpers import V, lc_parameters, make_lc, normalized

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

    import elisa.analytics.binary_fit.shared
    import elisa.base.error


@pytest.fixture(autouse=True)
def synthesis_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from elisa.analytics.models import rv

    original = rv.prepare_central_rv_binary

    def configured_binary(**kwargs) -> BinarySystem:
        binary = original(**kwargs)
        for component in (binary.primary, binary.secondary):
            component.atmosphere = "bb"
            component.limb_darkening_coefficients = {"rv_band": [0.1], "bolometric": [0.1]}
            component.discretization_factor = np.radians(10.0)
        return binary

    monkeypatch.setattr(rv, "prepare_central_rv_binary", configured_binary)
    previous = settings.settings_serializer()
    settings.configure(
        ATM_ATLAS="bb",
        LIMB_DARKENING_LAW="linear",
        NUMBER_OF_PROCESSES=1,
        REFLECTION_EFFECT=True,
        MESH_GENERATOR="auto",
        RV_METHOD="kinematic",
    )
    yield
    settings.configure(**previous)


def make_fit(
    method: str,
    phenomenon: str,
    invalid_case: str = "axis",
) -> (
    tuple[least_squares.DetachedLightCurveFit, NDArray[np.float64], NDArray[np.float64]]
    | tuple[mcmc.DetachedLightCurveFit, NDArray[np.float64], NDArray[np.float64]]
):
    cfg, _ = lc_parameters()
    if phenomenon == "pulsation":
        values = {
            "l": 1,
            "m": 1,
            "frequency": 16.0,
            "amplitude": 50.0,
            "start_phase": 0.0,
            "mode_axis_phi": 0.0,
            "mode_axis_theta": 30.0,
        }
        metadata = {key: {"value": value, "fixed": True} for key, value in values.items()}
        metadata["mode_axis_theta"] = {"value": 30.0, "fixed": False, "min": 0.0, "max": 200.0}
        if invalid_case == "fractional_l":
            metadata["l"] = {"constraint": "system@inclination - 84"}
        elif invalid_case == "m":
            metadata["m"] = {"constraint": "system@inclination - 84"}
        elif invalid_case in ("amplitude", "frequency"):
            metadata[invalid_case] = {
                "constraint": f"(system@inclination - 84) * {50 if invalid_case == 'amplitude' else 16}",
            }
        cfg["primary"]["pulsations"] = [dict(label="mode", **metadata)]
        proposal_key = "primary@pulsation@mode@mode_axis_theta"
        invalid = 190.0
        if invalid_case != "axis":
            proposal_key = "system@inclination"
            invalid = {"fractional_l": 85.5, "m": 86.0, "amplitude": 83.0, "frequency": 83.0}[invalid_case]
    else:
        metadata = {
            key: {"value": value, "fixed": True}
            for key, value in {
                "longitude": 90.0,
                "latitude": 60.0,
                "angular_radius": 25.0,
                "temperature_factor": 0.9,
            }.items()
        }
        metadata["angular_radius"] = {"value": 25.0, "fixed": False, "min": 0.0, "max": 40.0}
        cfg["primary"]["spots"] = [dict(label="spot", **metadata)]
        proposal_key, invalid = "primary@spot@spot@angular_radius", 0.0
    fit = (least_squares.DetachedLightCurveFit if method == "lsq" else mcmc.DetachedLightCurveFit)()
    fit.atmosphere_model = {"primary": "bb", "secondary": "bb"}
    fit.limb_darkening_coefficients = {c: {V: [0.1], "bolometric": [0.1]} for c in ("primary", "secondary")}
    fit.set_up(
        parameters.BinaryInitialParameters(**cfg),
        {V: make_lc([1.0, 0.9, 1.0], [0.01] * 3)},
        passband=[V],
        discretization=10.0,
        interp_treshold=100,
        samples="uniform",
        observer_system_cls=BinarySystem,
    )
    if method == "mcmc":
        fit.prepare_priors()
    xn = normalized(fit)
    bad = xn.copy()
    bad[list(fit.fitable).index(proposal_key)] = parameters.normalize_value(invalid, *fit.normalization[proposal_key])
    return fit, xn, bad


def evaluate(
    fit: least_squares.DetachedLightCurveFit | mcmc.DetachedLightCurveFit,
    xn: NDArray[np.float64],
) -> float | np.float64 | NDArray[np.float64]:
    if isinstance(fit, mcmc.MCMCFit):
        return fit.ln_probability(xn)
    return fit.model_to_fit(xn)


def assert_rejected(
    fit: least_squares.DetachedLightCurveFit | mcmc.DetachedLightCurveFit,
    value: float | NDArray[np.float64],
) -> None:
    if isinstance(fit, mcmc.MCMCFit):
        assert value == -np.inf
    else:
        count = sum(fit.num_of_points.values())
        np.testing.assert_array_equal(value, np.full(count, np.sqrt(const.MAX_USABLE_FLOAT / count)))
        assert value.shape == (count,)
        assert np.isfinite(value).all()


@pytest.mark.parametrize("method", ["lsq", "mcmc"])
@pytest.mark.parametrize("invalid_case", ["axis", "fractional_l", "m", "amplitude", "frequency"])
def test_real_pulsation_proposal_valid_invalid_valid(method: str, invalid_case: str) -> None:
    fit, xn, bad = make_fit(method, "pulsation", invalid_case)
    before = evaluate(fit, xn)
    assert np.isfinite(before).all()
    assert_rejected(fit, evaluate(fit, bad))
    np.testing.assert_array_equal(evaluate(fit, xn), before)


@pytest.mark.parametrize("method", ["lsq", "mcmc"])
def test_real_spot_mesh_proposal_valid_invalid_valid(method: str) -> None:
    fit, xn, bad = make_fit(method, "spot")
    before = evaluate(fit, xn)
    assert np.isfinite(before).all()
    assert_rejected(fit, evaluate(fit, bad))
    np.testing.assert_array_equal(evaluate(fit, xn), before)


@pytest.mark.parametrize("method", ["lsq", "mcmc"])
@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, TypeError, IndexError])
def test_unexpected_error_in_real_mode_constructor_terminates_fit(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    error_type: type[IndexError | RuntimeError | TypeError | ValueError],
) -> None:
    fit, xn, _ = make_fit(method, "pulsation")

    def broken(*_args, **_kwargs) -> None:
        raise error_type("mode implementation defect")

    monkeypatch.setattr("elisa.pulse.mode.PulsationMode.validate_mode", broken)
    with pytest.raises(error_type, match="mode implementation defect"):
        evaluate(fit, xn)


def _worker_evaluate(
    fit: elisa.analytics.binary_fit.shared.AbstractFit,
    xn: NDArray[np.float64],
) -> float | NDArray[np.float64]:
    return evaluate(fit, xn)


def _worker_recovery_sequence(
    fit: elisa.analytics.binary_fit.shared.AbstractFit,
    xn: NDArray[np.float64],
    bad: NDArray[np.float64],
) -> tuple[
    float | NDArray[np.float64],
    float | NDArray[np.float64],
    float | NDArray[np.float64],
]:
    """Evaluate all three proposals on this single deserialized fitting instance."""
    before = evaluate(fit, xn)
    rejected = evaluate(fit, bad)
    after = evaluate(fit, xn)
    return before, rejected, after


@pytest.mark.parametrize("method", ["lsq", "mcmc"])
def test_real_fitting_rejection_through_process_pool(method: str) -> None:
    """A worker transports rejection and recovers on the same deserialized fit."""
    fit, xn, bad = make_fit(method, "pulsation", "m")
    expected = evaluate(fit, xn)
    assert np.isfinite(expected).all()
    with multiprocessing.get_context("fork").Pool(1) as pool:
        before, rejected, after = pool.apply_async(_worker_recovery_sequence, (fit, xn, bad)).get(timeout=30)
    assert np.isfinite(before).all()
    assert_rejected(fit, rejected)
    np.testing.assert_array_equal(before, expected)
    np.testing.assert_array_equal(after, before)


@pytest.mark.parametrize(
    ("phenomenon", "error_type"),
    [("pulsation", ElisaParametersError), ("spot", ElisaParametersError)],
)
def test_direct_synthesis_keeps_meaningful_domain_error(
    phenomenon: str,
    error_type: type[elisa.base.error.ElisaParametersError],
) -> None:
    fit, _, bad = make_fit("lsq", phenomenon)
    kwargs = parameters.prepare_properties_set(
        parameters.vector_renormalizer(bad, fit.fitable, fit.normalization),
        fit.fitable,
        fit.constrained,
        fit.fixed,
    )
    with pytest.raises(error_type):
        lc.synthetic_binary(fit.x_data_reduced, fit.discretization, fit.observer, **kwargs)


def test_radiometric_rv_rejects_real_spot_mesh_failure() -> None:
    from elisa.analytics.models import rv
    from elisa.observer.observer import Observer

    fit, xn, bad = make_fit("lsq", "spot")

    def synthesize(vector: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
        kwargs = parameters.prepare_properties_set(
            parameters.vector_renormalizer(vector, fit.fitable, fit.normalization),
            fit.fitable,
            fit.constrained,
            fit.fixed,
        )
        # LC fixture uses q/a; direct radiometric RV takes physical component masses.
        binary = lc.prepare_binary(**kwargs)
        kwargs.pop("system@mass_ratio")
        kwargs.pop("system@semi_major_axis")
        kwargs.update({"primary@mass": binary.primary.mass * u.kg, "secondary@mass": binary.secondary.mass * u.kg})
        observer = Observer(system=None)
        curves = rv.central_rv_synthetic(fit.x_data_reduced, observer, **kwargs)
        for component in ("primary", "secondary"):
            star = getattr(observer._system, component)
            assert star.t_eff == kwargs[f"{component}@t_eff"]
            assert star.surface_potential == kwargs[f"{component}@surface_potential"]
            assert star.metallicity == kwargs[f"{component}@metallicity"]
        return curves

    settings.configure(RV_METHOD="radiometric", USE_SINGLE_LD_COEFFICIENTS=False)
    before = synthesize(xn)
    assert all(np.isfinite(values).all() for values in before.values())
    with pytest.raises(ElisaParametersError):
        synthesize(bad)
    after = synthesize(xn)
    for component in before:
        np.testing.assert_array_equal(after[component], before[component])


@pytest.mark.parametrize("method", ["lsq", "mcmc"])
def test_unexpected_constructor_error_crosses_fitting_process_boundary(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    fit, xn, _ = make_fit(method, "pulsation")

    def broken(*_args, **_kwargs) -> None:
        raise RuntimeError("remote mode implementation defect")

    monkeypatch.setattr("elisa.pulse.mode.PulsationMode.validate_mode", broken)
    with (
        multiprocessing.get_context("fork").Pool(1) as pool,
        pytest.raises(RuntimeError, match="remote mode implementation defect") as caught,
    ):
        pool.apply_async(_worker_evaluate, (fit, xn)).get(timeout=30)
    assert "broken" in str(caught.value.__cause__)
