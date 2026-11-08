"""One-time LC preflight orchestration and real single-point synthesis."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from elisa import settings
from elisa.analytics.binary_fit import initial_ensemble as init
from elisa.analytics.binary_fit.failure_policy import SYNTHESIS_FAILURES
from elisa.analytics.binary_fit.mcmc import CentralRadialVelocity, DetachedLightCurveFit, OvercontactLightCurveFit
from elisa.analytics.params import parameters as p
from elisa.base.error import (
    ElisaLimbDarkeningError,
    ElisaMonteCarloError,
    ElisaNumericError,
    ElisaNumericSolverError,
    ElisaSettingsError,
    MorphologyError,
)
from elisa.binary_system.curves import c_router
from elisa.binary_system.system import BinarySystem
from elisa.observer.observer import Observer
from tests.tests_analytics.fitting_helpers import B, V, lc_parameters, lc_setup, make_lc


def minimal_lc_fit() -> DetachedLightCurveFit:
    """Supply the real fitter type for retry tests with mocked physical validation."""
    fit = DetachedLightCurveFit()
    fit.y_data = {V: np.empty(0)}
    return fit


def test_valid_ensemble_preserves_x0_without_draws() -> None:
    ensemble = np.array([[0.5], [0.3], [0.8]])
    rng = Mock()
    with patch.object(init, "_validate_candidate") as validate:
        result = init.initialize_lc_ensemble(minimal_lc_fit(), ensemble, supplied=False, rng=rng)
    assert_array_equal(result, ensemble)
    assert validate.call_count == 3
    rng.uniform.assert_not_called()


@pytest.mark.parametrize("failure", [MorphologyError("geometry"), ElisaNumericError("synthesis")])
def test_invalid_x0_replaced_with_seeded_uniform_draw(failure: Exception) -> None:
    ensemble = np.array([[0.5], [0.8]])
    expected = np.random.default_rng(21).uniform(0, 1, 1)
    with patch.object(init, "_validate_candidate", side_effect=[failure, None, None]) as validate:
        result = init.initialize_lc_ensemble(
            minimal_lc_fit(),
            ensemble,
            supplied=False,
            rng=np.random.default_rng(21),
        )
    assert_allclose(result[0], expected)
    assert_array_equal(result[1], ensemble[1])
    assert result.shape == ensemble.shape
    assert validate.call_count == 3
    assert_array_equal(ensemble, [[0.5], [0.8]])


@pytest.mark.parametrize("valid", [True, False])
def test_explicit_ensemble_never_repaired(valid: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "MCMC_INITIAL_WALKER_MAX_ATTEMPTS", 1)
    ensemble = np.array([[0.3], [0.6], [0.9]])
    rng = Mock()
    outcomes = [None, None, None] if valid else [MorphologyError("bad"), None, ElisaNumericError("flux")]
    with patch.object(init, "_validate_candidate", side_effect=outcomes) as validate:
        if valid:
            assert_array_equal(
                init.initialize_lc_ensemble(minimal_lc_fit(), ensemble, supplied=True, rng=rng),
                ensemble,
            )
        else:
            with pytest.raises(ElisaMonteCarloError, match=r"\[0, 2\].*attempted=3, accepted=1, rejected=2"):
                init.initialize_lc_ensemble(minimal_lc_fit(), ensemble, supplied=True, rng=rng)
    assert validate.call_count == 3
    rng.uniform.assert_not_called()
    assert_array_equal(ensemble[:, 0], [0.3, 0.6, 0.9])


def test_retry_exhaustion_reports_counts() -> None:
    with (
        patch.object(init, "_validate_candidate", side_effect=MorphologyError("Roche geometry")),
        pytest.raises(ElisaMonteCarloError, match=r"attempted=5, accepted=0, rejected=5.*Roche geometry"),
    ):
        init.initialize_lc_ensemble(
            minimal_lc_fit(),
            np.array([[0.5], [0.6]]),
            supplied=False,
            max_attempts=5,
        )


@pytest.mark.parametrize("error", [RuntimeError("bug"), TypeError("bug"), FileNotFoundError("table")])
def test_unexpected_errors_escape_even_emergency_policy(error: Exception, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "PENALIZE_ALL_EXCEPTIONS", True)
    with patch.object(init, "_validate_candidate", side_effect=error), pytest.raises(type(error), match=str(error)):
        init.initialize_lc_ensemble(minimal_lc_fit(), np.array([[0.5]]), supplied=False)


@pytest.mark.parametrize(
    "cause", [FileNotFoundError("table"), ElisaSettingsError("config"), TypeError("bug"), AttributeError("bug")],
)
def test_wrapped_missing_resource_propagates(cause: Exception) -> None:
    error = ElisaLimbDarkeningError("missing table")
    error.__cause__ = cause
    with (
        patch.object(init, "_validate_candidate", side_effect=error),
        pytest.raises(ElisaLimbDarkeningError, match="missing table"),
    ):
        init.initialize_lc_ensemble(minimal_lc_fit(), np.array([[0.5]]), supplied=False)


@pytest.mark.parametrize(("eccentricity", "expected"), [(0.3, 0.62), (0.0, 0.12)])
def test_phase_and_actual_model_forwarding(eccentricity: float, expected: float) -> None:
    fit = Mock(MORPHOLOGY="detached", discretization=8, y_data={V: [], B: []})
    fit._prepare_likelihood_parameters.return_value = (None, {"physical": "with constraints"})
    binary = SimpleNamespace(
        morphology="detached",
        eccentricity=eccentricity,
        phase_shift=0.12,
        orbit=SimpleNamespace(periastron_phase=0.5),
    )
    observer = Mock()
    observer.observe.lc.return_value = ([], {V: [1.0], B: [2.0]})
    with (
        patch.object(init, "prepare_binary", return_value=binary) as prepare,
        patch.object(init.parameters, "extend_json_with_atm_params", return_value={"actual": 1}) as extend,
    ):
        init._validate_candidate(fit, observer, np.array([0.5]))
    extend.assert_called_once_with(
        {"physical": "with constraints"},
        atmosphere_model=fit.atmosphere_model,
        limb_darkening_coefficients=fit.limb_darkening_coefficients,
    )
    prepare.assert_called_once_with(discretization=8, actual=1)
    call = observer.observe.lc.call_args.kwargs
    assert_allclose(call["phases"], [expected])
    assert call["normalize"] is False
    assert call["flux_unit"].is_equivalent("W / m2")


@pytest.mark.parametrize("bad", [np.nan, np.inf, 0.0, -1.0, ElisaNumericError("surface synthesis")])
def test_bad_flux_in_any_band_replaced(bad: float | Exception) -> None:
    fit = Mock(MORPHOLOGY="detached", y_data={V: [], B: []}, atmosphere_model=None, limb_darkening_coefficients=None)
    fit._prepare_likelihood_parameters.return_value = (None, {})
    binary = SimpleNamespace(morphology="detached", eccentricity=0, phase_shift=0)
    observer = Mock()
    observer.observe.lc.side_effect = [
        bad if isinstance(bad, Exception) else ([], {V: [1.0], B: [bad]}),
        ([], {V: [1.0], B: [2.0]}),
    ]
    with (
        patch.object(init, "Observer", return_value=observer) as observer_cls,
        patch.object(init, "prepare_binary", return_value=binary),
    ):
        init.initialize_lc_ensemble(fit, np.array([[0.5]]), supplied=False, max_attempts=2)
    assert observer.observe.lc.call_count == 2
    observer_cls.assert_called_once_with(passband=[V, B])


def test_morphology_mismatch_skips_synthesis() -> None:
    fit = Mock(MORPHOLOGY="detached", atmosphere_model=None, limb_darkening_coefficients=None)
    fit._prepare_likelihood_parameters.return_value = (None, {})
    observer = Mock()
    with (
        patch.object(init, "prepare_binary", return_value=SimpleNamespace(morphology="over-contact")),
        pytest.raises(MorphologyError, match="Expected detached"),
    ):
        init._validate_candidate(fit, observer, np.array([0.5]))
    observer.observe.lc.assert_not_called()


def test_rv_hook_unchanged() -> None:
    ensemble = np.array([[0.5], [0.8]])
    with patch.object(init, "initialize_lc_ensemble") as validate:
        assert CentralRadialVelocity()._initialize_ensemble(ensemble, supplied=False) is ensemble
    validate.assert_not_called()


def test_hook_runs_once_before_sampler_and_never_during_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    fit = DetachedLightCurveFit()
    lc_setup(fit, {V: make_lc(errors=np.ones(3) * 0.01)})
    monkeypatch.setattr(settings, "NUMBER_OF_MCMC_PROCESSES", 1)
    with (
        patch.object(init, "_validate_candidate") as validate,
        patch.object(fit, "likelihood", return_value=-0.5) as likelihood,
    ):
        sampler = fit._fit(2, 1, 3, 1, np.array([[0.4], [0.6]]))
    assert sampler.get_chain().shape == (3, 2, 1)
    assert validate.call_count == 2
    assert likelihood.call_count > 2


@pytest.mark.usefixtures("science_settings")
@pytest.mark.requires_ld(passbands=(V, B, "bolometric"), laws=("linear",))
@pytest.mark.parametrize(
    ("contact", "spotted", "eccentric"),
    [(False, False, True), (False, True, False), (True, True, False), (False, True, True)],
)
def test_real_physical_preflight(contact: bool, spotted: bool, eccentric: bool) -> None:
    cfg, _ = lc_parameters()
    if eccentric:
        cfg["system"]["eccentricity"]["value"] = 0.2
        cfg["system"]["argument_of_periastron"]["value"] = 33.0
        cfg["system"]["phase_shift"] = {"value": 0.13, "fixed": True}
    if contact:
        physical = {g: {k: v["value"] for k, v in group.items() if "value" in v} for g, group in cfg.items()}
        del physical["nuisance"]
        physical["secondary"]["t_eff"] = 5000.0
        for component in ("primary", "secondary"):
            physical[component]["synchronicity"] = 1.0
        _, inner, outer = BinarySystem.from_json(physical).libration_potentials()
        cfg["primary"]["surface_potential"]["value"] = (inner + outer) / 2
        cfg["secondary"]["surface_potential"] = {"value": (inner + outer) / 2, "fixed": True}
    if spotted:
        cfg["primary"]["spots"] = [
            {
                "label": "cool",
                "longitude": {"value": 60.0, "fixed": True},
                "latitude": {"value": 70.0, "fixed": True},
                "angular_radius": {"value": 15.0, "fixed": True},
                "temperature_factor": {"value": 0.9, "fixed": True},
            },
        ]
    fit = OvercontactLightCurveFit() if contact else DetachedLightCurveFit()
    fit.atmosphere_model = {"primary": "bb", "secondary": "bb"}
    fit.limb_darkening_coefficients = {c: {b: [0.4] for b in (V, B, "bolometric")} for c in ("primary", "secondary")}
    data = {band: make_lc(errors=np.ones(3) * 0.01) for band in (V, B)}
    fit.set_up(
        p.BinaryInitialParameters(**cfg),
        data,
        passband=list(data),
        discretization=10,
        interp_treshold=100,
        samples="uniform",
        observer_system_cls=BinarySystem,
    )
    fit.prepare_priors()
    original_system, original_fluxes = fit.observer._system, fit.observer.fluxes
    ensemble = np.tile(fit.norm_init_vector, (2, 1))
    with patch.object(init, "prepare_binary", wraps=init.prepare_binary) as prepare:
        result = init.initialize_lc_ensemble(fit, ensemble, supplied=True)
    assert_array_equal(result, ensemble)
    assert prepare.call_count == 2
    assert fit.observer._system is original_system
    assert fit.observer.fluxes is original_fluxes
    if eccentric:
        # Independent real orbit check, using the same physical preparation.
        _, kwargs = fit._prepare_likelihood_parameters(ensemble[0])
        binary = init.prepare_binary(discretization=10, **kwargs)
        position = binary.orbit.orbital_motion(np.array([binary.orbit.periastron_phase]))
        assert_allclose(position[0, 0], binary.orbit.periastron_distance)


@pytest.mark.parametrize("failure_type", SYNTHESIS_FAILURES)
def test_domain_failure_retries_without_reclassifying_cause(failure_type: type[Exception]) -> None:
    error = failure_type("candidate synthesis failed")
    error.__cause__ = ElisaNumericSolverError("solver did not converge")
    with patch.object(init, "_validate_candidate", side_effect=[error, None]) as validate:
        result = init.initialize_lc_ensemble(
            minimal_lc_fit(),
            np.array([[0.5]]),
            supplied=False,
            rng=np.random.default_rng(21),
            max_attempts=2,
        )
    assert result.shape == (1, 1)
    assert validate.call_count == 2


def test_each_walker_gets_its_own_configured_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "MCMC_INITIAL_WALKER_MAX_ATTEMPTS", 2)
    error = MorphologyError("geometry")
    ensemble = np.array([[0.3], [0.6], [0.9]])
    rng = Mock()
    rng.uniform.side_effect = [np.array([0.1]), np.array([0.2]), np.array([0.4])]
    with patch.object(init, "_validate_candidate", side_effect=[error, None] * 3) as validate:
        result = init.initialize_lc_ensemble(minimal_lc_fit(), ensemble, supplied=False, rng=rng)
    assert_array_equal(result[:, 0], [0.1, 0.2, 0.4])
    assert validate.call_count == 6
    assert rng.uniform.call_count == 3
    assert_array_equal(ensemble[:, 0], [0.3, 0.6, 0.9])


def test_walker_cannot_borrow_unused_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "MCMC_INITIAL_WALKER_MAX_ATTEMPTS", 2)
    rng = Mock()
    rng.uniform.return_value = np.array([0.2])
    with (
        patch.object(
            init, "_validate_candidate", side_effect=[None, MorphologyError("geometry"), ElisaNumericError("flux")],
        ) as validate,
        pytest.raises(
            ElisaMonteCarloError, match=r"walker 1 .*after 2 attempts.*MorphologyError.*ElisaNumericError",
        ) as caught,
    ):
        init.initialize_lc_ensemble(
            minimal_lc_fit(), np.array([[0.3], [0.6], [0.9]]), supplied=False, rng=rng,
        )
    assert validate.call_count == 3
    rng.uniform.assert_called_once_with(0.0, 1.0, 1)
    assert "attempted=3, accepted=1, rejected=2" in str(caught.value)
    assert "flux" in str(caught.value)


def test_one_attempt_limit_counts_original_and_never_draws(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "MCMC_INITIAL_WALKER_MAX_ATTEMPTS", 1)
    rng = Mock()
    with (
        patch.object(init, "_validate_candidate", side_effect=MorphologyError("geometry")) as validate,
        pytest.raises(ElisaMonteCarloError, match=r"walker 0 .*after 1 attempts"),
    ):
        init.initialize_lc_ensemble(minimal_lc_fit(), np.array([[0.3]]), supplied=False, rng=rng)
    validate.assert_called_once()
    rng.uniform.assert_not_called()


@pytest.mark.usefixtures("science_settings")
@pytest.mark.requires_ld(passbands=(V, B, "bolometric"), laws=("linear",))
def test_direct_single_phase_eccentric_lc_uses_exact_router_path() -> None:
    _, flat = lc_parameters()
    flat.update({"system@eccentricity": 0.2, "system@argument_of_periastron": 33.0, "system@phase_shift": 0.13})
    flat = p.extend_json_with_atm_params(
        flat,
        atmosphere_model=dict.fromkeys(("primary", "secondary"), "bb"),
        limb_darkening_coefficients={c: {b: [0.4] for b in (V, B, "bolometric")} for c in ("primary", "secondary")},
    )
    binary = init.prepare_binary(discretization=10, **flat)
    observer = Observer(passband=[V, B], system=binary)
    phase = binary.orbit.periastron_phase + binary.phase_shift
    with (
        patch.object(c_router, "produce_ecc_curves_no_spots", wraps=c_router.produce_ecc_curves_no_spots) as route,
        patch.object(
            c_router.c_appx_router,
            "resolve_ecc_approximation_method",
            wraps=c_router.c_appx_router.resolve_ecc_approximation_method,
        ) as resolve,
    ):
        phases, fluxes = observer.observe.lc(phases=np.array([phase]), normalize=False)
    route.assert_called_once()
    assert resolve.call_args.kwargs["try_to_find_appx"] is False
    assert_allclose(phases, [phase])
    assert set(fluxes) == {V, B}
    for flux in fluxes.values():
        assert flux.shape == (1,)
        assert np.all(np.isfinite(flux) & (flux > 0))
