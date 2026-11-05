"""Atlas choices are resolved at input boundaries, not in surface synthesis."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from elisa import Star, settings
from elisa.analytics.binary_fit.least_squares import CentralRadialVelocity
from elisa.analytics.binary_fit.mcmc import CentralRadialVelocity as MCMCRadialVelocity
from elisa.analytics.params.parameters import BinaryInitialParameters
from elisa.atm import atm
from elisa.base.container import StarContainer
from elisa.base.parameter_tracking import parameter_is_explicit
from elisa.atm.const import Atmosphere, atm_file_prefix
from elisa.base.error import ElisaInitialParametersError
from elisa.binary_system.curves.community import RadialVelocitySystem
from tests.tests_analytics.fitting_helpers import rv_problem


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [("ck04", "ck04"), ("castelli", "ck04"), ("castelli-kurucz", "ck04"), ("k93", "k93"), ("kurucz", "k93"),
     ("black_body", "bb")],
)
def test_star_resolves_atlas_on_construction_and_reinitialization(alias: str, canonical: str) -> None:
    star = Star(mass=1.0, t_eff=6000.0, atmosphere=alias)
    assert star.atmosphere == canonical
    star.atmosphere = alias
    star.init()
    assert star.atmosphere == canonical


def test_invalid_star_atlas_is_initial_parameter_error() -> None:
    with pytest.raises(ElisaInitialParametersError, match="Incorrect atlas") as failure:
        Star(mass=1.0, t_eff=6000.0, atmosphere="invalid")
    assert isinstance(failure.value.__cause__, KeyError)


def test_invalid_atlas_assignment_is_rejected_and_keeps_previous_value() -> None:
    star = Star(mass=1.0, t_eff=6000.0, atmosphere="k93")
    with pytest.raises(ElisaInitialParametersError, match="Incorrect atlas"):
        star.atmosphere = "invalid"
    assert star.atmosphere == Atmosphere.K93
    star.init()
    assert star.atmosphere == Atmosphere.K93


def test_star_defaults_to_ck04_and_settings_have_no_global_atlas() -> None:
    assert Star(mass=1.0, t_eff=6000.0).atmosphere == Atmosphere.CK04
    assert not hasattr(settings, "ATM_ATLAS")


@pytest.mark.parametrize("value", [None, "", "invalid"])
def test_invalid_atmosphere_values_raise_initial_parameter_error(value: object) -> None:
    with pytest.raises(ElisaInitialParametersError):
        Star(mass=1.0, t_eff=6000.0, atmosphere=value)
    star = Star(mass=1.0, t_eff=6000.0)
    with pytest.raises(ElisaInitialParametersError):
        star.atmosphere = value


@pytest.mark.parametrize("atlas", ["black_body", "invalid"])
def test_fit_override_is_resolved_before_candidates(atlas: str) -> None:
    config, data = rv_problem()
    fit = CentralRadialVelocity()
    override = {"primary": atlas}
    fit.atmosphere_model = override
    if atlas == "invalid":
        with pytest.raises(ElisaInitialParametersError, match="Incorrect atlas"):
            fit.set_up(BinaryInitialParameters(**config), data, observer_system_cls=RadialVelocitySystem)
    else:
        fit.set_up(BinaryInitialParameters(**config), data, observer_system_cls=RadialVelocitySystem)
        assert fit.atmosphere_model == {"primary": Atmosphere.BB}
        assert override == {"primary": atlas}


def test_internal_radiance_uses_canonical_atlas_without_validation() -> None:
    with (
        patch.object(atm, "normalize_atlas", side_effect=AssertionError("revalidation")),
        patch.object(atm.NaiveInterpolatedAtm, "black_body_radiance", return_value={}) as blackbody,
    ):
        assert atm.NaiveInterpolatedAtm.radiance(np.array([6000.0]), np.array([2.0]), 0.0, "bb") == {}
    blackbody.assert_called_once()


def test_internal_table_selection_uses_canonical_atlas_without_validation() -> None:
    with patch.object(atm, "normalize_atlas", side_effect=AssertionError("revalidation")):
        paths = atm.NaiveInterpolatedAtm.atm_files(np.array([5000.0]), np.array([2.0]), 0.0, "ck04")
    assert paths


def test_initial_configuration_error_also_aborts_mcmc_if_encountered_in_candidate() -> None:
    config, data = rv_problem()
    fit = MCMCRadialVelocity()
    fit.set_up(BinaryInitialParameters(**config), data, observer_system_cls=RadialVelocitySystem)
    fit.prepare_priors()
    failure = ElisaInitialParametersError("Invalid fixed atlas configuration")
    with (
        patch.object(type(fit), "ln_prior", return_value=0.0),
        patch.object(type(fit), "likelihood", side_effect=failure),
        pytest.raises(ElisaInitialParametersError) as caught,
    ):
        fit.ln_probability(np.zeros(len(fit.fitable)))
    assert caught.value is failure


def test_canonical_atmosphere_values() -> None:
    assert [a.value for a in Atmosphere] == ["ck04", "k93", "bb"]
    assert all(isinstance(a, str) for a in Atmosphere)


@pytest.mark.parametrize(
    ("alias", "expected"),
    [("castelli", "ck04"), ("castelli-kurucz", "ck04"), ("kurucz", "k93"), ("ck04", "ck04"), ("k93", "k93"),
     ("bb", "bb"), ("black-body", "bb")],
)
def test_aliases_normalize_to_canonical(alias: str, expected: str) -> None:
    assert atm.validated_atlas(alias) is Atmosphere(expected)


@pytest.mark.parametrize(("atlas", "prefix"), [("ck04", "ck"), ("k93", "k"), ("castelli", "ck"), ("kurucz", "k")])
def test_file_prefix_resolution(atlas: str, prefix: str) -> None:
    assert atm_file_prefix(atlas) == prefix
    assert atm.get_atm_table_filename(5000.0, 2.0, 0.0, atlas).startswith(prefix)


def test_blackbody_has_no_file_prefix() -> None:
    with pytest.raises(ElisaInitialParametersError):
        atm_file_prefix("bb")


def test_atmosphere_reinitialization_tracking_and_serialization() -> None:
    star = Star(mass=1.0, t_eff=6000.0)
    assert not parameter_is_explicit(star, "atmosphere")
    star.atmosphere = "kurucz"
    assert parameter_is_explicit(star, "atmosphere")
    star.init()
    assert star.atmosphere == Atmosphere.K93
    assert star.kwargs_serializer()["atmosphere"] == "k93"
    clone = Star(**star.reinitialization_kwargs())
    assert clone.atmosphere == Atmosphere.K93
    assert StarContainer.from_star_instance(star).atmosphere == Atmosphere.K93


def test_default_atmosphere_survives_reinitialization_without_becoming_explicit() -> None:
    star = Star(mass=1.0, t_eff=6000.0)
    star.init()
    assert star.atmosphere == Atmosphere.CK04
    assert not parameter_is_explicit(star, "atmosphere")


@pytest.mark.parametrize("atmosphere", ["ck04", "k93", "bb"])
def test_synthesis_receives_component_atmosphere(atmosphere: str) -> None:
    from types import SimpleNamespace

    from elisa.single_system.curves import utils as single_utils

    star = SimpleNamespace(
        temperatures=np.array([6000.0]),
        log_g=np.array([4.4]),
        metallicity=0.0,
        atmosphere=Atmosphere(atmosphere),
        symmetry_faces=lambda x: x,
        mirror_face_values=lambda x: x,
    )
    system = SimpleNamespace(star=star, has_spots=lambda: True, has_pulsations=lambda: False)
    with patch.object(single_utils.atm.NaiveInterpolatedAtm, "radiance", return_value={}) as radiance:
        single_utils.get_normal_radiance(system)
    assert radiance.call_args.kwargs["atlas"] == atmosphere
