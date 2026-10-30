"""Atlas choices are resolved at input boundaries, not in surface synthesis."""

from unittest.mock import patch

import numpy as np
import pytest

from elisa import Star, settings
from elisa.atm import atm
from elisa.analytics.binary_fit.least_squares import CentralRadialVelocity
from elisa.analytics.binary_fit.mcmc import CentralRadialVelocity as MCMCRadialVelocity
from elisa.analytics.params.parameters import BinaryInitialParameters
from elisa.base.error import ElisaInitialParametersError
from elisa.binary_system.curves.community import RadialVelocitySystem
from tests.tests_analytics.fitting_helpers import rv_problem


@pytest.mark.parametrize("alias,canonical", [("ck04", "ck"), ("k93", "k"), ("black_body", "bb")])
def test_star_resolves_atlas_on_construction_and_reinitialization(alias, canonical):
    star = Star(mass=1.0, t_eff=6000.0, atmosphere=alias)
    assert star.atmosphere == canonical
    star.atmosphere = alias
    star.init()
    assert star.atmosphere == canonical


def test_invalid_star_atlas_is_initial_parameter_error():
    with pytest.raises(ElisaInitialParametersError, match="Incorrect atlas") as failure:
        Star(mass=1.0, t_eff=6000.0, atmosphere="invalid")
    assert isinstance(failure.value.__cause__, KeyError)


def test_invalid_atlas_is_rechecked_on_star_reinitialization():
    star = Star(mass=1.0, t_eff=6000.0)
    star.atmosphere = "invalid"
    with pytest.raises(ElisaInitialParametersError, match="Incorrect atlas"):
        star.init()


def test_default_atlas_is_normalized_and_invalid_updates_are_atomic():
    with settings.override(ATM_ATLAS="black_body"):
        assert settings.ATM_ATLAS == "bb"
        with pytest.raises(ElisaInitialParametersError, match="Incorrect atlas"):
            settings.configure(ATM_ATLAS="invalid")
        assert settings.ATM_ATLAS == "bb"


@pytest.mark.parametrize("atlas", ["black_body", "invalid"])
def test_fit_override_is_resolved_before_candidates(atlas):
    config, data = rv_problem()
    fit = CentralRadialVelocity()
    override = {"primary": atlas}
    fit.atmosphere_model = override
    if atlas == "invalid":
        with pytest.raises(ElisaInitialParametersError, match="Incorrect atlas"):
            fit.set_up(BinaryInitialParameters(**config), data, observer_system_cls=RadialVelocitySystem)
    else:
        fit.set_up(BinaryInitialParameters(**config), data, observer_system_cls=RadialVelocitySystem)
        assert fit.atmosphere_model == {"primary": "bb"}
        assert override == {"primary": atlas}


def test_internal_radiance_uses_canonical_atlas_without_validation():
    with (
        patch.object(atm, "normalize_atlas", side_effect=AssertionError("revalidation")),
        patch.object(atm.NaiveInterpolatedAtm, "black_body_radiance", return_value={}) as blackbody,
    ):
        assert atm.NaiveInterpolatedAtm.radiance(np.array([6000.0]), np.array([2.0]), 0.0, "bb") == {}
    blackbody.assert_called_once()


def test_internal_table_selection_uses_canonical_atlas_without_validation():
    with patch.object(atm, "normalize_atlas", side_effect=AssertionError("revalidation")):
        paths = atm.NaiveInterpolatedAtm.atm_files(np.array([5000.0]), np.array([2.0]), 0.0, "ck")
    assert paths


def test_initial_configuration_error_also_aborts_mcmc_if_encountered_in_candidate():
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
