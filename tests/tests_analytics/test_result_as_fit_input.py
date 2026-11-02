"""Public fitting workflows accept previous results as their next initial model."""

import json
from copy import deepcopy

import numpy as np
import pytest

from elisa import settings
from elisa.analytics import RVBinaryAnalyticsTask
from elisa.analytics.params.parameters import BinaryInitialParameters
from elisa.base.error import ElisaInvalidKwargsError
from tests.tests_analytics.fitting_helpers import rv_problem


def test_real_lsq_result_seeds_and_restarts_rv_mcmc(tmp_path):
    cfg, data = rv_problem(constrained=True)
    with settings.override(RV_METHOD="kinematic", NUMBER_OF_PROCESSES=1):
        lsq = RVBinaryAnalyticsTask(data=data, method="least_squares")
        result = lsq.fit(x0=cfg)
        before = deepcopy(result)
        initial = BinaryInitialParameters(**cfg)
        names = list(initial.get_fitable())
        bounds = initial.get_normalization_map()
        center = np.array(
            [
                (result["system"][name.split("@")[1]]["value"] - bounds[name][0]) / (bounds[name][1] - bounds[name][0])
                for name in names
            ]
        )
        state = center + np.random.default_rng(42).normal(0, 1e-4, (8, len(names)))
        task = RVBinaryAnalyticsTask(data=data, method="mcmc")
        posterior = task.fit(
            x0=result, nwalkers=8, nsteps=3, burn_in=0, initial_state=state, progress=False, save=False
        )
        assert result == before
        assert np.isfinite(posterior["r_squared"]["value"])
        assert "confidence_interval" in posterior["system"]["mass_ratio"]
        # Exercise the same handoff again with posterior metadata present.
        restarted = task.fit(
            x0=posterior,
            nwalkers=8,
            nsteps=3,
            burn_in=0,
            initial_state=task.fit_cls.flat_chain[-8:].copy(),
            progress=False,
            save=False,
        )
        assert np.isfinite(restarted["r_squared"]["value"])
        result_path = tmp_path / "result.json"
        chain_path = tmp_path / "chain.json"
        task.save_result(str(result_path))
        chain_path.write_text(
            json.dumps(
                {
                    "flat_chain": task.fit_cls.flat_chain.tolist(),
                    "fitable_parameters": names,
                    "normalization": bounds,
                    "fitable": {name: meta.to_dict() for name, meta in initial.get_fitable().items()},
                }
            )
        )
        loaded = RVBinaryAnalyticsTask(data=data, method="mcmc")
        loaded.load_result(filename=str(result_path))
        loaded.load_chain(str(chain_path), discard=8)
        resumed = loaded.fit(
            x0=result,
            nwalkers=8,
            nsteps=3,
            burn_in=0,
            initial_state=loaded.fit_cls.flat_chain[-8:].copy(),
            progress=False,
            save=False,
        )
        assert np.isfinite(resumed["r_squared"]["value"])
        assert result == before


def test_fit_result_does_not_hide_invalid_model_parameter():
    cfg, data = rv_problem()
    cfg["r_squared"] = {"value": 0.9, "unit": None}
    cfg["system"]["peroid"] = {"value": 3.0, "fixed": True}
    task = RVBinaryAnalyticsTask(data=data, method="mcmc")
    with pytest.raises(ElisaInvalidKwargsError, match="peroid"):
        task.fit(x0=cfg)


def test_plain_initial_input_still_rejects_unknown_section():
    cfg, data = rv_problem()
    cfg["systme"] = {"period": {"value": 3.0, "fixed": True}}
    task = RVBinaryAnalyticsTask(data=data, method="mcmc")
    with pytest.raises(ElisaInvalidKwargsError, match="systme"):
        task.fit(x0=cfg)
