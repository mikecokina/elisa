import logging
from copy import deepcopy
from pathlib import Path

# The tests package selects Astropy constants before test-module imports.
import numpy as np
import pytest

from elisa import settings
from elisa.analytics.tools.bvi import pogsons_formula
from elisa.base.error import ElisaAtmosphereError, ElisaLimbDarkeningError
from elisa.binary_system import system
from elisa.observer import observer
from tests import set_astropy_units, utils

set_astropy_units()


system_blueprint = {
    "system": {
        "eccentricity": 0.0,
        "argument_of_periastron": 90,
        "gamma": 0.0,
        "period": 100.0,
        "inclination": 90,
    },
    "primary": {
        "mass": 1.0,
        "t_eff": 5774.0,
        "surface_potential": 104.3,
        "synchronicity": 1.0,
        "gravity_darkening": 0.32,
        "discretization_factor": 3,
        "albedo": 0.6,
        "metallicity": 0.0,
    },
    "secondary": {
        "mass": 0.5,
        "t_eff": 4000.0,
        "surface_potential": 130.0,
        "synchronicity": 1.0,
        "gravity_darkening": 0.32,
        "discretization_factor": 5,
        "albedo": 0.6,
        "metallicity": 0.0,
    },
}

passband = ["Generic.Bessell.U", "Generic.Bessell.B", "Generic.Bessell.V", "Generic.Bessell.R", "Generic.Bessell.I"]
# passband = ['bolometric']
expected = np.array([0.164, 0.629, 0.387, 0.712])


# Retain the historical diagnostic export and verify that usable finite photometry is produced.
@pytest.mark.requires_ld(passbands=("Generic.Bessell.B", "Generic.Bessell.V", "bolometric"))
@pytest.mark.requires_atmosphere
class SunTestCase(utils.ElisaTestCase):
    @staticmethod
    def test_sun() -> None:
        _passband = ["Generic.Bessell.B", "Generic.Bessell.V"]
        temperatures = np.arange(3500, 50000, 500)

        #         for ld in ['logarithmic', 'square_root']:
        settings.configure(REFLECTION_EFFECT=False)

        res = []

        blueprint = deepcopy(system_blueprint)
        for t in temperatures:
            try:
                blueprint["primary"]["t_eff"] = t
                bs = system.BinarySystem.from_json(blueprint)
                o = observer.Observer(passband=_passband, system=bs)
                lc = o.observe.lc(phases=[0.5])

                b_v = pogsons_formula(lc[1]["Generic.Bessell.B"][-1], lc[1]["Generic.Bessell.V"][-1])
                res.append([t, b_v])

            except (ElisaAtmosphereError, ElisaLimbDarkeningError) as exc:
                logging.getLogger(__name__).debug("Temperature %s is outside the installed tables: %s", t, exc)

        assert res, "The temperature sweep must produce usable photometry"
        assert np.isfinite(res).all()
        x, y = np.array(res).T[1], np.array(res).T[0]

        with Path("cosine.sun.dat").open("a") as f:
            f.write(f"{x!r}\n")
            f.write(f"{y!r}\n")
        # u_b = pogsons_formula(lc[1]['Generic.Bessell.U'][-1], lc[1]['Generic.Bessell.B'][-1])
        # b_v = pogsons_formula(lc[1]['Generic.Bessell.B'][-1], lc[1]['Generic.Bessell.V'][-1])
        # v_r = pogsons_formula(lc[1]['Generic.Bessell.V'][-1], lc[1]['Generic.Bessell.R'][-1])
        # v_i = pogsons_formula(lc[1]['Generic.Bessell.V'][-1], lc[1]['Generic.Bessell.I'][-1])
        #
        # print(b_v)
