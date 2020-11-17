import numpy as np

from unittests import utils
from elisa import settings
from elisa.binary_system import system
from elisa.observer import observer
from importlib import reload
from elisa.analytics.tools.bvi import pogsons_formula
from unittest import skip


system_blueprint = {
    "system": {
        "eccentricity": 0.0,
        "argument_of_periastron": 90,
        "gamma": 0.0,
        "period": 100.0,
        "inclination": 90
    },
    "primary": {
        "mass": 1.0,
        "t_eff": 5774.0,
        "surface_potential": 104.3,
        "synchronicity": 1.0,
        "gravity_darkening": 0.32,
        "discretization_factor": 3,
        "albedo": 0.6,
        "metallicity": 0.0
    },
    "secondary": {
        "mass": 0.5,
        "t_eff": 4000.0,
        "surface_potential": 130.0,
        "synchronicity": 1.0,
        "gravity_darkening": 0.32,
        "discretization_factor": 5,
        "albedo": 0.6,
        "metallicity": 0.0
    }
}

passband = ['Generic.Bessell.U', 'Generic.Bessell.B', 'Generic.Bessell.V', 'Generic.Bessell.R', 'Generic.Bessell.I']
# passband = ['bolometric']
expected = np.array([0.164, 0.629, 0.387, 0.712])


@skip("Skipped...run it only localy with sets atmosphere and limb darkening")
class SunTestCase(utils.ElisaTestCase):

    @staticmethod
    def test_sun():
        _passband = ['Generic.Bessell.B', 'Generic.Bessell.V']
        TS = np.arange(3500, 50000, 500)

#         for ld in ['logarithmic', 'square_root']:
        settings.configure(REFLECTION_EFFECT=False)
        reload(observer)
        reload(system)

        res = []

        for t in TS:
            try:
                system_blueprint["primary"]["t_eff"] = t
                bs = system.BinarySystem.from_json(system_blueprint)
                o = observer.Observer(passband=_passband, system=bs)
                lc = o.observe.lc(phases=[0.5])

                b_v = pogsons_formula(lc[1]['Generic.Bessell.B'][-1], lc[1]['Generic.Bessell.V'][-1])
                res.append([t, b_v])

            except Exception as e:
                print(e)

        x, y = np.array(res).T[1], np.array(res).T[0]

        with open(f"cosine.sun.dat", "a") as f:
            f.write(f"{repr(x)}\n")
            f.write(f"{repr(y)}\n")
        print()
        # u_b = pogsons_formula(lc[1]['Generic.Bessell.U'][-1], lc[1]['Generic.Bessell.B'][-1])
        # b_v = pogsons_formula(lc[1]['Generic.Bessell.B'][-1], lc[1]['Generic.Bessell.V'][-1])
        # v_r = pogsons_formula(lc[1]['Generic.Bessell.V'][-1], lc[1]['Generic.Bessell.R'][-1])
        # v_i = pogsons_formula(lc[1]['Generic.Bessell.V'][-1], lc[1]['Generic.Bessell.I'][-1])
        #
        # print(b_v)
