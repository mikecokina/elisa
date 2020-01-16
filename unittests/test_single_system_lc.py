import os.path as op

from importlib import reload

from elisa.conf import config
from unittests.utils import (
    ElisaTestCase,
)
from elisa import (
    units
)
from elisa.single_system.curves import (
    lc,
    lcmp
)


class ComputeLightCurvesTestCase(ElisaTestCase):
    PARAMS = {
        'solar' :
        {
            "mass": 1.0,
            "t_eff": 5772 * units.K,
            "gravity_darkening": 0.32,
            "polar_log_g": 4.43775,
            "gamma": 0.0,
            # "inclination": 82.5 * units.deg,
            "inclination": 90.0 * units.deg,
            "rotation_period": 25.38 * units.d,
        },
    }

    SPOTS_META = {
    "standard":
        [
            {"longitude": 90,
             "latitude": 58,
             "angular_radius": 35,
             "temperature_factor": 0.98},
            {"longitude": 60,
             "latitude": 45,
             "angular_radius": 28,
             "temperature_factor": 0.97},
        ],
    }

    def setUp(self):
        # raise unittest.SkipTest(message)
        self.base_path = op.join(op.dirname(op.abspath(__file__)), "data", "light_curves")
        self.law = config.LIMB_DARKENING_LAW

        config.VAN_HAMME_LD_TABLES = op.join(self.base_path, "limbdarkening")
        config.CK04_ATM_TABLES = op.join(self.base_path, "atmosphere")
        config.ATM_ATLAS = "ck04"
        config._update_atlas_to_base_dir()

    def tearDown(self):
        config.LIMB_DARKENING_LAW = self.law
        reload(lc)
