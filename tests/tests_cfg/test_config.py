# The tests package selects Astropy constants before test-module imports.
import numpy as np
from numpy.testing import assert_array_equal

from elisa import settings
from tests import set_astropy_units
from tests.utils import ElisaTestCase

set_astropy_units()


class TestConfig(ElisaTestCase):
    def test_LD_LAW_TO_FILE_PREFIX(self) -> None:
        laws = "linear", "cosine", "logarithmic", "square_root"
        expected = ["lin", "lin", "log", "sqrt"]
        obtained = [settings.LD_LAW_TO_FILE_PREFIX[law] for law in laws]
        assert_array_equal(expected, obtained)

    def test_LD_LAW_CFS_COLUMNS(self) -> None:
        laws = "linear", "cosine", "logarithmic", "square_root"
        expected = [["xlin"], ["xlin"], ["xlog", "ylog"], ["xsqrt", "ysqrt"]]
        obtained = [settings.LD_LAW_CFS_COLUMNS[law] for law in laws]
        for expected_, obtained_ in zip(expected, obtained, strict=True):
            assert_array_equal(expected_, obtained_)

    def test_LD_DOMAIN_COLS(self) -> None:
        expected = ["temperature", "gravity"]
        obtained = settings.LD_DOMAIN_COLS
        assert_array_equal(expected, obtained)

    def test_atm_dataframe_main_cols(self) -> None:
        expected = ["flux", "wave"]
        obtained = [settings.ATM_MODEL_DATAFRAME_FLUX, settings.ATM_MODEL_DATAFRAME_WAVE]
        assert_array_equal(expected, obtained)

    def test_passband_main_cols(self) -> None:
        expected = ["throughput", "wavelength"]
        obtained = [settings.PASSBAND_DATAFRAME_THROUGHPUT, settings.PASSBAND_DATAFRAME_WAVE]
        assert_array_equal(expected, obtained)

    def test__update_atlas_to_base_dir(self) -> None:
        settings.configure(CK04_ATM_TABLES="x")
        settings._update_atlas_to_base_dir()
        ck_values = [v for k, v in settings.ATLAS_TO_BASE_DIR.items() if str(k).startswith("c")]
        assert np.all(ck_values == ["x"] * len(ck_values))
