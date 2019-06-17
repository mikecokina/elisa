import unittest

from numpy.testing import assert_array_equal
from elisa.conf import config


class TestConfig(unittest.TestCase):
    def test_LD_LAW_TO_FILE_PREFIX(self):
        laws = "linear", "cosine", "logarithmic", "square_root"
        expected = ["lin", "lin", "log", "sqrt"]
        obtained = [config.LD_LAW_TO_FILE_PREFIX[law] for law in laws]
        assert_array_equal(expected, obtained)

    def test_LD_LAW_CFS_COLUMNS(self):
        laws = "linear", "cosine", "logarithmic", "square_root"
        expected = [["xlin"], ["xlin"], ["xlog", "ylog"], ["xsqrt", "ysqrt"]]
        obtained = [config.LD_LAW_CFS_COLUMNS[law] for law in laws]
        assert_array_equal(expected, obtained)

    def test_LD_DOMAIN_COLS(self):
        expected = ["temperature", "gravity"]
        obtained = config.LD_DOMAIN_COLS
        assert_array_equal(expected, obtained)

    def test_atm_dataframe_main_cols(self):
        expected = ["flux", "wave"]
        obtained = [config.ATM_MODEL_DATAFRAME_FLUX, config.ATM_MODEL_DATAFRAME_WAVE]
        assert_array_equal(expected, obtained)

    def test_passband_main_cols(self):
        expected = ["throughput", "wavelength"]
        obtained = [config.PASSBAND_DATAFRAME_THROUGHPUT, config.PASSBAND_DATAFRAME_WAVE]
        assert_array_equal(expected, obtained)
