import os
import unittest
import pandas as pd
import numpy as np

from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from elisa.conf import config
from elisa import const, ld


class TestLimbDarkeningModule(unittest.TestCase):
    def test_get_metallicity_from_ld_table_filename(self):
        filenames = ["lin.bolometric.m02.csv", "lin.bolometric.p00.csv",
                     "lin.Generic.Bessell.B.p10.csv", os.path.join("path.path.csv", "lin.Generic.Bessell.B.p10.csv")]
        expected = [-0.2, 0.0, 1.0, 1.0]
        obtained = [ld.get_metallicity_from_ld_table_filename(filename) for filename in filenames]
        assert_array_equal(expected, obtained)

    def test_get_van_hamme_ld_table_filename(self):
        expected = ["lin.Generic.Bessell.B.m01.csv",
                    "lin.Generic.Stromgren.v.m50.csv",
                    "lin.Kepler.p01.csv",
                    "sqrt.Generic.Bessell.B.p10.csv"]
        params = [
            dict(passband="Generic.Bessell.B", metallicity=-0.1, law="linear"),
            dict(passband="Generic.Stromgren.v", metallicity=-5, law="linear"),
            dict(passband="Kepler", metallicity=0.1, law="linear"),
            dict(passband="Generic.Bessell.B", metallicity=1.0, law="square_root")
        ]
        obtained = [ld.get_van_hamme_ld_table_filename(**param) for param in params]

        assert_array_equal(expected, obtained)

    @staticmethod
    def expected_ld_tables():
        path = os.path.join(os.path.dirname(__file__), "data", "vh93")
        return [pd.read_csv(os.path.join(path, file)) for file in os.listdir(path)]

    def test_get_van_hamme_ld_table(self):
        config.VAN_HAMME_LD_TABLES = os.path.join(os.path.dirname(__file__), "data", "vh93")
        params = [
            dict(passband="Generic.Bessell.B", metallicity=-0.2, law="cosine"),
            dict(passband="Kepler", metallicity=-4, law="logarithmic")
        ]

        expected = self.expected_ld_tables()
        obtained = [ld.get_van_hamme_ld_table(**param) for param in params]

        for e, o in zip(expected, obtained):
            assert_frame_equal(e, o, check_less_precise=True, check_dtype=False, check_exact=True)

    def test_get_van_hamme_ld_table_by_name(self):
        config.VAN_HAMME_LD_TABLES = os.path.join(os.path.dirname(__file__), "data", "vh93")
        filenames = ["lin.Generic.Bessell.B.m02.csv", "log.Kepler.m40.csv"]

        expected = self.expected_ld_tables()
        obtained = [ld.get_van_hamme_ld_table_by_name(file) for file in filenames]

        for e, o in zip(expected, obtained):
            assert_frame_equal(e, o, check_less_precise=True, check_dtype=False, check_exact=True)

    def test_get_van_hamme_ld_table_by_name_same_as_get_van_hamme_ld_table(self):
        params = [
            dict(passband="Generic.Bessell.B", metallicity=-0.2, law="cosine"),
            dict(passband="Kepler", metallicity=-4, law="logarithmic")
        ]
        filenames = ["lin.Generic.Bessell.B.m02.csv", "log.Kepler.m40.csv"]

        obtained_by_params = [ld.get_van_hamme_ld_table(**param) for param in params]
        obtained_by_filename = [ld.get_van_hamme_ld_table_by_name(file) for file in filenames]

        for o1, o2 in zip(obtained_by_filename, obtained_by_params):
            assert_frame_equal(o1, o2)

    def test_get_relevant_ld_tables(self):
        params = [
            dict(metallicity=0.0, passband="Kepler", law="cosine"),
            dict(metallicity=-0.01, passband="Generic.Bessell.B", law="linear"),
            dict(metallicity=0.02, passband="SLOAN.SDSS.i", law="logarithmic"),
            dict(metallicity=1.0, passband="SLOAN.SDSS.i", law="square_root")
        ]
        obtained = [ld.get_relevant_ld_tables(**param) for param in params]
        expected = [
            ['lin.Kepler.p00.csv', 'lin.Kepler.p00.csv'],
            ['lin.Generic.Bessell.B.m01.csv', 'lin.Generic.Bessell.B.p00.csv'],
            ['log.SLOAN.SDSS.i.p00.csv', 'log.SLOAN.SDSS.i.p01.csv'],
            ['sqrt.SLOAN.SDSS.i.p10.csv', 'sqrt.SLOAN.SDSS.i.p10.csv']
        ]
        assert_array_equal(obtained, expected)

    def test_interpolate_on_ld_grid(self):
        config.VAN_HAMME_LD_TABLES = os.path.join(os.path.dirname(__file__), "data", "ld_grid")
        raise Exception("Unfinished unittest")

    def _raise_test_limb_darkening_factor(self, msg, **kwargs):
        with self.assertRaises(Exception) as context:
            ld.limb_darkening_factor(**kwargs)
        self.assertTrue(msg in str(context.exception))

    def test_limb_darkening_factor_raise(self):
        normals = np.array([[1, 1, 1], [0.5, 0.5, 0.5], [-1., -1, 1.3]])
        sov = const.LINE_OF_SIGHT

        # :warning: order of tests depend on order of check in ld.limb_darkening_factor method!!!

        # missing line_of_sight
        msg = 'Line of sight vector(s) was not supplied.'
        self._raise_test_limb_darkening_factor(msg, normal_vector=normals)

        # missing normal_vector
        msg = 'Normal vector(s) was not supplied.'
        self._raise_test_limb_darkening_factor(msg, line_of_sight=sov)

        # missing coefficients
        msg = 'Limb darkening coefficients were not supplied.'
        self._raise_test_limb_darkening_factor(msg, line_of_sight=sov, normal_vector=normals)

        # missing limb_darkening_law
        msg = 'Limb darkening rule was not supplied choose from'
        self._raise_test_limb_darkening_factor(msg, line_of_sight=sov, normal_vector=normals, coefficients=3)

    def test_limb_darkening_factor(self):
        raise Exception("Unfinished unittest")
