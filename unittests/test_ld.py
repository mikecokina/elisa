import os

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from elisa import const, ld
from elisa import settings
from unittests.utils import ElisaTestCase


class TestLimbDarkeningModule(ElisaTestCase):
    def setUp(self):
        super(TestLimbDarkeningModule, self).setUp()
        self.base_path = os.path.dirname(os.path.abspath(__file__))

    def test_get_metallicity_from_ld_table_filename(self):
        filenames = ["lin.bolometric.m02.csv", "lin.bolometric.p00.csv",
                     "lin.Generic.Bessell.B.p10.csv", os.path.join("path.path.csv", "lin.Generic.Bessell.B.p10.csv")]
        expected = [-0.2, 0.0, 1.0, 1.0]
        obtained = [ld.get_metallicity_from_ld_table_filename(filename) for filename in filenames]
        assert_array_equal(expected, obtained)

    def test_get_ld_table_filename(self):
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
        obtained = [ld.get_ld_table_filename(**param) for param in params]
        assert_array_equal(expected, obtained)

    def expected_ld_tables(self):
        path = os.path.join(self.base_path, "data", "vh93")
        return [pd.read_csv(os.path.join(path, file)) for file in sorted(os.listdir(path))]

    def test_get_ld_table(self):
        settings.configure(LD_TABLES=os.path.join(self.base_path, "data", "vh93"))
        params = [
            dict(passband="Generic.Bessell.B", metallicity=-0.2, law="cosine"),
            dict(passband="Kepler", metallicity=-4, law="logarithmic")
        ]

        expected = self.expected_ld_tables()
        obtained = [ld.get_ld_table(**param) for param in params]

        for e, o in zip(expected, obtained):
            assert_frame_equal(e, o, check_less_precise=True, check_dtype=False, check_exact=True)

    def test_get_ld_table_by_name(self):
        settings.configure(**{"LD_TABLES": os.path.join(self.base_path, "data", "vh93")})
        filenames = ["lin.Generic.Bessell.B.m02.csv", "log.Kepler.m40.csv"]

        expected = self.expected_ld_tables()
        obtained = [ld.get_ld_table_by_name(file) for file in filenames]

        for e, o in zip(expected, obtained):
            assert_frame_equal(e, o, check_less_precise=True, check_dtype=False, check_exact=True)

    def test_get_ld_table_by_name_same_as_get_ld_table(self):
        settings.configure(**{"LD_TABLES": os.path.join(self.base_path, "data", "vh93")})
        params = [
            dict(passband="Generic.Bessell.B", metallicity=-0.2, law="cosine"),
            dict(passband="Kepler", metallicity=-4, law="logarithmic")
        ]
        filenames = ["lin.Generic.Bessell.B.m02.csv", "log.Kepler.m40.csv"]

        obtained_by_params = [ld.get_ld_table(**param) for param in params]
        obtained_by_filename = [ld.get_ld_table_by_name(file) for file in filenames]

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

    def test_interpolate_on_ld_grid_lin(self):
        settings.configure(**{"LIMB_DARKENING_LAW": "cosine",
                              "LD_TABLES": os.path.join(self.base_path, "data", "ld_grid")})
        temperature = np.array([5500, 5561, 5582, 5932])
        log_g = np.array([4.2, 4.21, 4.223, 4.199]) - 2.0
        metallicity = -0.1
        passband = ["Generic.Bessell.B"]

        expected = np.array([0.7732, 0.76364, 0.76032, 0.70833])

        obtained = ld.interpolate_on_ld_grid(temperature, log_g, metallicity, passband)
        obtained = np.round(obtained["Generic.Bessell.B"].xlin.values, 5)

        self.assertTrue(np.all(obtained - expected) < 1e-5)

    def test_interpolate_on_ld_grid_log(self):
        settings.configure(**{"LIMB_DARKENING_LAW": "logarithmic",
                              "LD_TABLES": os.path.join(self.base_path, "data", "light_curves", "limbdarkening")})
        temperature = np.array([5500, 5561, 5582, 5932])
        log_g = np.array([4.2, 4.21, 4.223, 4.199]) - 2.0
        metallicity = 0.0
        passband = ["Generic.Bessell.V"]

        expected_xlog = np.round(np.array([0.776400, 0.772272, 0.770870, 0.747420]), 5)
        expected_ylog = np.round(np.array([0.204200, 0.211724, 0.214276, 0.251288]), 5)

        obtained = ld.interpolate_on_ld_grid(temperature, log_g, metallicity, passband)
        obtained_xlog = np.round(obtained["Generic.Bessell.V"].xlog.values, 5)
        obtained_ylog = np.round(obtained["Generic.Bessell.V"].ylog.values, 5)

        self.assertTrue(np.all(expected_ylog == obtained_ylog))
        self.assertTrue(np.all(expected_xlog == obtained_xlog))

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

    def test_limb_darkening_factor_linear(self):
        ld_law = 'linear'
        coefficients = np.array([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]])
        cos_theta = np.array([[-0.23], [0.25], [0.85], [0.55], [0.33], [1.0]])

        expected = np.array([0., 0.85, 0.955, 0.82, 0.665, 1.])
        obtained = ld.limb_darkening_factor(coefficients=coefficients,
                                            limb_darkening_law=ld_law,
                                            cos_theta=cos_theta)

        self.assertTrue(np.all(expected == np.round(obtained, 3)))

    def test_limb_darkening_factor_logarithmic(self):
        ld_law = 'logarithmic'
        coefficients = np.array([[0.1, 0.9],
                                 [0.2, 0.8],
                                 [0.3, 0.7],
                                 [0.4, 0.6],
                                 [0.5, 0.5],
                                 [0.2, 0.3]])
        cos_theta = np.array([[-0.23], [0.25], [0.85], [0.55], [0.33], [1.0]])

        obtained = ld.limb_darkening_factor(coefficients=coefficients,
                                            limb_darkening_law=ld_law,
                                            cos_theta=cos_theta)

        expected = np.array([0., 1.127, 1.052, 1.017, 0.848, 1.])
        self.assertTrue(np.all(expected == np.round(obtained, 3)))
