import os

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from elisa.atm import AtmModel
from elisa import settings
from elisa.observer.passband import PassbandContainer
from unittests.utils import ElisaTestCase
from elisa import (
    umpy as up,
    atm,
    const)


class TestMapDict(ElisaTestCase):
    def test_ATLAS_TO_ATM_FILE_PREFIX(self):
        supplied = ["castelli", "castelli-kurucz", "ck", "ck04", "kurucz", "k93", "k"]
        expected = ["ck", "ck", "ck", "ck", "k", "k", "k"]
        obtained = [settings.ATLAS_TO_ATM_FILE_PREFIX[s] for s in supplied]
        assert_array_equal(obtained, expected)

    def test_ATLAS_TO_BASE_DIR(self):
        settings.configure(**{
            "CK04_ATM_TABLES": os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ck04'),
            "K93_ATM_TABLES": os.path.join(os.path.dirname(os.path.abspath(__file__)), 'k93')
        })
        supplied = ["castelli", "castelli-kurucz", "ck", "ck04", "kurucz", "k93", "k"]
        expected = [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ck04')] * 4 + \
                   [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'k93')] * 3
        obtained = [settings.ATLAS_TO_BASE_DIR[s] for s in supplied]
        assert_array_equal(obtained, expected)

    def test_ATM_DOMAIN_QUANTITY_TO_VARIABLE_SUFFIX(self):
        supplied = ["temperature", "gravity", "metallicity"]
        expected = ["TEMPERATURE_LIST_ATM", "GRAVITY_LIST_ATM", "METALLICITY_LIST_ATM"]
        obtained = [settings.ATM_DOMAIN_QUANTITY_TO_VARIABLE_SUFFIX[s] for s in supplied]
        assert_array_equal(expected, obtained)


class TestAtmDataContainer(ElisaTestCase):
    def setUp(self):
        super(TestAtmDataContainer, self).setUp()
        df = pd.DataFrame({settings.ATM_MODEL_DATAFRAME_FLUX: np.array([1, 2, 3, 4, 5]),
                           settings.ATM_MODEL_DATAFRAME_WAVE: np.array([10, 20, 30, 40, 50])})
        self.container = atm.AtmDataContainer(df, 10, 10, 10, fpath="path")

    def test_bandwidth(self):
        assert_array_equal([10, 50], [self.container.left_bandwidth, self.container.right_bandwidth])

    def test_flux_to_si_mult(self):
        self.assertEqual(self.container.flux_to_si_mult, 1e-7 * 1e4 * 1e10)

    def test_wave_to_si_mult(self):
        self.assertEqual(self.container.wave_to_si_mult, 1e-10)


class TestAtmModuleGeneral(ElisaTestCase):
    def setUp(self):
        super(TestAtmModuleGeneral, self).setUp()
        settings.configure(CK04_ATM_TABLES=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'ck04'))

    def test_arange_atm_to_same_wavelength(self):
        c1 = atm.AtmDataContainer(AtmModel(**{
            "flux": np.array([1, 2, 3, 4, 5]),
            "wavelength": np.array([1, 2, 3, 4, 5])
        }), 10, 10, 10)

        c2 = atm.AtmDataContainer(AtmModel(**{
            "flux": np.array([0.4, 2, 3, 4.3, 5]),
            "wavelength": np.array([0.9, 2, 3, 4.1, 5])
        }), 5, 5, 5)

        result = atm.arange_atm_to_same_wavelength([c1, c2])
        expected_wavelength = np.array([0.9, 1, 2, 3, 4, 4.1, 5])

        # are aligned?
        assert_array_equal(result[0].model.wavelength,
                           result[1].model.wavelength)
        # are xpected
        assert_array_equal(result[0].model.wavelength, expected_wavelength)

    def test_strip_atm_container_by_bandwidth(self):
        c = atm.AtmDataContainer(AtmModel.from_dataframe(pd.DataFrame({
            settings.ATM_MODEL_DATAFRAME_FLUX: up.arange(0, 100, 10, dtype=np.float),
            settings.ATM_MODEL_DATAFRAME_WAVE: up.arange(10, dtype=np.float)
        })), 10, 10, 10)

        l_band, r_band = 3.1, 7.8
        result = atm.strip_atm_container_by_bandwidth(c, l_band, r_band)
        expected_df = pd.DataFrame({
            settings.ATM_MODEL_DATAFRAME_FLUX: [31, 40, 50, 60, 70, 78],
            settings.ATM_MODEL_DATAFRAME_WAVE: [3.1, 4., 5., 6., 7., 7.8]
        })
        assert_frame_equal(expected_df, result.model.to_dataframe(), check_dtype=False)

        # global is set but right and left are valid for given model
        gl_band, gr_band = 4.0, 6.5
        result = atm.strip_atm_container_by_bandwidth(c, l_band, r_band, global_left=gl_band, global_right=gr_band)
        assert_frame_equal(expected_df, result.model.to_dataframe(), check_dtype=False)

        # global is set
        l_band, r_band = -1, 10000
        gl_band, gr_band = 4.0, 6.5
        result = atm.strip_atm_container_by_bandwidth(c, l_band, r_band, global_left=gl_band, global_right=gr_band)
        expected_df = pd.DataFrame({
            settings.ATM_MODEL_DATAFRAME_FLUX: [40, 50, 60, 65],
            settings.ATM_MODEL_DATAFRAME_WAVE: [4, 5., 6, 6.5]
        })
        assert_frame_equal(expected_df, result.model.to_dataframe(), check_dtype=False)

    def test_find_global_atm_bandwidth(self):
        c1 = atm.AtmDataContainer(AtmModel(**{
            "flux": np.array([1, 2, 3, 4, 5]),
            "wavelength": np.array([1, 2, 3, 4, 5])
        }), 10, 10, 10)

        c2 = atm.AtmDataContainer(AtmModel(**{
            "flux": np.array([0.4, 2, 3, 4.3, 5]),
            "wavelength": np.array([0.9, 2, 3, 4.1, 5])
        }), 5, 5, 5)

        result = atm.find_global_atm_bandwidth([c1, c2])
        expected = (1, 5)
        self.assertTupleEqual(result, expected)

    def test_extend_atm_container_on_bandwidth_boundary(self):
        c = atm.AtmDataContainer(AtmModel.from_dataframe(pd.DataFrame({
            settings.ATM_MODEL_DATAFRAME_FLUX: up.arange(0, 100, 10, dtype=np.float),
            settings.ATM_MODEL_DATAFRAME_WAVE: up.arange(10, dtype=np.float)
        })), 10, 10, 10)

        l_band, r_band = 0.4, 8.8
        result = atm.extend_atm_container_on_bandwidth_boundary(c, l_band, r_band).\
            model.to_dataframe().sort_index(axis=1)
        expected = pd.DataFrame({
            settings.ATM_MODEL_DATAFRAME_WAVE: [0.4] + list(range(1, 9, 1)) + [8.8],
            settings.ATM_MODEL_DATAFRAME_FLUX: [4] + list(range(10, 90, 10)) + [88]
        }).sort_index(axis=1)
        assert_frame_equal(result, expected, check_dtype=False)

    def test_atm_file_prefix_to_quantity_list(self):
        atlas = ["ck04", "kurucz", "ck"]
        quantities = ["temperature", "gravity", "metallicity"]
        expected = [[3500.0, 3750.0, 4000.0],
                    [3500.0, 3750.0, 4000.0],
                    [3500.0, 3750.0, 4000.0],
                    [0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0],
                    [-2.5, -2.0, -1.5], [-5.0, -4.5, -4.0], [-2.5, -2.0, -1.5]]

        obtained = [atm.atm_file_prefix_to_quantity_list(q, a)[:3] for q in quantities for a in atlas]
        assert_array_equal(expected, obtained)

    def test_get_metallicity_from_atm_table_filename(self):
        filename = ["km05", "ckp02", "km10_3500_g05.csv", "km10_6500_g45", os.path.join("x", "y", "kp03_3500_g00.csv")]
        expected = [-0.5, 0.2, -1, -1, 0.3]
        obtained = [atm.get_metallicity_from_atm_table_filename(f) for f in filename]
        assert_array_equal(expected, obtained)

    def test_get_logg_from_atm_table_filename(self):
        filename = ["km10_3500_g05.csv", "km10_6500_g45", os.path.join("x", "y", "kp03_3500_g00.csv")]
        expected = [0.5, 4.5, 0.0]
        obtained = [atm.get_logg_from_atm_table_filename(f) for f in filename]
        assert_array_equal(expected, obtained)

    def test_get_temperature_from_atm_table_filename(self):
        filename = ["km10_3500_g05.csv", "km10_6500_g45", os.path.join("x", "y", "kp03_35000_g00.csv")]
        expected = [3500, 6500, 35000]
        obtained = [atm.get_temperature_from_atm_table_filename(f) for f in filename]
        assert_array_equal(expected, obtained)

    def test_get_atm_table_filename(self):
        t = [3500, 10250, 13000]
        log_g = [0.0, 2.5, 4]
        m_h = [-0.1, 0.0, 1]

        obtained = [atm.get_atm_table_filename(_t, _g, _m, atlas="ck04") for _t, _g, _m in zip(t, log_g, m_h)]
        expected = ['ckm01_3500_g00.csv', 'ckp00_10250_g25.csv', 'ckp10_13000_g40.csv']
        assert_array_equal(expected, obtained)

    def test_get_atm_directory(self):
        m_h = [-0.1, -5, 0.0, 0.1, 2]
        atlas = ["ck", "ck04", "k93", "k", "kurucz"]
        obtained = [atm.get_atm_directory(_m, _a) for _m, _a in zip(m_h, atlas)]
        expected = ['ckm01', 'ckm50', 'kp00', 'kp01', 'kp20']
        assert_array_equal(obtained, expected)

    def test_get_list_of_all_atm_tables(self):
        obtained = sorted([os.path.basename(f) for f in atm.get_list_of_all_atm_tables("ck04")])
        expected = sorted(['ckp00_10000_g40.csv', 'ckp00_10250_g40.csv', 'ckp00_25000_g45.csv', 'ckp00_26000_g45.csv',
                           'ckp00_4500_g50.csv', 'ckp00_5250_g30.csv', 'ckp00_5500_g30.csv',
                           'ckp05_4750_g15.csv', 'ckp05_5000_g15.csv', 'ckp05_11250_g20.csv', 'ckp05_11500_g20.csv'])
        assert_array_equal(expected, obtained)

    def test_get_atm_table(self):
        expected = pd.read_csv(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'data', 'ck04', 'ckp00', 'ckp00_10250_g40.csv'))
        obtained = atm.get_atm_table(10250, 4.0, 0, "ck")
        assert_frame_equal(expected, obtained, check_dtype=False)

    def test_compute_normal_intensity(self):
        spectral_flux = np.array([list(range(5)), list(range(5, 10, 1)), list(range(5))])
        wavelength = np.array(list(range(5)))
        result = np.round(atm.compute_normal_intensity(spectral_flux, wavelength), 4)
        expected = np.round(np.array([25.1327, 87.9646, 25.1327]) * (1.0 / const.PI), 4)
        result = np.round(result, 4)
        assert_array_equal(result, expected)

        result = np.round(atm.compute_normal_intensity(spectral_flux, wavelength, flux_mult=2.0), 4)
        expected = np.round(np.array([50.2655, 175.9292, 50.2655]) * (1.0 / const.PI), 4)
        result = np.round(result, 4)
        assert_array_equal(result, expected)

        result = np.round(atm.compute_normal_intensity(spectral_flux, wavelength, wave_mult=2.5), 4)
        expected = np.round(np.array([62.8319, 219.9115, 62.8319]) * (1.0 / const.PI), 4)
        result = np.round(result, 4)
        assert_array_equal(result, expected)

    def test_unique_atm_fpaths(self):
        paths = ["path1", "path1", "path2", "path3", "path2", "path4"]
        expected_fset = ['path1', 'path2', 'path3', 'path4']
        expected_fmap = {'path3': [3], 'path2': [2, 4], 'path4': [5], 'path1': [0, 1]}

        fset, fmap = atm.unique_atm_fpaths(paths)
        assert_array_equal(sorted(fset), expected_fset)
        self.assertDictEqual(fmap, expected_fmap)

    def test_remap_unique_atm_container_to_origin(self):
        class MockAtm(object):
            def __init__(self, p):
                self.fpath = p

        expected = ["path1", "path1", "path2", "path3", "path2", "path4"]

        models = [MockAtm(f) for f in ['path1', 'path2', 'path3', 'path4']]
        fmap = {'path3': [3], 'path2': [2, 4], 'path4': [5], 'path1': [0, 1]}

        obtained = [model.fpath for model in atm.remap_unique_atm_container_to_origin(models, fmap)]
        assert_array_equal(expected, obtained)

    def test_read_unique_atm_tables(self):
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'ck04', 'ckp00')
        paths = [os.path.join(base_path, 'ckp00_4500_g50.csv'),
                 os.path.join(base_path, 'ckp00_10250_g40.csv'),
                 os.path.join(base_path, 'ckp00_4500_g50.csv')]
        result, _ = atm.read_unique_atm_tables(paths)
        expected_paths = sorted([os.path.join(base_path, 'ckp00_4500_g50.csv'),
                                 os.path.join(base_path, 'ckp00_10250_g40.csv')])

        obtained_paths = sorted([r.fpath for r in result])
        assert_array_equal(obtained_paths, expected_paths)

    def test_find_atm_si_multiplicators(self):
        expected = (1e-7 * 1e4 * 1e10, 1e-10)
        cs = [atm.AtmDataContainer(AtmModel.from_dataframe(pd.DataFrame({settings.ATM_MODEL_DATAFRAME_WAVE: [1, 2, 3],
                                                           settings.ATM_MODEL_DATAFRAME_FLUX: [1, 2, 3]})), 0, 0, 0)] * 10
        mults = atm.find_atm_si_multiplicators(cs)
        self.assertTupleEqual(mults, expected)

    def test_find_atm_defined_wavelength(self):
        cs = [atm.AtmDataContainer(pd.DataFrame({settings.ATM_MODEL_DATAFRAME_WAVE: list(range(10)),
                                                 settings.ATM_MODEL_DATAFRAME_FLUX: list(range(10))}), 0, 0, 0)] * 10
        expected = list(range(10))
        obtained = atm.find_atm_defined_wavelength(cs)
        assert_array_equal(expected, obtained)

    def test_apply_passband(self):
        atmc = atm.AtmDataContainer(AtmModel.from_dataframe(pd.DataFrame({
            settings.ATM_MODEL_DATAFRAME_FLUX: up.arange(10, dtype=np.float),
            settings.ATM_MODEL_DATAFRAME_WAVE: up.arange(0, 100, 10, dtype=np.float)
        })), 0, 0, 0)

        bandc = PassbandContainer(
            pd.DataFrame({
                settings.PASSBAND_DATAFRAME_THROUGHPUT: [0.2, 1.0, 0.2, 0.2, 0.4],
                settings.PASSBAND_DATAFRAME_WAVE: [1, 10, 25, 40, 50]
            }),
            passband="bandc"
        )

        passband = dict(bandc=bandc)

        obtained = np.round(atm.apply_passband([atmc], passband)["bandc"][0].model.to_dataframe(), 4)
        expected = pd.DataFrame({
            settings.ATM_MODEL_DATAFRAME_FLUX: [0.02, 1., 0.8117, 0.5077, 0.8, 2.],
            settings.ATM_MODEL_DATAFRAME_WAVE: [1., 10., 20., 30., 40., 50.]
        })
        assert_frame_equal(expected, obtained, check_dtype=False)


class TestNaiveInterpolation(ElisaTestCase):
    def setUp(self):
        super(TestNaiveInterpolation, self).setUp()
        settings.configure(CK04_ATM_TABLES=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'ck04'))

    def test_atm_files(self):
        g = np.array([1.5, 2, 2])
        t = np.array([4999, 11300, 11500])

        obtained = sorted([os.path.basename(v) for v in atm.NaiveInterpolatedAtm.atm_files(t, g, 0.5, "ck04")])
        expected = ['ckp05_11250_g40.csv', 'ckp05_11500_g40.csv',
                    'ckp05_11500_g40.csv', 'ckp05_11500_g40.csv',
                    'ckp05_4750_g35.csv', 'ckp05_5000_g35.csv']
        assert_array_equal(obtained, expected)

    def test_compute_interpolation_weights(self):
        class MockAtm(object):
            def __init__(self, t):
                self.temperature = t

        top = [MockAtm(t) for t in [16000, 3750, 4750, 3500]]
        bottom = [MockAtm(t) for t in [15000, 3500, 4500, 3500]]
        temperatures = np.array([15010, 3555, 4562, 3500])
        weights = np.round(atm.NaiveInterpolatedAtm.compute_interpolation_weights(temperatures, top, bottom), 4)
        expected = [0.01, 0.22, 0.248, 1.]
        assert_array_equal(expected, weights)

    def test_compute_unknown_intensity_from_surounded_flux_matrices(self):
        bottom = np.array([[10, 11, 12, 13],
                           [100, 150, 180, 120]])

        top = np.array([[10, 12, 14, 16],
                        [110, 150, 180, 120]])

        weights = np.array([0.2, 1.0])

        obtained = atm.NaiveInterpolatedAtm.compute_unknown_intensity_from_surounded_flux_matrices(weights, top, bottom)
        expected = [[10., 11.2, 12.4, 13.6],
                    [110., 150., 180., 120.]]
        assert_array_equal(expected, obtained)
