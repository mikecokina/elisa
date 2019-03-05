import os
import unittest
from copy import copy

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from elisa.conf import config
from elisa.engine import atm
from elisa.engine.atm import NaiveInterpolatedAtm, AtmDataContainer
from elisa.engine.observer.observer import PassbandContainer

config.CK04_ATM_TABLES = os.path.join(os.path.dirname(__file__), 'ck04')


class TestNaiveInterpolation(unittest.TestCase):
    def setUp(self):
        self.temperatures = [5325.0, 10001, 25320.25, 4500]
        self.logg = [3.1, 4.2, 4.5, 5.0]
        self.metallicity = 0.1

    def test_atm_files_ck04(self):
        files = NaiveInterpolatedAtm.atm_files(temperature=self.temperatures,
                                               logg=self.logg,
                                               metallicity=self.metallicity,
                                               atlas="ck04")

        expected_dir = 'ckp00'
        expected = ['ckp00_5250_g30.csv', 'ckp00_10000_g40.csv', 'ckp00_25000_g45.csv', None,
                    'ckp00_5500_g30.csv', 'ckp00_10250_g40.csv', 'ckp00_26000_g45.csv', 'ckp00_4500_g50.csv']

        obtained = [os.path.basename(f) if f is not None else None for f in files]
        assert_array_equal(expected, obtained)

    def test_atm_files_k93(self):
        # files = NaiveInterpolatedAtm.atm_files(temperature=self.temperatures,
        #                                        logg=self.logg,
        #                                        metallicity=self.metallicity,
        #                                        atlas="k93")

        # todo: need implementation in config (uncomment and you will see)
        pass

    def test_atm_tables_ck04(self):
        fpaths = NaiveInterpolatedAtm.atm_files(temperature=self.temperatures,
                                                logg=self.logg,
                                                metallicity=self.metallicity,
                                                atlas="ck04")

        containers = atm.read_atm_tables(fpaths=fpaths)
        self.assertTrue(len(self.temperatures * 2) == len(containers))
        self.assertTrue(containers[3] is None)

        for t, container in zip(self.temperatures, containers[:int(len(containers) / 2)]):
            if container is not None:
                self.assertTrue(t > container.temperature)

        for t, container in zip(self.temperatures, containers[int(len(containers) / 2):]):
            self.assertTrue(t <= container.temperature)

    def test_strip_atm_container_by_bandwidth(self):
        model = pd.DataFrame({
            config.ATM_MODEL_DATAFRAME_FLUX: np.arange(0, 10, 1),
            config.ATM_MODEL_DATAFRAME_WAVE: np.arange(0, 10, 1)
        })
        container = AtmDataContainer(
            model=model,
            temperature=3600.0,
            logg=4.3,
            metallicity=0.0
        )

        striped = atm.strip_atm_container_by_bandwidth(container, left_bandwidth=2, right_bandwidth=5)
        assert_array_equal(np.arange(1, 7, 1), striped.model[config.ATM_MODEL_DATAFRAME_WAVE])

        container.model = model
        striped = atm.strip_atm_container_by_bandwidth(container, left_bandwidth=0, right_bandwidth=5)
        assert_array_equal(np.arange(0, 7, 1), striped.model[config.ATM_MODEL_DATAFRAME_WAVE])

        container.model = model
        striped = atm.strip_atm_container_by_bandwidth(container, left_bandwidth=0, right_bandwidth=25)
        assert_array_equal(np.arange(0, 10, 1), striped.model[config.ATM_MODEL_DATAFRAME_WAVE])

    def test_compute_interpolation_weights(self):
        model = pd.DataFrame({
            config.ATM_MODEL_DATAFRAME_FLUX: np.arange(0, 10, 1),
            config.ATM_MODEL_DATAFRAME_WAVE: np.arange(0, 10, 1)
        })
        temperature = np.array([3500.0, 5321, 10255.4])
        top_temp = np.array([3500.0, 5500.0, 10500.0])
        bottom_temp = np.array([None, 5250.0, 10250.0])

        top_atm = [AtmDataContainer(
            model=model,
            temperature=t,
            logg=4.3,
            metallicity=0.0
        ) for t in top_temp]
        bottom_atm = [AtmDataContainer(
            model=model,
            temperature=t,
            logg=4.3,
            metallicity=0.0
        ) if t is not None else None for t in bottom_temp]

        obtained = NaiveInterpolatedAtm.compute_interpolation_weights(temperature, top_atm, bottom_atm)

        bottom_temp[0] = 0.0
        expected = (temperature - bottom_temp) / (top_temp - bottom_temp)

        assert_array_equal(obtained, expected)

    def test_interpolate_return_exact(self):
        model_3500 = pd.DataFrame({
            config.ATM_MODEL_DATAFRAME_FLUX: np.arange(0, 10, 1),
            config.ATM_MODEL_DATAFRAME_WAVE: np.arange(0, 10, 1)
        })

        model_3500_orig = copy(model_3500)
        temperature = np.array([3500.0])
        atm_3500 = [
            None,
            AtmDataContainer(
                model=model_3500,
                temperature=3500.0,
                logg=0.0,
                metallicity=0.0
            )]

        atm_container = NaiveInterpolatedAtm.interpolate(
            atm_3500,
            temperature=temperature,
            logg=[0.0],
            metallicity=0.0,
            left_bandwidth=2,
            right_bandwidth=5
        )[0]

        obtained = atm_container.model
        expected = model_3500_orig.iloc[np.arange(1, 7, 1)]
        assert_frame_equal(obtained, expected.reset_index(drop=True))

    def test_interpolate_return_interpolated_on_same_wave(self):
        model_5250 = pd.DataFrame({
            config.ATM_MODEL_DATAFRAME_FLUX: np.arange(0, 10, 1),
            config.ATM_MODEL_DATAFRAME_WAVE: np.arange(0, 10, 1)
        })
        model_5500 = pd.DataFrame({
            config.ATM_MODEL_DATAFRAME_FLUX: np.arange(1, 11, 1),
            config.ATM_MODEL_DATAFRAME_WAVE: np.arange(0, 10, 1)
        })

        temperature = np.array([5400])
        atm_5xxx = [
            AtmDataContainer(
                model=model_5250,
                temperature=5250,
                logg=0.0,
                metallicity=0.0
            ),
            AtmDataContainer(
                model=model_5500,
                temperature=5500,
                logg=0.0,
                metallicity=0.0
            )]

        atm_container = NaiveInterpolatedAtm.interpolate(
            atm_5xxx,
            temperature=temperature,
            logg=[0.0],
            metallicity=0.0,
            left_bandwidth=2,
            right_bandwidth=6
        )[0]

        expected = pd.DataFrame({
            config.ATM_MODEL_DATAFRAME_FLUX: np.arange(1.6, 8.6, 1),
            config.ATM_MODEL_DATAFRAME_WAVE: np.arange(1, 8, 1)
        })
        obtained = atm_container.model
        assert_frame_equal(obtained, expected)

    def test_interpolate_return_interpolated_on_different_wave(self):
        model_5250 = pd.DataFrame({
            config.ATM_MODEL_DATAFRAME_FLUX: np.arange(0, 10, 0.25),
            config.ATM_MODEL_DATAFRAME_WAVE: np.arange(0, 20, 0.5)
        })
        model_5500 = pd.DataFrame({
            config.ATM_MODEL_DATAFRAME_FLUX: np.arange(0, 10, 1),
            config.ATM_MODEL_DATAFRAME_WAVE: np.arange(0, 10, 1)
        })

        temperature = np.array([5400])
        atm_5xxx = [
            AtmDataContainer(
                model=model_5250,
                temperature=5250,
                logg=0.0,
                metallicity=0.0
            ),
            AtmDataContainer(
                model=model_5500,
                temperature=5500,
                logg=0.0,
                metallicity=0.0
            )]

        atm_container = NaiveInterpolatedAtm.interpolate(
            atm_5xxx,
            temperature=temperature,
            logg=[0.0],
            metallicity=0.0,
            left_bandwidth=1.9,
            right_bandwidth=6.1
        )[0]

        obtained = atm_container.model
        expected = pd.DataFrame({
            config.ATM_MODEL_DATAFRAME_FLUX: (0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6),
            config.ATM_MODEL_DATAFRAME_WAVE: np.arange(1, 8, 1)
        })

        assert_frame_equal(obtained, expected)

    def test_apply_passband(self):
        model_3500 = pd.DataFrame({
            config.ATM_MODEL_DATAFRAME_FLUX: np.arange(2, 10, 1),
            config.ATM_MODEL_DATAFRAME_WAVE: np.arange(2, 10, 1)
        })

        atm_containers = [
            AtmDataContainer(
                model=model_3500,
                temperature=3500.0,
                logg=0.0,
                metallicity=0.0
            )]
        band_df = pd.DataFrame({
            config.PASSBAND_DATAFRAME_THROUGHPUT: [0.1] * 8,
            config.PASSBAND_DATAFRAME_WAVE: np.arange(2.4, 10.4, 1)
        })
        passband = {"passband": PassbandContainer(table=band_df)}

        obtained = atm.apply_passband(atm_containers, passband)["passband"][0].model
        expected = pd.DataFrame({
            config.ATM_MODEL_DATAFRAME_FLUX: (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            config.ATM_MODEL_DATAFRAME_WAVE: np.arange(2, 10, 1)
        })
        assert_frame_equal(expected, obtained)

    def test_inplace_in_strip_atm_container_by_bandwidth(self):
        model = pd.DataFrame({
            config.ATM_MODEL_DATAFRAME_FLUX: np.arange(0, 10, 1),
            config.ATM_MODEL_DATAFRAME_WAVE: np.arange(0, 10, 1)
        })
        container = AtmDataContainer(
            model=model,
            temperature=3600.0,
            logg=4.3,
            metallicity=0.0
        )

        # disable inplace
        striped = atm.strip_atm_container_by_bandwidth(container, left_bandwidth=2, right_bandwidth=5, inplace=False)
        assert_array_equal(np.arange(1, 7, 1), striped.model[config.ATM_MODEL_DATAFRAME_WAVE])

        # enable inplace and use same container as before (shouldn't be overwriten)
        striped = atm.strip_atm_container_by_bandwidth(container, left_bandwidth=0, right_bandwidth=5, inplace=True)
        assert_array_equal(np.arange(0, 7, 1), striped.model[config.ATM_MODEL_DATAFRAME_WAVE])

        assert_array_equal(striped.model[config.ATM_MODEL_DATAFRAME_WAVE],
                           container.model[config.ATM_MODEL_DATAFRAME_WAVE])
