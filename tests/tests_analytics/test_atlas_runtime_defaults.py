"""Atlas defaults are snapshots taken when stellar fitting parameters are made."""
from tests import set_astropy_units

from copy import deepcopy

import numpy as np

from elisa import const, settings
from elisa.atm import const as atm_const
from elisa.analytics.params import conf, parameters
from tests.utils import ElisaTestCase

set_astropy_units()


class AtlasRuntimeDefaultsTestCase(ElisaTestCase):
    @staticmethod
    def make_star(**kwargs):
        return parameters.StarInitialParameters(**{
            "t_eff": {"value": 6000.0},
            "metallicity": {"value": 0.0},
            **kwargs,
        })

    def assert_bounds(self, star, metallicity):
        self.assertEqual((star.t_eff.min, star.t_eff.max), (3500.0, 50000.0))
        self.assertEqual((star.metallicity.min, star.metallicity.max), metallicity)
        self.assertEqual(star.t_eff.value, 6000.0)
        self.assertEqual(star.metallicity.value, 0.0)

    def test_round_trip_keeps_existing_instances_and_shared_grids(self):
        grids = {name: (getattr(const, name), list(getattr(const, name))) for name in (
            "CK_TEMPERATURE_LIST_ATM", "CK_METALLICITY_LIST_ATM",
            "K_TEMPERATURE_LIST_ATM", "K_METALLICITY_LIST_ATM",
        )}
        constants = deepcopy(conf.DEFAULT_NORMALIZATION_STAR)
        stars = []
        for atlas, bounds in (("ck04", (-2.5, 0.5)), ("k93", (-5.0, 1.0)), ("ck04", (-2.5, 0.5))):
            settings.configure(ATM_ATLAS=atlas)
            star = self.make_star()
            stars.append((star, bounds, deepcopy(star.DEFAULT_NORMALIZATION)))
            for old, expected, defaults in stars:
                self.assert_bounds(old, expected)
                self.assertEqual(old.DEFAULT_NORMALIZATION, defaults)
        self.assertIsNot(stars[0][0].DEFAULT_NORMALIZATION, stars[2][0].DEFAULT_NORMALIZATION)
        self.assertEqual(conf.DEFAULT_NORMALIZATION_STAR, constants)
        for name, (original, values) in grids.items():
            self.assertIs(getattr(const, name), original)
            self.assertEqual(original, values)

    def test_all_supported_aliases(self):
        for alias, prefix in atm_const.ATM_ATLAS_NORMALIZER.items():
            with self.subTest(atlas=alias):
                settings.configure(ATM_ATLAS=alias)
                self.assert_bounds(self.make_star(), (-5.0, 1.0) if prefix == "k" else (-2.5, 0.5))

    def test_explicit_bounds_values_units_and_input_are_preserved(self):
        inputs = {
            "t_eff": {"value": 60, "min": 55, "max": 70, "unit": "kK", "sigma": 1},
            "metallicity": {"value": -5.5, "min": -6.0, "max": 2.0},
        }
        original = deepcopy(inputs)
        for atlas in ("ck04", "k93", "bb", "black_body", "ck04"):
            settings.configure(ATM_ATLAS=atlas)
            star = self.make_star(**inputs)
            self.assertEqual((star.t_eff.min, star.t_eff.value, star.t_eff.max), (55000, 60000, 70000))
            self.assertEqual(star.t_eff.sigma, 1000)
            self.assertEqual((star.metallicity.min, star.metallicity.value, star.metallicity.max), (-6, -5.5, 2))
            self.assertEqual(inputs, original)

    def test_partial_bounds_only_fill_missing_endpoint(self):
        for atlas, expected in (("ck04", (-2.5, 0.5)), ("k93", (-5.0, 1.0)), ("bb", (-2.5, 0.5))):
            settings.configure(ATM_ATLAS=atlas)
            lower = self.make_star(metallicity={"value": 0, "min": -8})
            upper = self.make_star(metallicity={"value": 0, "max": 3})
            self.assertEqual((lower.metallicity.min, lower.metallicity.max), (-8, expected[1]))
            self.assertEqual((upper.metallicity.min, upper.metallicity.max), (expected[0], 3))

    def test_fixed_and_constrained_parameters_keep_existing_semantics(self):
        for atlas in ("ck04", "k93", "bb"):
            settings.configure(ATM_ATLAS=atlas)
            star = self.make_star(t_eff={"value": 60000, "fixed": True},
                                  metallicity={"constraint": "primary@metallicity", "fixed": None})
            self.assertEqual(star.t_eff.value, 60000)
            self.assertIsNone(star.t_eff.min)
            self.assertIsNone(star.t_eff.max)
            self.assertEqual(star.metallicity.constraint, "primary@metallicity")
            self.assertIsNone(star.metallicity.min)
            self.assertIsNone(star.metallicity.max)

    def test_binary_initialization_and_normalization_follow_snapshot(self):
        previous = []
        for atlas, bounds in (("ck04", (-2.5, 0.5)), ("k93", (-5.0, 1.0)),
                              ("bb", (-2.5, 0.5)), ("ck04", (-2.5, 0.5))):
            settings.configure(ATM_ATLAS=atlas)
            fit = parameters.BinaryInitialParameters(
                system={"period": {"value": 1.0, "fixed": True}},
                primary={"t_eff": {"value": 6000}, "metallicity": {"value": 0}},
                secondary={"t_eff": {"value": 6000}, "metallicity": {"value": 0}},
            )
            for star in (fit.primary, fit.secondary):
                self.assert_bounds(star, bounds)
            normalization = fit.get_normalization_map()
            keys = ("primary@t_eff", "primary@metallicity", "secondary@t_eff", "secondary@metallicity")
            values = [6000, 0, 6000, 0]
            expected = [(6000 - 3500) / (50000 - 3500), -bounds[0] / (bounds[1] - bounds[0])] * 2
            normalized = parameters.vector_normalizer(values, keys, normalization)
            np.testing.assert_allclose(normalized, expected)
            np.testing.assert_allclose(parameters.vector_renormalizer(normalized, keys, normalization), values)
            previous.append((fit, normalization))
            for old, snapshot in previous:
                self.assertEqual(old.get_normalization_map(), snapshot)
