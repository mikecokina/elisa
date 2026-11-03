"""Atlas defaults are snapshots taken when stellar fitting parameters are made."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from elisa import const, settings
from elisa.analytics.params import conf, parameters
from elisa.atm import const as atm_const
from tests import set_astropy_units
from tests.utils import ElisaTestCase

set_astropy_units()


class AtlasRuntimeDefaultsTestCase(ElisaTestCase):
    @staticmethod
    def make_star(**kwargs) -> parameters.StarInitialParameters:
        return parameters.StarInitialParameters(
            **{
                "t_eff": {"value": 6000.0},
                "metallicity": {"value": 0.0},
                **kwargs,
            },
        )

    def assert_bounds(self, star: parameters.StarInitialParameters, metallicity: tuple[float, float]) -> None:
        assert ((star.t_eff.min, star.t_eff.max)) == ((3500.0, 50000.0))
        assert ((star.metallicity.min, star.metallicity.max)) == (metallicity)
        assert (star.t_eff.value) == (6000.0)
        assert (star.metallicity.value) == (0.0)

    def test_round_trip_keeps_existing_instances_and_shared_grids(self) -> None:
        grids = {
            name: (getattr(const, name), list(getattr(const, name)))
            for name in (
                "CK_TEMPERATURE_LIST_ATM",
                "CK_METALLICITY_LIST_ATM",
                "K_TEMPERATURE_LIST_ATM",
                "K_METALLICITY_LIST_ATM",
            )
        }
        constants = deepcopy(conf.DEFAULT_NORMALIZATION_STAR)
        stars = []
        for atlas, bounds in (("ck04", (-2.5, 0.5)), ("k93", (-5.0, 1.0)), ("ck04", (-2.5, 0.5))):
            settings.configure(ATM_ATLAS=atlas)
            star = self.make_star()
            stars.append((star, bounds, deepcopy(star.DEFAULT_NORMALIZATION)))
            for old, expected, defaults in stars:
                self.assert_bounds(old, expected)
                assert (defaults) == (old.DEFAULT_NORMALIZATION)
        assert (stars[0][0].DEFAULT_NORMALIZATION) is not (stars[2][0].DEFAULT_NORMALIZATION)
        assert (constants) == (conf.DEFAULT_NORMALIZATION_STAR)
        for name, (original, values) in grids.items():
            assert (getattr(const, name)) is (original)
            assert (original) == (values)

    def test_all_supported_aliases(self) -> None:
        for alias, prefix in atm_const.ATM_ATLAS_NORMALIZER.items():
            with self.subTest(atlas=alias):
                settings.configure(ATM_ATLAS=alias)
                self.assert_bounds(self.make_star(), (-5.0, 1.0) if prefix == "k" else (-2.5, 0.5))

    def test_explicit_bounds_values_units_and_input_are_preserved(self) -> None:
        inputs = {
            "t_eff": {"value": 60, "min": 55, "max": 70, "unit": "kK", "sigma": 1},
            "metallicity": {"value": -5.5, "min": -6.0, "max": 2.0},
        }
        original = deepcopy(inputs)
        for atlas in ("ck04", "k93", "bb", "black_body", "ck04"):
            settings.configure(ATM_ATLAS=atlas)
            star = self.make_star(**inputs)
            assert ((star.t_eff.min, star.t_eff.value, star.t_eff.max)) == ((55000, 60000, 70000))
            assert (star.t_eff.sigma) == (1000)
            assert ((star.metallicity.min, star.metallicity.value, star.metallicity.max)) == ((-6, -5.5, 2))
            assert (inputs) == (original)

    def test_partial_bounds_only_fill_missing_endpoint(self) -> None:
        for atlas, expected in (("ck04", (-2.5, 0.5)), ("k93", (-5.0, 1.0)), ("bb", (-2.5, 0.5))):
            settings.configure(ATM_ATLAS=atlas)
            lower = self.make_star(metallicity={"value": 0, "min": -8})
            upper = self.make_star(metallicity={"value": 0, "max": 3})
            assert ((lower.metallicity.min, lower.metallicity.max)) == ((-8, expected[1]))
            assert ((upper.metallicity.min, upper.metallicity.max)) == ((expected[0], 3))

    def test_fixed_and_constrained_parameters_keep_existing_semantics(self) -> None:
        for atlas in ("ck04", "k93", "bb"):
            settings.configure(ATM_ATLAS=atlas)
            star = self.make_star(
                t_eff={"value": 60000, "fixed": True},
                metallicity={"constraint": "primary@metallicity", "fixed": None},
            )
            assert (star.t_eff.value) == (60000)
            assert (star.t_eff.min) is None
            assert (star.t_eff.max) is None
            assert (star.metallicity.constraint) == ("primary@metallicity")
            assert (star.metallicity.min) is None
            assert (star.metallicity.max) is None

    def test_binary_initialization_and_normalization_follow_snapshot(self) -> None:
        previous = []
        for atlas, bounds in (("ck04", (-2.5, 0.5)), ("k93", (-5.0, 1.0)), ("bb", (-2.5, 0.5)), ("ck04", (-2.5, 0.5))):
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
                assert (old.get_normalization_map()) == (snapshot)
