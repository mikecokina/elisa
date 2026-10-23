"""Custom LD laws survive model reuse, container copies and reconstruction."""
from unittests import set_astropy_units

from copy import copy, deepcopy
from pathlib import Path
import pickle

import numpy as np

from elisa import BinarySystem, Observer, get_default_binary_definition, ld, settings
from elisa.base.container import StarContainer
from elisa.base.curves.utils import get_component_limbdarkening_cfs
from elisa.base.error import LimbDarkeningError
from elisa.base.star import Star
from elisa.base.transform import StarProperties
from unittests.utils import ElisaTestCase, prepare_single_system, SOLAR_MODEL

set_astropy_units()

SINGLE_PHASE = np.array([0.25])
BINARY_PHASES = np.array([0.1, 0.25])
INTERPOLATION_PHASES = np.array([0.0, 0.1, 0.25])


class CustomLDLawSafetyTestCase(ElisaTestCase):
    @staticmethod
    def coefficients(law):
        values = [0.5] if law in ("linear", "cosine") else [0.5, 0.2]
        return {band: list(values) for band in ("bolometric", "rv_band")}

    def binary(self, law, custom=True):
        settings.configure(LIMB_DARKENING_LAW=law, NUMBER_OF_PROCESSES=1)
        definition = get_default_binary_definition()
        for component in ("primary", "secondary"):
            definition[component]["atmosphere"] = "bb"
            definition[component]["discretization_factor"] = 10
            if custom:
                definition[component]["limb_darkening_coefficients"] = self.coefficients(law)
        return BinarySystem.from_json(definition)

    def assert_curves_equal(self, first, second):
        np.testing.assert_array_equal(first[0], second[0])
        for band in first[1]:
            np.testing.assert_array_equal(first[1][band], second[1][band])

    def test_log_sqrt_repeated_binary_lc_and_rv_reuse(self):
        for original, changed in (("logarithmic", "square_root"), ("square_root", "logarithmic")):
            for reflection in (True, False):
                with self.subTest(original=original, reflection=reflection):
                    model = self.binary(original)
                    settings.configure(REFLECTION_EFFECT=reflection)
                    observer = Observer(system=model, passband=["bolometric"])
                    lc = observer.lc(phases=BINARY_PHASES)
                    rv = observer.rv(phases=BINARY_PHASES, method="radiometric")
                    self.assert_curves_equal(lc, observer.lc(phases=BINARY_PHASES))
                    self.assert_curves_equal(rv, observer.rv(phases=BINARY_PHASES, method="radiometric"))
                    settings.configure(LIMB_DARKENING_LAW=changed)
                    for _ in range(2):
                        with self.assertRaisesRegex(LimbDarkeningError, f"defined for '{original}'.*'{changed}'"):
                            observer.lc(phases=BINARY_PHASES)
                        with self.assertRaisesRegex(LimbDarkeningError, "Supply new coefficients"):
                            observer.rv(phases=BINARY_PHASES, method="radiometric")
                    # Kinematic RV does not consume LD coefficients.
                    observer.rv(phases=BINARY_PHASES, method="kinematic")
                    settings.configure(LIMB_DARKENING_LAW=original)
                    self.assert_curves_equal(lc, observer.lc(phases=BINARY_PHASES))
                    self.assert_curves_equal(rv, observer.rv(phases=BINARY_PHASES, method="radiometric"))

    def test_container_and_bolometric_consumers_reject_same_length_law_change(self):
        for original, changed in (("logarithmic", "square_root"), ("square_root", "logarithmic")):
            model = self.binary(original)
            container = StarContainer.from_star_instance(model.primary)
            container.temperatures = np.array([5000.0, 6000.0])
            coefficients = container.limb_darkening_coefficients
            self.assertEqual(coefficients.law, original)
            np.testing.assert_array_equal(
                get_component_limbdarkening_cfs(container, ["bolometric"], symmetry_test=False)["bolometric"],
                [[0.5, 0.2], [0.5, 0.2]],
            )
            np.testing.assert_array_equal(ld.get_bolometric_ld_coefficients(
                container.temperatures, np.array([4.0, 4.0]), 0, coefficients), [[0.5, 0.5], [0.2, 0.2]])
            settings.configure(LIMB_DARKENING_LAW=changed)
            with self.assertRaises(LimbDarkeningError):
                get_component_limbdarkening_cfs(container, ["bolometric"], symmetry_test=False)
            with self.assertRaises(LimbDarkeningError):
                ld.get_bolometric_ld_coefficients(container.temperatures, np.array([4.0, 4.0]), 0, coefficients)

    def test_copy_pickle_and_reconstruction_preserve_provenance(self):
        model = self.binary("logarithmic")
        star = model.primary
        copies = [star.limb_darkening_coefficients.copy(), copy(star.limb_darkening_coefficients),
                  deepcopy(star.limb_darkening_coefficients),
                  pickle.loads(pickle.dumps(star.limb_darkening_coefficients))]
        kwargs = star.kwargs_serializer()
        definition = model.to_json()
        settings.configure(LIMB_DARKENING_LAW="square_root")
        for coefficients in copies:
            self.assertEqual(coefficients.law, "logarithmic")
            with self.assertRaises(LimbDarkeningError):
                StarProperties.limb_darkening_coefficients(coefficients)
        with self.assertRaises(LimbDarkeningError):
            Star(**kwargs)
        with self.assertRaises(LimbDarkeningError):
            BinarySystem.from_json(definition)
        with self.assertRaises(LimbDarkeningError):
            star.init()

    def test_explicit_replacement_and_none_are_supported(self):
        model = self.binary("logarithmic")
        old = model.primary.limb_darkening_coefficients
        settings.configure(LIMB_DARKENING_LAW="square_root")
        supplied = self.coefficients("square_root")
        snapshot = deepcopy(supplied)
        for star in (model.primary, model.secondary):
            star.limb_darkening_coefficients = supplied
            self.assertEqual(star.limb_darkening_coefficients.law, "square_root")
        Observer(system=model, passband=["bolometric"]).lc(phases=SINGLE_PHASE)
        self.assertEqual(supplied, snapshot)
        self.assertEqual(old.law, "logarithmic")
        model.primary.limb_darkening_coefficients = None
        self.assertIsNone(model.primary.limb_darkening_coefficients)

    def test_linear_cosine_alias_and_incompatible_coefficient_count(self):
        model = self.binary("linear")
        observer = Observer(system=model, passband=["bolometric"])
        first = observer.lc(phases=SINGLE_PHASE)
        settings.configure(LIMB_DARKENING_LAW="cosine")
        self.assert_curves_equal(first, observer.lc(phases=SINGLE_PHASE))
        for star in (model.primary, model.secondary):
            star.init()
            self.assertEqual(star.limb_darkening_coefficients.law, "linear")
        settings.configure(LIMB_DARKENING_LAW="logarithmic")
        with self.assertRaises(LimbDarkeningError):
            observer.lc(phases=SINGLE_PHASE)

    def test_single_system_repeated_reuse(self):
        for original, changed in (("logarithmic", "square_root"), ("square_root", "logarithmic")):
            settings.configure(LIMB_DARKENING_LAW=original)
            model = prepare_single_system(SOLAR_MODEL)
            model.star.atmosphere = "bb"
            model.star.limb_darkening_coefficients = self.coefficients(original)
            observer = Observer(system=model, passband=["bolometric"])
            first = observer.lc(phases=SINGLE_PHASE)
            self.assert_curves_equal(first, observer.lc(phases=SINGLE_PHASE))
            settings.configure(LIMB_DARKENING_LAW=changed)
            with self.assertRaises(LimbDarkeningError):
                observer.lc(phases=SINGLE_PHASE)
            with self.assertRaises(LimbDarkeningError):
                observer.rv(phases=SINGLE_PHASE, method="radiometric")
            settings.configure(LIMB_DARKENING_LAW=original)
            self.assert_curves_equal(first, observer.lc(phases=SINGLE_PHASE))

    def test_automatic_interpolation_switches_laws_on_same_model(self):
        settings.configure(LD_TABLES=str(Path(__file__).parent / "data/light_curves/limbdarkening"))
        model = self.binary("logarithmic", custom=False)
        observer = Observer(system=model, passband=["bolometric"])
        curves = []
        for law in ("logarithmic", "square_root", "logarithmic"):
            settings.configure(LIMB_DARKENING_LAW=law)
            current = observer.lc(phases=INTERPOLATION_PHASES)
            reference = Observer(system=self.binary(law, custom=False), passband=["bolometric"]).lc(
                phases=INTERPOLATION_PHASES)
            self.assert_curves_equal(current, reference)
            self.assertIsNone(model.primary.limb_darkening_coefficients)
            curves.append(current)
        self.assert_curves_equal(curves[0], curves[2])
        self.assertFalse(np.array_equal(curves[0][1]["bolometric"], curves[1][1]["bolometric"]))
