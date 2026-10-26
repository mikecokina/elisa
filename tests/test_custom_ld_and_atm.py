import os.path as op
import pickle
from copy import copy, deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal

from elisa import (
    settings,
    get_default_binary_definition
)

from elisa import BinarySystem, Observer
from elisa.base.curves.utils import get_component_limbdarkening_cfs

from elisa.base.curves import utils as curves_utils
from elisa import ld
from elisa.base.container import StarContainer
from elisa.base.error import ElisaValueError, LimbDarkeningError
from elisa.base.star import Star
from elisa.base.transform import StarProperties
from tests.utils import ElisaTestCase, prepare_single_system, SOLAR_MODEL


class DummyComponent:
    """Minimal stand-in for a StarContainer used in curves utils tests.

    Attributes are intentionally minimal and match names accessed by the
    functions under test.
    """

    def __init__(self, temperatures, log_g, metallicity, limb_darkening_coefficients=None):
        self.temperatures = np.asarray(temperatures)
        self.log_g = np.asarray(log_g)
        self.metallicity = metallicity
        self.limb_darkening_coefficients = limb_darkening_coefficients
        # surfaces-related arrays: use one value per face
        faces = self.temperatures.shape[0]
        self.indices = np.arange(faces)
        # Normal radiance and ld_cfs are dicts keyed by band name and have per-face arrays
        self.normal_radiance = {"bolometric": np.ones(faces)}
        self.ld_cfs = {"bolometric": np.ones((faces, 2))}
        self.los_cosines = np.ones(faces)
        self.coverage = np.ones(faces)
        # t_eff and log_g presence for other helpers
        self.t_eff = float(self.temperatures[0]) if self.temperatures.size else 5000.0

    # noinspection PyMethodMayBeStatic
    def symmetry_faces(self, arr):
        # Identity for tests which don't rely on symmetry transform
        return arr

    # noinspection PyMethodMayBeStatic
    def mirror_face_values(self, vals):
        # For testing, return vals unchanged; real implementation mirrors values
        return vals


class BinarySystemSeparatedAtmospheres(ElisaTestCase):
    def setUp(self):
        super(BinarySystemSeparatedAtmospheres, self).setUp()

    @staticmethod
    def test_atmospheres_of_components_differs():
        definition = get_default_binary_definition()
        definition["primary"].update({**definition["primary"], "atmosphere": "bb"})
        definition["secondary"].update({**definition["secondary"], "atmosphere": "ck04"})
        binary = BinarySystem.from_json(definition)
        assert binary.primary.atmosphere == "bb"
        assert binary.secondary.atmosphere == "ck04"

    @staticmethod
    def test_custom_lds_linear():
        definition = get_default_binary_definition()
        definition["primary"].update({**definition["primary"], "limb_darkening_coefficients": {'bolometric': 0.5}})
        binary = BinarySystem.from_json(definition)
        assert_array_equal(binary.primary.limb_darkening_coefficients['bolometric'], [0.5, ])

    @staticmethod
    def test_custom_lds_log():
        settings.configure(**{"LIMB_DARKENING_LAW": 'logarithmic'})
        definition = get_default_binary_definition()
        definition["primary"].update({**definition["primary"],
                                      "limb_darkening_coefficients": {'bolometric': [0.5, 0.4]}})
        binary = BinarySystem.from_json(definition)
        assert_array_equal(binary.primary.limb_darkening_coefficients['bolometric'], [0.5, 0.4])

    @staticmethod
    def test_custom_lds_sqrt():
        settings.configure(**{"LIMB_DARKENING_LAW": 'square_root'})
        definition = get_default_binary_definition()
        definition["primary"].update(
            {**definition["primary"],
             "limb_darkening_coefficients": {'bolometric': [0.5, 0.4]}}
        )
        binary = BinarySystem.from_json(definition)
        assert_array_equal(binary.primary.limb_darkening_coefficients['bolometric'], [0.5, 0.4])

    def test_raise_custom_lds_mismatch(self):
        settings.configure(**{"LIMB_DARKENING_LAW": 'square_root'})
        definition = get_default_binary_definition()
        definition["primary"].update({**definition["primary"],
                                      "limb_darkening_coefficients": {'bolometric': [0.5, ]}})
        with self.assertRaises(ElisaValueError) as context:
            BinarySystem.from_json(definition)

        length = len(definition['primary']['limb_darkening_coefficients']['bolometric'])
        self.assertTrue(f"You provided a vector with {length}" in str(context.exception))

    def test_raise_missing_passband_lds(self):
        self.lc_base_path = op.join(op.dirname(op.abspath(__file__)), "data", "light_curves")
        settings.configure(**{
            "LD_TABLES": op.join(self.lc_base_path, "limbdarkening"),
            "CK04_ATM_TABLES": op.join(self.lc_base_path, "atmosphere")
        })
        self.write_default_support(ld_tables=settings.LD_TABLES, atm_tables=settings.CK04_ATM_TABLES)

        definition = get_default_binary_definition()
        definition["primary"].update({**definition["primary"],
                                      "limb_darkening_coefficients": {'bolometric': 0.5}})

        bs = BinarySystem.from_json(definition)
        o = Observer(passband=["TESS"], system=bs)
        with self.assertRaises(KeyError) as context:
            o.lc(phases=[0.0, ])

        self.assertIn('TESS', str(context.exception))
        self.assertIn('limb-darkening factors', str(context.exception))
        self.assertIsInstance(context.exception.__cause__, KeyError)
        self.assertEqual(context.exception.__cause__.args, ('TESS',))

    def test_raise_missing_bolometric_passband_lds(self):
        definition = get_default_binary_definition()
        definition["primary"].update(
            {**definition["primary"], "limb_darkening_coefficients": {"TESS": 0.5}}
        )

        bs = BinarySystem.from_json(definition)
        o = Observer(passband=["TESS"], system=bs)

        with self.assertRaises(ValueError) as context:
            o.lc(phases=[0.0])

        self.assertIn(
            "Please add `bolometric` limb-darkening coefficients to your custom set "
            "of limb-darkening coefficients.",
            str(context.exception),
        )

    # noinspection PyMethodMayBeStatic
    def test_custom_ld_coeff_distribution(self):
        self.lc_base_path = op.join(op.dirname(op.abspath(__file__)), "data", "light_curves")
        settings.configure(**{
            "LD_TABLES": op.join(self.lc_base_path, "limbdarkening"),
            "CK04_ATM_TABLES": op.join(self.lc_base_path, "atmosphere")
        })
        self.write_default_support(ld_tables=settings.LD_TABLES, atm_tables=settings.CK04_ATM_TABLES)

        expected_ldc = [0.5, 0.4]
        passband = 'bolometric'
        settings.configure(**{"LIMB_DARKENING_LAW": 'square_root'})
        definition = get_default_binary_definition()
        definition["primary"].update(
            {**definition["primary"],
             "limb_darkening_coefficients": {passband: expected_ldc}}
        )

        bs = BinarySystem.from_json(definition)
        container = bs.build_container(phase=0.0)
        ldcfs = get_component_limbdarkening_cfs(
            container.primary,
            passbands=[passband, ],
            symmetry_test=False,
        )[passband]

        assert_array_equal(np.unique(ldcfs, axis=0)[0], expected_ldc)


class TestCurvesUtilsCachingAndBroadcast(ElisaTestCase):
    def test_explicit_coeffs_broadcast(self):
        """Explicit 1-D LD coefficients should be broadcast to (faces, coeffs)."""
        comp = DummyComponent(
            temperatures=[5000.0, 5000.0, 5000.0],
            log_g=[4.0, 4.0, 4.0],
            metallicity=0.0,
            limb_darkening_coefficients={"bolometric": np.array([0.5, 0.4])}
        )

        # noinspection PyTypeChecker
        ld_cfs = curves_utils.get_component_limbdarkening_cfs(comp, passbands=["bolometric"], symmetry_test=False)
        arr = ld_cfs["bolometric"]
        # Expect shape (faces, coeff_count)
        assert arr.shape == (3, 2)
        # Every row equals the provided coefficients
        assert_array_equal(arr, np.tile(np.array([0.5, 0.4]), (3, 1)))

    def test_interpolated_ld_cached(self):
        """When LD interpolation is used, results must be cached so the
        underlying interpolation function is invoked only once for identical
        inputs.
        """
        # Ensure caching is cleared at test start
        curves_utils.interpolate_on_ld_grid_cached.cache_clear()

        # Create a component without explicit LD coefficients so interpolation is used
        comp = DummyComponent(temperatures=[5500.0, 5500.0], log_g=[4.0, 4.0], metallicity=0.0,
                              limb_darkening_coefficients=None)

        # Force predictable behaviour: ensure single-coeff mode is off and restore later
        old_single = getattr(settings, "USE_SINGLE_LD_COEFFICIENTS", False)
        settings.USE_SINGLE_LD_COEFFICIENTS = False
        try:
            # Patch the underlying interpolation to a synthetic deterministic value
            with patch.object(curves_utils.ld, "interpolate_on_ld_grid",
                              return_value={"bolometric": np.array([[0.1, 0.2]])}) as mock_interp:
                # First call should invoke the interpolation
                # noinspection PyTypeChecker
                _ = curves_utils.get_component_limbdarkening_cfs(comp, passbands=["bolometric"], symmetry_test=False)
                # Second call with identical inputs should hit the cache and not call interpolation again
                # noinspection PyTypeChecker
                _ = curves_utils.get_component_limbdarkening_cfs(comp, passbands=["bolometric"], symmetry_test=False)

                self.assertEqual(mock_interp.call_count, 1,
                                 "LD interpolation should be called only once due to caching")
        finally:
            settings.USE_SINGLE_LD_COEFFICIENTS = old_single


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
