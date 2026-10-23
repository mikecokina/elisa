"""Exercise settings contexts and unittest's real setup/cleanup lifecycle."""
import unittest
from pathlib import Path
from unittest.mock import patch

from elisa import settings
from elisa.base.curves.utils import interpolate_on_ld_grid_cached
from unittests.utils import ElisaTestCase


class SettingsOverrideTestCase(ElisaTestCase):
    def test_nested_contexts_restore_complete_effective_configuration(self):
        original = settings.settings_serializer()
        identity = id(settings)
        with settings.override(LIMB_DARKENING_LAW="logarithmic"):
            settings.configure(CK04_ATM_TABLES="outer-ck", MAX_CURVE_DATAPOINTS=17)
            outer = settings.settings_serializer()
            with self.assertRaisesRegex(RuntimeError, "inner"):
                with settings.override(LIMB_DARKENING_LAW="square_root", CK04_ATM_TABLES="inner-ck"):
                    self.assertEqual(settings.ATLAS_TO_BASE_DIR["ck04"], "inner-ck")
                    settings.RV_LAMBDA_INTERVAL = (6000, 6500)
                    raise RuntimeError("inner")
            self.assertEqual(settings.settings_serializer(), outer)
            self.assertEqual(settings.ATLAS_TO_BASE_DIR["ck04"], "outer-ck")
        self.assertEqual(id(settings), identity)
        self.assertEqual(settings.settings_serializer(), original)
        self.assertEqual(settings.ATLAS_TO_BASE_DIR["ck04"], original["CK04_ATM_TABLES"])

    def test_exception_restores_additional_settings_and_logging(self):
        original = settings.settings_serializer()
        with self.assertRaisesRegex(RuntimeError, "body"):
            with settings.override(LOG_CONFIG="fit"):
                settings.configure(NUMBER_OF_THREADS=2, K93_ATM_TABLES="temporary-k")
                raise RuntimeError("body")
        self.assertEqual(settings.settings_serializer(), original)
        self.assertEqual(settings.ATLAS_TO_BASE_DIR["k93"], original["K93_ATM_TABLES"])

    def test_invalid_override_does_not_enter_or_publish(self):
        original = settings.__dict__
        with self.assertRaises(ValueError):
            with settings.override(REFLECTION_EFFECT=False, NUMBER_OF_THREADS=0):
                self.fail("invalid context entered")
        self.assertIs(settings.__dict__, original)


class SettingsIsolationTestCase(ElisaTestCase):
    def test_repeated_reordered_tests_and_failures_restore_canonical_defaults(self):
        expected = settings.settings_serializer()
        observed = []

        class Probe(ElisaTestCase):
            def setUp(self):
                super().setUp()
                observed.append(settings.settings_serializer())
                if self._testMethodName == "setup_error":
                    settings.configure(MAX_CURVE_DATAPOINTS=11, CK04_ATM_TABLES="setup-ck")
                    raise RuntimeError("setup error")

            def mutate(self):
                # Includes settings absent from the INI and the derived atlas map.
                settings.configure(LIMB_DARKENING_LAW="logarithmic", MAX_CURVE_DATAPOINTS=13,
                                   MAGNITUDE_SYSTEM="ab", CK04_ATM_TABLES="test-ck")
                Path(self.CONFIG_FILE).write_text("[physics]\nlimb_darkening_law = square_root\n")

            def body_error(self):
                self.mutate()
                raise RuntimeError("body error")

            def assertion_failure(self):
                self.mutate()
                self.fail("expected failure")

            def setup_error(self):
                self.fail("test should not run")

            def clean(self):
                self.assertEqual(settings.settings_serializer(), expected)

        orders = [
            ["mutate", "body_error", "clean", "setup_error", "assertion_failure", "clean"],
            ["clean", "assertion_failure", "setup_error", "clean", "body_error", "mutate"],
        ]
        for order in orders * 2:
            for name in order:
                # Even the first test must discard a contaminated runtime state.
                settings.configure(MAX_CURVE_DATAPOINTS=99, MESH_GENERATOR="auto")
                result = unittest.TestResult()
                Probe(name).run(result)
                self.assertEqual(len(result.errors), int(name in ("body_error", "setup_error")))
                self.assertEqual(len(result.failures), int(name == "assertion_failure"))
                self.assertEqual(settings.settings_serializer(), expected)
                self.assertEqual(settings.ATLAS_TO_BASE_DIR["ck04"], expected["CK04_ATM_TABLES"])
        self.assertTrue(all(state == expected for state in observed))

    def test_cleanup_registered_before_initial_reset_can_fail(self):
        expected = settings.settings_serializer()

        class Probe(ElisaTestCase):
            attempts = 0

            def reset_test_settings(self):
                self.attempts += 1
                if self.attempts == 1:
                    settings.MAX_CURVE_DATAPOINTS = 9
                    raise RuntimeError("initial reset failed")
                super().reset_test_settings()

            def runTest(self):
                self.fail("test should not run")

        probe = Probe()
        result = unittest.TestResult()
        probe.run(result)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(probe.attempts, 2)
        self.assertEqual(settings.settings_serializer(), expected)

    def test_reset_loads_test_ini_over_defaults(self):
        # Prove the INI is actually parsed; omitted values still use built-ins.
        settings.configure(MAX_CURVE_DATAPOINTS=123, MESH_GENERATOR="auto")
        with patch.object(self, "touch_default_config"):
            Path(self.CONFIG_FILE).write_text("[computational]\nmesh_generator = improved_trapezoidal\n")
            self.reset_test_settings()
        self.assertEqual(settings.MESH_GENERATOR, "improved_trapezoidal")
        self.assertEqual(settings.MAX_CURVE_DATAPOINTS, settings.DEFAULT_SETTINGS["MAX_CURVE_DATAPOINTS"])

    def test_mocked_scientific_results_do_not_survive_cleanup(self):
        from unittests.test_custom_ld_and_atm import TestCurvesUtilsCachingAndBroadcast

        result = unittest.TestResult()
        TestCurvesUtilsCachingAndBroadcast("test_interpolated_ld_cached").run(result)
        self.assertTrue(result.wasSuccessful(), result.errors or result.failures)
        self.assertEqual(interpolate_on_ld_grid_cached.cache_info().currsize, 0)
