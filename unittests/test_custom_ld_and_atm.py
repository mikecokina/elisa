import os.path as op
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal

from elisa import (
    settings,
    get_default_binary_definition
)

from elisa import BinarySystem, Observer
from unittests.utils import ElisaTestCase
from elisa.base.curves.utils import get_component_limbdarkening_cfs

from elisa.base.curves import utils as curves_utils


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
        with self.assertRaises(Exception) as context:
            BinarySystem.from_json(definition)

        length = len(definition['primary']['limb_darkening_coefficients']['bolometric'])
        self.assertTrue(f"however, you provided a vector with {length}" in str(context.exception))

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
        with self.assertRaises(Exception) as context:
            o.lc(phases=[0.0, ])

        self.assertTrue('Please supply limb-darkening factors for [\'TESS\'] '
                        'pasband(s) as well.' in str(context.exception))

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
        curves_utils._interpolate_on_ld_grid_cached.cache_clear()

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
