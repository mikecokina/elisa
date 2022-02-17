import numpy as np
from numpy.testing import assert_array_equal

from elisa import (
    settings,
    get_default_binary_definition
)

from elisa import BinarySystem, Observer
from unittests.utils import ElisaTestCase, prepare_binary_system
from elisa.base.curves.utils import get_component_limbdarkening_cfs


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
        definition["primary"].update({**definition["primary"],
                                      "limb_darkening_coefficients": {'bolometric': [0.5, 0.4]}})
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
        definition = get_default_binary_definition()
        definition["primary"].update({**definition["primary"],
                                      "limb_darkening_coefficients": {'bolometric': 0.5}})

        bs = BinarySystem.from_json(definition)
        o = Observer(passband=["TESS"], system=bs)
        with self.assertRaises(Exception) as context:
            o.lc(phases=[0.0,])

        self.assertTrue('Please supply limb-darkening factors for [\'TESS\'] '
                        'pasband(s) as well.' in str(context.exception))

    def test_raise_missing_bolometric_passband_lds(self):
        definition = get_default_binary_definition()
        definition["primary"].update({**definition["primary"],
                                      "limb_darkening_coefficients": {'TESS': 0.5}})

        bs = BinarySystem.from_json(definition)
        o = Observer(passband=["TESS"], system=bs)
        with self.assertRaises(Exception) as context:
            o.lc(phases=[0.0, ])

        self.assertTrue('Please ad `bolometric` limb-darkening coefficients to '
                        'your custom set of limb-darkening coefficients.' in str(context.exception))

    def test_custold_coeff_distribution(self):
        expected_ldc = [0.5, 0.4]
        passband = 'bolometric'
        settings.configure(**{"LIMB_DARKENING_LAW": 'square_root'})
        definition = get_default_binary_definition()
        definition["primary"].update({**definition["primary"],
                                      "limb_darkening_coefficients": {passband: expected_ldc}})

        bs = BinarySystem.from_json(definition)
        container = bs.build_container(phase=0.0)
        ldcfs = get_component_limbdarkening_cfs(
            container.primary,
            symmetry_test=False,
            passbands=[passband, ]
        )[passband]

        assert_array_equal(np.unique(ldcfs, axis=0)[0], expected_ldc)
