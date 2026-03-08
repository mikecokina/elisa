from unittests.utils import ElisaTestCase

from elisa.binary_system import utils as bsutils


class TestComponentToList(ElisaTestCase):
    def test_component_none_returns_empty(self):
        self.assertEqual(bsutils.component_to_list(None), [])

    def test_component_all_returns_both(self):
        self.assertEqual(bsutils.component_to_list("all"), ["primary", "secondary"])
        self.assertEqual(bsutils.component_to_list("both"), ["primary", "secondary"])

    def test_component_single_returns_list(self):
        self.assertEqual(bsutils.component_to_list("primary"), ["primary"])
        self.assertEqual(bsutils.component_to_list("secondary"), ["secondary"])

    def test_component_invalid_raises(self):
        with self.assertRaises(ValueError):
            # noinspection PyTypeChecker
            bsutils.component_to_list("invalid_component")
