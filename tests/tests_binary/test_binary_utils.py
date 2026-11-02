from tests.utils import ElisaTestCase

from elisa.binary_system import utils as bsutils
from elisa.base.error import ElisaOperationInputError, ElisaUnexpectedError


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
        with self.assertRaises(ElisaOperationInputError):
            # noinspection PyTypeChecker
            bsutils.component_to_list("invalid_component")


class TestJsonKindContract(ElisaTestCase):
    def test_incomplete_parameter_pair_is_unexpected(self):
        for use_asini, separation in [(False, "semi_major_axis"), (True, "asini")]:
            with self.subTest(separation=separation):
                data = {"system": {separation: 1.0}, "primary": {"mass": 1.0}}
                with self.assertRaises(ElisaUnexpectedError) as caught:
                    bsutils.resolve_json_kind(data, _sin=use_asini)
                message = str(caught.exception)
                self.assertIn(f"Found parameters: primary.mass, system.{separation}.", message)
                self.assertIn("missing secondary.mass", message)
                self.assertIn("missing system.mass_ratio", message)
