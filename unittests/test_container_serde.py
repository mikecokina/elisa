from elisa import (
    umpy as up
)
from elisa.base.container import (
    StarContainer,
    StarPropertiesContainer,
    SystemPropertiesContainer
)
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.binary_system.system import BinarySystem
from elisa.const import BINARY_POSITION_PLACEHOLDER
from unittests import utils as testutils
from unittests.utils import ElisaTestCase


class StarContainerSerDeTestCase(ElisaTestCase):
    props = ['mass', 't_eff', 'synchronicity', 'albedo', 'discretization_factor', 'polar_radius',
             'equatorial_radius', 'gravity_darkening', 'surface_potential', 'pulsations',
             'metallicity', 'polar_log_g', 'critical_surface_potential', 'side_radius']

    def setUp(self):
        self.s = testutils.prepare_binary_system(testutils.BINARY_SYSTEM_PARAMS['detached'])
        self.s.primary.discretization_factor = up.radians(10)
        self.s.secondary.discretization_factor = up.radians(10)

    def test_to_properties_container(self):
        props_container = self.s.primary.to_properties_container()
        self.assertTrue(isinstance(props_container, StarPropertiesContainer))
        has_attr = [hasattr(props_container, atr) for atr in self.props]
        self.assertTrue(all(has_attr))

    def test_from_properties_container(self):
        star = StarContainer.from_properties_container(self.s.primary.to_properties_container())
        self.assertTrue(isinstance(star, StarContainer))
        has_attr = [hasattr(star, atr) for atr in self.props]
        self.assertTrue(all(has_attr))


class OrbitalPositionContainerSerDeTestCase(ElisaTestCase):
    props = ["semi_major_axis", "morphology", "mass_ratio"] + BinarySystem.ALL_KWARGS

    def setUp(self):
        self.s = testutils.prepare_binary_system(testutils.BINARY_SYSTEM_PARAMS['detached'])
        self.s.primary.discretization_factor = up.radians(10)
        self.s.secondary.discretization_factor = up.radians(10)

    def test_to_properties_container(self):
        props_container = self.s.to_properties_container()
        self.assertTrue(isinstance(props_container, SystemPropertiesContainer))

        has_attr = [hasattr(props_container, atr) for atr in self.props]
        self.assertTrue(all(has_attr))

    def test_from_binary_system(self):
        system = OrbitalPositionContainer.from_binary_system(self.s, BINARY_POSITION_PLACEHOLDER(0, 1.0, 0.0, 0.0, 0.0))
        self.assertTrue(isinstance(system, OrbitalPositionContainer))
        self.assertTrue(isinstance(system.primary, StarContainer))
        self.assertTrue(isinstance(system.secondary, StarContainer))

        has_attr = [hasattr(system, atr) for atr in self.props]
        self.assertTrue(all(has_attr))
