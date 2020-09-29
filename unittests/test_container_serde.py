import os.path as op

from elisa import umpy as up, settings
from elisa.base.container import (
    StarContainer,
    StarPropertiesContainer,
    SystemPropertiesContainer
)
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.single_system.container import SystemContainer
from elisa.binary_system.system import BinarySystem
from elisa.const import (
    Position,
    SinglePosition
)
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
        self.s = testutils.prepare_binary_system(testutils.BINARY_SYSTEM_PARAMS['detached-physical'])
        self.s.primary.discretization_factor = up.radians(10)
        self.s.secondary.discretization_factor = up.radians(10)

    def test_to_properties_container(self):
        props_container = self.s.to_properties_container()
        self.assertTrue(isinstance(props_container, SystemPropertiesContainer))

        has_attr = [hasattr(props_container, atr) for atr in self.props]
        self.assertTrue(all(has_attr))

    def test_from_binary_system(self):
        system = OrbitalPositionContainer.from_binary_system(self.s, Position(0, 1.0, 0.0, 0.0, 0.0))
        self.assertTrue(isinstance(system, OrbitalPositionContainer))
        self.assertTrue(isinstance(system.primary, StarContainer))
        self.assertTrue(isinstance(system.secondary, StarContainer))

        has_attr = [hasattr(system, atr) for atr in self.props]
        self.assertTrue(all(has_attr))


class IndempotenceTestCase(ElisaTestCase):
    def setUp(self):
        super(IndempotenceTestCase, self).setUp()
        self.s = testutils.prepare_binary_system(testutils.BINARY_SYSTEM_PARAMS['detached-physical'],
                                                 testutils.SPOTS_META["primary"])
        self.s.primary.discretization_factor = up.radians(10)
        self.single = testutils.prepare_single_system(testutils.SINGLE_SYSTEM_PARAMS['spherical'],
                                                      testutils.SPOTS_META["primary"])
        self.base_path = op.dirname(op.abspath(__file__))
        settings.configure(LD_TABLES=op.join(self.base_path, "data", "light_curves", "limbdarkening"))

    def test_star_container_is_indempotent(self):
        system = OrbitalPositionContainer.from_binary_system(self.s, Position(0, 1.0, 0.0, 0.0, 0.0))
        system.build(components_distance=1.0)
        star = system.primary

        flatt_1 = star.flatt_it()
        flatt_2 = star.flatt_it()
        self.assertTrue(len(flatt_1.points) == len(flatt_2.points))

    def test_orbital_position_container_is_indempotent(self):
        settings.configure(LD_TABLES=op.join(self.base_path, "data", "light_curves", "limbdarkening"))

        system = OrbitalPositionContainer.from_binary_system(self.s, Position(0, 1.0, 0.0, 0.0, 0.0))
        system.build(components_distance=1.0)
        flatt_1 = system.flatt_it()
        flatt_2 = system.flatt_it()
        self.assertTrue(len(flatt_1.primary.points) == len(flatt_2.primary.points))

    def test_single_position_container_is_indempotent(self):
        system = SystemContainer.from_single_system(self.single, SinglePosition(0, 0.0, 0.0))
        system.build()
        flatt_1 = system.flatt_it()
        flatt_2 = system.flatt_it()
        self.assertTrue(len(flatt_1.star.points) == len(flatt_2.star.points))
