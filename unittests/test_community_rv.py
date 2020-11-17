import numpy as np

from elisa import (
    umpy as up,
    units as u,
    const
)
from elisa.binary_system.curves import rv
from elisa.binary_system.curves.community import RadialVelocitySystem
from elisa.observer.observer import Observer
from unittests.utils import (
    ElisaTestCase,
    prepare_binary_system,
    BINARY_SYSTEM_PARAMS
)

TOL = 5e-3


class RadialVelocityObserverTestCase(ElisaTestCase):
    def setUp(self):
        super(RadialVelocityObserverTestCase, self).setUp()
        self.phases = up.arange(-0.2, 1.25, 0.05)

    def test_all_init_values_in_expected_units(self):
        init_kwargs = dict(
            eccentricity=0.1,
            argument_of_periastron=90 * u.deg,
            period=10.0,
            mass_ratio=0.5,
            asini=9 * u.solRad,
            gamma=11 * u.m / u.s
        )

        expected = np.round([0.1, const.HALF_PI, 10.0, 0.5, 9.0, 11], 4)
        o = RadialVelocitySystem(**init_kwargs)
        obtained = np.round([[o.eccentricity, o.argument_of_periastron,
                              o.period, o.mass_ratio, o.asini, o.gamma]], 4)
        self.assertTrue(np.all(np.abs(expected - obtained)) < TOL)

    def test_rvs_from_binary_system_instance_are_same(self):
        s = prepare_binary_system(BINARY_SYSTEM_PARAMS["detached.ecc"])
        s.inclination = 1.1
        s.init()
        std_rvdict = rv.com_radial_velocity(s, position_method=s.calculate_orbital_motion, phases=self.phases)

        asini = np.float64((s.semi_major_axis * np.sin(s.inclination) * u.m).to(u.solRad))

        rv_system = RadialVelocitySystem(eccentricity=s.eccentricity,
                                         argument_of_periastron=np.degrees(s.argument_of_periastron),
                                         period=s.period,
                                         mass_ratio=s.mass_ratio,
                                         asini=asini,
                                         gamma=s.gamma)
        o = Observer(passband='bolometric', system=rv_system)

        phases, com_rv_dict = o.observe.rv(phases=self.phases, method='point_mass')

        self.assertTrue(np.all(np.abs(std_rvdict['primary'] - com_rv_dict['primary']) < TOL))
        self.assertTrue(np.all(np.abs(std_rvdict['secondary'] - com_rv_dict['secondary']) < TOL))
