import unittest
from elisa.engine.base.star import Star
from elisa.engine.single_system.system import SingleSystem
from astropy import units as u


class TestPulsations(unittest.TestCase):

    def test_renormalization(self):
        """
        tests if spherical harmonic functions are renormalized in a way that maximum value accros the surface is 1
        :return:
        """
        pulsations_metadata = [{'l': 0, 'm': 0, 'amplitude': 0 * u.K, 'frequency': 15 / u.d}]
        s = Star(mass=1.0 * u.solMass,
                 t_eff= 0 * u.K,
                 gravity_darkening=0.32,
                 discretization_factor=1,
                 pulsations=pulsations_metadata
                 )

        single = SingleSystem(star=s,
                              gamma=0 * u.km / u.s,
                              inclination=90 * u.deg,
                              rotation_period=9.99e10 * u.d,
                              polar_log_g=4.1 * u.dex(u.cm / u.s ** 2))

        single.build_surface()
        single.build_surface_map(colormap='temperature')
        single.star.pulsations[0].amplitude = 1 * u.K

        # tested up to max_l=86
        max_l = 100
        test_l = range(1, max_l)
        for l in test_l:
            single.star.pulsations[0].l = l
            for m in range(l+1):
                single.star.pulsations[0].m = m
                output = max(abs(s.add_pulsations()))
                self.assertLessEqual(output, 1.0, msg='{0} !< 1 for l={1}, m={2}'.format(output, l, m))
                bottom_val = 0.5
                self.assertGreater(output, bottom_val, msg='{0} !> {3} for l={1}, m={2}'.format(output, l, m,
                                                                                                bottom_val))
