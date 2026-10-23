# keep it first
# due to stupid astropy units/constants implementation
from tests import set_astropy_units

import numpy as np
import pytest
import os.path as op
from copy import copy

from elisa import umpy as up
from elisa import settings, BinarySystem, ld
from elisa.binary_system.surface import temperature
from elisa.utils import is_empty
from tests import utils as testutils
from tests.utils import ElisaTestCase

set_astropy_units()


class BuildSpotFreeTemperatureTestCase(ElisaTestCase):
    def generator_test_temperatures(self, key, allowed_range=None):

        settings.configure(**{
            "LIMB_DARKENING_LAW": "linear",
            "LD_TABLES": op.join(op.dirname(op.abspath(__file__)), "../data", "light_curves", "limbdarkening")
        })

        s = testutils.prepare_binary_system(testutils.BINARY_SYSTEM_PARAMS[key])
        s.primary.discretization_factor = up.radians(10)
        s.secondary.discretization_factor = up.radians(10)
        orbital_position_container = testutils.prepare_orbital_position_container(s)
        orbital_position_container.build_mesh(components_distance=1.0)
        orbital_position_container.build_faces(components_distance=1.0)
        orbital_position_container.build_surface_areas()
        orbital_position_container.build_surface_gravity(components_distance=1.0)
        orbital_position_container.build_faces_orientation(components_distance=1.0)
        orbital_position_container.build_temperature_distribution(components_distance=1.0)

        if allowed_range:
            obtained_primary = [np.min(orbital_position_container.primary.temperatures),
                                np.max(orbital_position_container.primary.temperatures)]
            obtained_secondary = [np.min(orbital_position_container.secondary.temperatures),
                                  np.max(orbital_position_container.secondary.temperatures)]

            self.assertTrue((obtained_primary[0] >= allowed_range[0][0]) &
                            (obtained_primary[1] <= allowed_range[0][1]))
            self.assertTrue((obtained_secondary[0] >= allowed_range[1][0]) &
                            (obtained_secondary[1] <= allowed_range[1][1]))

    def test_build_temperatures_detached(self):
        with self.assertRaises(Exception) as context:
            self.generator_test_temperatures('detached')
        msg = "Limb darkening interpolation produced numpy.nan/None."
        self.assertTrue(msg in str(context.exception))

    def test_build_temperatures_detached_physical(self):
        self.generator_test_temperatures('detached-physical', [[4998, 5002], [4999, 5004]])

    def test_build_temperatures_over_contact(self):
        self.generator_test_temperatures('over-contact', [[4155, 5406], [4240, 5436]])

    def test_build_temperatures_semi_detached(self):
        self.generator_test_temperatures('semi-detached', [[3760, 5335], [3840, 5450]])


class BuildSpottyTemperatureTestCase(ElisaTestCase):
    def setUp(self):
        super(BuildSpottyTemperatureTestCase, self).setUp()
        settings.configure(**{
            "LIMB_DARKENING_LAW": "linear",
            "LD_TABLES": op.join(op.dirname(op.abspath(__file__)), "../data", "light_curves", "limbdarkening")
        })
    
    def generator_test_temperatures(self, key):
        s = testutils.prepare_binary_system(testutils.BINARY_SYSTEM_PARAMS[key],
                                            spots_primary=testutils.SPOTS_META["primary"],
                                            spots_secondary=testutils.SPOTS_META["secondary"]
                                            )
        s.primary.discretization_factor = up.radians(10)
        s.secondary.discretization_factor = up.radians(10)
        orbital_position_container = testutils.prepare_orbital_position_container(s)
        orbital_position_container.build_mesh(components_distance=1.0)
        orbital_position_container.build_faces(components_distance=1.0)
        orbital_position_container.build_surface_areas()
        orbital_position_container.build_surface_gravity(components_distance=1.0)
        orbital_position_container.build_faces_orientation(components_distance=1.0)
        orbital_position_container.build_temperature_distribution(components_distance=1.0)

        self.assertTrue(not is_empty(orbital_position_container.primary.spots[0].temperatures))
        self.assertTrue(not is_empty(orbital_position_container.secondary.spots[0].temperatures))

    def test_build_temperatures_detached_physical(self):
        self.generator_test_temperatures('detached-physical')

    def test_build_temperatures_over_contact(self):
        self.generator_test_temperatures('over-contact')

    def test_build_temperatures_semi_detached(self):
        self.generator_test_temperatures('semi-detached')


class GravityDarkeningAlbedoTestCase(ElisaTestCase):
    def setUp(self):
        super(GravityDarkeningAlbedoTestCase, self).setUp()
        settings.configure(**{
            "LIMB_DARKENING_LAW": "linear",
            "LD_TABLES": op.join(op.dirname(op.abspath(__file__)), "../data", "light_curves", "limbdarkening")
        })
        self._base_model = {
              "system": {
                "inclination": 90.0,
                "period": 10.1,
                "argument_of_periastron": "90.0 deg",  # string representation of astropy quntity is also valid
                "gamma": 0.0,
                "eccentricity": 0.3,
                "primary_minimum_time": 0.0,
                "phase_shift": 0.0
              },
              "primary": {
                "mass": 2.0,
                "surface_potential": 100,
                "synchronicity": 1.0,
                "t_eff": 6500.0,
                "metallicity": 0.0
              },
              "secondary": {
                "mass": 1.0,
                "surface_potential": 100,
                "synchronicity": 1.0,
                "t_eff": 7500.0,
                "metallicity": 0.0
              }
            }

    def generator_test_betas(self, params, expected):
        s = BinarySystem.from_json(params)
        self.assertAlmostEqual(s.primary.gravity_darkening, expected[0], 5)
        self.assertAlmostEqual(s.secondary.gravity_darkening, expected[1], 5)

    def generator_test_albedo(self, params, expected):
        s = BinarySystem.from_json(params)
        self.assertAlmostEqual(s.primary.albedo, expected[0], 5)
        self.assertAlmostEqual(s.secondary.albedo, expected[1], 5)

    def test_user_defined(self):
        params = copy(self._base_model)
        params['primary'].update({'gravity_darkening': 1.0})
        params['secondary'].update({'gravity_darkening': 1.0})
        params['primary'].update({'albedo': 1.0})
        params['secondary'].update({'albedo': 1.0})

        self.generator_test_betas(params, (1.0, 1.0))
        self.generator_test_albedo(params, (1.0, 1.0))

    def test_auto_primary(self):
        params = copy(self._base_model)
        params['secondary'].update({'gravity_darkening': 1.0})
        params['primary'].update({'albedo': 1.0})

        self.generator_test_betas(params, (0.5302088, 1.0))
        self.generator_test_albedo(params, (1.0, 0.901515))

    def test_auto_both(self):
        self.generator_test_betas(self._base_model, (0.5302088, 0.968017))
        self.generator_test_albedo(self._base_model, (0.7126008, 0.901515))


LAWS = ('linear', 'cosine', 'logarithmic', 'square_root')


def original_factor(mu, coefficients, law):
    """Pre-optimization arithmetic, including both copies for matrix input."""
    coeffs = np.asarray(coefficients, dtype=float)
    mu = np.asarray(mu, dtype=float)
    mu = mu[:, np.newaxis] if mu.ndim == 1 else mu.copy()
    mu = mu.copy()
    neg = mu <= 0
    if law in ('linear', 'cosine'):
        mu[neg] = 0.0
        result = 1.0 - coeffs + coeffs * mu
    elif law == 'logarithmic':
        log_mu = mu.copy()
        mu[neg] = 0.0
        log_mu[neg] = 1.0
        result = 1.0 - coeffs[:, :1] * (1 - mu) - coeffs[:, 1:] * mu * np.log(log_mu)
    else:
        mu[neg] = 0.0
        result = 1.0 - coeffs[:, :1] * (1 - mu) - coeffs[:, 1:] * (1 - np.sqrt(mu))
    result[neg] = 0.0
    return result[:, 0] if result.shape[1] == 1 else result


@pytest.mark.parametrize('law', LAWS)
@pytest.mark.parametrize('shape,reduced', (((17, 11), (4, 3)), ((7, 19), (2, 5)), ((4, 3), (4, 3))))
def test_symmetrical_helper_default_and_reflection_blocks(monkeypatch, law, shape, reduced):
    monkeypatch.setattr(settings, 'LIMB_DARKENING_LAW', law)
    rng = np.random.default_rng(10)
    gamma = {k: rng.uniform(0, 1, shape) for k in ('primary', 'secondary')}
    n_coeff = 1 if law in ('linear', 'cosine') else 2
    ldc = {k: rng.uniform(.1, .4, (n_coeff, n)) for k, n in zip(('primary', 'secondary'), shape)}
    before = {k: v.tobytes() for k, v in gamma.items()}
    full = temperature.get_symmetrical_d_gamma(shape, reduced, ldc, gamma)
    calls = []
    factor = ld.limb_darkening_factor

    def counted(**kwargs):
        calls.append(kwargs['cos_theta'].shape)
        return factor(**kwargs)

    monkeypatch.setattr(ld, 'limb_darkening_factor', counted)
    optimized = temperature.get_symmetrical_d_gamma(shape, reduced, ldc, gamma, reflection_only=True)
    a, b = reduced
    assert calls == [(shape[0], b), (shape[1], a)]
    # All historically populated blocks remain available by default.
    slices = {'primary': [(slice(None), slice(None, b)), (slice(None, a), slice(b, None))],
              'secondary': [(slice(None, a), slice(None)), (slice(a, None), slice(None, b))]}
    for k, blocks in slices.items():
        for rows, cols in blocks:
            angles = gamma[k][rows, cols]
            count = angles.shape[0] if k == 'primary' else angles.shape[1]
            coeff = ldc[k][:, :count].T
            expected = original_factor(angles if k == 'primary' else angles.T, coeff, law)
            expected = expected if k == 'primary' else expected.T
            assert full[k][rows, cols].tobytes() == expected.tobytes()
        rows, cols = blocks[0]
        assert full[k][rows, cols].tobytes() == optimized[k][rows, cols].tobytes()
        assert full[k].shape == optimized[k].shape == shape
        assert gamma[k].tobytes() == before[k]


@pytest.mark.parametrize('iterations', (-1, 0, 1))
def test_disabled_reflection_returns_before_accessing_surface(monkeypatch, iterations):
    monkeypatch.setattr(settings, 'REFLECTION_EFFECT', iterations <= 0)
    system = object()
    assert temperature.apply_reflection_effect(system, 1., iterations) is system


@pytest.mark.parametrize('defect', ('law', 'missing_primary', 'missing_secondary', 'short_primary', 'short_secondary'))
def test_reflection_only_keeps_required_validation(monkeypatch, defect):
    # Deliberately bypass Settings validation to exercise the numerical defense.
    monkeypatch.setitem(settings.__dict__, 'LIMB_DARKENING_LAW', 'invalid' if defect == 'law' else 'cosine')
    gamma = {k: np.ones((7, 9)) for k in ('primary', 'secondary')}
    ldc = {'primary': np.ones((1, 7)), 'secondary': np.ones((1, 9))}
    if defect.startswith('missing_'):
        del ldc[defect.removeprefix('missing_')]
    elif defect.startswith('short_'):
        ldc[defect.removeprefix('short_')] = np.ones((1, 2))
    errors = []
    for reflection_only in (False, True):
        with pytest.raises(Exception) as caught:
            temperature.get_symmetrical_d_gamma((7, 9), (3, 4), ldc, gamma, reflection_only=reflection_only)
        errors.append((type(caught.value), str(caught.value)))
    assert errors[0] == errors[1]
