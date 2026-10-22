"""Contracts for the two reflection allocation/work reductions."""
from unittests import set_astropy_units

set_astropy_units()

import numpy as np
import pytest

from elisa import ld, settings
from elisa.binary_system.surface import temperature


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
@pytest.mark.parametrize('layout', ('vector', 'matrix', 'transpose', 'strided', 'readonly'))
@pytest.mark.parametrize('dtype', (np.float32, np.float64, np.int64))
def test_factor_bitwise_and_input_ownership(law, layout, dtype):
    values = [-1, 0, 1, 2] if dtype == np.int64 else [-1., -0., 0., .2, 1., np.nan, np.inf, -np.inf]
    mu = np.tile(np.asarray(values, dtype=dtype), (6, 1))
    if layout == 'vector':
        mu = mu[0]
    elif layout == 'transpose':
        mu = mu.T
    elif layout == 'strided':
        mu = mu[::2, ::2]
    elif layout == 'readonly':
        mu.flags.writeable = False
    coeffs = np.tile([.3] if law in ('linear', 'cosine') else [.3, .15], (mu.shape[0], 1))
    before = mu.tobytes()
    coeff_before = coeffs.tobytes()
    with np.errstate(invalid='ignore'):
        expected = original_factor(mu, coeffs, law)
        actual = ld.limb_darkening_factor(cos_theta=mu, coefficients=coeffs, limb_darkening_law=law)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()
    assert mu.tobytes() == before
    assert coeffs.tobytes() == coeff_before
    assert not np.shares_memory(actual, mu)


@pytest.mark.parametrize('law', LAWS)
@pytest.mark.parametrize('shape,reduced', (((17, 11), (4, 3)), ((7, 19), (2, 5)), ((4, 3), (4, 3))))
def test_symmetrical_helper_default_and_reflection_blocks(monkeypatch, law, shape, reduced):
    monkeypatch.setattr(type(settings), 'LIMB_DARKENING_LAW', law)
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


@pytest.mark.parametrize('law', (*LAWS, 'invalid', None))
def test_factor_errors_preserve_inputs(law):
    mu = np.array([[0., -.2], [.3, 1.]])
    before = mu.tobytes()
    from elisa.base.error import LimbDarkeningError
    if law in LAWS:
        with pytest.raises(LimbDarkeningError, match='coefficients were not supplied'):
            ld.limb_darkening_factor(cos_theta=mu, limb_darkening_law=law)
    else:
        with pytest.raises(LimbDarkeningError):
            ld.limb_darkening_factor(cos_theta=mu, coefficients=np.array([[.3], [.4]]), limb_darkening_law=law)
    assert mu.tobytes() == before


@pytest.mark.parametrize('iterations', (-1, 0, 1))
def test_disabled_reflection_returns_before_accessing_surface(monkeypatch, iterations):
    monkeypatch.setattr(type(settings), 'REFLECTION_EFFECT', iterations <= 0)
    system = object()
    assert temperature.apply_reflection_effect(system, 1., iterations) is system


@pytest.mark.parametrize('law', LAWS)
def test_factor_vector_route(law):
    normals = np.array([[0., 1., 0.], [1., 0., 0.], [-1., 0., 0.]])
    los = np.array([1., 0., 0.])
    coeff = np.tile([.4] if law in ('linear', 'cosine') else [.3, .15], (3, 1))
    expected = original_factor(np.sum(normals * los, axis=-1), coeff, law)
    result = ld.limb_darkening_factor(normal_vector=normals, line_of_sight=los,
                                    coefficients=coeff, limb_darkening_law=law)
    assert result.tobytes() == expected.tobytes()


@pytest.mark.parametrize('defect', ('law', 'missing_primary', 'missing_secondary', 'short_primary', 'short_secondary'))
def test_reflection_only_keeps_required_validation(monkeypatch, defect):
    monkeypatch.setattr(type(settings), 'LIMB_DARKENING_LAW', 'invalid' if defect == 'law' else 'cosine')
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
