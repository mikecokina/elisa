"""Scientific parity with the former binary bolometric passband construction."""
from tests import set_astropy_units

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from elisa import BinarySystem, Observer, Star, atm, ld, settings
from elisa.binary_system.curves import utils
from elisa.observer.passband import init_bolometric_passband

set_astropy_units()

BANDS = ['Generic.Bessell.V', 'Generic.Bessell.B']


def original_preparation(system, *, return_values=True, write_to_containers=False, **kwargs):
    """Pre-change implementation, including its unused bandwidth/atlas arguments."""
    ld_cfs = utils.get_limbdarkening_cfs(system, **kwargs)
    normal_radiance = utils._get_normal_radiance(system, **kwargs)
    if 'bolometric' in ld_cfs['primary']:
        bol_ld_cfs = {
            name: {'bolometric': ld_cfs[name]['bolometric']}
            for name in settings.BINARY_COUNTERPARTS
        }
    else:
        passband, left_bandwidth, right_bandwidth = init_bolometric_passband()
        bol_kwargs = {
            'passband': {'bolometric': passband},
            'left_bandwidth': left_bandwidth,
            'right_bandwith': right_bandwidth,
            'atlas': 'whatever',
        }
        bol_ld_cfs = utils.get_limbdarkening_cfs(system, **bol_kwargs)
    normal_radiance = atm.correct_normal_radiance_to_optical_depth(normal_radiance, bol_ld_cfs)
    if write_to_containers:
        for name in settings.BINARY_COUNTERPARTS:
            star = getattr(system, name)
            star.normal_radiance = normal_radiance[name]
            star.ld_cfs = ld_cfs[name]
    if return_values:
        return normal_radiance, ld_cfs
    return None


def make_observer(case, bands, custom, law):
    coefficients = [0.5] if law in ('linear', 'cosine') else [0.5, 0.2]
    common = dict(surface_potential=5., t_eff=7000., gravity_darkening=1.,
                  discretization_factor=10., albedo=0.6, metallicity=0., atmosphere='bb')
    if custom:
        common['limb_darkening_coefficients'] = {band: coefficients for band in [*BANDS, 'bolometric']}
    primary = Star(mass=2., synchronicity=0.8 if 'async' in case else 1., **common)
    secondary = Star(mass=1., synchronicity=1.2 if 'async' in case else 1., **common)
    if 'spot' in case:
        primary.spots = [dict(longitude=90., latitude=58., angular_radius=25., temperature_factor=0.95)]
    if case == 'pulsating':
        primary.pulsations = [dict(l=2, m=1, amplitude=50., frequency=16., start_phase=0.)]
    binary = BinarySystem(primary=primary, secondary=secondary, argument_of_periastron=58.,
                          gamma=0., period=2.5, eccentricity=0.2 if 'ecc' in case else 0.,
                          inclination=85., primary_minimum_time=0., phase_shift=0.)
    return Observer(passband=bands, system=binary)


def capture_science(preparation, observer, normalize):
    """Observe real arrays at both observed-band and bolometric LD stages."""
    arrays = []
    calculate = utils.get_limbdarkening_cfs

    def capture(system, *args, **kwargs):
        result = calculate(system, *args, **kwargs)
        for name, bands in result.items():
            for band, values in bands.items():
                assert values.shape[0] == len(getattr(system, name).temperatures)
                arrays.append((name, band, values.copy()))
        return result

    ld.clear_ld_caches()
    with patch.object(utils, 'prep_surface_params', preparation), \
         patch.object(utils, 'get_limbdarkening_cfs', capture):
        phases, flux = observer.observe.lc(phases=np.array([0., .13, .27, .49, .71]), normalize=normalize)
    return phases, flux, arrays


@pytest.mark.parametrize('law', ['linear', 'cosine', 'logarithmic', 'square_root'])
@pytest.mark.parametrize('custom', [False, True], ids=['tables', 'custom'])
@pytest.mark.parametrize('case,bands', [
    ('circular', BANDS),
    ('circular', ['bolometric', *BANDS]),
    ('circular_async_spot', BANDS),
    ('eccentric', BANDS),
    ('eccentric_async_spot', BANDS),
    ('pulsating', BANDS),
])
def test_bolometric_scientific_parity(law, custom, case, bands):
    directory = Path.home() / '.elisa/limbdarkening/ld'
    prefix = {'linear': 'lin', 'cosine': 'lin', 'logarithmic': 'log', 'square_root': 'sqrt'}[law]
    if not custom and not all((directory / f'{prefix}.{band}.p00.csv').is_file()
                              for band in [*BANDS, 'bolometric']):
        pytest.skip('Installed Van Hamme tables required')
    with settings.override(LD_TABLES=str(directory), LIMB_DARKENING_LAW=law,
                           NUMBER_OF_PROCESSES=1, USE_SINGLE_LD_COEFFICIENTS=False):
        try:
            for normalize in (False, True):
                before = capture_science(original_preparation, make_observer(case, bands, custom, law), normalize)
                after = capture_science(utils.prep_surface_params, make_observer(case, bands, custom, law), normalize)
                assert_array_equal(before[0], after[0], strict=True)
                assert list(before[1]) == list(after[1]) == bands
                for band in bands:
                    assert_array_equal(before[1][band], after[1][band], strict=True)
                assert len(before[2]) == len(after[2]) > 0
                for (name, band, expected), (actual_name, actual_band, actual) in zip(before[2], after[2]):
                    assert (name, band) == (actual_name, actual_band)
                    assert_array_equal(expected, actual, strict=True)
        finally:
            ld.clear_ld_caches()


@pytest.mark.parametrize('bands', [BANDS, ['bolometric', *BANDS]])
def test_surface_preparation_does_not_construct_passbands(bands):
    with settings.override(LIMB_DARKENING_LAW='cosine', NUMBER_OF_PROCESSES=1):
        observer = make_observer('circular', bands, True, 'cosine')
        # Observer initialization above must still construct actual passband containers.
        with patch('elisa.observer.passband.PassbandContainer.__init__',
                   side_effect=AssertionError('Unexpected passband construction during surface preparation')):
            _, flux = observer.observe.lc(phases=np.array([0., .25]), normalize=False)
        assert list(flux) == bands
        assert all(np.isfinite(values).all() for values in flux.values())
