# keep it first
# due to stupid astropy units/constants implementation
from unittests import set_astropy_units

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import griddata
from scipy.spatial import QhullError
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from elisa import const, ld, utils
from elisa import settings
from elisa.base.curves import utils as curve_utils
from elisa.base.error import LimbDarkeningError
from unittests.utils import ElisaTestCase

set_astropy_units()


class TestLimbDarkeningModule(ElisaTestCase):
    def setUp(self):
        super(TestLimbDarkeningModule, self).setUp()
        self.base_path = os.path.dirname(os.path.abspath(__file__))

    def test_get_metallicity_from_ld_table_filename(self):
        filenames = ["lin.bolometric.m02.csv", "lin.bolometric.p00.csv",
                     "lin.Generic.Bessell.B.p10.csv", os.path.join("path.path.csv", "lin.Generic.Bessell.B.p10.csv")]
        expected = [-0.2, 0.0, 1.0, 1.0]
        obtained = [ld.get_metallicity_from_ld_table_filename(filename) for filename in filenames]
        assert_array_equal(expected, obtained)

    def test_get_ld_table_filename(self):
        expected = ["lin.Generic.Bessell.B.m01.csv",
                    "lin.Generic.Stromgren.v.m50.csv",
                    "lin.Kepler.p01.csv",
                    "sqrt.Generic.Bessell.B.p10.csv"]
        params = [
            dict(passband="Generic.Bessell.B", metallicity=-0.1, law="linear"),
            dict(passband="Generic.Stromgren.v", metallicity=-5, law="linear"),
            dict(passband="Kepler", metallicity=0.1, law="linear"),
            dict(passband="Generic.Bessell.B", metallicity=1.0, law="square_root")
        ]
        obtained = [ld.get_ld_table_filename(**param) for param in params]
        assert_array_equal(expected, obtained)

    def expected_ld_tables(self):
        path = os.path.join(self.base_path, "data", "vh93")
        # noinspection PyArgumentList
        return [pd.read_csv(os.path.join(path, file)) for file in sorted(os.listdir(path))]

    def test_get_ld_table(self):
        settings.configure(LD_TABLES=os.path.join(self.base_path, "data", "vh93"))
        params = [
            dict(passband="Generic.Bessell.B", metallicity=-0.2, law="cosine"),
            dict(passband="Kepler", metallicity=-4, law="logarithmic")
        ]

        expected = self.expected_ld_tables()
        obtained = [ld.get_ld_table(**param) for param in params]

        for e, o in zip(expected, obtained):
            assert_frame_equal(e, o, rtol=1e-5, check_dtype=False, check_exact=True)

    def test_get_ld_table_by_name(self):
        settings.configure(**{"LD_TABLES": os.path.join(self.base_path, "data", "vh93")})
        filenames = ["lin.Generic.Bessell.B.m02.csv", "log.Kepler.m40.csv"]

        expected = self.expected_ld_tables()
        obtained = [ld.get_ld_table_by_name(file) for file in filenames]

        for e, o in zip(expected, obtained):
            assert_frame_equal(e, o, rtol=1e-5, check_dtype=False, check_exact=True)

    def test_get_ld_table_by_name_same_as_get_ld_table(self):
        settings.configure(**{"LD_TABLES": os.path.join(self.base_path, "data", "vh93")})
        params = [
            dict(passband="Generic.Bessell.B", metallicity=-0.2, law="cosine"),
            dict(passband="Kepler", metallicity=-4, law="logarithmic")
        ]
        filenames = ["lin.Generic.Bessell.B.m02.csv", "log.Kepler.m40.csv"]

        obtained_by_params = [ld.get_ld_table(**param) for param in params]
        obtained_by_filename = [ld.get_ld_table_by_name(file) for file in filenames]

        for o1, o2 in zip(obtained_by_filename, obtained_by_params):
            assert_frame_equal(o1, o2)

    def test_get_relevant_ld_tables(self):
        params = [
            dict(metallicity=0.0, passband="Kepler", law="cosine"),
            dict(metallicity=-0.01, passband="Generic.Bessell.B", law="linear"),
            dict(metallicity=0.02, passband="SLOAN.SDSS.i", law="logarithmic"),
            dict(metallicity=1.0, passband="SLOAN.SDSS.i", law="square_root")
        ]
        obtained = [ld.get_relevant_ld_tables(**param) for param in params]
        expected = [
            ['lin.Kepler.p00.csv', 'lin.Kepler.p00.csv'],
            ['lin.Generic.Bessell.B.m01.csv', 'lin.Generic.Bessell.B.p00.csv'],
            ['log.SLOAN.SDSS.i.p00.csv', 'log.SLOAN.SDSS.i.p01.csv'],
            ['sqrt.SLOAN.SDSS.i.p10.csv', 'sqrt.SLOAN.SDSS.i.p10.csv']
        ]
        assert_array_equal(obtained, expected)

    def test_interpolate_on_ld_grid_lin(self):
        settings.configure(**{"LIMB_DARKENING_LAW": "cosine",
                              "LD_TABLES": os.path.join(self.base_path, "data", "ld_grid")})
        temperature = np.array([5500, 5561, 5582, 5932])
        log_g = np.array([4.2, 4.21, 4.223, 4.199]) - 2.0
        metallicity = -0.1
        passband = ["Generic.Bessell.B"]

        # Direct legacy griddata gives 0.708598 for the final query on this fixture.
        expected = np.array([0.7732, 0.76364, 0.76032, 0.70860])

        obtained = ld.interpolate_on_ld_grid(temperature, log_g, metallicity, passband)
        ld_column = settings.LD_LAW_CFS_COLUMNS[settings.LIMB_DARKENING_LAW].index('xlin')
        obtained = np.round(obtained["Generic.Bessell.B"][:, ld_column], 5)

        np.testing.assert_allclose(obtained, expected, rtol=0, atol=1e-5)

    def test_interpolate_on_ld_grid_log(self):
        settings.configure(**{"LIMB_DARKENING_LAW": "logarithmic",
                              "LD_TABLES": os.path.join(self.base_path, "data", "light_curves", "limbdarkening")})
        temperature = np.array([5500, 5561, 5582, 5932])
        log_g = np.array([4.2, 4.21, 4.223, 4.199]) - 2.0
        metallicity = 0.0
        passband = ["Generic.Bessell.V"]

        expected_xlog = np.round(np.array([0.776400, 0.772272, 0.770870, 0.747420]), 5)
        expected_ylog = np.round(np.array([0.204200, 0.211724, 0.214276, 0.251288]), 5)

        obtained = ld.interpolate_on_ld_grid(temperature, log_g, metallicity, passband)
        ld_column_x = settings.LD_LAW_CFS_COLUMNS[settings.LIMB_DARKENING_LAW].index('xlog')
        ld_column_y = settings.LD_LAW_CFS_COLUMNS[settings.LIMB_DARKENING_LAW].index('ylog')
        obtained_xlog = np.round(obtained["Generic.Bessell.V"][:, ld_column_x], 5)
        obtained_ylog = np.round(obtained["Generic.Bessell.V"][:, ld_column_y], 5)

        self.assertTrue(np.all(expected_ylog == obtained_ylog))
        self.assertTrue(np.all(expected_xlog == obtained_xlog))

    def _raise_test_limb_darkening_factor(self, msg, **kwargs):
        with self.assertRaises(Exception) as context:
            ld.limb_darkening_factor(**kwargs)
        self.assertTrue(msg in str(context.exception))

    def test_limb_darkening_factor_raise(self):
        normals = np.array([[1, 1, 1], [0.5, 0.5, 0.5], [-1., -1, 1.3]])
        sov = const.LINE_OF_SIGHT

        # :warning: order of tests depend on order of check in ld.limb_darkening_factor method!!!

        # missing line_of_sight
        msg = 'Line of sight vector(s) was not supplied.'
        self._raise_test_limb_darkening_factor(msg, normal_vector=normals)

        # missing normal_vector
        msg = 'Normal vector(s) was not supplied.'
        self._raise_test_limb_darkening_factor(msg, line_of_sight=sov)

        # missing coefficients
        msg = 'Limb darkening coefficients were not supplied.'
        self._raise_test_limb_darkening_factor(msg, line_of_sight=sov, normal_vector=normals)

        # missing limb_darkening_law
        msg = 'Limb darkening rule was not supplied choose from'
        self._raise_test_limb_darkening_factor(msg, line_of_sight=sov, normal_vector=normals, coefficients=3)

    def test_limb_darkening_factor_linear(self):
        ld_law = 'linear'
        coefficients = np.array([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]])
        cos_theta = np.array([[-0.23], [0.25], [0.85], [0.55], [0.33], [1.0]])

        expected = np.array([0., 0.85, 0.955, 0.82, 0.665, 1.])
        obtained = ld.limb_darkening_factor(coefficients=coefficients,
                                            limb_darkening_law=ld_law,
                                            cos_theta=cos_theta)

        self.assertTrue(np.all(expected == np.round(obtained, 3)))

    def test_limb_darkening_factor_logarithmic(self):
        ld_law = 'logarithmic'
        coefficients = np.array([[0.1, 0.9],
                                 [0.2, 0.8],
                                 [0.3, 0.7],
                                 [0.4, 0.6],
                                 [0.5, 0.5],
                                 [0.2, 0.3]])
        cos_theta = np.array([[-0.23], [0.25], [0.85], [0.55], [0.33], [1.0]])

        obtained = ld.limb_darkening_factor(coefficients=coefficients,
                                            limb_darkening_law=ld_law,
                                            cos_theta=cos_theta)

        expected = np.array([0., 1.127, 1.052, 1.017, 0.848, 1.])
        self.assertTrue(np.all(expected == np.round(obtained, 3)))


# Synthetic tables keep cache and boundary regressions independent of installed assets.


@pytest.fixture(name="ld_sources")
def synthetic_ld_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'LD_TABLES', str(tmp_path))
    monkeypatch.setattr(settings, 'LIMB_DARKENING_LAW', 'cosine')
    monkeypatch.setattr(settings, 'USE_SINGLE_LD_COEFFICIENTS', False)
    points = np.array([[5000., 3.], [6000., 3.], [5000., 4.], [6000., 4.]])
    for law in ('cosine', 'logarithmic', 'square_root'):
        for band in ('Generic.Bessell.U', 'bolometric', 'Generic.Bessell.V'):
            for metal in (0., .1):
                values = {name: (i + 1) * (points[:, 0] / 10000 + points[:, 1] / 10) + metal
                          for i, name in enumerate(settings.LD_LAW_CFS_COLUMNS[law])}
                frame = pd.DataFrame(dict(temperature=points[:, 0], gravity=points[:, 1], **values))
                # Include an identical full-row duplicate; intermediate metals conflict at coordinates.
                pd.concat([frame, frame.iloc[:1]]).to_csv(
                    tmp_path / ld.get_ld_table_filename(band, metal, law), index=False)
    curve_utils.interpolate_on_ld_grid_cached.cache_clear()
    from elisa.buffer import buffer
    buffer.LD_CFS_TABLES.clear()
    yield tmp_path
    if hasattr(ld, 'clear_ld_caches'):
        ld.clear_ld_caches()
    curve_utils.interpolate_on_ld_grid_cached.cache_clear()
    buffer.LD_CFS_TABLES.clear()


def legacy_source(band, metal):
    cols = settings.LD_LAW_COLS_ORDER[settings.LIMB_DARKENING_LAW]
    frame = pd.concat([pd.read_csv(Path(settings.LD_TABLES) / name)[cols]
                       for name in ld.get_relevant_ld_tables(band, metal)], ignore_index=True).drop_duplicates()
    return (frame[settings.LD_DOMAIN_COLS].to_numpy(),
            frame[settings.LD_LAW_CFS_COLUMNS[settings.LIMB_DARKENING_LAW]].to_numpy())


@pytest.mark.parametrize('law', ['linear', 'cosine', 'logarithmic', 'square_root'])
@pytest.mark.parametrize('metal', [0., .05])
@pytest.mark.parametrize('band', ['Generic.Bessell.U', 'bolometric', 'Generic.Bessell.V'])
def test_unrounded_griddata_equivalence(ld_sources, monkeypatch, law, metal, band):
    monkeypatch.setattr(settings, 'LIMB_DARKENING_LAW', law)
    points, values = legacy_source(band, metal)
    queries = np.array([[5000., 3.], [6000., 4.], [5500., 3.], [5000., 3.5],
                        [5523.123456789123, 3.456789123456], [5500., 3.5], [5500., 3.5],
                        [5000.000001, 3.5]])
    expected = griddata(points, values, queries, method='linear')
    actual = ld.interpolate_on_ld_grid(queries[:, 0], queries[:, 1] - 2, metal, [band])[band]
    assert_array_equal(actual, expected)
    if metal == 0.:
        # Independent affine-field oracle (not merely another interpolation implementation).
        np.testing.assert_allclose(actual[:, 0], queries[:, 0] / 10000 + queries[:, 1] / 10,
                                   rtol=0, atol=4e-16)
    alias = ld.interpolate_on_ld_grid(queries[:, 0], queries[:, 1] - 2, metal, {'rv_band': None})
    assert alias['rv_band'].shape == (len(queries), values.shape[1])


@pytest.mark.parametrize('defect, error', [('outside', LimbDarkeningError), ('query_nan', LimbDarkeningError),
    ('value_nan', LimbDarkeningError), ('domain_nan', ValueError), ('degenerate', QhullError),
    ('missing_column', KeyError), ('missing_file', FileNotFoundError)])
def test_interpolation_errors(ld_sources, defect, error):
    path = ld_sources / ld.get_ld_table_filename('bolometric', 0.)
    frame = pd.read_csv(path)
    query = [5500., 1.5]
    if defect == 'outside':
        query[0] = 4999.999999
    elif defect == 'query_nan':
        query[0] = np.nan
    elif defect == 'value_nan':
        frame['xlin'] = np.nan
    elif defect == 'domain_nan':
        frame.loc[0, 'temperature'] = np.nan
    elif defect == 'degenerate':
        frame['gravity'] = 3.
    elif defect == 'missing_column':
        frame = frame.drop(columns='xlin')
    if defect == 'missing_file':
        path.unlink()
    else:
        frame.to_csv(path, index=False)
    with pytest.raises(error):
        ld.interpolate_on_ld_grid(np.array([query[0]]), np.array([query[1]]), 0., ['bolometric'])


def component(temperature=5500.1234567891):
    return SimpleNamespace(temperatures=np.array([temperature]), log_g=np.array([1.543210987654]),
                           metallicity=0., limb_darkening_coefficients=None)


def evaluate(comp):
    return curve_utils.get_component_limbdarkening_cfs(comp, ['bolometric'], symmetry_test=False)['bolometric']


def test_exact_keys_and_output_ownership(ld_sources):
    a, b = component(), component(5500.1234567892)
    with patch.object(ld, 'interpolate_on_ld_grid', wraps=ld.interpolate_on_ld_grid) as call:
        first = evaluate(a)
        second = evaluate(b)
        assert call.call_count == 2
        assert_array_equal(call.call_args.kwargs['temperature'], b.temperatures)
        points, values = legacy_source('bolometric', 0.)
        assert_array_equal(second, griddata(points, values, np.column_stack((b.temperatures, b.log_g + 2)),
                                            method='linear'))
        c = component()
        c.log_g += 1e-10
        evaluate(c)
        assert call.call_count == 3
        assert_array_equal(call.call_args.kwargs['log_g'], c.log_g)
        first[:] = -999
        assert evaluate(a)[0, 0] > 0
        assert call.call_count == 3
    key = utils.float64_array_to_cache_key
    assert key(np.array([1], dtype='f4')) == key(np.array([1], dtype='f8'))
    assert key(np.array([1., 2.])) != key(np.array([[1., 2.]]))
    assert key(np.array([0.])) != key(np.array([-0.]))
    assert key(np.array([np.nan])) == key(np.array([np.nan]))


def test_source_reuse_and_invalidation(ld_sources, monkeypatch, tmp_path):
    with patch.object(ld.interpolate, 'LinearNDInterpolator', wraps=ld.interpolate.LinearNDInterpolator) as build:
        original = evaluate(component()).copy()
        evaluate(component(5555.))
        assert build.call_count == 1
        # A same-sized rewrite, even with restored mtime, must invalidate both layers.
        path = ld_sources / ld.get_ld_table_filename('bolometric', 0.)
        stat = path.stat()
        frame = pd.read_csv(path)
        frame['xlin'] += 1
        frame.to_csv(path, index=False)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert_array_equal(evaluate(component()), original + 1)
        assert build.call_count == 2
        monkeypatch.setattr(settings, 'LIMB_DARKENING_LAW', 'logarithmic')
        assert evaluate(component()).shape == (1, 2)
        assert build.call_count == 3
        other = ld_sources / 'other'
        other.mkdir()
        frame['xlin'] += 1
        frame.to_csv(other / ld.get_ld_table_filename('bolometric', 0., 'cosine'), index=False)
        monkeypatch.setattr(settings, 'LIMB_DARKENING_LAW', 'cosine')
        monkeypatch.setattr(settings, 'LD_TABLES', str(other))
        assert_array_equal(evaluate(component()), original + 2)
        assert build.call_count == 4
        ld.clear_ld_caches()
        evaluate(component())
        assert build.call_count == 5


def test_result_lru_eviction(ld_sources):
    evaluate(component())
    for i in range(128):
        evaluate(component(5501. + i))
    before = curve_utils.interpolate_on_ld_grid_cached.cache_info()
    evaluate(component())
    after = curve_utils.interpolate_on_ld_grid_cached.cache_info()
    assert after.currsize == 128
    assert after.misses == before.misses + 1


def test_source_lru_eviction_and_schema(ld_sources, monkeypatch):
    import shutil
    for i in range(33):
        band = f'test{i}'
        shutil.copyfile(ld_sources / ld.get_ld_table_filename('bolometric', 0.),
                        ld_sources / ld.get_ld_table_filename(band, 0.))
        ld.interpolate_on_ld_grid(np.array([5500.]), np.array([1.5]), 0., [band])
    info = ld.get_ld_source_interpolator.cache_info()
    assert info.currsize == 32 and info.misses == 33
    ld.interpolate_on_ld_grid(np.array([5500.]), np.array([1.5]), 0., ['test0'])
    assert ld.get_ld_source_interpolator.cache_info().misses == 34
    monkeypatch.setattr(settings, 'LIMB_DARKENING_LAW', 'logarithmic')
    first = evaluate(component())
    monkeypatch.setitem(settings.LD_LAW_CFS_COLUMNS, 'logarithmic', ['ylog', 'xlog'])
    assert_array_equal(evaluate(component()), first[:, ::-1])


def test_empty_and_nan_payload_keys(ld_sources):
    key = utils.float64_array_to_cache_key
    assert key(np.array([1., 2.], dtype='>f8')) == key(np.array([1., 2.], dtype='<f8'))
    assert key(np.arange(6.)[::2]) == key(np.array([0., 2., 4.]))
    nan1 = np.array([0x7ff8000000000001], dtype=np.uint64).view(np.float64)
    nan2 = np.array([0x7ff8000000000002], dtype=np.uint64).view(np.float64)
    assert key(nan1) != key(nan2)
    for values in (np.array([[0., -0., 1.]]), nan1, nan2, np.arange(6.)[::2], np.array([])):
        decoded = utils.float64_array_from_cache_key(key(values))
        assert key(decoded) == key(values)
        assert decoded.shape == values.shape
        assert not decoded.flags.writeable
    result = ld.interpolate_on_ld_grid(np.array([]), np.array([]), 0., ['bolometric'])
    assert result['bolometric'].shape == (0, 1)
    with pytest.raises(ValueError):
        ld.interpolate_on_ld_grid(np.array([5500.]), np.array([1.5]), 99., ['bolometric'])


def test_circular_complete_curve_lru(monkeypatch):
    from elisa.binary_system.curves import c_managed
    table_dir = Path.home() / '.elisa/limbdarkening/ld'
    if not (table_dir / 'lin.Generic.Bessell.U.p00.csv').is_file():
        pytest.skip('Configured Van Hamme assets are required for the audit baseline')
    monkeypatch.setattr(settings, 'LD_TABLES', str(table_dir))
    monkeypatch.setattr(settings, 'LIMB_DARKENING_LAW', 'cosine')
    monkeypatch.setattr(settings, 'NUMBER_OF_PROCESSES', 1)
    monkeypatch.setattr(settings, 'USE_SINGLE_LD_COEFFICIENTS', False)
    ld.clear_ld_caches()
    original_worker = c_managed.produce_circ_sync_curves_mp
    def worker(*worker_args, **worker_kwargs):
        before = curve_utils.interpolate_on_ld_grid_cached.cache_info()
        result = original_worker(*worker_args, **worker_kwargs)
        assert curve_utils.interpolate_on_ld_grid_cached.cache_info() == before
        return result
    lc_kwargs = dict(from_phase=-.6, to_phase=.6, phase_step=.01, normalize=True)
    try:
        with patch.object(c_managed, 'produce_circ_sync_curves_mp', worker), \
             patch.object(ld.interpolate, 'LinearNDInterpolator', wraps=ld.interpolate.LinearNDInterpolator) as build, \
             patch.object(ld, 'interpolate_on_ld_grid', wraps=ld.interpolate_on_ld_grid) as query:
            obs = audit_observer()
            first = obs.observe.lc(**lc_kwargs)
            assert curve_utils.interpolate_on_ld_grid_cached.cache_info()[:2] == (0, 4)
            assert query.call_count == 6 and build.call_count == 2
            repeated = obs.observe.lc(**lc_kwargs)
            assert curve_utils.interpolate_on_ld_grid_cached.cache_info()[:2] == (4, 4)
            assert query.call_count == 8 and build.call_count == 2
            equivalent = audit_observer().observe.lc(**lc_kwargs)
            assert curve_utils.interpolate_on_ld_grid_cached.cache_info()[:2] == (8, 4)
            assert query.call_count == 10 and build.call_count == 2
            for actual in (repeated, equivalent):
                assert_array_equal(first[0], actual[0])
                assert_array_equal(first[1]['Generic.Bessell.U'], actual[1]['Generic.Bessell.U'])
    finally:
        ld.clear_ld_caches()


@pytest.mark.parametrize('law', ['linear', 'cosine', 'logarithmic', 'square_root'])
@pytest.mark.parametrize('metal', [-5., 0., .05, 1.])
@pytest.mark.parametrize('band', ['Generic.Bessell.U', 'bolometric', 'Generic.Bessell.V'])
def test_installed_asset_hull_parity(monkeypatch, law, metal, band):
    from scipy.spatial import ConvexHull
    directory = Path.home() / '.elisa/limbdarkening/ld'
    names = ld.get_relevant_ld_tables(band, metal, law)
    if not all((directory / name).is_file() for name in names):
        pytest.skip('Installed Van Hamme asset not available')
    monkeypatch.setattr(settings, 'LD_TABLES', str(directory))
    monkeypatch.setattr(settings, 'LIMB_DARKENING_LAW', law)
    points, values = legacy_source(band, metal)
    hull = points[ConvexHull(points).vertices]
    center = hull.mean(axis=0)
    midpoints = (hull + np.roll(hull, -1, axis=0)) / 2
    # Hull nodes, edges, all source rows (including conflicting coordinates), interiors.
    queries = np.vstack((points, midpoints, center, center, midpoints + 1e-8 * (center - midpoints)))
    gravity = queries[:, 1] - 2
    expected = griddata(points, values, np.column_stack((queries[:, 0], gravity + 2)), method='linear')
    assert_array_equal(ld.interpolate_on_ld_grid(queries[:, 0], gravity, metal, [band])[band], expected)
    # An outward displacement may lie within Qhull's tolerance: follow the oracle's decision exactly.
    for query in (midpoints[0] - 1e-8 * (center - midpoints[0]),
                  np.nextafter(hull[0], np.inf)):
        gravity = np.array([query[1] - 2])
        expected = griddata(points, values, np.column_stack(([query[0]], gravity + 2)), method='linear')
        if np.isnan(expected).any():
            with pytest.raises(LimbDarkeningError, match='produced numpy.nan/None'):
                ld.interpolate_on_ld_grid(np.array([query[0]]), gravity, metal, [band])
        else:
            assert_array_equal(ld.interpolate_on_ld_grid(np.array([query[0]]), gravity, metal, [band])[band], expected)
    ld.clear_ld_caches()


def audit_observer(eccentricity=0., bands=('Generic.Bessell.U',)):
    from astropy import units as u
    from elisa import BinarySystem, Observer, Star
    primary = Star(mass=2.15*u.solMass, surface_potential=3.6, synchronicity=1., t_eff=10000*u.K,
                   gravity_darkening=1., discretization_factor=5, albedo=.6, metallicity=0., atmosphere='bb')
    secondary = Star(mass=1.2*u.solMass, surface_potential=4., synchronicity=1., t_eff=7000*u.K,
                     gravity_darkening=1., albedo=.6, metallicity=0., atmosphere='bb')
    binary = BinarySystem(primary=primary, secondary=secondary, argument_of_periastron=58*u.deg,
                          gamma=-30.7*u.km/u.s, period=2.5*u.d, eccentricity=eccentricity, inclination=85*u.deg,
                          distance=155*u.pc, primary_minimum_time=2440000*u.d, phase_shift=0.)
    return Observer(passband=list(bands), system=binary)


@pytest.mark.parametrize('eccentricity,bands', [(0., ['Generic.Bessell.U']),
    (.2, ['Generic.Bessell.U']), (0., ['Generic.Bessell.U', 'Generic.Bessell.V'])])
def test_complete_scientific_unrounded_parity(monkeypatch, eccentricity, bands):
    from astropy import units as u
    from elisa.binary_system.curves import utils as binary_curve_utils
    directory = Path.home() / '.elisa/limbdarkening/ld'
    if not (directory / 'lin.Generic.Bessell.U.p00.csv').is_file():
        pytest.skip('Configured Van Hamme assets required')
    monkeypatch.setattr(settings, 'LD_TABLES', str(directory))
    monkeypatch.setattr(settings, 'LIMB_DARKENING_LAW', 'cosine')
    monkeypatch.setattr(settings, 'NUMBER_OF_PROCESSES', 1)
    monkeypatch.setattr(settings, 'USE_SINGLE_LD_COEFFICIENTS', False)
    lc_kwargs = dict(from_phase=-.6, to_phase=.6, phase_step=.01)
    optimized = ld.interpolate_on_ld_grid
    original_prep = binary_curve_utils.prep_surface_params
    def oracle(temperature, log_g, metallicity, passband):
        result = {}
        for oracle_band in passband:
            points, values = legacy_source(oracle_band, metallicity)
            result[oracle_band] = griddata(points, values, np.column_stack((temperature, np.asarray(log_g) + 2)),
                                    method='linear')
            assert not np.isnan(result[oracle_band]).any()
        return result
    results = []
    try:
        for interpolate in (oracle, optimized):
            ld.clear_ld_caches()
            captured_coeffs, captured_temperatures = [], []
            def capture(*interpolation_args, **interpolation_kwargs):
                result = interpolate(*interpolation_args, **interpolation_kwargs)
                captured_coeffs.extend(value.copy() for value in result.values())
                return result
            def capture_prep(system, **preparation_kwargs):
                result = original_prep(system, **preparation_kwargs)
                captured_temperatures.extend(getattr(system, name).temperatures.copy()
                                             for name in ('primary', 'secondary'))
                return result
            with patch.object(ld, 'interpolate_on_ld_grid', capture), \
                 patch.object(binary_curve_utils, 'prep_surface_params', capture_prep):
                obs = audit_observer(eccentricity, bands)
                phases, raw = obs.observe.lc(**lc_kwargs, normalize=False, flux_unit=u.W/u.m**2)
                _, normalized = obs.observe.lc(**lc_kwargs, normalize=True)
            results.append((phases, raw, normalized, captured_coeffs, captured_temperatures))
        for before, after in zip(results[0][3], results[1][3]):
            assert_array_equal(before, after)
        assert len(results[0][3]) == len(results[1][3])
        for before, after in zip(results[0][4], results[1][4]):
            assert_array_equal(before, after)
        assert_array_equal(results[0][0], results[1][0])
        for band in bands:
            assert_array_equal(results[0][1][band], results[1][1][band])
            assert_array_equal(results[0][2][band], results[1][2][band])
        if eccentricity:
            assert ld.get_ld_source_interpolator.cache_info().misses == 2
            assert ld.get_ld_source_interpolator.cache_info().hits == 982
            assert len(results[1][3]) == 984
            assert not np.array_equal(results[1][4][0], results[1][4][162])
    finally:
        ld.clear_ld_caches()


def test_same_size_restored_mtime_revision(ld_sources):
    original = evaluate(component()).copy()
    path = ld_sources / ld.get_ld_table_filename('bolometric', 0.)
    before = path.stat()
    contents = path.read_text()
    changed = contents.replace('0.8', '1.8')
    assert contents != changed and len(contents) == len(changed)
    path.write_text(changed)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    assert not np.array_equal(evaluate(component()), original)
    # Atomic replacement has a new inode even when size and mtime match.
    replacement = path.with_suffix('.new')
    replacement.write_text(contents)
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    replacement.replace(path)
    assert_array_equal(evaluate(component()), original)
