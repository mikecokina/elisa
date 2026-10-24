"""Regressions found by real fitting browser workflows."""
import pytest

from elisa.analytics.params.parameters import BinaryInitialParameters
from elisa.ui.tabs.rv_fitting.components.param_inputs import PARAM_SPEC
from elisa.ui.tabs.rv_fitting.logic.compute import build_x0, result_to_dataframe


@pytest.mark.parametrize('mode', ['free', 'fixed', 'constrained'])
def test_rv_modes_are_accepted_by_scientific_parameter_validator(mode):
    values = {}
    for name, (_, value, fixed, lo, hi, _) in PARAM_SPEC.items():
        values.update({f'{name}_value': value, f'{name}_mode': 'fixed' if fixed else 'free',
                       f'{name}_min': lo, f'{name}_max': hi})
    values.update(eccentricity_mode=mode, eccentricity_constraint='0 * system@asini')
    x0 = build_x0(values)
    params = BinaryInitialParameters(**x0)
    if mode == 'constrained':
        assert 'fixed' not in x0['system']['eccentricity']
        assert 'system@eccentricity' in params.get_constrained()
        assert 'system@eccentricity' not in params.get_fitable()
    elif mode == 'fixed':
        assert 'system@eccentricity' in params.get_fixed()
    else:
        assert 'system@eccentricity' in params.get_fitable()


def test_rv_result_table_identifies_constraint_and_expression():
    frame = result_to_dataframe({'system@eccentricity': {'value': 0, 'constraint': '0 * system@asini'}})
    assert frame.loc[0, 'Status'] == 'constrained: 0 * system@asini'
