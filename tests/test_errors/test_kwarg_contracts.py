from __future__ import annotations

from typing import TYPE_CHECKING

# The tests package selects Astropy constants before test-module imports.
import pytest

from elisa import utils
from elisa.base.error import ElisaError, ElisaInputKwargsError, ElisaInvalidKwargsError, ElisaMissingKwargsError
from tests import set_astropy_units

if TYPE_CHECKING:
    import elisa.base.error


set_astropy_units()


@pytest.mark.parametrize("error_type", [ElisaInvalidKwargsError, ElisaMissingKwargsError])
def test_kwarg_error_hierarchy_and_common_catch(
    error_type: type[elisa.base.error.ElisaInvalidKwargsError | elisa.base.error.ElisaMissingKwargsError],
) -> None:
    assert ElisaInputKwargsError.__bases__ == (ElisaError,)
    assert error_type.__bases__ == (ElisaInputKwargsError,)
    failure = error_type("argument diagnostics")
    assert not isinstance(failure, ValueError)
    with pytest.raises(ElisaInputKwargsError) as caught:
        raise failure
    assert caught.value is failure


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"unknown": 1}, ElisaInvalidKwargsError),
        ({}, ElisaMissingKwargsError),
    ],
)
def test_dataset_constructor_distinguishes_invalid_and_missing_kwargs(
    kwargs: dict[object, object] | dict[str, int],
    error_type: type[elisa.base.error.ElisaInvalidKwargsError | elisa.base.error.ElisaMissingKwargsError],
) -> None:
    from elisa.analytics.dataset.base import RVData

    with pytest.raises(error_type) as caught:
        RVData(**kwargs)
    assert str(caught.value)


@pytest.mark.parametrize(
    ("supplied", "required", "error_type"),
    [
        ({"system@unknown": 1.0}, [], ElisaInvalidKwargsError),
        ({}, ["system@period"], ElisaMissingKwargsError),
    ],
)
def test_initial_parameter_name_checks_propagate_specific_kwargs_errors(
    supplied: dict[object, object] | dict[str, float],
    required: list[object] | list[str],
    error_type: type[elisa.base.error.ElisaInvalidKwargsError | elisa.base.error.ElisaMissingKwargsError],
) -> None:
    from types import SimpleNamespace

    from elisa.analytics.params.parameters import check_initial_param_validity

    parameters = {name: SimpleNamespace(value=value) for name, value in supplied.items()}
    with pytest.raises(error_type) as caught:
        check_initial_param_validity(parameters, ["system@period"], required)
    assert type(caught.value) is error_type
    assert isinstance(caught.value, ElisaInputKwargsError)
    assert "system@" in str(caught.value)


def test_parameter_name_checks_accept_allowed_and_complete_inputs() -> None:
    utils.invalid_param_checker({"system@period": 1.0}, ["system@period"], "x0")
    utils.check_missing_params(["system@period"], ["system@period"], "x0")
