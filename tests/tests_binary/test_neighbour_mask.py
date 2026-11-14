"""Rebuild decisions must follow physical positions through sorted worker batches."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from elisa import settings
from elisa.binary_system import dynamic
from elisa.binary_system.curves import c_appx_router, c_managed, utils
from tests.utils import BINARY_SYSTEM_PARAMS, prepare_binary_system


@pytest.mark.parametrize("repeat", [False, True])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("batches", [1, 3, 8])
def test_rebuild_decisions(
    monkeypatch: pytest.MonkeyPatch, shuffle: bool, batches: int, repeat: bool,
) -> None:
    monkeypatch.setattr(settings, "USE_SIMILAR_NEIGHBOURS_APPROXIMATION", True)
    monkeypatch.setattr(settings, "MAX_D_FLUX", 0.001)
    binary = prepare_binary_system(BINARY_SYSTEM_PARAMS["detached.ecc"])
    phases = np.linspace(0, 1, 31, endpoint=False)
    if repeat:
        phases = np.concatenate((phases, phases[[0, 4, 15]]))
    if shuffle:
        phases = np.random.default_rng(42).permutation(phases)
    positions = binary.calculate_orbital_motion(phases, return_nparray=True)
    potentials = binary.correct_potentials(phases)
    radii = utils.forward_radii_from_distances(binary, positions[:, 1], potentials)
    enabled, mask, ordered = c_appx_router.eval_similar_neighbours_approximation(binary, radii, positions)
    assert enabled
    ids = ordered[:, 0].astype(int)
    sorted_radii = radii[:, ids]
    expected = dynamic.resolve_object_geometry_update(
        False, len(phases), utils.compute_rel_d_geometry(binary, sorted_radii[:, :-1], sorted_radii[:, 1:]),
    ) | dynamic.resolve_irrad_update(utils.compute_rel_d_irradiation(binary, ordered[:, 1]), len(phases))
    assert_array_equal(mask[ids], expected)
    for batch in np.array_split(ordered, batches):
        container = MagicMock()
        with (
            patch.object(c_managed.OrbitalPositionContainer, "from_binary_system", return_value=container),
            patch.object(c_managed, "_update_surface_in_ecc_orbits") as update,
            patch.object(c_managed.bsutils, "move_sys_onpos", return_value=container),
            patch.object(c_managed.crv_utils, "update_surface_params", return_value=(container, None, None)),
            patch.object(c_managed, "compute_surface_coverage"),
        ):
            c_managed.similar_neighbour_approximation_ecc_curve_integration(
                binary, potentials, batch, mask, [], lambda curves, *_args: curves, {},
            )
        batch_ids = batch[:, 0].astype(int)
        decisions = mask[batch_ids].copy()
        decisions[0] = True
        assert_array_equal([call.kwargs["new_geometry_test"] for call in update.call_args_list], decisions)
        for call, idx in zip(container.set_on_position_params.call_args_list, batch_ids, strict=True):
            assert call.args[1] == potentials["primary"][idx]
            assert call.args[2] == potentials["secondary"][idx]


@pytest.mark.requires_ld(passbands=("bolometric", "Generic.Bessell.V"))
@pytest.mark.requires_atmosphere
@pytest.mark.parametrize("observable", ["lc", "rv"])
def test_observer_phase_order(observable: str) -> None:
    from elisa.observer.observer import Observer

    with settings.override(
        NUMBER_OF_PROCESSES=3,
        NUMBER_OF_MCMC_PROCESSES=1,
        USE_INTERPOLATION_APPROXIMATION=False,
        USE_SYMMETRICAL_COUNTERPARTS_APPROXIMATION=False,
        USE_SIMILAR_NEIGHBOURS_APPROXIMATION=True,
        MAX_D_FLUX=0.001,
    ):
        binary = prepare_binary_system(BINARY_SYSTEM_PARAMS["detached.ecc"])
        observer = Observer(system=binary, passband=["Generic.Bessell.V"])
        phases = np.linspace(0, 1, 31, endpoint=False)
        permutation = np.random.default_rng(42).permutation(len(phases))
        kwargs = {"method": "radiometric"} if observable == "rv" else {}
        compute = getattr(observer, observable)
        baseline = compute(phases=phases, **kwargs)[1]
        reordered = compute(phases=phases[permutation], **kwargs)[1]
    for label, values in baseline.items():
        assert_array_equal(values, reordered[label][np.argsort(permutation)])
