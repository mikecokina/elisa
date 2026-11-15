"""Regenerate only the single-star V fixture after checking its LD provenance.

Run from the repository root with PYTHONPATH=src:. and installed science tables.
The historical file has an arbitrary absolute scale; compare shapes first.
"""
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from elisa import Observer, ld, settings
from elisa import units as u
from elisa.atm import atm
from elisa.single_system.curves import utils as curves
from tests import utils
from tests.tests_pulse.test_synthesis import TestPulsationModule


def regenerate() -> None:
    """Use the regression's canonical settings, mesh, mode and phase grid."""
    case = TestPulsationModule()
    case.setUp()
    try:
        star = case.prepare_system([{
            "l": 2, "m": 1, "amplitude": 100 * u.m / u.s, "frequency": 15 / u.d,
            "start_phase": 0.0, "horizontal_to_radial_amplitude_ratio": 0.0,
            "temperature_amplitude_factor": 0.01,
        }])
        band = "Generic.Bessell.V"
        phases = np.linspace(0.0, 1 / (15 * star.rotation_period))
        obtained = Observer(passband=[band], system=star).lc(phases=phases)
        original = curves.get_normal_radiance
        bolometric = []

        def capture(system: Any, **kwargs: Any) -> dict:
            bolometric[:] = [ld.get_bolometric_ld_coefficients(
                system.star.temperatures, system.star.log_g,
                system.star.metallicity, system.star.limb_darkening_coefficients,
            )]
            return original(system, **kwargs)

        def old_normalization(fluxes: dict, _coefficients: dict) -> dict:
            factor = ld.calculate_integrated_limb_darkening_factor(
                settings.LIMB_DARKENING_LAW, bolometric[0],
            )
            return {component: {name: flux / factor for name, flux in bands.items()}
                    for component, bands in fluxes.items()}

        with (patch.object(curves, "get_normal_radiance", side_effect=capture),
              patch.object(atm, "correct_normal_radiance_to_optical_depth", side_effect=old_normalization)):
            old = Observer(passband=[band], system=star).lc(phases=phases)[1][band]
        reference = utils.load_light_curve("single.pulsating.v.json")[1][band]
        normalize = utils.normalize_lc_for_unittests
        # The legacy reference differs slightly in historical numerical precision.
        # Permit only that measured discrepancy before replacing the fixture.
        if not np.allclose(normalize(obtained[1][band]), normalize(reference), atol=1e-12, rtol=0):
            np.testing.assert_allclose(normalize(old), normalize(reference), atol=1.5e-6, rtol=0)
        destination = Path(__file__).parents[1] / "data" / "light_curves" / "curves" / "single.pulsating.v.json"
        destination.write_text(json.dumps(
            [obtained[0].tolist(), {band: obtained[1][band].tolist()}], indent=4,
        ) + "\n")
    finally:
        case.tearDown()


if __name__ == "__main__":
    regenerate()
