"""Scalar reference for public orbital outputs, independent of batch dispatch."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from elisa.binary_system.orbit.orbit import Orbit


def scalar_orbital_motion(orbit: Orbit, phase: ArrayLike) -> NDArray:
    """Assemble unrounded positions with the public scalar Kepler solver.

    Calling the scalar solver directly retains its single 2pi correction and
    keeps this reference independent of the batch orbital-motion method.
    """
    phases = np.atleast_1d(np.asarray(phase))
    periastron_phases = orbit.true_phase(
        phase=phases,
        phase_shift=orbit.conjunctions["primary_eclipse"]["true_phase"],
    )
    mean_anomalies = orbit.phase_to_mean_anomaly(phase=periastron_phases)
    eccentric_anomalies = np.array(
        [orbit.mean_anomaly_to_eccentric_anomaly(mean_anomaly) for mean_anomaly in mean_anomalies]
    )
    true_anomalies = orbit.eccentric_anomaly_to_true_anomaly(eccentric_anomalies)
    separations = orbit.relative_radius(true_anomalies)
    azimuths = orbit.true_anomaly_to_azimuth(true_anomalies)
    return np.column_stack((separations, azimuths, true_anomalies, phases))
