"""Pulsation surface displacement, velocity, and acceleration calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from elisa import settings
from elisa.base.error import raise_elisa_not_implemented_error, raise_elisa_pulsation_mode_error
from elisa.pulse.utils import generate_phase_shift

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.pulse.mode import PulsationMode


def calculate_radial_displacement(mode: PulsationMode, harmonics: NDArray) -> NDArray:
    """Calculate radial displacement of surface points.

    Computes the radial component of pulsation-induced surface displacement
    by multiplying the radial amplitude by the spherical harmonic values.
    The result is a complex scalar length in metres, not a Cartesian x
    component. Multiplying it by the equilibrium unit direction gives the
    radial vector; container operations divide by the system length scale.

    :param mode: Pulsation mode object with radial amplitude.
    :type mode: elisa.pulse.mode.PulsationMode
    :param harmonics: Spherical harmonic values Y_l^m at surface points.
    :type harmonics: NDArray
    :returns: Complex radial displacement at each surface point.
    :rtype: NDArray
    """
    return mode.radial_amplitude * harmonics


def calculate_mode_derivatives(
    displacement: NDArray,
    angular_frequency: float,
) -> NDArray:
    """Calculate velocity from complex displacement.

    With D(t)=D_0*exp(-i*omega*t), d(Re(D))/dt=Re(-i*omega*D)
    equals omega*Im(D). Equilibrium coordinates and the pulsation-axis
    orientation are held fixed. The result retains the input length unit per
    second; container operations convert mesh lengths to metres.

    This is an intrinsic harmonic derivative. Moving-frame transport and
    material velocities of an evolving eccentric equilibrium surface are not
    included.

    :param displacement: Complex displacement vector.
    :type displacement: NDArray
    :param angular_frequency: Angular frequency of pulsation (omega = 2*pi*frequency).
    :type angular_frequency: float
    :returns: Velocity perturbation at each point.
    :rtype: NDArray
    """
    return angular_frequency * np.imag(displacement)


def calculate_mode_second_derivatives(
    displacement: NDArray,
    angular_frequency: float,
) -> NDArray:
    """Calculate acceleration from complex displacement.

    Differentiating exp(-i*omega*t) twice gives -omega**2. Thus the second
    derivative of Re(D) is -omega**2*Re(D), with the same frozen equilibrium
    and orientation convention as calculate_mode_derivatives. Units are the
    input length unit per second squared; no frame-transport terms are added.

    :param displacement: Complex displacement vector.
    :type displacement: NDArray
    :param angular_frequency: Angular frequency of pulsation (omega = 2*pi*frequency).
    :type angular_frequency: float
    :returns: Acceleration perturbation at each point.
    :rtype: NDArray
    """
    return -(angular_frequency**2) * np.real(displacement)


def calculate_temperature_pert_factor(
    mode: PulsationMode,
    scale: float,
) -> NDArray:
    """Calculate temperature perturbation factor for pulsating star surface.

    Computes the temperature perturbation following Townsend (2003) treatment,
    using the explicit radial scalar coefficient D_r, not a Cartesian vector
    component. The implemented fractional perturbation is
    temperature_amplitude_factor*Re(exp(-i*shift)*D_r)*scale/A_r.
    D_r is stored in mesh units, so scale/A_r makes this factor dimensionless
    and cancels the radial amplitude already present in D_r. The temperature
    amplitude factor retains its configured or previously assigned meaning.

    :param mode: Pulsation mode with displacement and temperature data.
    :type mode: elisa.pulse.mode.PulsationMode
    :param scale: Scaling factor for the system (semi-major axis for binary systems).
    :type scale: float
    :returns: Temperature perturbation factor (delta_T = factor * T_eff).
    :rtype: NDArray
    """
    # Apply phase shift between geometry and temperature perturbations
    hrm_shift = np.real(
        generate_phase_shift(mode.temperature_perturbation_phase_shift)
        * mode.complex_radial_displacement,
    )

    # Calculate temperature perturbation following Townsend (2003)
    return (
        mode.temperature_amplitude_factor
        * hrm_shift
        * scale
        / mode.radial_amplitude
    )


def calculate_cartesian_displacement(
        mode: PulsationMode,
        directions: NDArray,
        harmonics: NDArray,
        gradient: NDArray,
        radius: NDArray,
        scale: float = 1.0,
) -> NDArray:
    """Construct a linear physical vector in the pulsation-axis frame.

    For harmonic Y, radial unit direction n and dimensionless tangent gradient
    g, construct ``D=A_r*Y*n + A_h*R*g/sqrt(mean(R**2*|g|**2))``, then divide
    physical lengths by scale. The complex norm ``|g|**2`` sums the squared
    magnitudes of all three Cartesian components, not just their real parts.
    This discrete vertex RMS uses local equilibrium radii and no area weights.
    In a distorted mesh it therefore differs from normalizing g alone.

    Both input vectors are in pulsation axes, and g.n=0: horizontal means
    tangent to the radial sphere, not to the distorted surface normal. A_r and
    signed A_h are physical lengths in metres. Radii may be in metres or SMA
    units because their common length scale cancels in the horizontal quotient.
    The l=0 tangent contribution is absent, including when an explicit ratio
    was supplied; amplitude assignment is not redefined here.

    :param mode: Mode carrying physical radial and horizontal amplitude scales.
    :param directions: Equilibrium unit directions, shape (n_points, 3).
    :param harmonics: Complex normalized Y values, shape (n_points,).
    :param gradient: Dimensionless complex tangent gradients, shape (n_points, 3).
    :param radius: Local equilibrium radii, shape (n_points,).
    :param scale: Metres per mesh length unit: 1 for singles, binary SMA for binaries.
    :returns: Complex Cartesian vectors in the input frame, shape (n_points, 3),
              in mesh length units (metres for singles, SMA units for binaries).
    """
    if settings.PULSATION_MODEL != "uniform":
        return raise_elisa_not_implemented_error(
            f"Surface displacement calculation is not implemented for pulsation model {settings.PULSATION_MODEL!r}.")
    displacement = (calculate_radial_displacement(mode, harmonics)/scale)[:, None]*directions
    if mode.l != 0 and mode.horizontal_amplitude != 0:
        horizontal = radius[:, None]*gradient
        rms = np.sqrt(np.mean(np.sum(np.abs(horizontal)**2, axis=1)))
        if rms == 0:
            raise_elisa_pulsation_mode_error("Horizontal displacement has zero discrete RMS.")
        displacement += (mode.horizontal_amplitude/(scale*rms))*horizontal
    return displacement
