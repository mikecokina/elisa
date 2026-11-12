"""Pulsation mode spherical harmonics and amplitude calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from elisa import const, settings
from elisa.base.error import raise_elisa_pulsation_mode_error
from elisa.logger import getLogger
from elisa.umpy import sph_harm

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.base.body import Body
    from elisa.base.container import StarContainer
    from elisa.pulse.mode import PulsationMode

# Binary64 finite magnitudes: normals start at 2**-1022; subnormals extend
# down to 2**-1074 with reduced relative precision. Smaller results may round
# to zero. The largest finite value is (2-2**-52)*2**1023, below 2**1024.
# For nonzero M, frexp(M)=(f, b) means M=f*2**b, 1/2 <= f < 1. Thus the
# accepted b=450..550 band is M in [2**449, 2**550), and target b=512 places
# M in [2**511, 2**512). These b values are NOT the stored field exponent E.
#
# Magnitude powers, log2(M); the working band is checked periodically:
#    below min             normal range                         overflow
#       |                                                        |
#     -1074   -1022       0                 449  511 550        1024
#       |--------|-------|-------------------[----^---)------------|
#        subnormal                         band target
#
# The positive band lets a large mantissa amplify a subnormal coordinate
# before rounding: e.g. 2**511 * 2**-1074 = 2**-563, a normal result. It also
# leaves hundreds of bits above for products, sums and tangent projection.
# This protects dynamic range, not every tiny or cancelled component.
# Band placement and -900 are engineering margins, not optimality bounds.
_HARMONIC_SEED_EXPONENT_FLOOR = -900
_HARMONIC_SCALE_LOWER = 450
_HARMONIC_SCALE_UPPER = 550
_HARMONIC_SCALE_TARGET = 512
# After scale checks M < 2**550; a degree block with <=128 growth bits keeps the
# exact-arithmetic bound below 2**678. Projection adds at most two bits on
# unit directions. See _degree_rescale_interval for the bound's assumptions.
_HARMONIC_DEGREE_GROWTH_BUDGET = 128
# For nonzero q and order>0, factoring gives |q_hat| in [1/2, sqrt(2)). A
# seed step grows the largest value/derivative magnitude by at most
# sqrt(3/2)*(sqrt(2)+1) < 3, including intermediates. Thus 32 steps add less
# than 51 bits in exact arithmetic, comfortably within the upper headroom.
_HARMONIC_SEED_RESCALE_INTERVAL = 32


logger = getLogger(__name__)


def spherical_harmonics(
    mode: PulsationMode,
    points: NDArray,
    time_exponential: complex,
    *,
    order: int | None = None,
    degree: int | None = None,
) -> NDArray:
    """Evaluate scalar harmonics with ELISA's full-sphere RMS normalization.

    Multiplication by sqrt(4*pi) converts the standard unit-integral harmonic
    to unit mean squared magnitude over a sphere. This is an analytic spherical
    normalization, not a discrete mesh RMS. The backend includes the
    Condon--Shortley phase; the supplied complex time factor is applied last.

    :param mode: Pulsation mode object containing degree and order information.
    :type mode: object
    :param points: Spherical coordinates (r, phi, theta) where harmonics are evaluated,
                   shape (n_points, 3).
    :type points: NDArray
    :param time_exponential: Time-dependent complex exponential factor.
    :type time_exponential: complex
    :param order: Azimuthal order m of the mode (overrides mode.m if provided).
    :type order: int | None
    :param degree: Spherical harmonic degree l (overrides mode.l if provided).
    :type degree: int | None
    :returns: Normalized spherical harmonics at input points with shape (n_points,).
    :rtype: NDArray
    """
    l = mode.l if degree is None else degree  # noqa: E741
    m = mode.m if order is None else order
    return (
        mode.renorm_const
        * sph_harm(m, l, points[:, 1], points[:, 2])
        * time_exponential
    )


def diff_spherical_harmonics_by_phi(
    mode: PulsationMode,
    harmonics: list[NDArray],
) -> NDArray:
    """Calculate azimuthal derivative of spherical harmonics.

    Computes the partial derivative of spherical harmonics with respect to
    the azimuthal coordinate phi: d Y_m^l / d phi.

    :param mode: Pulsation mode object containing azimuthal order information.
    :type mode: object
    :param harmonics: List of harmonics [Y_l^m, Y_l^(m+1)].
    :type harmonics: list[NDArray]
    :returns: Azimuthal derivative of spherical harmonics with shape (n_points,).
    :rtype: NDArray
    """
    return (0 + 1j) * mode.m * harmonics[0]


def assign_amplitudes(
    star_container: StarContainer | Body,
    normalization_constant: float = 1.0,
) -> None:
    """Assign radial and horizontal displacement amplitudes to pulsation modes.

    Calculates and assigns the radial and horizontal (non-radial) displacement
    amplitudes for each pulsation mode based on the mode parameters and stellar
    properties. For modes without an explicit horizontal-to-radial amplitude ratio,
    it is calculated from the mode properties. Temperature amplitude factors are
    computed for modes where not explicitly provided.

    The input amplitude V is a velocity coefficient scale in m/s. For angular
    frequency omega and signed horizontal/radial ratio q, the physical length
    scales are A_r=V/(omega*sqrt(1+q**2)) and A_h=q*A_r, both in metres.
    A_r is not in general the sampled radial RMS or a pointwise maximum.
    Horizontal mesh RMS normalization is performed later in kinematics.

    :param star_container: Star container object with pulsation modes.
    :type star_container: object
    :param normalization_constant: Scaling factor for amplitudes. For binary systems,
                                   this is the semi-major axis; for single stars, it
                                   should remain 1.0. Defaults to 1.0.
    :type normalization_constant: float
    :raises PulsationModeError: If a radial mode (l=0) or mode with only radial motion
                        lacks an explicit temperature_amplitude_factor.
    """
    r_equiv = star_container.equivalent_radius * normalization_constant
    if r_equiv == 0:
        error_msg = "Pulsation amplitudes require a nonzero equivalent radius."
        raise_elisa_pulsation_mode_error(error_msg)
    mult = const.G * star_container.mass / r_equiv**3

    for mode_index, mode in star_container.pulsations.items():
        if mode.angular_frequency == 0:
            error_msg = "Pulsation amplitudes require a nonzero frequency."
            raise_elisa_pulsation_mode_error(error_msg)
        # Calculate horizontal/radial amplitude ratio if not provided (Aerts 2010, p. 198)
        if mode.horizontal_to_radial_amplitude_ratio is None:
            amp_ratio = (
                np.sqrt(mode.l * (mode.l + 1))
                * mult
                / mode.angular_frequency**2
            )
            mode.horizontal_to_radial_amplitude_ratio = amp_ratio

        amplitude = mode.amplitude / mode.angular_frequency

        # Calculate radial and horizontal amplitudes
        mode.radial_amplitude = amplitude / np.sqrt(
            mode.horizontal_to_radial_amplitude_ratio**2 + 1,
        )
        mode.horizontal_amplitude = (
            mode.horizontal_to_radial_amplitude_ratio * mode.radial_amplitude
        )

        # Assign or calculate temperature amplitude factor
        if mode.temperature_amplitude_factor is None:
            if mode.l == 0 or mode.horizontal_to_radial_amplitude_ratio == 0.0:
                error_msg = (
                    "Parameter `temperature_amplitude_factor` needs to be "
                    "supplied in case of radial modes or in case of modes "
                    "with radial motion."
                )
                raise_elisa_pulsation_mode_error(error_msg)
            mode.temperature_amplitude_factor = (
                temp_amplitude(mode) * mode.radial_amplitude / r_equiv
            )

        # Check for excessive surface displacement
        surf_ampl = mode.horizontal_amplitude / r_equiv
        if surf_ampl > settings.SURFACE_DISPLACEMENT_TOL:
            prec = int(-np.log10(surf_ampl) + 2)
            if not settings.SUPPRESS_WARNINGS:
                warning_msg = (
                    f"Relative horizontal surface displacement amplitude "
                    f"({round(surf_ampl, prec)}) for the mode {mode_index} "
                    f"exceeded safe tolerances ({settings.SURFACE_DISPLACEMENT_TOL}) "
                    f"given by the use of linear approximation. This can lead to "
                    f"invalid surface discretization. Use this result with caution."
                )
                logger.warning(warning_msg)


def temp_amplitude(mode: PulsationMode) -> float:
    """Calculate temperature perturbation amplitude for a pulsation mode.

    Computes the temperature perturbation amplitude according to equation 22
    in Townsend 2003:

        delta T = temp_amplitude * (delta r / r) * T
        temp_amplitude = nabla_ad * (K*l*(l+1) - 4 - 1/K)

    where K is the horizontal-to-radial amplitude ratio and nabla_ad is the
    adiabatic temperature gradient.

    :param mode: Pulsation mode object containing degree and amplitude ratio.
    :type mode: object
    :returns: Temperature perturbation amplitude factor.
    :rtype: float
    :raises PulsationModeError: If horizontal_to_radial_amplitude_ratio is zero.
    """
    # The prescription contains 1/K. Amplitude assignment requires an explicit
    # temperature factor for radial modes or K=0 rather than evaluating it.
    if mode.horizontal_to_radial_amplitude_ratio == 0:
        error_msg = "Temperature perturbation requires a nonzero horizontal-to-radial amplitude ratio."
        raise_elisa_pulsation_mode_error(error_msg)

    return const.IDEAL_ADIABATIC_GRADIENT * (
        mode.horizontal_to_radial_amplitude_ratio * mode.l * (mode.l + 1)
        - 4
        - 1 / mode.horizontal_to_radial_amplitude_ratio
    )


def cartesian_harmonics(mode: PulsationMode, directions: NDArray, time_exponential: complex) -> tuple[NDArray, NDArray]:
    """Evaluate Y and its dimensionless Cartesian tangent gradient on unit n.

    Regular solid harmonics S_lm=r**l Y_lm are homogeneous Cartesian
    polynomials. Starting at S_00=1, build S_mm with x+i*y, S_(m+1,m) with z,
    then advance degree with the normalized z*S_(l-1,m), r**2*S_(l-2,m)
    recurrence. The derivative array follows the same recurrence by the product
    rule. Its coefficients include ELISA's sqrt(4*pi) harmonic normalization
    and the Condon--Shortley sign in the sectoral seed.

    At unit n, Euler's identity n.grad(S)=l*S gives the dimensionless tangent
    gradient g=grad(S)-(n.grad(S))*n. It is tangent to the equilibrium radial
    sphere, not necessarily tangent to the distorted stellar surface.
    Polynomial evaluation needs neither angular division nor a chosen pole
    meridian: the same Cartesian expression applies at and near both poles.
    On a sphere of physical radius R, the spatial surface gradient is g/R;
    kinematics uses the dimensionless g in its radius-weighted displacement law.

    Negative orders use Y_(l,-m)=(-1)**m*conj(Y_(l,m)) for the spatial harmonic
    and its gradient. Apply the supplied exp(-i*(omega*t+start_phase)) afterward;
    conjugating that temporal factor would reverse the mode's time convention.

    Seeds at risk of underflow use per-direction binary exponents shared by
    values and derivatives, including both adjacent-degree recurrence states.
    The final tangent projection precedes reconstruction in float64.

    :param mode: Mode supplying degree l and signed order m.
    :param directions: Unit Cartesian directions, shape (n_points, 3), in
                       pulsation axes. Radius is deliberately excluded.
    :param time_exponential: Common complex temporal factor at this snapshot.
    :returns: Complex harmonic values (n_points,) and tangent gradients
              (n_points, 3), both dimensionless and in the input frame.
    """
    value, gradient = _spatial_cartesian_harmonics(mode.l, abs(mode.m), directions)
    # Spatial conjugacy precedes the shared temporal factor for both outputs.
    if mode.m < 0:
        value = (-1) ** abs(mode.m) * value.conj()
        gradient = (-1) ** abs(mode.m) * gradient.conj()
    return time_exponential * value, time_exponential * gradient


def _harmonic_scaling_mask(degree: int, order: int, directions: NDArray) -> NDArray | None:
    """Identify samples requiring stabilization; None denotes an ordinary batch."""
    if degree <= 1:
        return None
    # For order>0, max(|x|,|y|)**order is a lower bound on |q|**order. Testing it
    # against 2**-900 avoids forming the tiny power and conservatively flags
    # small sectoral seeds, with 122 bits to the normal boundary at -1022.
    # This margin is not a bound on every component or later cancellation.
    coordinate_size = np.abs(directions)
    transverse_size = np.maximum(coordinate_size[:, 0], coordinate_size[:, 1])
    risky_seed = (
        (transverse_size != 0) & (transverse_size < 2.0 ** (_HARMONIC_SEED_EXPONENT_FLOOR / order))
        if order
        else np.zeros(len(directions), dtype=bool)
    )
    # Tiny components can be amplified independently, including zonal values
    # close to the equator; inspecting only |x+iy| would miss these samples.
    coordinate_floor = 2.0**_HARMONIC_SEED_EXPONENT_FLOOR
    if np.min(coordinate_size, initial=1.0) < coordinate_floor:
        risky_seed |= np.any((coordinate_size != 0) & (coordinate_size < coordinate_floor), axis=1)
    return risky_seed if np.any(risky_seed) else None


def _spatial_cartesian_harmonics(degree: int, order: int, directions: NDArray) -> tuple[NDArray, NDArray]:
    """Evaluate positive-order harmonics, keeping stabilization local to risky samples."""
    scaling_mask = _harmonic_scaling_mask(degree, order, directions)
    if scaling_mask is None:
        return _evaluate_cartesian_harmonic(degree, order, directions, scaled=False)
    if np.all(scaling_mask):
        return _evaluate_cartesian_harmonic(degree, order, directions, scaled=True)

    values = np.empty(len(directions), dtype=complex)
    gradients = np.empty(directions.shape, dtype=complex)
    for subset, scaled in ((scaling_mask, True), (~scaling_mask, False)):
        values[subset], gradients[subset] = _evaluate_cartesian_harmonic(
            degree,
            order,
            directions[subset],
            scaled=scaled,
        )
    return values, gradients


def _evaluate_cartesian_harmonic(
    degree: int,
    order: int,
    directions: NDArray,
    *,
    scaled: bool,
) -> tuple[NDArray, NDArray]:
    """Project solid-harmonic derivatives before reconstructing the spatial field."""
    value, derivative, exponent = _regular_solid_harmonic(degree, order, directions, scaled=scaled)
    # Project using the computed radial derivative (Euler: n.grad(S)=l*S).
    # This avoids leaving cancellation residuals in the tangent field at high l.
    gradient = derivative - np.sum(derivative * directions, axis=1)[:, None] * directions
    if exponent is not None:
        # Projection must cancel radial terms while derivatives are safely
        # scaled. Reconstruct each component without forming 2**exponent.
        _shift_complex_inplace(value, exponent)
        _shift_complex_inplace(gradient, exponent[:, None])
    return value, gradient


def _multiply_harmonic_coordinate(coefficient: float, coordinate: NDArray, state: NDArray, *, scaled: bool) -> NDArray:
    """Preserve ordinary rounding order or protect tiny coordinates in scaled states."""
    if scaled:
        # Multiplying the coefficient by a subnormal coordinate first can lose
        # information before the large scaled state has a chance to amplify it.
        return coefficient * (coordinate * state)
    return coefficient * coordinate * state


def _factored_transverse_coordinate(order: int, directions: NDArray) -> tuple[NDArray, NDArray]:
    """Return q_hat and e for q=x+iy=2**e*q_hat in a fully scaled batch.

    For nonzero max(|x|,|y|), frexp puts its scaled magnitude in [1/2, 1),
    hence |q_hat| is in [1/2, sqrt(2)). Its exponent e can be -1073 for the
    smallest subnormal coordinate; it is distinct from the stored exponent E.
    Order zero uses e=0 because no transverse seed multiplication is needed.
    Scaling down a much smaller component can itself round; this norm bound
    does not guarantee preservation of each real and imaginary component.
    """
    x, y, _ = directions.T
    transverse_size = np.maximum(np.abs(x), np.abs(y))
    seed_exponent = np.where(order > 0, np.frexp(transverse_size)[1], 0)
    transverse = np.ldexp(x, -seed_exponent) + 1j * np.ldexp(y, -seed_exponent)
    return transverse, seed_exponent


def _align_sectoral_scale(value: NDArray, exponent: NDArray, seed_exponent: NDArray, order: int) -> None:
    """Mutate value and exponent to align S_mm and all its derivatives."""
    # The seed loop differentiates in the factored transverse coordinates:
    # S_mm=V_hat*2**(E+m*e), D_mm=D_hat*2**(E+(m-1)*e).
    # Absorb one factor 2**e into V_hat, leaving a common exponent. Derivatives
    # are unchanged; their z component is zero before degree advancement.
    _shift_complex_inplace(value, seed_exponent)
    exponent += (order - 1) * seed_exponent


def _shift_complex_inplace(state: NDArray, powers: NDArray) -> None:
    """Multiply each complex component by 2**powers, mutating only state.

    Binary shifts are exact when no significand bits are discarded and the
    result stays finite. Subnormal results can round; unrepresentable results
    can become zero or infinity. ldexp avoids forming a separate 2**powers
    factor that could overflow or underflow before multiplication.
    """
    state.real = np.ldexp(state.real, powers)
    state.imag = np.ldexp(state.imag, powers)


def _rescale_harmonic_states(
    value: NDArray,
    derivative: NDArray,
    exponent: NDArray,
    *,
    previous: tuple[NDArray, NDArray] | None = None,
) -> None:
    """Mutate live mantissas and their shared exponent without changing represented fields.

    Each sample represents S=value*2**E and D=derivative*2**E, where E is
    the stored exponent, not the frexp exponent of the largest mantissa M.
    Shifting every mantissa by 2**s and replacing E by E-s preserves the
    represented field algebraically; _shift_complex_inplace states when the
    floating-point shift is exact. For a power-of-two M, frexp's exponent is
    one ABOVE its magnitude power, so this example targets 2**511, not 2**512:

        represented value = mantissa * 2**E
        before: mantissa = 2**(-800), E = 0       (frexp exponent -799)
        shift:  s = 512 - (-799) = 1311
        after:  mantissa = 2**511,    E = -1311   (frexp exponent 512)
                -----------------------------------------------
                represented value remains 2**(-800)

    A shared E keeps value/derivative product-rule terms at the same scale.
    During seed construction these are fields in factored coordinates. Once
    degree advancement starts, previous contains the adjacent-degree value and
    derivative at exactly the SAME E: rescaling only the current degree or
    selected derivative components would change the recurrence's mathematics.
    All arrays are owned by the caller; current and previous states must not
    alias one another. Zero states are left alone; the maximum-based check
    cannot prevent loss of a much smaller component through rounding.
    """
    magnitude = np.maximum(np.abs(value), np.max(np.abs(derivative), axis=1))
    if previous is not None:
        previous_value, previous_derivative = previous
        magnitude = np.maximum(magnitude, np.abs(previous_value))
        magnitude = np.maximum(magnitude, np.max(np.abs(previous_derivative), axis=1))
    binary_exponent = np.frexp(magnitude)[1]
    outside_band = (binary_exponent < _HARMONIC_SCALE_LOWER) | (binary_exponent > _HARMONIC_SCALE_UPPER)
    shift = np.where((magnitude != 0) & outside_band, _HARMONIC_SCALE_TARGET - binary_exponent, 0)
    if not np.any(shift):
        return
    _shift_complex_inplace(value, shift)
    _shift_complex_inplace(derivative, shift[:, None])
    if previous is not None:
        _shift_complex_inplace(previous_value, shift)
        _shift_complex_inplace(previous_derivative, shift[:, None])
    exponent -= shift


def _degree_rescale_interval(degree: int) -> int:
    """Choose scale-check spacing from a conservative degree-growth estimate.

    For unit directions in exact arithmetic, r**2=1, |n_i|<=1. At each
    two-degree step, a<=sqrt(2*degree+1), b<2. If M bounds both live values
    and every derivative component, each next component and intermediate is
    bounded by (2*a+3*b)*M, hence by (2*sqrt(2*degree+1)+6)*M. Keeping the
    previous state cannot increase this bound. The adjacent-degree seed has
    its own immediate check; its formation is not covered by this block bound.

    When growth_bits<=128, block_length*growth_bits<=128. For larger estimates
    the minimum interval of one cannot enforce a 128-bit growth budget. This
    is an upper-growth estimate, not a lower bound against cancellation or
    component underflow. It assumes finite coefficients and unit inputs;
    coefficient evaluation, rounding and arbitrarily large degrees have no
    universal safety proof here. The large unused exponent range is an
    engineering margin, not a floating-point error bound.
    """
    growth_bits = int(np.ceil(np.log2(2 * np.sqrt(2 * degree + 1) + 6)))
    return max(1, _HARMONIC_DEGREE_GROWTH_BUDGET // growth_bits)


def _regular_solid_harmonic(
    degree: int,
    order: int,
    directions: NDArray,
    *,
    scaled: bool,
) -> tuple[NDArray, NDArray, NDArray | None]:
    """Evaluate S_lm and its Cartesian derivative through one normalized recurrence.

    Returned arrays are owned mantissas. With an exponent array, both represent
    physical values multiplied by 2**(-exponent); otherwise they are unscaled.
    """
    x, y, z = directions.T
    value = np.ones(len(x), dtype=complex)
    derivative = np.zeros((len(x), 3), dtype=complex)
    transverse = x + 1j * y
    transverse_gradient = np.array([1.0, 1j, 0.0])
    exponent = None
    degree_block = 1
    if scaled:
        transverse, seed_exponent = _factored_transverse_coordinate(order, directions)
        exponent = np.zeros(len(x), dtype=np.int64)
        _rescale_harmonic_states(value, derivative, exponent)
        degree_block = _degree_rescale_interval(degree)

    for m in range(1, order + 1):
        factor = -np.sqrt((2 * m + 1) / (2 * m))
        derivative = factor * (transverse[:, None] * derivative + value[:, None] * transverse_gradient)
        value = _multiply_harmonic_coordinate(factor, transverse, value, scaled=scaled)
        if exponent is not None and m % _HARMONIC_SEED_RESCALE_INTERVAL == 0:
            _rescale_harmonic_states(value, derivative, exponent)
    if exponent is not None:
        _align_sectoral_scale(value, exponent, seed_exponent, order)

    # The adjacent-degree seed starts the recurrence at fixed |m|.
    if degree > order:
        previous, previous_derivative = value, derivative
        factor = np.sqrt(2 * order + 3)
        derivative = _multiply_harmonic_coordinate(factor, z[:, None], derivative, scaled=scaled)
        derivative[:, 2] += factor * value
        value = _multiply_harmonic_coordinate(factor, z, value, scaled=scaled)
        if exponent is not None:
            _rescale_harmonic_states(value, derivative, exponent, previous=(previous, previous_derivative))
        # Retain r**2 as a polynomial even though evaluation uses unit vectors:
        # its derivative contributes 2*n*S_(l-2,m) to the spatial gradient.
        radius_squared = np.sum(directions**2, axis=1)
        for next_degree in range(order + 2, degree + 1):
            a = np.sqrt((4 * next_degree**2 - 1) / (next_degree**2 - order**2))
            b = np.sqrt(
                (2 * next_degree + 1)
                * ((next_degree - 1) ** 2 - order**2)
                / ((2 * next_degree - 3) * (next_degree**2 - order**2)),
            )
            axial_derivative = _multiply_harmonic_coordinate(a, z[:, None], derivative, scaled=scaled)
            next_derivative = axial_derivative - b * (
                radius_squared[:, None] * previous_derivative + 2 * directions * previous[:, None]
            )
            next_derivative[:, 2] += a * value
            axial_value = _multiply_harmonic_coordinate(a, z, value, scaled=scaled)
            previous, value = value, axial_value - b * radius_squared * previous
            previous_derivative, derivative = derivative, next_derivative
            if exponent is not None and (next_degree - order) % degree_block == 0:
                _rescale_harmonic_states(value, derivative, exponent, previous=(previous, previous_derivative))
    return value, derivative, exponent
