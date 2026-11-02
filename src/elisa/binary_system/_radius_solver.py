"""First intersections on component-local Roche rays.

A ray starts at the selected star's centre and follows one fixed direction.
Its potential starts at +infinity and initially decreases. We seek the first
surface intersection on that decreasing branch, before the first radial minimum.
Searching beyond that minimum could select the outer surface branch; crossing
the companion's centre could select an entirely different surface.

The calculation has three stages:

1. ``RocheRay.first_turn`` determines the end of the decreasing branch.
2. ``solve_radius`` brackets the intersection between the centre and that end.
3. Near a critical potential, ``_critical_radius`` uses Decimal arithmetic to
   distinguish an existing intersection from a rounded critical limit or a
   genuinely missing intersection.

Principal directions have analytic uniqueness properties. Other directions
require interval bounds and subdivision to locate their *first* minimum.
"""

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, localcontext
from math import cos, inf, isfinite, sin, sqrt

import numpy as np
from scipy.optimize import brentq

from elisa.base.error import raise_elisa_max_iteration_error, raise_elisa_numeric_error

EPS: float = float(np.finfo(float).eps)
# Safety ceiling inherited from the public radius API, in semi-major-axis units.
RADIUS_LIMIT = 30.0
# An unresolvable general-ray interval fails after this many subdivisions.
MAX_SUBDIVISION_DEPTH = 60
# Switching affects evaluation precision only, never acceptance of a root.
PRECISE_POTENTIAL_GAP = 1e-7
# Maximum relative radius uncertainty allowed when accepting a critical limit.
CRITICAL_RADIUS_RTOL = 1e-7
# Potential roundoff budget for a critical value supplied by float arithmetic.
CRITICAL_ROUNDING_UNITS = 4
# Conservative arithmetic slack for derivative/curvature interval decisions.
INTERVAL_ROUNDING_UNITS = 32
PRECISE_DIGITS = 50


def bracket_root(function: Callable[[float], float], left: float, right: float) -> float:
    """Solve a geometrically selected bracket to float relative precision.

    The caller must establish a sign change (or an endpoint root) and select
    the desired branch. This helper serves both potential and derivative roots;
    it does not determine their physical meaning. At the stellar centre the
    callbacks supply the analytic infinite limit instead of dividing by zero.

    Raise ElisaMaxIterationError on nonconvergence and ElisaNumericError for
    a nonpositive, nonfinite or out-of-bracket result. Other callback errors
    propagate to the caller.
    """

    # The smallest positive xtol avoids imposing an absolute length scale.
    # scipy requires rtol >= 4*eps; geometry, rather than solver status, selects
    # the physical branch before this helper is entered.
    def objective(radius: float) -> float:
        # Present SciPy with a concrete function, including when the supplied
        # Callable is a bound method or a callable object.
        return function(radius)

    root, result = brentq(objective, left, right, xtol=5e-324, rtol=4 * EPS, full_output=True, disp=False)
    if not result.converged:
        raise_elisa_max_iteration_error("Roche radius bracket solver did not converge.")
    if not isfinite(root) or not left <= root <= right or root <= 0:
        raise_elisa_numeric_error("Roche solver returned a radius outside its geometric bracket.")
    return root


@dataclass(frozen=True)
class RocheRay:
    """Immutable Roche potential along a component-local radial direction.

    ``Omega(r) = a/r + b/s - b*c*r/d**2 + rotation*r**2 + offset``,
    where ``s`` is the distance from the point to the companion's centre.
    Lengths r, s and d are in semi-major-axis units.

    Attributes:
        a: Own-star gravity coefficient: 1 for primary, q for secondary.
        b: Companion gravity coefficient: q for primary, 1 for secondary.
        d: Instantaneous separation between the stellar centres.
        c: Direction cosine along local +x, which points to the companion.
        rotation: F**2*(1+q)*sin(theta)**2/2, the quadratic coefficient.
        offset: Potential zero-point correction: 0 or (1-q)/2.
        transverse_squared: Squared direction perpendicular to local x.
            Evaluated from transverse components to retain very small angles,
            even when c has rounded to exactly one.

    """

    a: float
    b: float
    d: float
    c: float
    rotation: float
    offset: float
    transverse_squared: float

    def separation(self, r: float) -> float:
        """Return the distance to the companion, including off-axis displacement."""
        # This form avoids cancellation near the companion.
        return sqrt((r - self.d * self.c) ** 2 + self.d**2 * self.transverse_squared)

    def potential(self, r: float) -> float:
        """Evaluate Omega(r), using +infinity at either stellar centre."""
        if r == 0:
            # Own-star singularity: use the inward endpoint limit.
            return inf
        s = self.separation(r)
        if s == 0:
            # Companion singularity; only an exactly forward ray reaches it.
            return inf
        return self.a / r + self.b / s - self.b * self.c * r / self.d**2 + self.rotation * r * r + self.offset

    def derivative(self, r: float) -> float:
        """Evaluate dOmega/dr on the interval connected to the stellar centre."""
        if r == 0:
            # Omega decreases outward from the own-star singularity.
            return -inf
        s = self.separation(r)
        if s == 0:
            # At r=d on a forward ray, use the limit approached from r<d.
            # This is not the derivative on the far side of the companion.
            return inf
        return (
            -self.a / r**2 - self.b * (r - self.d * self.c) / s**3 - self.b * self.c / self.d**2 + 2 * self.rotation * r
        )

    def first_turn(self) -> float:
        """Return the end of the first decreasing branch, as a radius.

        Usually this is a stationary minimum. If the potential keeps decreasing
        throughout the supported radius range, return RADIUS_LIMIT instead.
        Thus the returned endpoint must not always be interpreted as critical.

        Forward/backward potentials are strictly convex. For perpendicular
        rays, Omega'(r)/r is increasing (or remains negative without rotation).
        These properties make an endpoint derivative test sufficient. General
        rays lack that guarantee and use interval subdivision instead.
        """
        # Only the exactly forward ray has a second singularity at positive r.
        # Backward and off-axis rays may extend beyond the separation d.
        end = min(self.d, RADIUS_LIMIT) if self.c == 1 and self.transverse_squared == 0 else RADIUS_LIMIT
        if self.c == 0 or self.transverse_squared == 0:
            # Fast path: axial/perpendicular geometry permits at most one turn.
            if self.derivative(end) <= 0:
                # No turn before the ceiling; polar is the usual example.
                return end
            # Negative derivative at the centre, positive at end: unique minimum.
            return bracket_root(self.derivative, 0.0, end)
        # None means every interval up to the ceiling was shown to decrease.
        return self._turn_interval(0.0, end, 0) or end

    def _turn_interval(self, lo: float, hi: float, depth: int) -> float | None:
        """Find the first derivative zero in [lo, hi], or prove there is none.

        Earlier intervals have already been shown to decrease, so the search
        enters this interval on the inner branch. A negative derivative bound
        excludes a turn. A positive curvature bound makes the derivative
        increasing and allows a sign bracket. Otherwise subdivide, visiting
        the inner half first so a later minimum cannot hide the first one.

        Raise ElisaNumericError if subdivision cannot resolve the geometry.
        """
        # Distance to the companion is smallest at its projection onto the ray
        # (clamped to this interval), and largest at one of the endpoints.
        center = self.d * self.c
        nearest = min(hi, max(lo, center))
        smin = self.separation(nearest)
        smax = max(self.separation(lo), self.separation(hi))
        # Outward slack covers arithmetic roundoff in interval bounds.
        # Bound each term of Omega' from above. For the companion term both
        # signs of (lo-center) are possible, hence the maximum over distances.
        tidal = max(-(lo - center) / smin**3, -(lo - center) / smax**3)
        terms = (-self.a / hi**2, self.b * tidal, -self.b * self.c / self.d**2, 2 * self.rotation * hi)
        slack = INTERVAL_ROUNDING_UNITS * EPS * sum(abs(t) for t in terms)
        if sum(terms) < -slack:
            # Omega' is negative everywhere in this interval: skip it.
            return None
        # Retain a lower bound for the positive axial curvature term. Dropping
        # it entirely does NOT converge to the true curvature as hi-lo -> 0,
        # and can leave a perfectly regular first minimum uncertifiable.
        # |r-center| >= axial_min and s <= smax throughout the interval, so
        # 3*axial_min**2/smax**5 <= 3*(r-center)**2/s**5.
        axial_min = abs(nearest - center)
        curvature_terms = (
            2 * self.a / hi**3,
            3 * self.b * (axial_min / smax) ** 2 / smax**3,
            -self.b / smin**3,
            2 * self.rotation,
        )
        curvature = sum(curvature_terms)
        curvature_slack = INTERVAL_ROUNDING_UNITS * EPS * sum(abs(term) for term in curvature_terms)
        if curvature > curvature_slack:
            if self.derivative(hi) <= 0:
                # Even the largest derivative is nonpositive: no turn here.
                return None
            # The derivative changes sign exactly once in this convex interval.
            return bracket_root(self.derivative, lo, hi)
        # Neither bound decided the interval. Refine rather than assuming
        # a single minimum for an arbitrary direction.
        if depth >= MAX_SUBDIVISION_DEPTH:
            raise_elisa_numeric_error("Cannot certify the first stationary point on this Roche ray.")
        mid = (lo + hi) / 2
        if mid in (lo, hi):
            raise_elisa_numeric_error("Cannot certify this Roche interval at floating-point resolution.")
        first = self._turn_interval(lo, mid, depth + 1)
        return first if first is not None else self._turn_interval(mid, hi, depth + 1)


def solve_radius(spin: float, q: float, d: float, omega: float, *, secondary: bool, phi: float, theta: float) -> float:
    """Return a component-local surface radius on the first decreasing branch.

    Args:
        spin: Component rotational synchronicity F.
        q: Binary mass ratio M_secondary/M_primary, strictly positive.
        d: Instantaneous separation in semi-major-axis units, strictly positive.
        omega: Requested component surface potential in ELISA's convention.
        secondary: Select secondary local coefficients when True.
        phi: Local longitude in radians, measured from the companion direction.
        theta: Colatitude in radians, measured from the north pole.

    Returns:
        Positive radius at the first inner-branch intersection. For a target
        rounded slightly below a stationary minimum, may return that limiting
        radius if both potential and radius uncertainty budgets permit it.

    Raises:
        ElisaNumericError: Invalid inputs, no supported inner intersection,
            unresolved geometry, or an inadmissible critical limit.
        ElisaMaxIterationError: A bracketed numerical solve fails to converge.

    """
    if not all(isfinite(v) for v in (spin, q, d, omega, phi, theta)) or q <= 0 or d <= 0:
        raise_elisa_numeric_error("Roche radius requires finite inputs and positive mass ratio and separation.")
    c = cos(phi) * sin(theta)
    # Exact canonical angles avoid artificial transverse components on axes.
    if phi == np.pi / 2 or theta == 0:
        c = 0.0
    transverse_squared = (sin(phi) * sin(theta)) ** 2 + cos(theta) ** 2
    if theta == np.pi / 2:
        # In the equatorial plane the z component is exactly zero.
        transverse_squared = sin(phi) ** 2
    if phi in (0.0, np.pi) and theta == np.pi / 2:
        # Forward/backward must have no spurious sin(pi) transverse component.
        transverse_squared = 0.0
    # r=0 is the chosen component's own centre in BOTH coordinate systems.
    a, b = (q, 1.0) if secondary else (1.0, q)
    offset = (1 - q) / 2 if secondary else 0.0
    ray = RocheRay(a, b, d, c, spin**2 * (1 + q) * sin(theta) ** 2 / 2, offset, transverse_squared)
    end = ray.first_turn()
    # "minimum" is the lowest potential on the supported inner interval.
    # At the radius ceiling it need not be a stationary/global minimum.
    minimum = ray.potential(end)
    # Use term magnitudes: the final potential alone can hide cancellation.
    scale = (
        abs(a / end)
        + abs(b / ray.separation(end))
        + abs(b * c * end / d**2)
        + abs(ray.rotation * end**2)
        + abs(offset)
        + abs(omega)
    )
    if abs(minimum - omega) > PRECISE_POTENTIAL_GAP * scale:
        # Well separated from criticality: float arithmetic resolves the sign.
        if minimum > omega:
            # The entire inner interval lies above the target: no intersection.
            raise_elisa_numeric_error("No component-local Roche radius on the inner radial branch.")
        # Omega(0+) > target >= Omega(end); solve on this decreasing interval.
        return bracket_root(lambda r: ray.potential(r) - omega, 0.0, end)
    # Close values can cancel in float64. Re-evaluate before deciding existence.
    return _critical_radius(ray, spin, q, theta, omega, end)


def _critical_radius(ray: RocheRay, spin: float, q: float, theta: float, omega: float, end: float) -> float:  # noqa: RET503 - raising helper is NoReturn
    """Resolve a near-critical target with 50-digit potential arithmetic.

    ``end`` is the previously selected inner-branch endpoint. Its radius stays
    a float; coefficients and potential differences are recomputed in Decimal
    from the supplied float inputs to avoid losing a small difference between
    large potential terms. The local context does not change caller precision.

    Three outcomes are distinguished:

    * Omega(end) <= target: a bracketed intersection exists; solve for it,
      including targets just above the minimum instead of snapping them to it.
    * Omega(end) > target by an admissible roundoff-sized amount: return the
      stationary limiting radius only if its radius uncertainty is acceptable.
    * Otherwise: raise ElisaNumericError because no supported inner root exists.

    The radius ceiling is never accepted as a rounded stationary limit.
    """
    with localcontext(prec=PRECISE_DIGITS):
        # Reconstruct the rotational coefficient before float multiplication
        # could round it. Direction components are those of the selected ray.
        a, b, d, c = map(Decimal, (ray.a, ray.b, ray.d, ray.c))
        rotation = Decimal(spin) ** 2 * (1 + Decimal(q)) * Decimal(sin(theta)) ** 2 / 2
        offset = (1 - Decimal(q)) / 2 if ray.offset != 0 else Decimal(0)

        def residual(r: float) -> Decimal:
            if r == 0:
                return Decimal(inf)
            radius = Decimal(r)
            separation = ((radius - d * c) ** 2 + d * d * Decimal(ray.transverse_squared)).sqrt()
            return (
                a / radius
                + b / separation
                - b * c * radius / d**2
                + rotation * radius * radius
                + offset
                - Decimal(omega)
            )

        # Sign convention: positive gap means the target is BELOW the minimum.
        gap = residual(end)
        if gap <= 0:
            # Existing intersection. Subtract in Decimal, then convert the small
            # residual to float for Brent; converting potentials first loses it.
            return bracket_root(lambda r: float(residual(r)), 0.0, end)
        # No exact intersection: assess whether the target can represent the
        # rounded critical potential, rather than silently inventing a root.
        endpoint_radius = Decimal(end)
        endpoint_separation = ((endpoint_radius - d * c) ** 2 + d * d * Decimal(ray.transverse_squared)).sqrt()
        scale = (
            abs(a / endpoint_radius)
            + abs(b / endpoint_separation)
            + abs(b * c * endpoint_radius / d**2)
            + abs(rotation * endpoint_radius**2)
            + abs(offset)
            + abs(Decimal(omega))
        )
        curvature = (
            2 * a / endpoint_radius**3
            + b * (3 * (endpoint_radius - d * c) ** 2 / endpoint_separation**5 - 1 / endpoint_separation**3)
            + 2 * rotation
        )
        # Four rounding units per signed term allow float coefficient evaluation
        # in library critical potentials; this is not an absolute potential band.
        # Near a minimum: DeltaOmega ~= curvature * DeltaRadius**2 / 2.
        # This converts potential ambiguity to a local radius uncertainty;
        # nonpositive curvature cannot justify a minimum and is rejected.
        uncertainty = (2 * gap / curvature).sqrt() if curvature > 0 else Decimal("Infinity")
        # All three conditions are required: a stationary endpoint, a plausible
        # float rounding discrepancy, and an acceptable geometric uncertainty.
        if (
            end < RADIUS_LIMIT
            and gap <= Decimal(CRITICAL_ROUNDING_UNITS * EPS) * scale
            and uncertainty <= Decimal(CRITICAL_RADIUS_RTOL) * endpoint_radius
        ):
            return end
        raise_elisa_numeric_error("Surface potential is below the resolvable Roche critical limit.")
