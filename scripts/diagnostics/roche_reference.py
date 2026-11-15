"""Independent Decimal references for the four principal Roche radius directions.

No ELISA or SciPy equations/solvers are called. Lengths are in semi-major-axis
units. All input floats are converted exactly; 65-digit bisection determines the
first positive radial intersection, not an arbitrary root of the equation.
"""

# ruff: noqa: INP001
from collections.abc import Callable
from decimal import Decimal, getcontext, localcontext
from typing import TypedDict

ANGULAR_MAX_DEPTH = 180


class Reference(TypedDict):
    """High-precision values; absent roots are explicit, never estimated."""

    stationary: Decimal | None
    critical: Decimal | None
    root: Decimal | None


def decimal(value: float | Decimal) -> Decimal:
    """Convert a binary float exactly, retaining deliberate ULP perturbations."""
    return Decimal(value)


class Ray:
    """Component-local Kopal potential on a principal ray."""

    def __init__(self, q: float, spin: float, distance: float, component: str, direction: str) -> None:
        self.q, self.spin, self.distance = map(decimal, (q, spin, distance))
        self.component, self.direction = component, direction
        self.a, self.b = (Decimal(1), self.q) if component == "primary" else (self.q, Decimal(1))
        self.offset = Decimal(0) if component == "primary" else (1 - self.q) / 2
        self.sign = {"forward": 1, "backward": -1, "side": 0, "polar": 0}[direction]
        self.rotation = self.spin**2 * (1 + self.q) / 2 if direction != "polar" else Decimal(0)

    def separation(self, r: Decimal) -> Decimal:
        return (self.distance**2 + r * r - 2 * self.distance * self.sign * r).sqrt()

    def potential(self, r: Decimal) -> Decimal:
        return (
            self.a / r
            + self.b / self.separation(r)
            - self.b * self.sign * r / self.distance**2
            + self.rotation * r * r
            + self.offset
        )

    def derivative(self, r: Decimal) -> Decimal:
        return (
            -self.a / r**2
            - self.b * (r - self.distance * self.sign) / self.separation(r) ** 3
            - self.b * self.sign / self.distance**2
            + 2 * self.rotation * r
        )

    def curvature(self, r: Decimal) -> Decimal:
        """Evaluate the second radial derivative independently of ELISA."""
        separation = self.separation(r)
        return (
            2 * self.a / r**3
            + self.b * (3 * (r - self.distance * self.sign) ** 2 / separation**5 - 1 / separation**3)
            + 2 * self.rotation
        )

    def stationary(self) -> Decimal | None:
        """Find the unique minimum: forward in (0,d), backward/side in (0,infinity).

        Forward/backward potentials are strictly convex. For the side ray,
        derivative/r = -a/r^3-b/(d^2+r^2)^(3/2)+2A is strictly increasing.
        Polar is strictly decreasing and has no finite stationary point.
        """
        if self.direction == "polar":
            return None
        lo = self.distance * Decimal("1e-25")
        hi = self.distance * (1 - Decimal("1e-25")) if self.direction == "forward" else self.distance
        while self.derivative(hi) <= 0:
            hi *= 2
        return bisect(self.derivative, lo, hi)

    def inner_root(self, omega: float | Decimal, stationary: Decimal | None = None) -> Decimal | None:
        """First radial intersection; None if target is below this ray's minimum."""
        omega = decimal(omega)
        lo = self.distance * Decimal("1e-25")
        if self.direction == "polar":
            if omega <= self.offset:
                return None
            hi = self.distance
            while self.potential(hi) > omega:
                hi *= 2
        else:
            hi = stationary if stationary is not None else self.stationary()
            if omega < self.potential(hi):
                return None
        return bisect(lambda r: self.potential(r) - omega, lo, hi)


def bisect(function: Callable[[Decimal], Decimal], lo: Decimal, hi: Decimal) -> Decimal:
    """Bracket a unique root and retain 60+ decimal digits without fsolve."""
    left, right = function(lo), function(hi)
    if left == 0:
        return lo
    if right == 0:
        return hi
    if left * right > 0:
        message = "Reference root is not bracketed"
        raise ValueError(message)
    for _ in range(4 * getcontext().prec):
        mid = (lo + hi) / 2
        if mid in (lo, hi):
            break
        value = function(mid)
        if value == 0:
            return mid
        if (value > 0) == (left > 0):
            lo, left = mid, value
        else:
            hi = mid
    return (lo + hi) / 2


def reference(
    q: float,
    spin: float,
    distance: float,
    component: str,
    direction: str,
    omega: float | Decimal | None = None,
    precision: int = 65,
) -> Reference:
    """Return independently derived stationary and first-intersection data."""
    with localcontext() as ctx:
        ctx.prec = precision
        ray = Ray(q, spin, distance, component, direction)
        stationary = ray.stationary()
        critical = None if stationary is None else ray.potential(stationary)
        root = None if omega is None else ray.inner_root(omega, stationary)
        return {"stationary": stationary, "critical": critical, "root": root}


def decimal_sincos(angle: float) -> tuple[Decimal, Decimal]:
    """Independent Taylor evaluation at the exact input float (diagnostic angles)."""
    x = Decimal(angle)
    sine, cosine = x, Decimal(1)
    sin_term, cos_term = sine, cosine
    for n in range(1, 4 * getcontext().prec + 100):
        sin_term *= -x * x / ((2 * n) * (2 * n + 1))
        cos_term *= -x * x / ((2 * n - 1) * (2 * n))
        previous = sine, cosine
        sine += sin_term
        cosine += cos_term
        if previous == (sine, cosine):
            return sine, cosine
    message = "Reference trigonometric series did not converge"
    raise ArithmeticError(message)


class AngularRay(Ray):
    """Independent arbitrary-angle oracle with analytic companion-term extrema.

    Unlike production's separate term bounds, this reference bounds the complete
    companion derivative and curvature by evaluating all their internal extrema.
    Decimal arithmetic and bisection are used throughout. No production geometry,
    trigonometry, bracket or stationary radius is reused.
    """

    def __init__(self, q: float, spin: float, distance: float, component: str, phi: float, theta: float) -> None:
        super().__init__(q, spin, distance, component, "forward")
        sin_phi, cos_phi = decimal_sincos(phi)
        sin_theta, cos_theta = decimal_sincos(theta)
        self.sign = cos_phi * sin_theta
        self.transverse = (sin_phi * sin_theta) ** 2 + cos_theta**2
        self.rotation *= sin_theta**2

    def separation(self, r: Decimal) -> Decimal:
        return ((r - self.distance * self.sign) ** 2 + self.distance**2 * self.transverse).sqrt()

    def stationary(self) -> Decimal | None:
        center = self.distance * self.sign
        width2 = self.distance**2 * self.transverse

        def companion_derivative(u: Decimal) -> Decimal:
            return -u / (u * u + width2) ** Decimal("1.5")

        def companion_curvature(u: Decimal) -> Decimal:
            return (2 * u * u - width2) / (u * u + width2) ** Decimal("2.5")

        def search(lo: Decimal, hi: Decimal, depth: int) -> Decimal | None:
            left, right = lo - center, hi - center
            # The curvature has maxima at +/-sqrt(3/2)*width and a minimum
            # at zero. Only zero and the endpoints can attain its minimum.
            candidates = [companion_curvature(left), companion_curvature(right)]
            if left <= 0 <= right:
                candidates.append(companion_curvature(Decimal(0)))
            lower = 2 * self.a / hi**3 + self.b * min(candidates) + 2 * self.rotation
            if lower > 0:
                if self.derivative(hi) <= 0:
                    return None
                return bisect(self.derivative, lo, hi)
            # Companion derivative extrema occur at +/-width/sqrt(2).
            extrema = [left, right]
            turn = (width2 / 2).sqrt()
            extrema.extend(u for u in (-turn, turn) if left <= u <= right)
            upper = (
                -self.a / hi**2
                + self.b * max(map(companion_derivative, extrema))
                - self.b * self.sign / self.distance**2
                + 2 * self.rotation * hi
            )
            if upper < 0:
                return None
            if depth > ANGULAR_MAX_DEPTH:
                message = "Independent angular stationary isolation failed"
                raise ArithmeticError(message)
            mid = (lo + hi) / 2
            first = search(lo, mid, depth + 1)
            return first if first is not None else search(mid, hi, depth + 1)

        return search(self.distance * Decimal("1e-25"), Decimal(30), 0)


def angular_reference(
    q: float,
    spin: float,
    distance: float,
    component: str,
    phi: float,
    theta: float,
    omega: float | None = None,
    precision: int = 65,
) -> Reference:
    """Return first angular minimum and inner intersection, independently."""
    with localcontext() as ctx:
        ctx.prec = precision
        ray = AngularRay(q, spin, distance, component, phi, theta)
        stationary = ray.stationary()
        end = stationary if stationary is not None else Decimal(30)
        critical = ray.potential(end)
        root = None
        if omega is not None and Decimal(omega) >= critical:
            root = bisect(lambda r: ray.potential(r) - Decimal(omega), Decimal(distance) * Decimal("1e-25"), end)
        return {"stationary": stationary, "critical": critical, "root": root}
