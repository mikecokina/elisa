"""Independent Decimal references for the four principal Roche radius directions.

No ELISA or SciPy equations/solvers are called. Lengths are in semi-major-axis
units. All input floats are converted exactly; 65-digit bisection determines the
first positive radial intersection, not an arbitrary root of the equation.
"""

# ruff: noqa: INP001
from collections.abc import Callable
from decimal import Decimal, getcontext, localcontext
from typing import TypedDict


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
