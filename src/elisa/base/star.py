from __future__ import annotations

from copy import copy, deepcopy
from typing import TYPE_CHECKING, Any

from elisa import umpy as up
from elisa import units as u
from elisa import utils
from elisa.base.body import Body
from elisa.base.container import StarPropertiesContainer
from elisa.base.transform import StarProperties
from elisa.logger import getLogger
from elisa.pulse.mode import PulsationMode

if TYPE_CHECKING:
    from collections.abc import Iterable

    from elisa.types import Float
    from elisa.units import _DefaultStarInputUnits, _DefaultStarUnits

logger = getLogger("elisa.base.star")


class Star(Body):
    """Star component used in stellar system models.

    Overview
    --------
    Child class of :class:`elisa.base.body.Body` representing a single
    stellar component. The class provides initialization, input
    transformation and serialization helpers used by different system
    types (for example :class:`elisa.single_system.SingleSystem` and
    :class:`elisa.binary_system.BinarySystem`). Use this class directly
    via::

        from elisa import Star

    The constructor accepts a set of keyword arguments (documented
    below). Values may be plain numbers or :class:`astropy.units.Quantity`
    instances; when units are omitted, the library assumes sensible
    defaults (for example temperature in Kelvin).

    Mandatory arguments (SingleSystem)
    ---------------------------------
    These arguments are required when the star is created as a
    component of a ``SingleSystem``::

    :param mass: Stellar mass. If given as a plain number (``int`` or
        ``float``) it is assumed to be in solar masses; if an
        :class:`astropy.units.Quantity` is supplied it is converted to
        the library internal units.
    :type mass: float | astropy.units.Quantity

    :param t_eff: Effective temperature. Accepts a numeric value or a
        :class:`astropy.units.Quantity`. When given without units, the
        value is interpreted as Kelvin.
    :type t_eff: float | astropy.units.Quantity

    :param polar_log_g: Base-10 logarithm of the polar surface gravity.
    :type polar_log_g: float

    Optional / derived (SingleSystem)
    ---------------------------------
    The following optional arguments influence model behavior; when
    omitted, default values or interpolated tables are used:

    :param metallicity: Metallicity [M/H]. Default is ``0.0``.
    :type metallicity: float

    :param gravity_darkening: Gravity-darkening exponent. If omitted
        the value is interpolated from Claret (2003) tables based on
        ``t_eff``.
    :type gravity_darkening: float

    :param limb_darkening_coefficients: Optional limb-darkening
        coefficients for the entire star (useful when the object lies
        outside the supported atmosphere grid). It may be a single numeric
        value or a mapping of passband name to coefficients, e.g.
        ``{ 'V': [a, b, ...] }``.
    :type limb_darkening_coefficients: float | dict[str, list[float]]

    Mandatory arguments (BinarySystem)
    ---------------------------------
    When the star is a component of a ``BinarySystem`` the following
    keywords are typically required::

    :param mass: See description above.
    :type mass: float | astropy.units.Quantity

    :param t_eff: See description above.
    :type t_eff: float | astropy.units.Quantity

    :param surface_potential: Generalised surface potential (Wilson 1979
        convention) defining the Roche equipotential.
    :type surface_potential: float

    :param synchronicity: Rotation/orbital frequency ratio
        (omega_rot / omega_orb). Equals ``1`` for synchronous
        rotation.
    :type synchronicity: float

    :param albedo: Surface albedo in the interval (0, 1).
    :type albedo: float

    Additional derived attributes (available after system initialization)
    -------------------------------------------------------------------
    After the containing system is initialized (single or binary), the
    :class:`Star` instance will expose several derived attributes (these
    are computed by the system initialization code rather than by the
    :class:`Star` constructor):

    - ``critical_surface_potential`` (:class:`float`) -- potential at
      which the star fills its Roche lobe;
    - ``equivalent_radius`` (:class:`float`) -- radius of a sphere with
      the same volume as the component (units: semi-major axis);
    - ``filling_factor`` (:class:`float`) -- defined as
      ``(Omega_inner - Omega) / (Omega_inner - Omega_outer)`` and
      interpreted as::

          filling_factor < 0  -> component does not fill its Roche lobe
          filling_factor = 0  -> component fills precisely its Roche lobe
          0 < filling_factor < 1 -> component overflows its Roche lobe
          filling_factor = 1  -> upper boundary (further increase implies mass loss)

    Radii at periastron (for eccentric systems)
    -------------------------------------------
    The following radii (in units of the semi-major axis) are made
    available after system initialization:

    :polar_radius: Radius measured toward the stellar pole.
    :side_radius: Radius perpendicular to the pole and the direction to the companion.
    :backward_radius: Radius opposite the direction toward the companion.
    :forward_radius: Radius toward the companion (it may be ``numpy.nan`` for
        over-contact systems).

    Optional parameters
    -------------------
    Additional optional parameters accepted by the constructor include:

    :param spots: A list of spot definitions. The order of list items
        defines layering (the first spot lies below subsequently
        defined overlapping spots). Each spot is a mapping with keys
        such as ``longitude``, ``latitude``, ``angular_radius`` and
        ``temperature_factor``. Example::

            [
                {"longitude": 90, "latitude": 58, "angular_radius": 15, "temperature_factor": 0.9},
                {"longitude": 85, "latitude": 80, "angular_radius": 30, "temperature_factor": 1.05},
                {"longitude": 45, "latitude": 90, "angular_radius": 30, "temperature_factor": 0.95},
            ]

    :type spots: list[dict[str, float]]

    :param pulsations: Pulsation mode descriptions (list of dicts).
        Each dict must contain mode metadata (for example ``l``, ``m``,
        ``amplitude``, ``frequency``). The property is stored as a
        mapping index → :class:`PulsationMode` internally.
    :type pulsations: list[dict[str, float]]

    :param atmosphere: Atmosphere identifier used for spectral models.
        Supported aliases include::

            - ``castelli``, ``castelli-kurucz``, ``ck``, ``ck04``
            - ``kurucz``, ``k``, ``k93``

    :type atmosphere: str

    Notes
    -----
    - The constructor accepts :class:`astropy.units.Quantity` values
      where meaningful; when units are omitted the default input units
      (see :meth:`default_input_units`) are assumed.
    - The class exposes convenience methods to serialize and convert
      instance state to :class:`elisa.base.container.StarPropertiesContainer`
      via :meth:`to_properties_container`.

    """

    MANDATORY_KWARGS = ("mass", "t_eff")
    OPTIONAL_KWARGS = (
        "surface_potential",
        "synchronicity",
        "albedo",
        "pulsations",
        "atmosphere",
        "spots",
        "metallicity",
        "polar_log_g",
        "discretization_factor",
        "gravity_darkening",
        "limb_darkening_coefficients",
    )
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, *, name: str | None = None, **kwargs: Any) -> None:
        utils.invalid_kwarg_checker(kwargs, Star.ALL_KWARGS, Star)
        super().__init__(name, **kwargs)
        kwargs = self.transform_input(**kwargs)

        # default values of properties
        self.filling_factor = up.NaN
        self.critical_surface_potential = up.NaN
        self.surface_potential = up.NaN
        self.metallicity: Float = 0.0
        self.polar_log_g = up.NaN
        self.gravity_darkening = up.NaN
        # Test against None value is provided across the codebase, so we need to
        # set it to None here instead of any other value (for example an empty dict) to avoid confusion.
        self.limb_darkening_coefficients: dict[str, list[float]] | None = None
        self._pulsations: dict[int, PulsationMode] | list[PulsationMode] = []

        self.side_radius = up.NaN
        self.forward_radius = up.NaN
        self.backward_radius = up.NaN
        self.equivalent_radius = up.NaN

        self.init_parameters(**kwargs)

    @property
    def default_input_units(self) -> _DefaultStarInputUnits:
        """Return the default units used for input parameters.

        Returned object defines the units that are assumed when numerical
        input values are provided without units (for example the
        ``t_eff`` parameter is interpreted in Kelvin by default).

        :returns: Default input units descriptor.
        :rtype: elisa.units.DefaultStarInputUnits
        """
        return u.DefaultStarInputUnits

    @property
    def default_internal_units(self) -> _DefaultStarUnits:
        """Return the internal units used within the Star instance.

        These units are used to store and compute internal physical
        quantities.

        :returns: Default internal units' descriptor.
        :rtype: elisa.units.DefaultStarUnits
        """
        return u.DefaultStarUnits

    def transform_input(self, **kwargs) -> dict[str, Any]:
        """Transform and validate initialization keyword arguments.

        Uses :class:`elisa.base.transform.StarProperties` to normalize and
        validate input values provided to the constructor.

        :param kwargs: Keyword arguments forwarded from :meth:`__init__`.
        :type kwargs: dict
        :returns: Transformed keyword arguments ready to be consumed by
            the rest of the initialization logic.
        :rtype: dict
        """
        return StarProperties.transform_input(**kwargs)

    def init_parameters(self, **kwargs) -> None:
        """Set initial attribute values from transformed keyword args.

        Iterates over all recognized keys and assigns attribute values on
        the instance when present in ``kwargs``.

        :param kwargs: Transformed initialization parameters.
        :type kwargs: dict
        :returns: None
        :rtype: None
        """
        logger.debug("initialising properties of class instance %s", self.__class__.__name__)
        for kwarg in Star.ALL_KWARGS:
            if kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])

    def kwargs_serializer(self) -> dict[str, Any]:
        """Serialize keyword-arguments representing the current instance.

        Produces a dict that can be passed back to the constructor to
        re-create the instance with the same parameters. Numerical values
        that should carry units are converted to quantities using the
        ``default_units`` mapping when needed.

        :returns: Dictionary of serialized keyword arguments.
        :rtype: dict[str, Any]
        """
        default_units = {
            "mass": u.kg,
            "t_eff": u.K,
            "discretization_factor": u.rad,
        }

        serialized_kwargs: dict[str, Any] = {}
        for kwarg in self.ALL_KWARGS:
            if kwarg == "spots":
                # important: this relies on dict ordering
                spots_attr = getattr(self, kwarg)
                # Support both dict and list-like spot collections
                if hasattr(spots_attr, "values"):
                    value = [spot.kwargs_serializer() for spot in spots_attr.values()]
                else:
                    value = [spot.kwargs_serializer() for spot in spots_attr]
            elif kwarg in default_units:
                value = getattr(self, kwarg)
                if not isinstance(value, u.Quantity):
                    value = value * default_units[kwarg]
            else:
                value = getattr(self, kwarg)

            serialized_kwargs[kwarg] = value
        return serialized_kwargs

    def init(self) -> None:
        """Re-initialize the instance from its serialized kwargs."""
        self.__init__(**self.kwargs_serializer())

    def has_pulsations(self) -> bool:
        """Return True when the star has defined pulsation modes.

        :returns: True if at least one pulsation mode is present.
        :rtype: bool
        """
        return len(self._pulsations) > 0

    @property
    def pulsations(self) -> dict[int, PulsationMode] | list[PulsationMode]:
        """Return pulsation modes attached to the star.

        The property returns either a list or a dict-like mapping of
        pulsation modes. When set via the :paramref:`pulsations` setter,
        a mapping {index: :class:`PulsationMode`} is used.

        :returns: Pulsation modes collection.
        :rtype: dict[int, PulsationMode] | list[PulsationMode]
        """
        return self._pulsations

    @pulsations.setter
    def pulsations(self, pulsations: Iterable[dict] | None) -> None:
        """Set pulsation modes from a sequence of mode descriptors.

        The expected input is an iterable of dict-like mode descriptors,
        for example ``[{"l":0, "m":0, "amplitude":0.1, "frequency":5.0}, ...]``.

        :param pulsations: Iterable of pulsation metadata dicts, or ``None``/``[]``.
        :type pulsations: Iterable[dict] | None
        :returns: None
        :rtype: None
        """
        if pulsations in [None, []]:
            self._pulsations = {}
        elif pulsations:
            self._pulsations = {idx: PulsationMode(**pulsation_meta) for idx, pulsation_meta in enumerate(pulsations)}

    def properties_serializer(self) -> dict[str, Any]:
        """Prepare a dict of properties suitable for constructing a :class:`StarPropertiesContainer`.

        :returns: Mapping of property name to value used by the container
            constructor.
        :rtype: dict[str, Any]
        """
        properties_list = [
            "mass",
            "t_eff",
            "synchronicity",
            "albedo",
            "discretization_factor",
            "equivalent_radius",
            "polar_radius",
            "equatorial_radius",
            "gravity_darkening",
            "surface_potential",
            "pulsations",
            "metallicity",
            "polar_log_g",
            "critical_surface_potential",
            "atmosphere",
            "side_radius",
            "limb_darkening_coefficients",
        ]
        props: dict[str, Any] = {prop: copy(getattr(self, prop)) for prop in properties_list}
        props.update(
            {
                "name": self.name,
                "spots": deepcopy(self.spots),
            },
        )
        return props

    def to_properties_container(self) -> StarPropertiesContainer:
        """Return a :class:`StarPropertiesContainer` representing this star.

        :returns: Serialized star properties' container.
        :rtype: elisa.base.container.StarPropertiesContainer
        """
        return StarPropertiesContainer(**self.properties_serializer())
