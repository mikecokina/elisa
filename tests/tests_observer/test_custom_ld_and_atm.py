from __future__ import annotations

import pickle
from copy import copy, deepcopy
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from elisa import BinarySystem, Observer, get_default_binary_definition, ld, settings
from elisa.base.container import StarContainer
from elisa.base.curves import utils as curves_utils
from elisa.base.curves.utils import get_component_limbdarkening_cfs
from elisa.base.error import ElisaInitialParametersError
from elisa.base.star import Star
from elisa.base.transform import StarProperties
from tests.utils import SOLAR_MODEL, ElisaTestCase, prepare_single_system

if TYPE_CHECKING:
    from numpy.typing import NDArray

    import elisa.binary_system.system


@pytest.mark.parametrize(
    ("law", "coefficients"),
    [
        ("linear", "not a coefficient mapping"),
        ("logarithmic", {"bolometric": 0.5}),
        ("square_root", {"bolometric": [0.5]}),
    ],
)
def test_invalid_custom_coefficient_definition_is_initial_configuration_error(
    law: str,
    coefficients: dict[str, float] | dict[str, list[float]] | str,
) -> None:
    with settings.override(LIMB_DARKENING_LAW=law), pytest.raises(ElisaInitialParametersError):
        StarProperties.limb_darkening_coefficients(coefficients)


class DummyComponent:
    """Minimal stand-in for a StarContainer used in curves utils tests.

    Attributes are intentionally minimal and match names accessed by the
    functions under test.
    """

    def __init__(
        self,
        temperatures: list[float],
        log_g: list[float],
        metallicity: float,
        limb_darkening_coefficients: dict[str, NDArray[np.float64]] | None = None,
    ) -> None:
        self.temperatures = np.asarray(temperatures)
        self.log_g = np.asarray(log_g)
        self.metallicity = metallicity
        self.limb_darkening_coefficients = limb_darkening_coefficients
        # surfaces-related arrays: use one value per face
        faces = self.temperatures.shape[0]
        self.indices = np.arange(faces)
        # Normal radiance and ld_cfs are dicts keyed by band name and have per-face arrays
        self.normal_radiance = {"bolometric": np.ones(faces)}
        self.ld_cfs = {"bolometric": np.ones((faces, 2))}
        self.los_cosines = np.ones(faces)
        self.coverage = np.ones(faces)
        # t_eff and log_g presence for other helpers
        self.t_eff = float(self.temperatures[0]) if self.temperatures.size else 5000.0

    # noinspection PyMethodMayBeStatic
    def symmetry_faces(self, arr: NDArray[np.generic]) -> NDArray[np.generic]:
        # Identity for tests which don't rely on symmetry transform
        return arr

    # noinspection PyMethodMayBeStatic
    def mirror_face_values(self, vals: NDArray[np.generic]) -> NDArray[np.generic]:
        # For testing, return vals unchanged; real implementation mirrors values
        return vals


class BinarySystemSeparatedAtmospheres(ElisaTestCase):
    def setUp(self) -> None:
        super().setUp()

    @staticmethod
    def test_atmospheres_of_components_differs() -> None:
        definition = get_default_binary_definition()
        definition["primary"].update({**definition["primary"], "atmosphere": "bb"})
        definition["secondary"].update({**definition["secondary"], "atmosphere": "ck04"})
        binary = BinarySystem.from_json(definition)
        assert binary.primary.atmosphere == "bb"
        assert binary.secondary.atmosphere == "ck"

    @staticmethod
    def test_custom_lds_linear() -> None:
        definition = get_default_binary_definition()
        definition["primary"].update({**definition["primary"], "limb_darkening_coefficients": {"bolometric": 0.5}})
        binary = BinarySystem.from_json(definition)
        assert_array_equal(binary.primary.limb_darkening_coefficients["bolometric"], [0.5])

    @staticmethod
    def test_custom_lds_log() -> None:
        settings.configure(LIMB_DARKENING_LAW="logarithmic")
        definition = get_default_binary_definition()
        definition["primary"].update(
            {**definition["primary"], "limb_darkening_coefficients": {"bolometric": [0.5, 0.4]}},
        )
        binary = BinarySystem.from_json(definition)
        assert_array_equal(binary.primary.limb_darkening_coefficients["bolometric"], [0.5, 0.4])

    @staticmethod
    def test_custom_lds_sqrt() -> None:
        settings.configure(LIMB_DARKENING_LAW="square_root")
        definition = get_default_binary_definition()
        definition["primary"].update(
            {**definition["primary"], "limb_darkening_coefficients": {"bolometric": [0.5, 0.4]}},
        )
        binary = BinarySystem.from_json(definition)
        assert_array_equal(binary.primary.limb_darkening_coefficients["bolometric"], [0.5, 0.4])

    def test_raise_custom_lds_mismatch(self) -> None:
        settings.configure(LIMB_DARKENING_LAW="square_root")
        definition = get_default_binary_definition()
        definition["primary"].update({**definition["primary"], "limb_darkening_coefficients": {"bolometric": [0.5]}})
        with pytest.raises(ElisaInitialParametersError) as context:
            BinarySystem.from_json(definition)

        length = len(definition["primary"]["limb_darkening_coefficients"]["bolometric"])
        assert f"You provided a vector with {length}" in str(context.value)

    def test_raise_missing_passband_lds(self) -> None:
        # Isolated LD fixture for reflection before the specific custom-coefficient assertion.
        self.lc_base_path = str(
            Path(__file__).resolve().parent.parent.joinpath("data", "light_curves"),
        )
        settings.configure(LD_TABLES=str(Path(self.lc_base_path).joinpath("limbdarkening")))

        definition = get_default_binary_definition()
        definition["primary"].update({**definition["primary"], "limb_darkening_coefficients": {"bolometric": 0.5}})

        bs = BinarySystem.from_json(definition)
        o = Observer(passband=["TESS"], system=bs)
        with pytest.raises(ElisaInitialParametersError) as context:
            o.lc(phases=[0.0])

        assert ("TESS") in (str(context.value))
        assert ("limb-darkening factors") in (str(context.value))
        assert isinstance(context.value.__cause__, KeyError)
        assert (context.value.__cause__.args) == (("TESS",))

    def test_raise_missing_bolometric_passband_lds(self) -> None:
        definition = get_default_binary_definition()
        definition["primary"].update(
            {**definition["primary"], "limb_darkening_coefficients": {"TESS": 0.5}},
        )

        bs = BinarySystem.from_json(definition)
        o = Observer(passband=["TESS"], system=bs)

        with pytest.raises(ElisaInitialParametersError) as context:
            o.lc(phases=[0.0])

        assert (
            "Please add `bolometric` limb-darkening coefficients to your custom set of limb-darkening coefficients."
        ) in (str(context.value))

    # noinspection PyMethodMayBeStatic
    def test_custom_ld_coeff_distribution(self) -> None:
        # Isolated LD fixture for reflection before the specific custom-coefficient assertion.
        self.lc_base_path = str(
            Path(__file__).resolve().parent.parent.joinpath("data", "light_curves"),
        )
        settings.configure(LD_TABLES=str(Path(self.lc_base_path).joinpath("limbdarkening")))

        expected_ldc = [0.5, 0.4]
        passband = "bolometric"
        settings.configure(LIMB_DARKENING_LAW="square_root")
        definition = get_default_binary_definition()
        definition["primary"].update(
            {**definition["primary"], "limb_darkening_coefficients": {passband: expected_ldc}},
        )

        bs = BinarySystem.from_json(definition)
        container = bs.build_container(phase=0.0)
        ldcfs = get_component_limbdarkening_cfs(
            container.primary,
            passbands=[passband],
            symmetry_test=False,
        )[passband]

        assert_array_equal(np.unique(ldcfs, axis=0)[0], expected_ldc)


class TestCurvesUtilsCachingAndBroadcast(ElisaTestCase):
    def test_explicit_coeffs_broadcast(self) -> None:
        """Explicit 1-D LD coefficients should be broadcast to (faces, coeffs)."""
        comp = DummyComponent(
            temperatures=[5000.0, 5000.0, 5000.0],
            log_g=[4.0, 4.0, 4.0],
            metallicity=0.0,
            limb_darkening_coefficients={"bolometric": np.array([0.5, 0.4])},
        )

        # noinspection PyTypeChecker
        ld_cfs = curves_utils.get_component_limbdarkening_cfs(comp, passbands=["bolometric"], symmetry_test=False)
        arr = ld_cfs["bolometric"]
        # Expect shape (faces, coeff_count)
        assert arr.shape == (3, 2)
        # Every row equals the provided coefficients
        assert_array_equal(arr, np.tile(np.array([0.5, 0.4]), (3, 1)))

    def test_interpolated_ld_cached(self) -> None:
        """When LD interpolation is used, results must be cached so the.

        underlying interpolation function is invoked only once for identical
        inputs.
        """
        # Ensure caching is cleared at test start
        curves_utils.interpolate_on_ld_grid_cached.cache_clear()

        # Create a component without explicit LD coefficients so interpolation is used
        comp = DummyComponent(
            temperatures=[5500.0, 5500.0],
            log_g=[4.0, 4.0],
            metallicity=0.0,
            limb_darkening_coefficients=None,
        )

        # Force predictable behaviour: ensure single-coeff mode is off and restore later
        old_single = getattr(settings, "USE_SINGLE_LD_COEFFICIENTS", False)
        settings.USE_SINGLE_LD_COEFFICIENTS = False
        try:
            # Patch the underlying interpolation to a synthetic deterministic value
            with (
                patch.object(curves_utils.ld, "get_ld_source_identities", return_value=()),
                patch.object(
                    curves_utils.ld,
                    "interpolate_on_ld_grid",
                    return_value={"bolometric": np.array([[0.1, 0.2]])},
                ) as mock_interp,
            ):
                # First call should invoke the interpolation
                # noinspection PyTypeChecker
                _ = curves_utils.get_component_limbdarkening_cfs(comp, passbands=["bolometric"], symmetry_test=False)
                # Second call with identical inputs should hit the cache and not call interpolation again
                # noinspection PyTypeChecker
                _ = curves_utils.get_component_limbdarkening_cfs(comp, passbands=["bolometric"], symmetry_test=False)

                assert (mock_interp.call_count) == (1), "LD interpolation should be called only once due to caching"
        finally:
            settings.USE_SINGLE_LD_COEFFICIENTS = old_single


SINGLE_PHASE = np.array([0.25])
BINARY_PHASES = np.array([0.1, 0.25])
INTERPOLATION_PHASES = np.array([0.0, 0.1, 0.25])


class CustomLDLawSafetyTestCase(ElisaTestCase):
    @staticmethod
    def coefficients(law: str) -> dict[str, list[float]]:
        values = [0.5] if law in ("linear", "cosine") else [0.5, 0.2]
        return {band: list(values) for band in ("bolometric", "rv_band")}

    def binary(self, law: str, custom: bool = True) -> elisa.binary_system.system.BinarySystem:
        settings.configure(LIMB_DARKENING_LAW=law, NUMBER_OF_PROCESSES=1)
        definition = get_default_binary_definition()
        for component in ("primary", "secondary"):
            definition[component]["atmosphere"] = "bb"
            definition[component]["discretization_factor"] = 10
            if custom:
                definition[component]["limb_darkening_coefficients"] = self.coefficients(law)
        return BinarySystem.from_json(definition)

    def assert_curves_equal(
        self,
        first: tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]],
        second: tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]],
    ) -> None:
        np.testing.assert_array_equal(first[0], second[0])
        for band in first[1]:
            np.testing.assert_array_equal(first[1][band], second[1][band])

    def test_log_sqrt_repeated_binary_lc_and_rv_reuse(self) -> None:
        for original, changed in (("logarithmic", "square_root"), ("square_root", "logarithmic")):
            for reflection in (True, False):
                with self.subTest(original=original, reflection=reflection):
                    model = self.binary(original)
                    settings.configure(REFLECTION_EFFECT=reflection)
                    observer = Observer(system=model, passband=["bolometric"])
                    lc = observer.lc(phases=BINARY_PHASES)
                    rv = observer.rv(phases=BINARY_PHASES, method="radiometric")
                    self.assert_curves_equal(lc, observer.lc(phases=BINARY_PHASES))
                    self.assert_curves_equal(rv, observer.rv(phases=BINARY_PHASES, method="radiometric"))
                    settings.configure(LIMB_DARKENING_LAW=changed)
                    for _ in range(2):
                        with pytest.raises(ElisaInitialParametersError, match=f"defined for '{original}'.*'{changed}'"):
                            observer.lc(phases=BINARY_PHASES)
                        with pytest.raises(ElisaInitialParametersError, match="Supply new coefficients"):
                            observer.rv(phases=BINARY_PHASES, method="radiometric")
                    # Kinematic RV does not consume LD coefficients.
                    observer.rv(phases=BINARY_PHASES, method="kinematic")
                    settings.configure(LIMB_DARKENING_LAW=original)
                    self.assert_curves_equal(lc, observer.lc(phases=BINARY_PHASES))
                    self.assert_curves_equal(rv, observer.rv(phases=BINARY_PHASES, method="radiometric"))

    def test_container_and_bolometric_consumers_reject_same_length_law_change(self) -> None:
        for original, changed in (("logarithmic", "square_root"), ("square_root", "logarithmic")):
            model = self.binary(original)
            container = StarContainer.from_star_instance(model.primary)
            container.temperatures = np.array([5000.0, 6000.0])
            coefficients = container.limb_darkening_coefficients
            assert (coefficients.law) == (original)
            np.testing.assert_array_equal(
                get_component_limbdarkening_cfs(container, ["bolometric"], symmetry_test=False)["bolometric"],
                [[0.5, 0.2], [0.5, 0.2]],
            )
            np.testing.assert_array_equal(
                ld.get_bolometric_ld_coefficients(container.temperatures, np.array([4.0, 4.0]), 0, coefficients),
                [[0.5, 0.5], [0.2, 0.2]],
            )
            settings.configure(LIMB_DARKENING_LAW=changed)
            with pytest.raises(ElisaInitialParametersError):
                get_component_limbdarkening_cfs(container, ["bolometric"], symmetry_test=False)
            with pytest.raises(ElisaInitialParametersError):
                ld.get_bolometric_ld_coefficients(container.temperatures, np.array([4.0, 4.0]), 0, coefficients)

    def test_copy_pickle_and_reconstruction_preserve_provenance(self) -> None:
        model = self.binary("logarithmic")
        star = model.primary
        copies = [
            star.limb_darkening_coefficients.copy(),
            copy(star.limb_darkening_coefficients),
            deepcopy(star.limb_darkening_coefficients),
            pickle.loads(pickle.dumps(star.limb_darkening_coefficients)),
        ]
        kwargs = star.reinitialization_kwargs()
        assert ("polar_log_g") not in (kwargs)
        reconstructed = Star(**kwargs)
        assert (reconstructed.limb_darkening_coefficients.law) == ("logarithmic")
        definition = model.to_json()
        settings.configure(LIMB_DARKENING_LAW="square_root")
        for coefficients in copies:
            assert (coefficients.law) == ("logarithmic")
            with pytest.raises(ElisaInitialParametersError):
                StarProperties.limb_darkening_coefficients(coefficients)
        with pytest.raises(ElisaInitialParametersError):
            Star(**kwargs)
        with pytest.raises(ElisaInitialParametersError):
            BinarySystem.from_json(definition)
        with pytest.raises(ElisaInitialParametersError):
            star.init()

    def test_explicit_replacement_and_none_are_supported(self) -> None:
        model = self.binary("logarithmic")
        old = model.primary.limb_darkening_coefficients
        settings.configure(LIMB_DARKENING_LAW="square_root")
        supplied = self.coefficients("square_root")
        snapshot = deepcopy(supplied)
        for star in (model.primary, model.secondary):
            star.limb_darkening_coefficients = supplied
            assert (star.limb_darkening_coefficients.law) == ("square_root")
        Observer(system=model, passband=["bolometric"]).lc(phases=SINGLE_PHASE)
        assert (supplied) == (snapshot)
        assert (old.law) == ("logarithmic")
        model.primary.limb_darkening_coefficients = None
        assert (model.primary.limb_darkening_coefficients) is None

    def test_linear_cosine_alias_and_incompatible_coefficient_count(self) -> None:
        model = self.binary("linear")
        observer = Observer(system=model, passband=["bolometric"])
        first = observer.lc(phases=SINGLE_PHASE)
        settings.configure(LIMB_DARKENING_LAW="cosine")
        self.assert_curves_equal(first, observer.lc(phases=SINGLE_PHASE))
        for star in (model.primary, model.secondary):
            star.init()
            assert (star.limb_darkening_coefficients.law) == ("linear")
        settings.configure(LIMB_DARKENING_LAW="logarithmic")
        with pytest.raises(ElisaInitialParametersError):
            observer.lc(phases=SINGLE_PHASE)

    def test_single_system_repeated_reuse(self) -> None:
        for original, changed in (("logarithmic", "square_root"), ("square_root", "logarithmic")):
            settings.configure(LIMB_DARKENING_LAW=original)
            model = prepare_single_system(SOLAR_MODEL)
            model.star.atmosphere = "bb"
            model.star.limb_darkening_coefficients = self.coefficients(original)
            observer = Observer(system=model, passband=["bolometric"])
            first = observer.lc(phases=SINGLE_PHASE)
            self.assert_curves_equal(first, observer.lc(phases=SINGLE_PHASE))
            settings.configure(LIMB_DARKENING_LAW=changed)
            with pytest.raises(ElisaInitialParametersError):
                observer.lc(phases=SINGLE_PHASE)
            with pytest.raises(ElisaInitialParametersError):
                observer.rv(phases=SINGLE_PHASE, method="radiometric")
            settings.configure(LIMB_DARKENING_LAW=original)
            self.assert_curves_equal(first, observer.lc(phases=SINGLE_PHASE))

    @pytest.mark.requires_ld(laws=("logarithmic", "square_root"))
    def test_automatic_interpolation_switches_laws_on_same_model(self) -> None:
        model = self.binary("logarithmic", custom=False)
        observer = Observer(system=model, passband=["bolometric"])
        curves = []
        for law in ("logarithmic", "square_root", "logarithmic"):
            settings.configure(LIMB_DARKENING_LAW=law)
            current = observer.lc(phases=INTERPOLATION_PHASES)
            reference = Observer(system=self.binary(law, custom=False), passband=["bolometric"]).lc(
                phases=INTERPOLATION_PHASES,
            )
            self.assert_curves_equal(current, reference)
            assert (model.primary.limb_darkening_coefficients) is None
            curves.append(current)
        self.assert_curves_equal(curves[0], curves[2])
        assert not (np.array_equal(curves[0][1]["bolometric"], curves[1][1]["bolometric"]))
