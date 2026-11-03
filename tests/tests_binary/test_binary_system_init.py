from __future__ import annotations

# The tests package selects Astropy constants before test-module imports.
from copy import copy
from typing import ClassVar

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from elisa import const as c
from elisa import units as u
from elisa.base.error import (
    ElisaInitialParametersError,
    ElisaInvalidKwargsError,
    ElisaMissingKwargsError,
    ElisaUnexpectedError,
)
from elisa.base.star import Star
from elisa.binary_system.system import BinarySystem
from tests import set_astropy_units
from tests.utils import ElisaTestCase, prepare_binary_system

set_astropy_units()


class BinarySystemInitTestCase(ElisaTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.params_combination = [
            {
                "primary_mass": 2.0,
                "secondary_mass": 1.0,
                "primary_surface_potential": 100.0,
                "secondary_surface_potential": 100.0,
                "primary_synchronicity": 1.0,
                "secondary_synchronicity": 1.0,
                "argument_of_periastron": c.HALF_PI * u.rad,
                "gamma": 0.0,
                "period": 200.0,
                "eccentricity": 0.0,
                "inclination": c.HALF_PI * u.rad,
                "primary_minimum_time": 0.0,
                "phase_shift": 0.0,
                "primary_t_eff": 6000,
                "secondary_t_eff": 5000,
                "primary_gravity_darkening": 1.0,
                "secondary_gravity_darkening": 1.0,
                "primary_albedo": 0.6,
                "secondary_albedo": 0.6,
            },  # compact spherical components on circular orbit
            {
                "primary_mass": 2.0,
                "secondary_mass": 1.0,
                "primary_surface_potential": 100.0,
                "secondary_surface_potential": 80.0,
                "primary_synchronicity": 400,
                "secondary_synchronicity": 550,
                "argument_of_periastron": c.HALF_PI * u.rad,
                "gamma": 0.0,
                "period": 300.0,
                "eccentricity": 0.0,
                "inclination": 90.0 * u.deg,
                "primary_minimum_time": 0.0,
                "phase_shift": 0.0,
                "primary_t_eff": 5000,
                "secondary_t_eff": 6000,
                "primary_gravity_darkening": 1.0,
                "secondary_gravity_darkening": 1.0,
                "primary_albedo": 0.6,
                "secondary_albedo": 0.6,
                "primary_discretization_factor": 6,
            },  # rotationally squashed compact spherical components
            {
                "primary_mass": 2.0,
                "secondary_mass": 1.0,
                "primary_surface_potential": 3.5,
                "secondary_surface_potential": 3.0,
                "primary_synchronicity": 1.5,
                "secondary_synchronicity": 1.2,
                "argument_of_periastron": c.HALF_PI,
                "gamma": 0.0,
                "period": 1.0,
                "eccentricity": 0.0,
                "inclination": 90.0 * u.deg,
                "primary_minimum_time": 0.0,
                "phase_shift": 0.0,
                "primary_t_eff": 5000,
                "secondary_t_eff": 5000,
                "primary_gravity_darkening": 1.0,
                "secondary_gravity_darkening": 1.0,
                "primary_albedo": 0.6,
                "secondary_albedo": 0.6,
                "secondary_discretization_factor": 7,
            },  # close tidally deformed components with asynchronous rotation on circular orbit
            {
                "primary_mass": 2.0,
                "secondary_mass": 1.0,
                "primary_surface_potential": 4.8,
                "secondary_surface_potential": 4.0,
                "primary_synchronicity": 1.5,
                "secondary_synchronicity": 1.2,
                "argument_of_periastron": c.HALF_PI * u.rad,
                "gamma": 0.0,
                "period": 1.0,
                "eccentricity": 0.3,
                "inclination": 90.0 * u.deg,
                "primary_minimum_time": 0.0,
                "phase_shift": 0.0,
                "primary_t_eff": 9000,
                "secondary_t_eff": 5000,
                "primary_gravity_darkening": 1.0,
                "secondary_gravity_darkening": 1.0,
                "primary_albedo": 0.6,
                "secondary_albedo": 0.6,
            },  # close tidally deformed components with asynchronous rotation on eccentric orbit
            {
                "primary_mass": 2.0,
                "secondary_mass": 1.0,
                "primary_surface_potential": 2.875844632141054,
                "secondary_surface_potential": 2.875844632141054,
                "primary_synchronicity": 1.0,
                "secondary_synchronicity": 1.0,
                "argument_of_periastron": c.HALF_PI * u.rad,
                "gamma": 0.0,
                "period": 1.0,
                "eccentricity": 0.0,
                "inclination": 90.0 * u.deg,
                "primary_minimum_time": 0.0,
                "phase_shift": 0.0,
                "primary_t_eff": 6000,
                "secondary_t_eff": 5000,
                "primary_gravity_darkening": 1.0,
                "secondary_gravity_darkening": 1.0,
                "primary_albedo": 0.6,
                "secondary_albedo": 0.6,
            },  # synchronous contact system
            {
                "primary_mass": 2.0,
                "secondary_mass": 1.0,
                "primary_surface_potential": 3.159639848886489,
                "secondary_surface_potential": 3.229240544834036,
                # "primary_surface_potential": 3.1596399,
                # "secondary_surface_potential": 3.2292406,
                "primary_synchronicity": 1.5,
                "secondary_synchronicity": 2.0,
                "argument_of_periastron": c.HALF_PI * u.rad,
                "gamma": 0.0,
                "period": 1.0,
                "eccentricity": 0.0,
                "inclination": 90.0 * u.deg,
                "primary_minimum_time": 0.0,
                "phase_shift": 0.0,
                "primary_t_eff": 6500,
                "secondary_t_eff": 6000,
                "primary_gravity_darkening": 1.0,
                "secondary_gravity_darkening": 1.0,
                "primary_albedo": 0.6,
                "secondary_albedo": 0.6,
            },  # asynchronous contact system (improbable but whatever...)
            {
                "primary_mass": 2.0,
                "secondary_mass": 1.0,
                "primary_surface_potential": 2.7,
                "secondary_surface_potential": 2.7,
                "primary_synchronicity": 1.0,
                "secondary_synchronicity": 1.0,
                "argument_of_periastron": 90 * u.deg,
                "gamma": 0.0,
                "period": 1.0,
                "eccentricity": 0.0,
                "inclination": 90.0 * u.deg,
                "primary_minimum_time": 0.0,
                "phase_shift": 0.0,
                "primary_t_eff": 5000,
                "secondary_t_eff": 5000,
                "primary_gravity_darkening": 1.0,
                "secondary_gravity_darkening": 1.0,
                "primary_albedo": 0.6,
                "secondary_albedo": 0.6,
            },  # over-contact system
        ]

    def prepare_systems(self) -> list[BinarySystem]:
        return [prepare_binary_system(combo) for combo in self.params_combination]

    def test_calculate_semi_major_axis(self) -> None:
        expected = [
            144406544570.0,
            189226104475.0,
            4222472978.0,
            4222472978.0,
            4222472978.0,
            4222472978.0,
            4222472978.0,
        ]
        obtained = [np.round(bs.semi_major_axis, 0) for bs in self.prepare_systems()]
        assert_array_equal(expected, obtained)

    def test_critical_surface_potential(self) -> None:
        expected_potentials = np.round(
            np.array(
                [
                    [2.875844632141054, 2.875844632141054],
                    [93.717106763853593, 73.862399105365014],
                    [3.159639848886489, 2.935086409515319],
                    [4.027577786299736, 3.898140726941630],
                    [2.875844632141054, 2.875844632141054],
                    [3.159639848886489, 3.229240544834036],
                    [2.875844632141054, 2.875844632141054],
                ],
            ),
            5,
        )
        obtained_potentials = [
            [bs.primary.critical_surface_potential, bs.secondary.critical_surface_potential]
            for bs in self.prepare_systems()
        ]
        obtained_potentials = np.round(np.array(obtained_potentials), 5)
        assert_array_equal(expected_potentials, obtained_potentials)

    def test_compute_morphology(self) -> None:
        expected = ["detached", "detached", "detached", "detached", "semi-detached", "double-contact", "over-contact"]
        obtained = [bs.morphology for bs in self.prepare_systems()]
        assert_array_equal(expected, obtained)

    def test_setup_components_radii(self) -> None:
        radii = ["forward_radius", "side_radius", "equatorial_radius", "backward_radius", "polar_radius"]
        obtained = {}

        expected = {
            "primary": {
                "forward_radius": [0.01005, 0.01229, 0.3783, 0.26257, 0.57075, 0.53029, np.nan],
                "side_radius": [0.01005, 0.01229, 0.35511, 0.24859, 0.43994, 0.41553, 0.48182],
                "equatorial_radius": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                "backward_radius": [0.01005, 0.01229, 0.36723, 0.25608, 0.46794, 0.44207, 0.52573],
                "polar_radius": [0.01005, 0.01005, 0.33055, 0.24242, 0.41427, 0.37162, 0.44577],
            },
            "secondary": {
                "forward_radius": [0.00506, 0.00763, 0.34966, 0.25693, 0.42925, 0.36743, np.nan],
                "side_radius": [0.00506, 0.00763, 0.29464, 0.21419, 0.31288, 0.28094, 0.35376],
                "equatorial_radius": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                "backward_radius": [0.00506, 0.00763, 0.32018, 0.23324, 0.34537, 0.30842, 0.41825],
                "polar_radius": [0.00506, 0.00635, 0.2798, 0.20994, 0.29977, 0.2489, 0.33306],
            },
        }

        for component in ["primary", "secondary"]:
            obtained[component] = {}
            for radius in radii:
                obtained[component][radius] = []
                for i, bs in enumerate(self.prepare_systems()):
                    value = (
                        np.round(getattr(getattr(bs, component), radius), 5)
                        if hasattr(getattr(bs, component), radius)
                        else np.nan
                    )
                    assert_array_equal([value], [expected[component][radius][i]])
                    if bs.eccentricity and radius != "equatorial_radius":
                        from scripts.diagnostics.roche_reference import reference

                        star = getattr(bs, component)
                        oracle = reference(
                            bs.mass_ratio,
                            star.synchronicity,
                            1 - bs.eccentricity,
                            component,
                            radius.removesuffix("_radius"),
                            star.surface_potential,
                        )
                        np.testing.assert_allclose(getattr(star, radius), float(oracle["root"]), rtol=1e-9)

    def test_lagrangian_points(self) -> None:

        expected_points = np.round(
            np.array(
                [
                    [-0.80302796065835425, 0.57075157151852673, 1.5823807222136623],
                    [-0.80302796065835425, 0.57075157151852673, 1.5823807222136623],
                    [-0.80302796065835425, 0.57075157151852673, 1.5823807222136623],
                    [-0.7308068505479407, 0.41566688133312363, 1.4990376377419574],
                    [-0.80302796065835425, 0.57075157151852673, 1.5823807222136623],
                    [-0.80302796065835425, 0.57075157151852673, 1.5823807222136623],
                    [-0.80302796065835425, 0.57075157151852673, 1.5823807222136623],
                ],
            ),
            5,
        )

        obtained_points = [bs.lagrangian_points() for bs in self.prepare_systems()]

        obtained_points = np.round(np.array(obtained_points), 5)
        assert_array_equal(expected_points, obtained_points)

    def test_components(self) -> None:
        bs = self.prepare_systems()[0]
        assert hasattr(bs, "components")
        assert bs.components["primary"]
        assert bs.components["secondary"]

    def test_discretization_factor(self) -> None:
        expected_p = [5.0, 6.0, 5.96, 5.0, 5.0, 5.0, 5.0]
        expected_s = [12.0, 6.67, 7.0, 12.0, 9.93, 8.63, 6.61]
        obtained_p, obtained_s = [], []
        for _i, bs in enumerate(self.prepare_systems()):
            obtained_p.append(np.degrees(bs.primary.discretization_factor))
            obtained_s.append(np.degrees(bs.secondary.discretization_factor))

        assert_array_equal(np.round(obtained_p, 2), expected_p)
        assert_array_equal(np.round(obtained_s, 2), expected_s)


class ValidityTestCase(ElisaTestCase):
    MANDATORY_KWARGS: ClassVar[list[str]] = ["inclination", "period", "eccentricity", "argument_of_periastron"]

    def setUp(self) -> None:
        super().setUp()
        self._initial_params = {
            "primary_mass": 2.0,
            "secondary_mass": 1.0,
            "primary_surface_potential": 100.0,
            "secondary_surface_potential": 100.0,
            "primary_synchronicity": 1.0,
            "secondary_synchronicity": 1.0,
            "argument_of_periastron": c.HALF_PI * u.rad,
            "gamma": 0.0,
            "period": 1.0,
            "eccentricity": 0.0,
            "inclination": c.HALF_PI * u.deg,
            "primary_minimum_time": 0.0,
            "phase_shift": 0.0,
            "primary_t_eff": 5000,
            "secondary_t_eff": 5000,
            "primary_gravity_darkening": 1.0,
            "secondary_gravity_darkening": 1.0,
            "primary_albedo": 1.0,
            "secondary_albedo": 1.0,
        }

        self._primary = Star(
            mass=self._initial_params["primary_mass"],
            surface_potential=self._initial_params["primary_surface_potential"],
            synchronicity=self._initial_params["primary_synchronicity"],
            t_eff=self._initial_params["primary_t_eff"],
            gravity_darkening=self._initial_params["primary_gravity_darkening"],
            albedo=self._initial_params["primary_albedo"],
            metallicity=0.0,
        )

        self._secondary = Star(
            mass=self._initial_params["secondary_mass"],
            surface_potential=self._initial_params["secondary_surface_potential"],
            synchronicity=self._initial_params["secondary_synchronicity"],
            t_eff=self._initial_params["secondary_t_eff"],
            gravity_darkening=self._initial_params["secondary_gravity_darkening"],
            albedo=self._initial_params["secondary_albedo"],
            metallicity=0.0,
        )

    def test__star_params_validity_check(self) -> None:
        with pytest.raises(ElisaUnexpectedError) as context:
            BinarySystem(
                primary=42,
                secondary=self._secondary,
                argument_of_periastron=self._initial_params["argument_of_periastron"],
                gamma=self._initial_params["gamma"],
                period=self._initial_params["period"],
                eccentricity=self._initial_params["eccentricity"],
                inclination=self._initial_params["inclination"],
                primary_minimum_time=self._initial_params["primary_minimum_time"],
                phase_shift=self._initial_params["phase_shift"],
            )

        assert "Component `primary` is not instance of class" in str(context.value)

        star_mandatory_kwargs = [
            "mass",
            "surface_potential",
            "synchronicity",
            "albedo",
            "metallicity",
            "gravity_darkening",
        ]

        for kw in star_mandatory_kwargs[:1]:
            p = copy(self._primary)
            setattr(p, f"{kw}", None)

            with pytest.raises(ElisaInitialParametersError) as context:
                BinarySystem(
                    primary=p,
                    secondary=self._secondary,
                    argument_of_periastron=self._initial_params["argument_of_periastron"],
                    gamma=self._initial_params["gamma"],
                    period=self._initial_params["period"],
                    eccentricity=self._initial_params["eccentricity"],
                    inclination=self._initial_params["inclination"],
                    primary_minimum_time=self._initial_params["primary_minimum_time"],
                    phase_shift=self._initial_params["phase_shift"],
                )

            assert f"Missing argument(s): `{kw}`" in str(context.value)

    def test_invalid_kwarg_checker(self) -> None:
        with pytest.raises(ElisaInvalidKwargsError) as context:
            BinarySystem(
                primary=self._primary,
                secondary=self._secondary,
                argument_of_periastron=self._initial_params["argument_of_periastron"],
                gamma=self._initial_params["gamma"],
                period=self._initial_params["period"],
                eccentricity=self._initial_params["eccentricity"],
                inclination=self._initial_params["inclination"],
                primary_minimum_time=self._initial_params["primary_minimum_time"],
                phase_shift=self._initial_params["phase_shift"],
                adhoc_param="xxx",
            )
        assert "Invalid keyword argument(s): adhoc_param" in str(context.value)

    def test_check_missing_kwargs(self) -> None:
        initial_kwargs = {
            "primary": self._primary,
            "secondary": self._secondary,
            "argument_of_periastron": self._initial_params["argument_of_periastron"],
            "period": self._initial_params["period"],
            "eccentricity": self._initial_params["eccentricity"],
            "inclination": self._initial_params["inclination"],
            "primary_minimum_time": self._initial_params["primary_minimum_time"],
        }

        for kw in self.MANDATORY_KWARGS:
            kwargs = copy(initial_kwargs)
            del kwargs[kw]
            with pytest.raises(ElisaMissingKwargsError) as context:
                BinarySystem(**kwargs)
            assert f"Missing argument(s): `{kw}`" in str(context.value)


class BinarySystemSerializersTestCase(ElisaTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.params_combination = [
            {
                "primary_mass": 2.0,
                "secondary_mass": 1.0,
                "primary_surface_potential": 100.0,
                "secondary_surface_potential": 100.0,
                "primary_synchronicity": 1.0,
                "secondary_synchronicity": 1.0,
                "argument_of_periastron": c.HALF_PI * u.rad,
                "gamma": 0.0,
                "period": 1.0,
                "eccentricity": 0.0,
                "inclination": c.HALF_PI * u.deg,
                "primary_minimum_time": 0.0,
                "phase_shift": 0.0,
                "primary_t_eff": 5000,
                "secondary_t_eff": 5000,
                "primary_gravity_darkening": 1.0,
                "secondary_gravity_darkening": 1.0,
                "primary_albedo": 0.6,
                "secondary_albedo": 0.6,
            },
            # compact spherical components on circular orbit
            {
                "primary_mass": 2.0,
                "secondary_mass": 1.0,
                "primary_surface_potential": 4.8,
                "secondary_surface_potential": 4.0,
                "primary_synchronicity": 1.5,
                "secondary_synchronicity": 1.2,
                "argument_of_periastron": c.HALF_PI * u.rad,
                "gamma": 0.0,
                "period": 1.0,
                "eccentricity": 0.3,
                "inclination": 90.0 * u.deg,
                "primary_minimum_time": 0.0,
                "phase_shift": 0.0,
                "primary_t_eff": 5000,
                "secondary_t_eff": 5000,
                "primary_gravity_darkening": 1.0,
                "secondary_gravity_darkening": 1.0,
                "primary_albedo": 0.6,
                "secondary_albedo": 0.6,
            },  # close tidally deformed components with asynchronous rotation on eccentric orbit
        ]

        self._binaries = self.prepare_systems()

    def prepare_systems(self) -> list[BinarySystem]:
        return [prepare_binary_system(combo) for combo in self.params_combination]

    def test_kwargs_serializer(self) -> None:
        bs = self._binaries[-1]
        obtained = bs.kwargs_serializer()

        expected = {
            "inclination": c.HALF_PI,
            "period": 1.0,
            "eccentricity": 0.3,
            "argument_of_periastron": c.HALF_PI,
            "gamma": 0.0,
            "primary_minimum_time": 0.0,
            "phase_shift": 0.0,
        }

        obtained_array, expectedd_array = [], []

        for key, expected_value in expected.items():
            if isinstance(obtained[key], u.Quantity):
                obtained[key] = obtained[key].value
            obtained_array.append(round(obtained[key], 5))
            expectedd_array.append(round(expected_value, 5))

        assert_array_equal(expectedd_array, obtained_array)

    def test_properties_serializer(self) -> None:
        bs = self._binaries[-1]
        obtained = bs.properties_serializer()
        expected = ["semi_major_axis", "morphology", "mass_ratio"]
        for e in expected:
            assert e in obtained

    @staticmethod
    def _get_std() -> BinarySystem:
        data = {
            "system": {
                "inclination": "90.0 deg",
                "period": "10.1 d",
                "argument_of_periastron": 90.0,
                "gamma": 0.0,
                "eccentricity": 0.3,
                "primary_minimum_time": 0.0,
                "phase_shift": 0.0,
            },
            "primary": {
                "mass": "2.0 solMass",
                "surface_potential": 7.1,
                "synchronicity": 1.0,
                "t_eff": "6500.0 K",
                "discretization_factor": 5,
            },
            "secondary": {
                "mass": 1.5,
                "surface_potential": 7.1,
                "synchronicity": 1.0,
                "t_eff": 6500.0,
                "gravity_darkening": 1.0,
                "discretization_factor": 5,
                "albedo": 1.0,
                "metallicity": 0.0,
            },
        }
        return BinarySystem.from_json(data)

    @staticmethod
    def _get_community() -> BinarySystem:
        data = {
            "system": {
                "inclination": "90.0 deg",
                "period": "10.1 d",
                "argument_of_periastron": 90.0,
                "gamma": 0.0,
                "eccentricity": 0.3,
                "primary_minimum_time": 0.0,
                "phase_shift": 0.0,
                "mass_ratio": 0.75,
                "semi_major_axis": 29.854,
            },
            "primary": {
                "surface_potential": 7.1,
                "synchronicity": 1.0,
                "t_eff": "6500.0 K",
                "discretization_factor": 5,
            },
            "secondary": {
                "surface_potential": 7.1,
                "synchronicity": 1.0,
                "t_eff": 6500.0,
                "gravity_darkening": 1.0,
                "discretization_factor": 5,
                "albedo": 1.0,
                "metallicity": 0.0,
            },
        }

        return BinarySystem.from_json(data, _kind_of="community")

    @classmethod
    def test_init_from_json_std(cls) -> None:
        cls._get_std()

    @classmethod
    def test_init_from_json_community(cls) -> None:
        cls._get_community()

    def test_init_from_json_community_and_std_are_equivalent(self) -> None:
        com = self._get_community()
        std = self._get_std()

        com_a1 = np.float64((com.semi_major_axis * u.m).to(u.solRad))
        com_a2 = np.float64((std.semi_major_axis * u.m).to(u.solRad))
        assert np.round(com_a1, 2) == np.round(com_a2, 2)
        assert np.round(com.mass_ratio, 2) == np.round(std.mass_ratio, 2)

    def test_to_json(self) -> None:
        data = {
            "system": {
                "inclination": 90.0,
                "period": 10.1,
                "argument_of_periastron": 90.0,
                "gamma": 0.0,
                "eccentricity": 0.3,
                "primary_minimum_time": "0.0 d",
                "phase_shift": 0.0,
                "mass_ratio": 0.75,
                "semi_major_axis": 29.854,
                "distance": "1 pc",
            },
            "primary": {
                "surface_potential": 7.1,
                "synchronicity": 1.0,
                "t_eff": "6500.0 K",
                "gravity_darkening": 1.0,
                "discretization_factor": 5,
                "albedo": 1.0,
                "metallicity": 0.0,
                "atmosphere": "bb",
                "limb_darkening_coefficients": {"bolometric": 0.5},
                "spots": [
                    {
                        "longitude": 90,
                        "latitude": 90,
                        "angular_radius": 45,
                        "temperature_factor": 0.9,
                        "discretization_factor": 5,
                    },
                    {
                        "longitude": 30,
                        "latitude": 80,
                        "angular_radius": 70,
                        "temperature_factor": 1.0,
                    },
                ],
            },
            "secondary": {
                "surface_potential": 7.1,
                "synchronicity": 1.0,
                "t_eff": 6500.0,
                "discretization_factor": 5,
                "pulsations": [
                    {
                        "l": 2,
                        "m": 2,
                        # "amplitude": 500,
                        "amplitude": "0.5 km / s",
                        # "frequency": 0.5,
                        "frequency": "0.5 / d",
                        "start_phase": 0.0 * c.PI,
                        "mode_axis_theta": 90,
                        "mode_axis_phi": "0 rad",
                        "horizontal_to_radial_amplitude_ratio": 0.1,
                        "temperature_perturbation_phase_shift": f"{c.PI / 2} rad",
                        "temperature_amplitude_factor": 0.03,
                        "tidally_locked": True,
                    },
                    {
                        "l": 10,
                        "m": 5,
                        "amplitude": 500,
                        "frequency": 0.5,
                    },
                ],
            },
        }

        expected = {
            "system": {
                "inclination": 90.0,
                "period": 10.1,
                "argument_of_periastron": 90.0,
                "gamma": 0.0,
                "eccentricity": 0.3,
                "primary_minimum_time": 0.0,
                "phase_shift": 0.0,
                "semi_major_axis": 29.854,
                "additional_light": 0.0,
                "distance": 1.0,
            },
            "primary": {
                "mass": 2.0,
                "surface_potential": 7.1,
                "synchronicity": 1.0,
                "t_eff": 6500.0,
                "gravity_darkening": 1.0,
                "discretization_factor": 5,
                "albedo": 1.0,
                "metallicity": 0.0,
                "atmosphere": "bb",
                "limb_darkening_coefficients": {"bolometric": [0.5]},
                "spots": [
                    {
                        "longitude": 90,
                        "latitude": 90,
                        "angular_radius": 45,
                        "temperature_factor": 0.9,
                        "discretization_factor": 5,
                    },
                    {
                        "longitude": 30,
                        "latitude": 80,
                        "angular_radius": 70,
                        "temperature_factor": 1.0,
                        "discretization_factor": 5,
                    },
                ],
            },
            "secondary": {
                "mass": 1.5,
                "surface_potential": 7.1,
                "synchronicity": 1.0,
                "t_eff": 6500.0,
                "discretization_factor": 5,
                "metallicity": 0.0,
                "gravity_darkening": 0.5302,
                "albedo": 0.7126,
                "pulsations": [
                    {
                        "l": 2,
                        "m": 2,
                        "amplitude": 500,
                        "frequency": 0.5,
                        "start_phase": 0.0,
                        "mode_axis_theta": 90,
                        "mode_axis_phi": 0,
                        "horizontal_to_radial_amplitude_ratio": 0.1,
                        "temperature_perturbation_phase_shift": c.PI / 2,
                        "temperature_amplitude_factor": 0.03,
                        "tidally_locked": True,
                    },
                    {
                        "l": 10,
                        "m": 5,
                        "amplitude": 500,
                        "frequency": 0.5,
                        "start_phase": 0.0,
                        "mode_axis_theta": 0,
                        "mode_axis_phi": 0,
                        "horizontal_to_radial_amplitude_ratio": 70.931415717,
                        "temperature_perturbation_phase_shift": 1.0471975,
                        "temperature_amplitude_factor": 0.21494597573438942,
                        "tidally_locked": False,
                    },
                ],
            },
        }

        bs = BinarySystem.from_json(data)
        js = bs.to_json()
        # Independently integrate the periastron volume before accepting derived
        # pulsation defaults; the slice approximation has finite resolution.
        from elisa.pulse.pulsations import spherical_harmonics
        from tests.tests_binary.test_periastron_volume import volume

        star = bs.secondary
        mode = star.pulsations[1]
        radius = (3 * volume(bs, "secondary", star.surface_potential, 0.7) / (4 * c.PI)) ** (1 / 3)
        radius *= bs.semi_major_axis
        ratio = np.sqrt(110) * c.G * star.mass / (radius**3 * mode.angular_frequency**2)
        temperature = c.IDEAL_ADIABATIC_GRADIENT * (ratio * 110 - 4 - 1 / ratio)
        temperature *= mode.amplitude / (mode.angular_frequency * np.sqrt(1 + ratio**2) * radius)
        np.testing.assert_allclose(mode.horizontal_to_radial_amplitude_ratio, ratio, rtol=0.002)
        np.testing.assert_allclose(mode.temperature_amplitude_factor, temperature, rtol=0.002)

        restored = BinarySystem.from_json(js)
        points = np.array([[1.0, 0.2, 0.5], [1.0, 1.2, 2.0]])
        for index, original in star.pulsations.items():
            recovered = restored.secondary.pulsations[index]
            for name in (
                "amplitude",
                "frequency",
                "start_phase",
                "mode_axis_theta",
                "mode_axis_phi",
                "horizontal_to_radial_amplitude_ratio",
                "temperature_amplitude_factor",
                "temperature_perturbation_phase_shift",
                "radial_amplitude",
                "horizontal_amplitude",
                "l",
                "m",
                "tidally_locked",
            ):
                np.testing.assert_allclose(getattr(recovered, name), getattr(original, name), rtol=1e-12)
            np.testing.assert_allclose(
                spherical_harmonics(recovered, points, np.exp(0.3j)),
                spherical_harmonics(original, points, np.exp(0.3j)),
            )

        for item in js["system"]:
            assert (js["system"][item]) == (expected["system"][item])

        for component in ["primary", "secondary"]:
            for item in js[component]:
                if item not in ["spots", "pulsations"]:
                    self.assertAlmostEqual(js[component][item], expected[component][item], 3)
                else:
                    for ii, feature in enumerate(js[component][item]):
                        for fitem in feature:
                            self.assertAlmostEqual(
                                feature[fitem],
                                expected[component][item][ii][fitem],
                                3,
                            )
