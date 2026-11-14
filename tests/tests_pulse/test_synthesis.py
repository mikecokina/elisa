from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_less

from elisa import Observer, const, utils
from elisa import units as u
from elisa.base.star import Star
from elisa.binary_system.system import BinarySystem
from elisa.pulse import container_ops
from elisa.pulse import utils as putils
from elisa.single_system.system import SingleSystem
from tests import utils as testutils
from tests.tests_pulse.helpers import BINARY_STAR, BINARY_SYSTEM, STAR_PARAMS, SYSTEM_PARMAS, TOL
from tests.utils import ElisaTestCase


class TestPulsationModule(ElisaTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.base_path = str(Path(__file__).resolve().parent.parent)
        self.lc_base_path = str(Path(__file__).resolve().parent.joinpath("data", "light_curves"))

    def prepare_system(self, pulse_meta: list[dict[str, object]]) -> SingleSystem:
        star = Star(pulsations=pulse_meta, **STAR_PARAMS)
        return SingleSystem(star=star, **SYSTEM_PARMAS)

    def prepare_binary(self, pulse_meta: list[dict[str, object]]) -> BinarySystem:
        primary = Star(pulsations=pulse_meta, **BINARY_STAR)
        secondary = Star(pulsations=pulse_meta, **BINARY_STAR)
        return BinarySystem(primary=primary, secondary=secondary, **BINARY_SYSTEM)

    def test_complex_displacement_amplitudes(self) -> None:
        in_ratio = 2
        pulse_meta = [
            {
                "l": 10,
                "m": 2,
                "amplitude": 1 * u.m / u.s,
                "frequency": 1 / u.d,
                "start_phase": 0.0,
                "horizontal_to_radial_amplitude_ratio": in_ratio,
            },
        ]

        single = self.prepare_system(pulse_meta)
        system_container = SingleSystem.build_container(single, phase=0)

        mode = system_container.star.pulsations[0]
        directions = system_container.star.pulsation_equilibrium_points
        directions = directions / np.linalg.norm(directions, axis=1)[:, None]
        radial_component = np.sum(mode.complex_displacement * directions, axis=1)
        radial = np.sqrt(np.mean(abs(radial_component) ** 2))
        tangent = mode.complex_displacement - radial_component[:, None] * directions
        horizontal = np.sqrt(np.mean(np.sum(abs(tangent) ** 2, axis=1)))
        np.testing.assert_allclose(horizontal, abs(mode.horizontal_amplitude), rtol=1e-12)

        ratio = horizontal / radial
        assert abs(ratio - in_ratio) < 0.05

    def test_kinematics_single(self) -> None:
        amplitude = 1.0
        freq = 1
        omega = const.FULL_ARC * freq
        pulse_meta = [
            {
                "l": 2,
                "m": 1,
                "amplitude": amplitude * u.m / u.s,
                "frequency": freq / u.s,
                "start_phase": 0.0,
                "horizontal_to_radial_amplitude_ratio": 0.0,
                "temperature_amplitude_factor": 0.01,
            },
        ]
        overshoot = 2.0  # this value may change for different modes

        single = self.prepare_system(pulse_meta)
        system_container = SingleSystem.build_container(single, phase=0.2854)

        # displacement
        r_ampl = amplitude / omega
        args = (system_container.star, 0.0)
        kwargs_ = {
            "update_container": False,
            "return_perturbation": True,
        }
        displacement = container_ops.position_perturbation(*args, **kwargs_)
        radius = np.abs(putils.spherical_vector_components(displacement, system_container.star.points_spherical)[:, 0])
        assert_array_less(radius, overshoot * r_ampl)
        assert_array_less(r_ampl, radius.max())

        # Project Cartesian face vectors at the equilibrium face centers.
        star = system_container.star
        coordinates = utils.cartesian_to_spherical(star.pulsation_equilibrium_points[star.faces].mean(axis=1))
        velocity = container_ops.velocity_perturbation(star, 1.0, return_perturbation=True)
        velocities = np.abs(putils.spherical_vector_components(velocity, coordinates)[:, 0])
        assert_array_less(velocities, overshoot * amplitude)
        assert_array_less(amplitude, velocities.max())

        a_amp = amplitude * omega
        acceleration = container_ops.gravity_acc_perturbation(star, 1.0, return_perturbation=True)
        acc = np.abs(putils.spherical_vector_components(acceleration, coordinates)[:, 0])
        assert_array_less(acc, overshoot * a_amp)
        assert_array_less(a_amp, acc.max())

        # teff
        t_ampl = pulse_meta[0]["temperature_amplitude_factor"] * system_container.star.t_eff
        args = (system_container.star, 1.0)
        ts = np.abs(
            container_ops.temp_perturbation(
                *args,
                update_container=False,
                return_perturbation=True,
            ),
        )
        assert_array_less(ts, overshoot * t_ampl)
        assert_array_less(t_ampl, ts.max())

    def test_container_time_and_perturbations_not_nan(self) -> None:
        """Ensure container.time is computed and pulsation perturbation arrays are finite.

        Regression test guarding against accidental overwrites of the container time
        (which previously produced NaNs in time-dependent pulsation arrays).
        """
        pulse_meta = [
            {
                "l": 2,
                "m": 1,
                "amplitude": 1.0 * u.m / u.s,
                "frequency": 1 / u.s,
                "start_phase": 0.0,
                "horizontal_to_radial_amplitude_ratio": 0.0,
                "temperature_amplitude_factor": 0.01,
            },
        ]

        single = self.prepare_system(pulse_meta)
        system_container = SingleSystem.build_container(single, phase=0.2854)

        # time must be finite (not NaN or inf)
        assert np.isfinite(system_container.time), "Container time must be finite and not NaN"

        # position perturbation (point-wise) must not contain NaNs
        pert = container_ops.position_perturbation(
            system_container.star,
            0.0,
            update_container=False,
            return_perturbation=True,
        )
        assert pert is not None
        assert not np.isnan(pert).any(), "Position perturbation contains NaNs"

        # velocity perturbation must not contain NaNs
        vel = container_ops.velocity_perturbation(
            system_container.star,
            1.0,
            update_container=False,
            return_perturbation=True,
        )
        assert vel is not None
        assert not np.isnan(vel).any(), "Velocity perturbation contains NaNs"

    @pytest.mark.requires_ld(passbands=("Generic.Bessell.V", "bolometric"))
    @pytest.mark.requires_atmosphere
    def test_single_pulsating_lc(self) -> None:
        freq = 15
        pulse_meta = [
            {
                "l": 2,
                "m": 1,
                "amplitude": 100 * u.m / u.s,
                "frequency": freq / u.d,
                "start_phase": 0.0,
                "horizontal_to_radial_amplitude_ratio": 0.0,
                "temperature_amplitude_factor": 0.01,
            },
        ]

        single = self.prepare_system(pulse_meta)
        o = Observer(passband=["Generic.Bessell.V"], system=single)

        phases = np.linspace(0.0, 1 / (freq * single.rotation_period))

        expected = testutils.load_light_curve("single.pulsating.v.json")
        expected_phases = expected[0]
        expected_flux = testutils.normalize_lc_for_unittests(expected[1]["Generic.Bessell.V"])

        obtained = o.lc(phases=phases)
        obtained_phases = obtained[0]
        obtained_flux = testutils.normalize_lc_for_unittests(obtained[1]["Generic.Bessell.V"])

        assert np.all(np.abs(np.round(obtained_phases, 3) - np.round(expected_phases, 3)) < TOL)
        assert np.all(np.abs(np.round(obtained_flux, 3) - np.round(expected_flux, 3)) < TOL)

        # plt.plot(obtained_phases, obtained_flux)
        # plt.plot(expected_phases, expected_flux)
        # plt.show()

    @pytest.mark.requires_ld
    @pytest.mark.requires_atmosphere
    def test_single_pulsating_rv(self) -> None:
        freq = 15
        pulse_meta = [
            {
                "l": 1,
                "m": 1,
                "amplitude": 1 * u.km / u.s,
                "frequency": freq / u.d,
                "start_phase": 0.0,
                "horizontal_to_radial_amplitude_ratio": 0.0,
                "temperature_amplitude_factor": 0.01,
            },
        ]

        single = self.prepare_system(pulse_meta)
        o = Observer(system=single)

        phases = np.linspace(0.0, 1 / (freq * single.period), 50)

        expected = testutils.load_radial_curve("single.pulsating.json")
        expected_rv = testutils.normalize_single_rv_for_unittests(expected["star"])

        obtained = o.rv(phases=phases, method="radiometric")
        obtained_rv = testutils.normalize_single_rv_for_unittests(obtained[1]["star"])

        assert np.all(np.abs(obtained_rv - expected_rv) < TOL)

        # plt.plot(phases, obtained_rv)
        # plt.plot(phases, expected_rv)
        # plt.plot(expected_rv-obtained_rv)
        # plt.show()

    @pytest.mark.requires_ld
    def test_kinematics_binary(self) -> None:
        amplitude = 1.0
        freq = 1
        omega = const.FULL_ARC * freq
        pulse_meta = [
            {
                "l": 4,
                "m": 2,
                "amplitude": amplitude * u.m / u.s,
                "frequency": freq / u.s,
                "start_phase": 0.0,
                "horizontal_to_radial_amplitude_ratio": 1.0,
                "temperature_amplitude_factor": 0.01,
            },
        ]
        overshoot = 2.0  # this value may change for different modes

        binary = self.prepare_binary(pulse_meta)
        system_container = BinarySystem.build_container(binary, phase=0.6741)

        # displacement
        r_ampl = amplitude / omega
        args = (system_container.secondary, system_container.position.distance)
        kwargs_ = {
            "update_container": False,
            "return_perturbation": True,
        }
        star = system_container.secondary
        displacement = container_ops.position_perturbation(*args, **kwargs_)
        radius = np.abs(putils.spherical_vector_components(displacement, star.points_spherical)[:, 0])
        radius *= binary.semi_major_axis
        coordinates = utils.cartesian_to_spherical(star.pulsation_equilibrium_points[star.faces].mean(axis=1))
        assert_array_less(radius, overshoot * r_ampl)
        assert_array_less(r_ampl, radius.max())

        # velocity
        args = (system_container.secondary, binary.semi_major_axis)
        velocity = container_ops.velocity_perturbation(*args, **kwargs_)
        velocities = np.abs(putils.spherical_vector_components(velocity, coordinates)[:, 0])
        assert_array_less(velocities, overshoot * amplitude)
        assert_array_less(amplitude, velocities.max())

        # acceleration
        a_amp = amplitude * omega
        args = (system_container.secondary, binary.semi_major_axis)
        acceleration = container_ops.gravity_acc_perturbation(*args, **kwargs_)
        acc = np.abs(putils.spherical_vector_components(acceleration, coordinates)[:, 0])
        assert_array_less(acc, overshoot * a_amp)
        assert_array_less(a_amp, acc.max())

        # teff
        t_ampl = pulse_meta[0]["temperature_amplitude_factor"] * system_container.secondary.t_eff
        args = (system_container.secondary, binary.semi_major_axis)
        kwargs_ = {
            "update_container": False,
            "return_perturbation": True,
        }
        ts = np.abs(container_ops.temp_perturbation(*args, **kwargs_))
        assert_array_less(ts, overshoot * t_ampl)
        assert_array_less(t_ampl, ts.max())

    @pytest.mark.requires_ld(passbands=("bolometric", "Generic.Bessell.V"))
    @pytest.mark.requires_atmosphere
    def test_binary_pulsating_lc(self) -> None:
        freq = 15
        pulse_meta = [
            {
                "l": 2,
                "m": 1,
                "amplitude": 100 * u.m / u.s,
                "frequency": freq / u.d,
                "start_phase": 0.0,
                "horizontal_to_radial_amplitude_ratio": 0.0,
                "temperature_amplitude_factor": 0.01,
            },
        ]

        binary = self.prepare_binary(pulse_meta)
        o = Observer(passband=["Generic.Bessell.V"], system=binary)

        start_phase = 0.025
        phases = np.linspace(start_phase, start_phase + 2 / (freq * binary.period), 10)

        expected = testutils.load_light_curve("binary.pulsating.v.json")
        expected_phases = expected[0]
        # The historical flux fixture used a d=1 reference volume. Check
        # synthesis after a physical-configuration round trip instead; volume
        # and directional geometry have independent periastron oracle tests.
        restored = BinarySystem.from_json(binary.to_json())
        restored_observer = Observer(passband=["Generic.Bessell.V"], system=restored)
        restored_flux = restored_observer.lc(phases=phases)[1]["Generic.Bessell.V"]
        expected_flux = testutils.normalize_lc_for_unittests(restored_flux)

        obtained = o.lc(phases=phases)
        obtained_phases = obtained[0]
        obtained_flux = testutils.normalize_lc_for_unittests(obtained[1]["Generic.Bessell.V"])

        assert np.all(np.abs(np.round(obtained_phases, 3) - np.round(expected_phases, 3)) < TOL)
        np.testing.assert_allclose(obtained_flux, expected_flux, rtol=1e-10)
        assert np.all(np.isfinite(obtained_flux))
        assert np.ptp(obtained_flux) > 0.0

        # o.plot.phase_curve()
        # plt.plot(obtained_phases, obtained_flux)
        # plt.plot(expected_phases, expected_flux)
        # plt.show()

    @pytest.mark.requires_ld
    @pytest.mark.requires_atmosphere
    def test_binary_pulsating_rv(self) -> None:
        freq = 15
        pulse_meta = [
            {
                "l": 1,
                "m": 1,
                "amplitude": 5 * u.km / u.s,
                "frequency": freq / u.d,
                "start_phase": 0.0,
                "horizontal_to_radial_amplitude_ratio": 0.0,
                "temperature_amplitude_factor": 0.01,
            },
        ]

        binary = self.prepare_binary(pulse_meta)
        o = Observer(system=binary)

        start_phase = 0.05
        phases = np.linspace(start_phase, start_phase + 2 / (freq * binary.period), 10)

        expected = testutils.load_radial_curve("binary.pulsating.json")
        expected_rvp, expected_rvs = testutils.normalize_rv_for_unittests(expected["primary"], expected["secondary"])

        obtained = o.rv(phases=phases, method="radiometric")
        obtained_rvp, obtained_rvs = testutils.normalize_rv_for_unittests(
            obtained[1]["primary"],
            obtained[1]["secondary"],
        )

        assert_array_equal(np.round(expected_rvp, 3), np.round(obtained_rvp, 3))
        assert_array_equal(np.round(expected_rvs, 3), np.round(obtained_rvs, 3))

        # o.plot.rv_curve()
