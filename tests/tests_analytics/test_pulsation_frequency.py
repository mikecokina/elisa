"""Fitting frequencies stay in cycles/day until mode construction converts to Hz."""

import numpy as np

from astropy import units as u
from elisa.analytics.models import lc, serializers
from elisa.analytics.params import parameters
from elisa.analytics.params.transform import PulsationModeInitialProperties
from elisa.base.error import ElisaInitialParametersError
from elisa.pulse.mode import PulsationMode
from tests.utils import ElisaTestCase


class PulsationFrequencyTestCase(ElisaTestCase):
    def frequency_inputs(self):
        return (1.0, 1 / u.d, (1 / u.d).to(u.Hz), "1 / d")

    def fit_parameters(self, frequency, *, constrained=False):
        mode = {
            "label": "mode1",
            **{
                key: {"value": value, "fixed": True}
                for key, value in {
                    "l": 1, "m": 1, "amplitude": 1.0,
                    "start_phase": 0.0, "mode_axis_theta": 0.0, "mode_axis_phi": 0.0,
                }.items()
            },
            "frequency": frequency,
        }
        modes = [mode]
        if constrained:
            modes.append({
                **mode,
                "label": "mode2",
                "frequency": {"constraint": "primary@pulsation@mode1@frequency * 2.0"},
            })
        star = {
            key: {"value": value, "fixed": True}
            for key, value in {
                "mass": 2.0, "t_eff": 6500.0, "surface_potential": 10.0,
                "albedo": 1.0, "gravity_darkening": 1.0,
            }.items()
        }
        fit = parameters.BinaryInitialParameters(
            system={
                "period": {"value": 2.0, "min": 1.0, "max": 3.0, "fixed": False},
                "eccentricity": {"value": 0.0, "fixed": True},
                "argument_of_periastron": {"value": 0.0, "fixed": True},
                "inclination": {"value": 85.0, "fixed": True},
            },
            primary={**star, "pulsations": modes},
            secondary=star.copy(),
        )
        fit.validate_lc_parameters("detached")
        return fit

    def construct_fitting_model(self, fit):
        # Follow the callbacks' normalization, constraint and serialization path.
        free = fit.get_fitable()
        normalization = fit.get_normalization_map()
        normalized = parameters.vector_normalizer(
            [parameter.value for parameter in free.values()], free, normalization,
        )
        values = parameters.vector_renormalizer(normalized, free, normalization)
        kwargs = parameters.prepare_properties_set(
            np.asarray(values), free, fit.get_constrained(), fit.get_fixed(),
        )
        serialized = serializers.serialize_primary_kwargs(**kwargs)
        binary = lc.prepare_binary(**kwargs)
        return kwargs, serialized, binary

    def test_fit_transform_keeps_cycles_per_day(self):
        for value in self.frequency_inputs():
            with self.subTest(value=value):
                self.assertAlmostEqual(PulsationModeInitialProperties.frequency(value), 1.0)

    def test_bounds_sigma_and_metadata_use_cycles_per_day(self):
        for unit in (None, u.d**-1, u.Hz):
            for explicit_unit in (False, True):
                if unit is None and explicit_unit:
                    continue
                with self.subTest(unit=unit, explicit_unit=explicit_unit):
                    values = {"value": 1.0, "min": 0.5, "max": 2.0, "sigma": 0.1}
                    if unit is not None:
                        values = {key: (value / u.d).to(unit) for key, value in values.items()}
                    if explicit_unit:
                        values = {key: value.value for key, value in values.items()}
                        values["unit"] = unit
                    parameter = parameters.InitialParameter(
                        transform_cls=PulsationModeInitialProperties,
                        param="frequency", fixed=False, **values,
                    )
                    for key, expected in (("value", 1.0), ("min", 0.5), ("max", 2.0), ("sigma", 0.1)):
                        self.assertAlmostEqual(getattr(parameter, key), expected)
                    self.assertEqual(parameter.unit, u.d**-1)
                    metadata = parameter.to_dict()
                    self.assertEqual(u.Unit(metadata["unit"]), u.d**-1)
                    self.assertAlmostEqual(metadata["value"], 1.0)

    def test_free_and_fixed_frequencies_survive_model_construction(self):
        for fixed in (False, True):
            for value in self.frequency_inputs():
                with self.subTest(fixed=fixed, value=value):
                    fit = self.fit_parameters({"value": value, "fixed": fixed, "min": 0.5, "max": 2.0})
                    kwargs, serialized, binary = self.construct_fitting_model(fit)
                    self.assertAlmostEqual(kwargs["primary@pulsation@mode1@frequency"], 1.0)
                    self.assertAlmostEqual(serialized["pulsations"][0]["frequency"], 1.0)
                    self.assertAlmostEqual(binary.primary.pulsations[0].frequency, 1 / 86400, places=14)
                    if fixed:
                        self.assertIsNone(fit.primary.pulsations[0].frequency.min)
                        self.assertIsNone(fit.primary.pulsations[0].frequency.sigma)

    def test_constrained_frequency_uses_same_convention(self):
        for fixed in (False, True):
            with self.subTest(source_fixed=fixed):
                fit = self.fit_parameters(
                    {"value": 1 / 86400, "unit": "Hz", "fixed": fixed,
                     "min": 0.5 / 86400, "max": 2 / 86400},
                    constrained=True,
                )
                kwargs, serialized, binary = self.construct_fitting_model(fit)
                self.assertAlmostEqual(kwargs["primary@pulsation@mode2@frequency"], 2.0)
                self.assertAlmostEqual(serialized["pulsations"][1]["frequency"], 2.0)
                self.assertAlmostEqual(binary.primary.pulsations[0].frequency, 1 / 86400, places=14)
                self.assertAlmostEqual(binary.primary.pulsations[1].frequency, 2 / 86400, places=14)

    def test_direct_mode_frequency_convention_is_unchanged(self):
        for value in self.frequency_inputs():
            with self.subTest(value=value):
                mode = PulsationMode(l=1, m=1, amplitude=1.0, frequency=value)
                self.assertAlmostEqual(mode.frequency, 1 / 86400, places=14)
        self.assertEqual(PulsationMode(l=1, m=1, amplitude=1.0, frequency=1 * u.Hz).frequency, 1.0)

    def test_existing_frequency_validation_is_preserved(self):
        for value in (-1.0, -1 / u.d, -1 * u.Hz):
            with self.subTest(value=value):
                with self.assertRaises(ElisaInitialParametersError):
                    PulsationModeInitialProperties.frequency(value)
        with self.assertRaises(ElisaInitialParametersError):
            PulsationModeInitialProperties.frequency(object())
        with self.assertRaises(u.UnitConversionError):
            PulsationModeInitialProperties.frequency(1 * u.m)
