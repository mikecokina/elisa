import json
import numpy as np
from abc import abstractmethod
from typing import Union, Dict
from tqdm import tqdm

from elisa import units
from elisa.conf.config import BINARY_COUNTERPARTS
from elisa.analytics.params.parameters import BinaryInitialParameters
from elisa.analytics.params import parameters
from elisa.analytics.binary_fit import least_squares
from elisa.analytics.binary_fit import mcmc
from elisa.analytics.binary_fit import io_tools
from elisa.analytics.binary_fit.mixins import MCMCMixin
from elisa.binary_system.surface.gravity import calculate_polar_gravity_acceleration
from elisa.analytics.models import lc as lc_model
from elisa.analytics.binary_fit.shared import check_for_boundary_surface_potentials
from elisa.base.error import MaxIterationError, MorphologyError

from elisa.logger import getLogger

logger = getLogger('analytics.binary_fit.lc_fit')

DASH_N = 126


class LCFit(object):

    def __init__(self, morphology):
        self.morphology = morphology

        self.result = None
        self.flat_result = None
        self.fit_method_instance: Union[LCFitLeastSquares, LCFitMCMC, None] = None

    def get_result(self):
        return self.result

    def set_result(self, result):
        self.result = result
        self.flat_result = parameters.deserialize_result(self.result)

    def load_result(self, path):
        """
        Function loads fitted parameters of given model.

        :param path: str;
        """
        with open(path, 'r') as f:
            loaded_result = json.load(f)
        self.result = loaded_result
        self.flat_result = parameters.deserialize_result(self.result)

    def save_result(self, path):
        """
        Save result as json.

        :param path: str; path to file
        """
        if self.result is None:
            raise IOError("No result to store.")

        with open(path, 'w') as f:
            json.dump(self.result, f, separators=(',\n', ': '))

    @abstractmethod
    def resolve_fit_cls(self, morphology: str):
        pass


class LCFitMCMC(LCFit):
    def __init__(self, morphology):
        super().__init__(morphology)
        self.fit_method_instance = self.resolve_fit_cls(morphology)()

        self.flat_chain = None
        self.flat_chain_path = None
        self.normalization = None
        self.variable_labels = None

    def resolve_fit_cls(self, morphology: str):
        _cls = {"detached": mcmc.DetachedLightCurveFit, "over-contact": mcmc.OvercontactLightCurveFit}
        return _cls[morphology]

    def fit(self, x0: BinaryInitialParameters, data, **kwargs):
        x0.validate_lc_parameters(morphology=self.morphology)
        self.result = self.fit_method_instance.fit(data=data, x0=x0, **kwargs)
        self.flat_result = self.fit_method_instance.flat_result

        self.flat_chain = self.fit_method_instance.last_sampler.get_chain(flat=True)
        self.flat_chain_path = self.fit_method_instance.flat_chain_path
        self.normalization = self.fit_method_instance.normalization
        self.variable_labels = list(self.fit_method_instance.fitable.keys())

        logger.info('Fitting and processing of results finished successfully.')
        self.fit_summary()

        return self.result

    def load_chain(self, filename, discard=0, percentiles=None):
        """
        Function loads MCMC chain along with auxiliary data from json file created after each MCMC run.

        :param percentiles: List;
        :param self: Union[] instance of fitting cls based on method (mcmc, lsqr) and type(lc, rv)
        :param discard: int; Discard the first discard steps in the chain as burn-in. (default: 0)
        :param filename: str; full name of the json file
        :return: Tuple[numpy.ndarray, List, Dict]; flattened mcmc chain, labels of variables in `flat_chain` columns,
                                                  {var_name: (min_boundary, max_boundary), ...} dictionary of
                                                  boundaries defined by user for each variable needed
                                                  to reconstruct real values from normalized `flat_chain` array
        """
        return io_tools.load_chain(self, filename, discard, percentiles)

    def fit_summary(self, path=None, chain_file=None, parameters_file=None, percentiles=None):
        f = None
        if path is not None:
            f = open(path, 'w')
            write_fn = f.write
            line_sep = '\n'
        else:
            write_fn = print
            line_sep = ''

        if chain_file is None:
            flat_chain = self.flat_chain
            variable_labels = self.variable_labels
            normalization = self.normalization
        else:
            data = MCMCMixin.load_flat_chain(fit_id=chain_file)
            flat_chain = data['flat_chain']
            variable_labels = data['variable_labels']
            normalization = data['normalization']

        if parameters_file is None:
            params = self.flat_result
        else:
            with open(parameters_file, 'r') as f:
                loaded_result = json.load(f)
            params = parameters.deserialize_result(loaded_result)

        init_binary_kwargs = {key: val['value'] for key, val in params.items()}
        # binary_instance = lc_model.prepare_binary(_verify=False, **init_binary_kwargs)

        if flat_chain is None:
            raise ValueError('MCMC chain was not found.')

        renormalized_chain = np.empty(flat_chain.shape)
        for ii, lbl in enumerate(variable_labels):
            renormalized_chain[:, ii] = parameters.renormalize_value(flat_chain[:, ii], normalization[lbl][0],
                                                                     normalization[lbl][1])
        stop_idx = {}
        complete_param_list = [
            'system@mass_ratio', 'system@semi_major_axis', 'system@inclination', 'system@eccentricity',
            'system@argument_of_periastron', 'system@period', 'system@primary_minimum_time', 'system@additional_light',
            'system@phase_shift']

        component_param_list = [
            'mass', 'surface_potential', 'synchronicity', 'equivalent_radius',
            'polar_radius', 'backward_radius', 'side_radius', 'forward_radius', 't_eff',
            'gravity_darkening', 'albedo', 'metallicity', 'critical_potential', 'polar_log_g', 'bolometric_luminosity'
        ]

        stop_idx['system'] = len(complete_param_list)
        spot_numbers, pulsation_numbers = {}, {}

        for component in BINARY_COUNTERPARTS.keys():
            complete_param_list += [f'{component}@{prm}' for prm in component_param_list]

            stop_idx[f'{component}_params'] = len(complete_param_list)
            spot_lbls = set([lbl.split('@')[2] for lbl in params.keys() if f'{component}@spots' in lbl])
            spot_numbers[component] = len(spot_lbls)
            for spot in spot_lbls:
                complete_param_list += [
                    f'{component}@spots@{spot}@longitude', f'{component}@spots@{spot}@latitude',
                    f'{component}@spots@{spot}@radius', f'{component}@spots@{spot}@temperature_factor',
                ]
            stop_idx[f'{component}_spots'] = len(complete_param_list)
            pulse_lbls = set([lbl.split('@')[2] for lbl in params.keys() if f'{component}@pulsations' in lbl])
            pulsation_numbers[component] = len(pulse_lbls)
            for spot in pulse_lbls:
                complete_param_list += [
                    f'{component}@pulsations@{spot}@l', f'{component}@pulsations@{spot}@m',
                    f'{component}@pulsations@{spot}@amplitude', f'{component}@pulsations@{spot}@frequency',
                    f'{component}@pulsations@{spot}@start_phase', f'{component}@pulsations@{spot}@mode_axis_phi',
                    f'{component}@pulsations@{spot}@mode_axis_theta',
                ]
            stop_idx[f'{component}_pulsations'] = len(complete_param_list)

        param_columns = {lbl: ii for ii, lbl in enumerate(complete_param_list)}

        full_chain = np.empty((flat_chain.shape[0], len(complete_param_list)))

        for ii in tqdm(range(flat_chain.shape[0])):
            init_binary_kwargs.update({lbl: renormalized_chain[ii, jj] for jj, lbl in enumerate(variable_labels)})
            try:
                binary_instance = lc_model.prepare_binary(_verify=False, **init_binary_kwargs)
                for var_label in complete_param_list[:stop_idx['system']]:
                    full_chain[ii, param_columns[var_label]] = getattr(binary_instance, var_label.split('@')[1])

                for component in BINARY_COUNTERPARTS.keys():
                    star_instance = getattr(binary_instance, component)
                    for var_label in complete_param_list[stop_idx[f'{component}_params'] - len(component_param_list):
                    stop_idx[f'{component}_params'] - 3]:
                        full_chain[ii, param_columns[var_label]] = getattr(star_instance, var_label.split('@')[1])

                    polar_g = calculate_polar_gravity_acceleration(star_instance,
                                                                   1 - binary_instance.eccentricity,
                                                                   binary_instance.mass_ratio,
                                                                   component=component,
                                                                   semi_major_axis=binary_instance.semi_major_axis,
                                                                   synchronicity=star_instance.synchronicity,
                                                                   logg=True) + 2
                    full_chain[ii, param_columns[f'{component}@polar_log_g']] = polar_g

                    crit_pot = binary_instance.critical_potential(component, 1.0 - binary_instance.eccentricity)
                    full_chain[ii, param_columns[f'{component}@critical_potential']] = crit_pot

                    l_bol = binary_instance.calculate_bolometric_luminosity(components=component)[component]
                    full_chain[ii, param_columns[f'{component}@bolometric_luminosity']] = l_bol

                    if spot_numbers[component] == 0 and pulsation_numbers[component] == 0:
                        continue

                    ref_idx = stop_idx[f'{component}_params']
                    for spot_idx in range(spot_numbers[component]):
                        full_chain[ii, ref_idx + 4 * spot_idx] = star_instance.spots[spot_idx].longitude
                        full_chain[ii, ref_idx + 4 * spot_idx + 1] = star_instance.spots[spot_idx].latitude
                        full_chain[ii, ref_idx + 4 * spot_idx + 2] = star_instance.spots[spot_idx].angular_radius
                        full_chain[ii, ref_idx + 4 * spot_idx + 3] = star_instance.spots[spot_idx].temperature_factor

                    ref_idx = stop_idx[f'{component}_spots']
                    for pulse_idx in range(pulsation_numbers[component]):
                        full_chain[ii, ref_idx + 7*pulse_idx] = star_instance.pulsations[pulse_idx].l
                        full_chain[ii, ref_idx + 7*pulse_idx + 1] = star_instance.pulsations[pulse_idx].m
                        full_chain[ii, ref_idx + 7*pulse_idx + 2] = star_instance.pulsations[pulse_idx].amplitude
                        full_chain[ii, ref_idx + 7*pulse_idx + 3] = star_instance.pulsations[pulse_idx].frequency
                        full_chain[ii, ref_idx + 7*pulse_idx + 4] = star_instance.pulsations[pulse_idx].start_phase
                        full_chain[ii, ref_idx + 7*pulse_idx + 5] = star_instance.pulsations[pulse_idx].mode_axis_phi
                        full_chain[ii, ref_idx + 7*pulse_idx + 6] = star_instance.pulsations[pulse_idx].mode_axis_theta

            except (MaxIterationError, MorphologyError) as e:
                continue

        full_chain_mask = (~np.isnan(full_chain)).any(axis=1)
        full_chain = full_chain[full_chain_mask]

        percentiles = [16, 50, 84] if percentiles is None else percentiles
        calculated_percentiles = np.percentile(full_chain, percentiles, axis=0)
        full_chain_results = np.row_stack((calculated_percentiles[1, :],
                                           calculated_percentiles[1, :] - calculated_percentiles[0, :],
                                           calculated_percentiles[2, :] - calculated_percentiles[1, :]))

        intro = (write_fn, 'Parameter', 'value', '-1 sigma', '+1 sigma', 'unit', 'status', line_sep)
        write_fn(f"\nBINARY SYSTEM{line_sep}")
        io_tools.write_ln(*intro)
        write_fn(f"{'-' * DASH_N}{line_sep}")

        io_tools.write_propagated_ln(full_chain_results[:, param_columns['system@mass_ratio']], params,
                                     'system@mass_ratio', 'Mass ratio (q=M_2/M_1):', write_fn, line_sep, '-')

        sma = (full_chain_results[:, param_columns['system@semi_major_axis']] *
               units.DISTANCE_UNIT).to(units.solRad).value
        io_tools.write_propagated_ln(sma, params, 'system@semi_major_axis', 'Semi major axis (a):', write_fn, line_sep,
                                     'solRad')

        incl = (full_chain_results[:, param_columns['system@inclination']] *
                units.ARC_UNIT).to(units.deg).value
        io_tools.write_propagated_ln(incl, params, 'system@inclination', 'Inclination (i):', write_fn, line_sep, 'deg')

        io_tools.write_propagated_ln(full_chain_results[:, param_columns['system@eccentricity']], params,
                                     'system@eccentricity', 'Eccentricity (e):', write_fn, line_sep, '-')

        omega = (full_chain_results[:, param_columns['system@argument_of_periastron']] *
                 units.ARC_UNIT).to(units.deg).value
        io_tools.write_propagated_ln(omega, params, 'system@argument_of_periastron', 'Argument of periastron (omega):',
                                     write_fn, line_sep, 'deg')

        io_tools.write_propagated_ln(full_chain_results[:, param_columns['system@period']], params,
                                     'system@period', 'Orbital period (P):', write_fn, line_sep, 'd')

        if 'system@primary_minimum_time' in params:
            io_tools.write_param_ln(params, 'system@primary_minimum_time', 'Time of primary minimum (T0):',
                                    write_fn, line_sep)

        io_tools.write_propagated_ln(full_chain_results[:, param_columns['system@additional_light']], params,
                                     'system@additional_light', 'Additional light (l_3):', write_fn,
                                     line_sep, '-')

        io_tools.write_propagated_ln(full_chain_results[:, param_columns['system@phase_shift']], params,
                                     'system@phase_shift', 'Phase shift:', write_fn,
                                     line_sep, '-')

        write_fn(f"{'-' * DASH_N}{line_sep}")

        for component in BINARY_COUNTERPARTS:
            comp_n = 1 if component == 'primary' else 2
            write_fn(f"{component.upper()} COMPONENT{line_sep}")
            io_tools.write_ln(*intro)
            write_fn(f"{'-' * DASH_N}{line_sep}")

            mass = (full_chain_results[:, param_columns[f'{component}@mass']] * units.MASS_UNIT).to(units.solMass).value
            io_tools.write_propagated_ln(mass, params, f'{component}@mass', f'Mass (M_{comp_n}):', write_fn, line_sep,
                                         'solMas')

            io_tools.write_propagated_ln(full_chain_results[:, param_columns[f'{component}@surface_potential']], params,
                                         f'{component}@surface_potential', f'Surface potential (Omega_{comp_n}):',
                                         write_fn, line_sep, '-')
            io_tools.write_propagated_ln(full_chain_results[:, param_columns[f'{component}@critical_potential']], params,
                                         f'{component}@critical_potential', 'Critical potential at L_1:',
                                         write_fn, line_sep, '-')
            io_tools.write_propagated_ln(full_chain_results[:, param_columns[f'{component}@synchronicity']],
                                         params, f'{component}@synchronicity', f'Synchronicity (F_{comp_n}):',
                                         write_fn, line_sep, '-')
            io_tools.write_propagated_ln(full_chain_results[:, param_columns[f'{component}@polar_log_g']],
                                         params, f'{component}@polar_log_g', 'Polar gravity (log g):',
                                         write_fn, line_sep, 'log(cgs)')

            r_equiv = (full_chain_results[:, param_columns[f'{component}@equivalent_radius']] *
                       full_chain_results[:, param_columns['system@semi_major_axis']] *
                       units.DISTANCE_UNIT).to(units.solRad).value

            io_tools.write_propagated_ln(r_equiv, params, f'{component}@equivalent_radius',
                                         'Equivalent radius (R_equiv):', write_fn, line_sep, 'solRad')
            write_fn(f"\nPeriastron radii{line_sep}")

            r_polar = (full_chain_results[:, param_columns[f'{component}@polar_radius']] *
                       full_chain_results[:, param_columns['system@semi_major_axis']] *
                       units.DISTANCE_UNIT).to(units.solRad).value
            r_backw = (full_chain_results[:, param_columns[f'{component}@backward_radius']] *
                       full_chain_results[:, param_columns['system@semi_major_axis']] *
                       units.DISTANCE_UNIT).to(units.solRad).value
            r_side = (full_chain_results[:, param_columns[f'{component}@side_radius']] *
                      full_chain_results[:, param_columns['system@semi_major_axis']] *
                      units.DISTANCE_UNIT).to(units.solRad).value

            io_tools.write_propagated_ln(r_polar, params, f'{component}@polar_radius',
                                         'Polar radius:', write_fn, line_sep, 'solRad')
            io_tools.write_propagated_ln(r_backw, params, f'{component}@backw_radius',
                                         'Backward radius:', write_fn, line_sep, 'solRad')
            io_tools.write_propagated_ln(r_side, params, f'{component}@side_radius',
                                         'Backward radius:', write_fn, line_sep, 'solRad')
            if self.morphology is not 'over-contact':
                r_forw = (full_chain_results[:, param_columns[f'{component}@forward_radius']] *
                          full_chain_results[:, param_columns['system@semi_major_axis']] *
                          units.DISTANCE_UNIT).to(units.solRad).value
                io_tools.write_propagated_ln(r_forw, params, f'{component}@forward_radius',
                                             'Forward radius:', write_fn, line_sep, 'solRad')

            write_fn(f"\nAtmospheric parameters{line_sep}")
            io_tools.write_propagated_ln(full_chain_results[:, param_columns[f'{component}@t_eff']],
                                         params, f'{component}@t_eff', f'Effective temperature (T_eff{comp_n}):',
                                         write_fn, line_sep, 'K')

            l_bol = (full_chain_results[:, param_columns[f'{component}@bolometric_luminosity']] *
                     units.LUMINOSITY_UNIT).to('L_sun').value
            io_tools.write_propagated_ln(l_bol, params, f'{component}@bolometric_luminosity',
                                         'Bolometric luminosity (L_bol): ', write_fn, line_sep, 'L_sol')
            io_tools.write_propagated_ln(full_chain_results[:, param_columns[f'{component}@gravity_darkening']],
                                         params, f'{component}@gravity_darkening',
                                         f'Gravity darkening factor (G_{comp_n}):', write_fn, line_sep, '-')
            io_tools.write_propagated_ln(full_chain_results[:, param_columns[f'{component}@albedo']],
                                         params, f'{component}@albedo',
                                         f'Albedo (A_{comp_n}):', write_fn, line_sep, '-')
            io_tools.write_propagated_ln(full_chain_results[:, param_columns[f'{component}@metallicity']],
                                         params, f'{component}@metallicity',
                                         'Metallicity (log10(X_Fe/X_H)):', write_fn, line_sep, '-')

            write_fn(f"{'-' * DASH_N}{line_sep}")


class LCFitLeastSquares(LCFit):
    def __init__(self, morphology):
        super().__init__(morphology)
        self.fit_method_instance = self.resolve_fit_cls(morphology)()

    def resolve_fit_cls(self, morphology: str):
        _cls = {"detached": least_squares.DetachedLightCurveFit, "over-contact": least_squares.OvercontactLightCurveFit}
        return _cls[morphology]

    def fit(self, x0: BinaryInitialParameters, data, **kwargs):
        x0.validate_lc_parameters(morphology=self.morphology)
        self.result = self.fit_method_instance.fit(data=data, x0=x0, **kwargs)
        self.flat_result = self.fit_method_instance.flat_result
        logger.info('Fitting and processing of results finished successfully.')
        self.fit_summary()
        return self.result

    def fit_summary(self, path=None):
        f = None
        if path is not None:
            f = open(path, 'w')
            write_fn = f.write
            line_sep = '\n'
        else:
            write_fn = print
            line_sep = ''

        try:
            result_dict: Dict = self.flat_result
            result_dict = check_for_boundary_surface_potentials(result_dict)
            b_kwargs = {key: val['value'] for key, val in result_dict.items()}
            binary_instance = lc_model.prepare_binary(_verify=False, **b_kwargs)

            intro = (write_fn, 'Parameter', 'value', '-1 sigma', '+1 sigma', 'unit', 'status', line_sep)
            write_fn(f"\nBINARY SYSTEM{line_sep}")
            io_tools.write_ln(*intro)
            write_fn(f"{'-' * DASH_N}{line_sep}")

            q_desig = 'Mass ratio (q=M_2/M_1):'
            a_desig = 'Semi major axis (a):'

            if 'system@mass_ratio' in result_dict:
                io_tools.write_param_ln(result_dict, 'system@mass_ratio', q_desig, write_fn, line_sep, 3)
                io_tools.write_param_ln(result_dict, 'system@semi_major_axis', a_desig, write_fn, line_sep, 3)
            else:
                io_tools.write_ln(write_fn, q_desig, binary_instance.mass_ratio, '', '', '', 'derived', line_sep, 3)
                sma = (binary_instance.semi_major_axis * units.DISTANCE_UNIT).to(units.solRad).value
                io_tools.write_ln(write_fn, a_desig, sma, '', '', 'AU', 'derived', line_sep, 3)

            io_tools.write_param_ln(result_dict, 'system@inclination', 'Inclination (i):', write_fn, line_sep, 2)
            io_tools.write_param_ln(result_dict, 'system@eccentricity', 'Eccentricity (e):', write_fn, line_sep, 2)
            io_tools.write_param_ln(result_dict, 'system@argument_of_periastron', 'Argument of periastron (omega):',
                                    write_fn, line_sep, 2)
            io_tools.write_param_ln(result_dict, 'system@period', 'Orbital period (P):', write_fn, line_sep)

            if 'system@primary_minimum_time' in result_dict:
                io_tools.write_param_ln(result_dict, 'system@primary_minimum_time', 'Time of primary minimum (T0):',
                                        write_fn, line_sep)
            if 'system@additional_light' in result_dict:
                io_tools.write_param_ln(result_dict, 'system@additional_light', 'Additional light (l_3):',
                                        write_fn, line_sep, 4)

            p_desig = 'Phase shift:'
            if 'system@phase_shift' in result_dict:
                io_tools.write_param_ln(result_dict, 'system@phase_shift', p_desig, write_fn, line_sep)
            else:
                io_tools.write_ln(write_fn, p_desig, 0.0, '-', '-', '-', 'Fixed', line_sep)

            # tool_shared.write_ln(write_fn, 'Morphology: ', binary_instance.morphology, '-', '-', '-', 'derived',
            #                      line_sep)

            write_fn(f"{'-' * DASH_N}{line_sep}")

            for component in BINARY_COUNTERPARTS:
                comp_n = 1 if component == 'primary' else 2
                star_instance = getattr(binary_instance, component)
                write_fn(f"{component.upper()} COMPONENT{line_sep}")
                io_tools.write_ln(*intro)
                write_fn(f"{'-' * DASH_N}{line_sep}")

                m_desig = f'Mass (M_{comp_n}):'
                if f'{component}@mass' in result_dict:
                    io_tools.write_param_ln(result_dict, f'{component}@mass', m_desig, write_fn, line_sep, 3)
                else:
                    mass = (star_instance.mass * units.MASS_UNIT).to(units.solMass).value
                    io_tools.write_ln(write_fn, m_desig, mass,
                                      '-', '-', 'solMass', 'Derived', line_sep, 3)

                io_tools.write_param_ln(result_dict, f'{component}@surface_potential',
                                        f'Surface potential (Omega_{comp_n}):', write_fn, line_sep, 4)

                crit_pot = binary_instance.critical_potential(component, 1.0 - binary_instance.eccentricity)
                io_tools.write_ln(write_fn, 'Critical potential at L_1:', crit_pot, '-', '-', '-', 'Derived', line_sep, 4)

                f_desig = f'Synchronicity (F_{comp_n}):'

                if f'{component}@synchronicity' in result_dict:
                    io_tools.write_param_ln(result_dict, f'{component}@synchronicity', f_desig, write_fn, line_sep, 3)
                else:
                    io_tools.write_ln(write_fn, f_desig, star_instance.synchronicity, '-', '-', '-', 'Fixed', line_sep, 3)

                polar_g = calculate_polar_gravity_acceleration(star_instance,
                                                               1 - binary_instance.eccentricity,
                                                               binary_instance.mass_ratio,
                                                               component=component,
                                                               semi_major_axis=binary_instance.semi_major_axis,
                                                               synchronicity=star_instance.synchronicity,
                                                               logg=True) + 2

                io_tools.write_ln(write_fn, 'Polar gravity (log g):', polar_g, '-', '-', 'log(cgs)', 'Derived',
                                  line_sep, 3)

                r_equiv = (star_instance.equivalent_radius * binary_instance.semi_major_axis *
                           units.DISTANCE_UNIT).to(units.solRad).value

                io_tools.write_ln(write_fn, 'Equivalent radius (R_equiv):', r_equiv,
                                  '-', '-', 'solRad', 'Derived', line_sep, 5)

                write_fn(f"\nPeriastron radii{line_sep}")
                polar_r = (star_instance.polar_radius * binary_instance.semi_major_axis
                           * units.DISTANCE_UNIT).to(units.solRad).value

                back_r = (star_instance.backward_radius * binary_instance.semi_major_axis
                          * units.DISTANCE_UNIT).to(units.solRad).value
                side_r = (star_instance.side_radius * binary_instance.semi_major_axis
                          * units.DISTANCE_UNIT).to(units.solRad).value

                io_tools.write_ln(write_fn, 'Polar radius:', polar_r, '-', '-', 'solRad', 'Derived', line_sep, 5)
                io_tools.write_ln(write_fn, 'Backward radius:', back_r, '-', '-', 'solRad', 'Derived', line_sep, 5)
                io_tools.write_ln(write_fn, 'Side radius:', side_r, '-', '-', 'solRad', 'Derived', line_sep, 5)

                if getattr(star_instance, 'forward_radius', None) is not None:
                    forward_r = (star_instance.forward_radius * binary_instance.semi_major_axis
                                 * units.DISTANCE_UNIT).to(units.solRad).value
                    io_tools.write_ln(write_fn, 'Forward radius:', forward_r, '-',
                                      '-', 'solRad', 'Derived', line_sep, 5)

                write_fn(f"\nAtmospheric parameters{line_sep}")

                io_tools.write_param_ln(result_dict, f'{component}@t_eff', f'Effective temperature (T_eff{comp_n}):',
                                        write_fn, line_sep, 0)

                l_bol = (binary_instance.calculate_bolometric_luminosity(components=component)[component] *
                         units.LUMINOSITY_UNIT).to('L_sun').value

                io_tools.write_ln(write_fn, 'Bolometric luminosity (L_bol): ', l_bol,
                                  '-', '-', 'L_Sol', 'Derived', line_sep, 2)
                io_tools.write_param_ln(result_dict, f'{component}@gravity_darkening',
                                        f'Gravity darkening factor (G_{comp_n}):', write_fn, line_sep, 3)
                io_tools.write_param_ln(result_dict, f'{component}@albedo', f'Albedo (A_{comp_n}):',
                                        write_fn, line_sep, 3)

                met_desig = 'Metallicity (log10(X_Fe/X_H)):'
                if f'{component}@metallicity' in result_dict:
                    io_tools.write_param_ln(result_dict, f'{component}@metallicity', met_desig, write_fn, line_sep, 2)
                else:
                    io_tools.write_ln(write_fn, met_desig, star_instance.metallicity,
                                      '-', '-', '-', 'Fixed', line_sep, 2)

                if star_instance.has_spots():
                    write_fn(f"{'-' * DASH_N}{line_sep}")
                    write_fn(f"{component.upper()} SPOTS{line_sep}")
                    io_tools.write_ln(*intro)

                    for spot in self.result[component]["spots"]:
                        write_fn(f'{"-" * DASH_N}{line_sep}')
                        write_fn(f'Spot: {spot["label"]} {line_sep}')
                        write_fn(f'{"-" * DASH_N}{line_sep}')

                        io_tools.write_param_ln(spot, 'longitude', 'Longitude: ', write_fn, line_sep, 3)
                        io_tools.write_param_ln(spot, 'latitude', 'Latitude: ', write_fn, line_sep, 3)
                        io_tools.write_param_ln(spot, 'angular_radius', 'Angular radius: ', write_fn, line_sep, 3)
                        io_tools.write_param_ln(spot, 'temperature_factor', 'Temperature factor (T_spot/T_eff): ',
                                                write_fn, line_sep, 3)

                if star_instance.has_pulsations():
                    write_fn(f"{'-' * DASH_N}{line_sep}")
                    write_fn(f"{component.upper()} PULSATIONS{line_sep}")
                    io_tools.write_ln(*intro)

                    for mode in self.result[component]["pulsations"]:
                        write_fn(f'{"-" * DASH_N}{line_sep}')
                        write_fn(f'Pulsation: {mode["label"]} {line_sep}')
                        write_fn(f'{"-" * DASH_N}{line_sep}')

                        io_tools.write_param_ln(mode, 'l', 'Angular degree (l): ', write_fn, line_sep, 0)
                        io_tools.write_param_ln(mode, 'm', 'Azimuthal order (m): ', write_fn, line_sep, 0)
                        io_tools.write_param_ln(mode, 'amplitude', 'Amplitude (A): ', write_fn, line_sep, 0)
                        io_tools.write_param_ln(mode, 'frequency', 'Frequency (f): ', write_fn, line_sep, 0)
                        io_tools.write_param_ln(mode, 'start_phase', 'Initial phase (at T_0): ', write_fn, line_sep, 3)
                        io_tools.write_param_ln(mode, 'mode_axis_phi', 'Longitude of mode axis: ',
                                                write_fn, line_sep, 1)
                        io_tools.write_param_ln(mode, 'mode_axis_theta', 'Latitude of mode axis: ',
                                                write_fn, line_sep, 1)

                write_fn(f"{'-' * DASH_N}{line_sep}")

            if result_dict.get('r_squared', False):
                if result_dict['r_squared']['value'] is not None:
                    io_tools.write_param_ln(result_dict, 'r_squared', 'Fit R^2: ', write_fn, line_sep, 6)

                write_fn(f"{'-' * DASH_N}{line_sep}")
        finally:
            if f is not None:
                f.close()
