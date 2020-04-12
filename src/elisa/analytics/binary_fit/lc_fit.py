import json
from ...logger import getLogger
from copy import copy
from astropy import units as u

from ... import utils
from elisa.analytics.binary_fit.plot import (
    LCPlot, params
)
from elisa.analytics import utils as autils
from elisa.analytics.binary.least_squares import (
    binary_detached as lst_detached_fit,
    binary_overcontact as lst_over_contact_fit
)
from elisa.analytics.binary.mcmc import (
    binary_detached as mcmc_detached_fit,
    binary_overcontact as mcmc_over_contact_fit
)
from elisa import units
from elisa.analytics.binary_fit import shared
from elisa.analytics.binary.models import prepare_binary
from elisa.binary_system.surface.gravity import calculate_polar_gravity_acceleration

logger = getLogger('analytics.binary_fit.lc_fit')


class LCFit(object):
    MANDATORY_KWARGS = ['light_curves', ]
    OPTIONAL_KWARGS = []
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    MANDATORY_FIT_PARAMS = ['eccentricity', 'argument_of_periastron', 'period', 'inclination', 'p__t_eff', 's__t_eff',
                            'p__surface_potential', 's__surface_potential', 'p__gravity_darkening',
                            's__gravity_darkening', 'p__albedo', 's__albedo', 'period']
    OPTIONAL_FIT_PARAMS = ['primary_minimum_time', 'p__mass', 's__mass', 'semi_major_axis',
                           'asini', 'mass_ratio', 'additional_light', 'phase_shift', 'p__synchronicity',
                           's__synchronicity', 'p__metallicity', 's__metallicity', 'p__spots', 's__spots',
                           'p__pulsations', 's__pulsations']
    ALL_FIT_PARAMS = MANDATORY_FIT_PARAMS + OPTIONAL_FIT_PARAMS

    FIT_PARAMS_COMBINATIONS = {
        'standard': ['p__mass', 's__mass', 'inclination', 'eccentricity', 'argument_of_periastron', 'period',
                     'primary_minimum_time', 'p__t_eff', 's__t_eff', 'p__surface_potential',
                     's__surface_potential', 'p__gravity_darkening', 's__gravity_darkening', 'p__albedo',
                     's__albedo', 'additional_light', 'phase_shift', 'p__synchronicity', 's__synchronicity',
                     'p__metallicity', 's__metallicity', 'p__spots', 's__spots', 'p__pulsations', 's__pulsations'],
        'community': ['mass_ratio', 'semi_major_axis', 'inclination', 'eccentricity', 'argument_of_periastron',
                      'period', 'primary_minimum_time', 'p__t_eff', 's__t_eff', 'p__surface_potential',
                      's__surface_potential', 'p__gravity_darkening', 's__gravity_darkening', 'p__albedo',
                      's__albedo', 'additional_light', 'phase_shift', 'p__synchronicity', 's__synchronicity',
                      'p__metallicity', 's__metallicity', 'p__spots', 's__spots', 'p__pulsations',
                      's__pulsations']
    }

    def __init__(self, **kwargs):
        utils.invalid_kwarg_checker(kwargs, LCFit.ALL_KWARGS, LCFit)
        utils.check_missing_kwargs(self.__class__.MANDATORY_KWARGS, kwargs, instance_of=self.__class__)

        self.light_curves = None
        self.fit_params = None
        self.ranges = None
        self.period = None

        # MCMC specific variables
        self.flat_chain = None
        self.flat_chain_path = None
        self.normalization = None
        self.variable_labels = None

        self.plot = LCPlot(self)

        # values of properties
        logger.debug(f"setting properties of LCFit")
        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])

    def fit(self, x0, morphology='detached', method='least_squares', discretization=3, interp_treshold=None, **kwargs):
        """
        Function solves an inverse task of inferring parameters of the eclipsing binary from the observed light curve.
        Least squares method is adopted from scipy.optimize.least_squares.
        MCMC uses emcee package to perform sampling.

        :param x0: Dict; initial state (metadata included) {param_name: {`value`: value, `unit`: astropy.unit, ...},
                   ...}
        :param morphology: str; `detached` (default) or `over-contact`
        :param method: str; 'least_squares` (default) or `mcmc`
        :param discretization: Union[int, float]; discretization factor of the primary component, default value: 3
        :param kwargs: dict; method-dependent
        :param interp_treshold: bool; Above this total number of datapoints, light curve will be interpolated using
            model containing `interp_treshold` equidistant points per epoch
        :**kwargs options for least_squares**: passes arguments of scipy.optimize.least_squares method except
                                               `fun`, `x0` and `bounds`
        :**kwargs options for mcmc**:
            * ** nwalkers (int) ** * - The number of walkers in the ensemble. Minimum is 2 * number of free parameters.
            * ** nsteps (int) ** * - The number of steps to run. Default is 1000.
            * ** initial_state (numpy.ndarray) ** * - The initial state or position vector made of free parameters with
                    shape (nwalkers, number of free parameters). The order of the parameters is specified by their order
                    in `x0`. Be aware that initial states should be supplied in normalized form (0, 1). For example, 0
                    means value of the parameter at `min` and 1.0 at `max` value in `x0`. By default, they are generated
                    randomly uniform distribution.
            * ** burn_in ** * - The number of steps to run to achieve equilibrium where useful sampling can start.
                    Default value is nsteps / 10.
            * ** percentiles ** * - List of percentiles used to create error results and confidence interval from MCMC
                    chain. Default value is [16, 50, 84] (1-sigma confidence interval)
        :return: dict; resulting parameters {param_name: {`value`: value, `unit`: astropy.unit, ...}, ...}
        """
        # treating a lack of `value` key in constrained parameters
        x0 = autils.prep_constrained_params(x0)

        shared.check_initial_param_validity(x0, LCFit.ALL_FIT_PARAMS, LCFit.MANDATORY_FIT_PARAMS)

        # transforming initial parameters to base units
        x0 = autils.transform_initial_values(x0)

        x_data, y_data, yerr = dict(), dict(), dict()
        for fltr, data in self.light_curves.items():
            x_data[fltr] = data.x_data
            y_data[fltr] = data.y_data
            yerr[fltr] = data.yerr

        if method == 'least_squares':
            fit_fn = lst_detached_fit if morphology == 'detached' else lst_over_contact_fit
            self.fit_params = fit_fn.fit(xs=x_data, ys=y_data, x0=x0, yerr=yerr,
                                         discretization=discretization, interp_treshold=interp_treshold, **kwargs)

        elif str(method).lower() in ['mcmc']:
            fit_fn = mcmc_detached_fit if morphology == 'detached' else mcmc_over_contact_fit
            self.fit_params = fit_fn.fit(xs=x_data, ys=y_data, x0=x0, yerr=yerr,
                                         discretization=discretization, interp_treshold=interp_treshold, **kwargs)
            self.flat_chain = fit_fn.last_sampler.get_chain(flat=True)
            self.flat_chain_path = fit_fn.last_fname
            self.normalization = fit_fn.last_normalization
            self.variable_labels = fit_fn.labels

        else:
            raise ValueError('Method name not recognised. Available methods `least_squares`, `mcmc` or `MCMC`.')

        self.fit_summary()
        return self.fit_params

    def load_chain(self, filename, discard=0):
        """
        Function loads MCMC chain along with auxiliary data from json file created after each MCMC run.

        :param discard: Discard the first `discard` steps in the chain as burn-in. (default: 0)
        :param filename: str; full name of the json file
        :return: Tuple[numpy.ndarray, list, Dict]; flattened mcmc chain, labels of variables in `flat_chain` columns,
                                                   {var_name: (min_boundary, max_boundary), ...} dictionary of 
                                                   boundaries defined by user for each variable needed
                                                   to reconstruct real values from normalized `flat_chain` array
        """
        return shared.load_mcmc_chain(self, filename, discard=discard)

    def store_parameters(self, parameters=None, filename=None):
        """
        Function converts model parameters to json compatibile format and stores model parameters.

        :param parameters: Dict; {'name': {'value': numpy.ndarray, 'unit': Union[astropy.unit, str], ...}, ...}
        :param filename: str;
        """
        parameters = copy(self.fit_params) if parameters is None else parameters

        parameters = autils.unify_unit_string_representation(parameters)

        json_params = autils.convert_dict_to_json_format(parameters)
        with open(filename, 'w') as f:
            json.dump(json_params, f, separators=(',\n', ': '))

    def load_parameters(self, filename=None):
        """
        Function loads fitted parameters of given model.

        :param filename: str;
        :return: Dict; {'name': {'value': numpy.ndarray, 'unit': Union[astropy.unit, str], ...}, ...}
        """
        with open(filename, 'r') as f:
            prms = json.load(f)

        prms = autils.convert_json_to_dict_format(prms)
        self.period = prms['period']['value']
        self.fit_params = prms

        return prms

    def fit_summary(self, filename=None, fit_params=None):
        """
        Producing a summary of the fit in more human readable form.

        :param filename: Union[str, None]; if not None, summary is stored in file, otherwise it is printed into console
        :return:
        """
        def prep_input_params(fit_params):
            r2 = fit_params.pop('r_squared')['value'] if 'r_squared' in fit_params.keys() else None

            # transforming initial parameters to base units
            fit_params = autils.transform_initial_values(fit_params)

            x0_vector, labels = params.x0_vectorize(fit_params)
            fixed = params.x0_to_fixed_kwargs(fit_params)
            constraint = params.x0_to_constrained_kwargs(fit_params)

            processed_params = {lbl: {'value': x0_vector[ii], 'fixed': False} for ii, lbl in enumerate(labels)}

            for lbl, val in processed_params.items():
                s_lbl = lbl.split(params.PARAM_PARSER)
                if s_lbl[0] in params.COMPOSITE_PARAMS:
                    if 'min' in fit_params[s_lbl[0]][s_lbl[1]][s_lbl[2]].keys() and \
                            'max' in fit_params[s_lbl[0]][s_lbl[1]][s_lbl[2]].keys():
                        val.update({'min': fit_params[s_lbl[0]][s_lbl[1]][s_lbl[2]]['min'],
                                    'max': fit_params[s_lbl[0]][s_lbl[1]][s_lbl[2]]['max']})

                else:
                    if 'min' in fit_params[lbl].keys() and 'max' in fit_params[lbl].keys():
                        val.update({'min': fit_params[lbl]['min'], 'max': fit_params[lbl]['max']})

            processed_params.update({lbl: {'value': val, 'fixed': True} for lbl, val in fixed.items()})

            results = {lbl: val['value'] for lbl, val in processed_params.items()}
            constraint_dict = params.constraints_evaluator(results, constraint)

            processed_params.update({lbl: {'value': val, 'constraint': constraint[lbl]}
                                     for lbl, val in constraint_dict.items()})


            processed_params = params.extend_result_with_units(processed_params)
            processed_params = params.dict_to_user_format(processed_params)

            processed_params.update({'r_squared': {'value': r2, 'unit': ''}})
            return processed_params

        def component_summary(binary_instance, component):
            """
            Unified summary for binary system component fit summary
            :param binary_instance: BinarySystem;
            :param component: str;
            :return:
            """
            comp_n = 1 if component == 'primary' else 2
            comp_prfx = 'p__' if component == 'primary' else 's__'
            star_instance = getattr(binary_instance, component)

            write_fn(f"#{'-'*125}{line_sep}")
            write_fn(f"# {component.upper()} COMPONENT{line_sep}")
            shared.write_ln(*intro)
            write_fn(f"#{'-'*125}{line_sep}")

            m_desig = f'Mass (M_{comp_n}):'
            shared.write_param_ln(processed_params, f'{comp_prfx}mass', m_desig, write_fn, line_sep, 3) \
                if f'{comp_prfx}mass' in processed_params.keys() else \
                shared.write_ln(write_fn, m_desig, (star_instance.mass * units.MASS_UNIT).to(u.solMass).value,
                                '', '', 'solMass', 'Derived', line_sep, 3)

            shared.write_param_ln(processed_params, f'{comp_prfx}surface_potential', 'Surface potential (Omega_1):',
                                  write_fn, line_sep, 4)
            crit_pot = binary_instance.critical_potential(component=component,
                                                          components_distance=1-binary_instance.eccentricity)
            shared.write_ln(write_fn, 'Critical potential at L_1:', crit_pot, '', '', '', 'Derived', line_sep, 4)

            f_desig = f'Synchronicity (F_{comp_n}):'
            shared.write_param_ln(processed_params, f'{comp_prfx}synchronicity', f_desig, write_fn, line_sep, 3) \
                if f'{comp_prfx}synchronicity' in processed_params.keys() else \
                shared.write_ln(write_fn, f_desig, star_instance.synchronicity,
                                '', '', '', 'Fixed', line_sep, 3)

            polar_g = calculate_polar_gravity_acceleration(star_instance,
                                                           1 - binary_instance.eccentricity,
                                                           binary_instance.mass_ratio,
                                                           component=component,
                                                           semi_major_axis=binary_instance.semi_major_axis,
                                                           logg=True) + 2
            shared.write_ln(write_fn, 'Polar gravity (log g):', polar_g,
                            '', '', 'cgs', 'Derived', line_sep, 3)

            write_fn(f"# Periastron radii{line_sep}")
            polar_r = \
                (star_instance.polar_radius * binary_instance.semi_major_axis * units.DISTANCE_UNIT). \
                    to(u.solRad).value
            back_r = \
                (star_instance.backward_radius * binary_instance.semi_major_axis * units.DISTANCE_UNIT). \
                    to(u.solRad).value
            side_r = \
                (star_instance.side_radius * binary_instance.semi_major_axis * units.DISTANCE_UNIT). \
                    to(u.solRad).value

            shared.write_ln(write_fn, 'Polar radius:', polar_r, '', '', 'solRad', 'Derived', line_sep, 5)
            shared.write_ln(write_fn, 'Backward radius:', back_r, '', '', 'solRad', 'Derived', line_sep, 5)
            shared.write_ln(write_fn, 'Side radius:', side_r, '', '', 'solRad', 'Derived', line_sep, 5)

            if getattr(star_instance, 'forward_radius', None) is not None:
                forward_r = \
                    (star_instance.forward_radius * binary_instance.semi_major_axis * units.DISTANCE_UNIT). \
                        to(u.solRad).value
                shared.write_ln(write_fn, 'Forward radius:', forward_r, '', '', 'solRad', 'Derived', line_sep, 5)

            write_fn(f"# Atmospheric parameters{line_sep}")
            shared.write_param_ln(processed_params, f'{comp_prfx}t_eff', f'Effective temperature (T_eff{comp_n}):',
                                  write_fn, line_sep, 0)
            shared.write_param_ln(processed_params, f'{comp_prfx}gravity_darkening',
                                  f'Gravity darkening factor (G_{comp_n}):', write_fn, line_sep, 3)
            shared.write_param_ln(processed_params, f'{comp_prfx}albedo', f'Albedo (A_{comp_n}):', write_fn, line_sep, 3)

            met_desig = 'Metallicity (log(X_Fe/X_H)):'
            shared.write_param_ln(processed_params, f'{comp_prfx}metallicity', met_desig, write_fn, line_sep, 2) \
                if f'{comp_prfx}metallicity' in processed_params.keys() else \
                shared.write_ln(write_fn, met_desig, star_instance.metallicity,
                                '', '', '', 'Fixed', line_sep, 2)

            if star_instance.has_spots():
                write_fn(f"#{'-'*125}{line_sep}")
                write_fn(f"# {component.upper()} SPOTS{line_sep}")
                shared.write_ln(*intro)

                for spot_name, spot in processed_params[f'{comp_prfx}spots'].items():
                    write_fn(f"#{'-'*125}{line_sep}")
                    write_fn(f"# Spot: {spot_name} {line_sep}")
                    write_fn(f"#{'-'*125}{line_sep}")

                    shared.write_param_ln(spot, 'longitude', 'Longitude: ', write_fn, line_sep, 3)
                    shared.write_param_ln(spot, 'latitude', 'Latitude: ', write_fn, line_sep, 3)
                    shared.write_param_ln(spot, 'angular_radius', 'Angular radius: ', write_fn, line_sep, 3)
                    shared.write_param_ln(spot, 'temperature_factor', 'Temperature factor (T_spot/T_eff): ',
                                          write_fn, line_sep, 3)

            if star_instance.has_pulsations():
                write_fn(f"#{'-'*125}{line_sep}")
                write_fn(f"# {component.upper()} PULSATIONS{line_sep}")
                shared.write_ln(*intro)

                for mode_name, mode in processed_params[f'{comp_prfx}pulsations'].items():
                    write_fn(f"#{'-'*125}{line_sep}")
                    write_fn(f"# Spot: {mode_name} {line_sep}")
                    write_fn(f"#{'-'*125}{line_sep}")

                    shared.write_param_ln(mode, 'l', 'Angular degree (l): ', write_fn, line_sep, 0)
                    shared.write_param_ln(mode, 'm', 'Azimuthal order (m): ', write_fn, line_sep, 0)
                    shared.write_param_ln(mode, 'amplitude', 'Amplitude (A): ', write_fn, line_sep, 0)
                    shared.write_param_ln(mode, 'frequency', 'Frequency (f): ', write_fn, line_sep, 0)
                    shared.write_param_ln(mode, 'start_phase', 'Initial phase (at T_0): ', write_fn, line_sep, 3)
                    shared.write_param_ln(mode, 'mode_axis_phi', 'Longitude of mode axis: ', write_fn, line_sep, 1)
                    shared.write_param_ln(mode, 'mode_axis_theta', 'Latitude of mode axis: ', write_fn, line_sep, 1)

        fit_params = copy(self.fit_params) if fit_params is None else fit_params
        # preparing binary instance to calculate derived values

        processed_params = prep_input_params(fit_params)

        b_kwargs = params.x0_to_kwargs(processed_params)
        binary_instance = prepare_binary(verify=False, **b_kwargs)

        if filename is not None:
            f = open(filename, 'w')
            write_fn = f.write
            line_sep = '\n'
        else:
            write_fn = print
            line_sep = ''

        intro = (write_fn, '# Parameter', 'value', '-1 sigma', '+1 sigma', 'unit', 'status', line_sep)

        write_fn(f"# BINARY SYSTEM{line_sep}")
        shared.write_ln(*intro)
        write_fn(f"#{'-'*125}{line_sep}")
        q_desig = 'Mass ratio (q=M_2/M_1):'
        a_desig = 'Semi major axis (a):'
        if 'mass_ratio' in processed_params.keys():
            shared.write_param_ln(processed_params, 'mass_ratio', q_desig, write_fn, line_sep, 3)
            shared.write_param_ln(processed_params, 'semi_major_axis', a_desig, write_fn, line_sep, 3)
        else:
            shared.write_ln(write_fn, q_desig, binary_instance.mass_ratio, '', '', '', 'derived',
                            line_sep, 3)
            sma = (binary_instance.semi_major_axis * units.DISTANCE_UNIT).to(u.solRad).value
            shared.write_ln(write_fn, a_desig, sma, '', '', 'AU', 'derived', line_sep, 3)

        shared.write_param_ln(processed_params, 'inclination', 'Inclination (i):', write_fn, line_sep, 2)
        shared.write_param_ln(processed_params, 'eccentricity', 'Eccentricity (e):', write_fn, line_sep, 2)
        shared.write_param_ln(processed_params, 'argument_of_periastron', 'Argument of periastron (omega):', write_fn,
                              line_sep, 2)
        shared.write_param_ln(processed_params, 'period', 'Orbital period (P):', write_fn, line_sep)

        if 'primary_minimum_time' in processed_params.keys():
            shared.write_param_ln(processed_params, 'primary_minimum_time', 'Time of primary minimum (T0):', write_fn,
                                  line_sep)
        if 'additional_light' in processed_params.keys():
            shared.write_param_ln(processed_params, 'additional_light', 'Additional light (l_3):', write_fn, line_sep,
                                  4)

        p_desig = 'Phase shift:'
        if 'phase_shift' in processed_params.keys():
            shared.write_param_ln(processed_params, 'phase_shift', p_desig, write_fn, line_sep)
        else:
            shared.write_ln(write_fn, p_desig, 0.0, '', '', '', 'Fixed', line_sep)

        if processed_params['r_squared']['value'] is not None:
            shared.write_param_ln(processed_params, 'r_squared', 'Fit R^2: ', write_fn, line_sep, 4)

        component_summary(binary_instance, 'primary')
        component_summary(binary_instance, 'secondary')

        if filename is not None:
            f.close()
