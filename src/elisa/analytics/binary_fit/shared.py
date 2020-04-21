import numpy as np

from ... import utils

from elisa.analytics.binary.mcmc import McMcMixin
from elisa.analytics.binary import params

from elisa.base.spot import Spot
from elisa.pulse.mode import PulsationMode

from elisa.conf import config

MANDATORY_SPOT_PARAMS = Spot.MANDATORY_KWARGS
OPTIONAL_SPOT_PARAMS = []

MANDATORY_PULSATION_PARAMS = PulsationMode.MANDATORY_KWARGS
OPTIONAL_PULSATION_PARAMS = PulsationMode.OPTIONAL_KWARGS


def load_mcmc_chain(fit_instance, filename, discard=0):
    filename = filename[:-5] if filename[-5:] == '.json' else filename
    data = McMcMixin.restore_flat_chain(fname=filename)
    fit_instance.flat_chain = np.array(data['flat_chain'])[discard:, :]
    fit_instance.variable_labels = data['labels']
    fit_instance.normalization = data['normalization']

    # reproducing results from chain
    params.update_normalization_map(fit_instance.normalization)
    dict_to_add = McMcMixin.resolve_mcmc_result(flat_chain=fit_instance.flat_chain,
                                                labels=fit_instance.variable_labels)
    dict_to_add = params.dict_to_user_format(dict_to_add)
    if fit_instance.fit_params is not None:
        fit_instance.fit_params.update(dict_to_add)
    else:
        raise ValueError('Load fit parameters before loading the chain. '
                         'For eg. by `fit_instance.fit_parameters = your_X0`.')

    return fit_instance.flat_chain, fit_instance.variable_labels, fit_instance.normalization


def check_initial_param_validity(x0, params_distribution):
    """
    Checking if initial parameters dictionary is containing all necessary values and
    no invalid ones.

    :param x0: dict; dictionary of initial parameters
    :param params_distribution; dict; dictionary of necessary and allowed parameters
    :return:
    """
    # checking types of variables
    param_types = {key: None for key, _ in x0.items()}
    utils.invalid_param_checker(param_types, params_distribution['ALL_TYPES'], 'FIT TYPE')
    utils.check_missing_params(params_distribution['MANDATORY_TYPES'], param_types, 'FIT TYPE')

    # checking parameters in system fit parameters
    system_param_names = {key: None for key, _ in x0['system'].items()}
    utils.invalid_param_checker(system_param_names, params_distribution['ALL_SYSTEM_PARAMS'], 'System')
    utils.check_missing_params(params_distribution['MANDATORY_SYSTEM_PARAMS'], system_param_names, 'System')

    # checking parameters in star fit parameters
    composite_names = []
    for component in config.BINARY_COUNTERPARTS.keys():
        star_param_names = {key: None for key, _ in x0[component].items()}
        utils.invalid_param_checker(star_param_names, params_distribution['ALL_STAR_PARAMS'],
                                    f'{component} component')
        utils.check_missing_params(params_distribution['MANDATORY_STAR_PARAMS'], star_param_names,
                                   f'{component} component')

        # checking validity of parameters in spots
        if 'spots' in x0[component].keys():
            for spot_name, spot in x0[component]['spots'].items():
                if spot_name in composite_names:
                    raise NameError(f'Spot name `{spot_name}` is duplicate.')
                composite_names.append(spot_name)
                spot_param_names = {key: None for key, _ in spot.items()}
                utils.invalid_param_checker(spot_param_names, params_distribution['ALL_SPOT_PARAMS'],
                                            f'{component} component spot `{spot_name}`')
                utils.check_missing_params(params_distribution['MANDATORY_SPOT_PARAMS'], spot_param_names,
                                           f'{component} component spot `{spot_name}`')

        # checking validity of parameters in spots
        if 'pulsations' in x0[component].keys():
            for mode_name, mode in x0[component]['pulsations'].items():
                if mode_name in composite_names:
                    raise NameError(f'Pulsations mode name `{mode_name}` is duplicate.')
                composite_names.append(mode_name)
                mode_param_names = {key: None for key, _ in mode.items()}
                utils.invalid_param_checker(mode_param_names, params_distribution['ALL_PULSATIONS_PARAMS'],
                                            f'{component} pulsation mode `{mode_name}`')
                utils.check_missing_params(params_distribution['MANDATORY_SPOT_PARAMS'], mode_param_names,
                                           f'{component} pulsation mode `{mode_name}`')


def write_param_ln(fit_params, param_name, designation, write_fn, line_sep, precision=8):
    """
    Auxiliary function to the fit_summary functions, produces one line in output for given parameter that is present
    in `fit_params`.

    :param fit_params: dict;
    :param param_name: str; name os the parameter in `fit_params`
    :param designation: str; displayed name of the parameter
    :param write_fn: function used to write into console or to the file
    :param line_sep: str; symbols to finish the line
    :return:
    """
    if 'min' in fit_params[param_name].keys() and 'max' in fit_params[param_name].keys():
        bot = fit_params[param_name]['min'] - fit_params[param_name]['value']
        top = fit_params[param_name]['max'] - fit_params[param_name]['value']

        aux = np.abs([bot, top])
        aux[aux == 0] = 1e6
        sig_figures = -int(np.log10(np.min(aux))//1) + 1

        bot = round(bot, sig_figures)
        top = round(top, sig_figures)
    else:
        bot, top = '', '',
        sig_figures = precision

    status = 'not recognized'
    if 'fixed' in fit_params[param_name].keys():
        status = 'Fixed' if fit_params[param_name]['fixed'] else 'Variable'
    elif 'constraint' in fit_params[param_name].keys():
        status = fit_params[param_name]['constraint']

    return write_ln(write_fn,
                    designation,
                    round(fit_params[param_name]['value'], sig_figures),
                    bot, top, fit_params[param_name]['unit'],
                    status, line_sep)


def write_ln(write_fn, designation, value, bot, top, unit, status, line_sep, precision=8):
    val = round(value, precision) if type(value) is not str else value
    return write_fn(f"{designation:<35} "
                    f"{val:>20}"
                    f"{bot:>20}"
                    f"{top:>20}"
                    f"{unit:>20}    "
                    f"{status:<50}{line_sep}")
