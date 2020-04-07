import numpy as np

from ... import utils

from elisa.analytics.binary.mcmc import McMcMixin
from elisa.analytics.binary import params
from elisa.analytics.binary_fit.lc_fit import LCFit


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
    if fit_instance.fit_params is not None:
        fit_instance.fit_params.update(dict_to_add)
    else:
        raise ValueError('Load fit parameters before loading the chain. '
                         'For eg. by `fit_instance.fit_parameters = your_X0`.')

    return fit_instance.flat_chain, fit_instance.variable_labels, fit_instance.normalization


def check_initial_param_validity(x0, all_fit_params, mandatory_fit_params):
    """
    Checking if initial parameters system and composite (spots and pulsations) are containing all necessary values and
    no invalid ones.

    :param x0: dict; dictionary of initial parameters
    :param all_fit_params: list; list of all valid system parameters (spot and pulsation parameters excluded)
    :param mandatory_fit_params: list; list of mandatory system parameters (spot and pulsation parameters excluded)
    :return:
    """
    param_names = {key: value['value'] for key, value in x0.items() if key not in params.COMPOSITE_PARAMS}
    utils.invalid_param_checker(param_names, all_fit_params, 'x0')
    utils.check_missing_params(mandatory_fit_params, param_names, 'x0')

    # checking validity of spot parameters
    spots = {key: value for key, value in x0.items() if key in params.SPOT_PARAMS}
    spot_names = []
    for spot_object in spots.values():
        for spot_name, spot in spot_object.items():
            # checking for duplicate names
            if spot_name in spot_names:
                raise NameError(f'Spot name `{spot_name}` is duplicate.')
            spot_names.append(spot_name)

            spot_condensed = {key: value['value'] for key, value in spot.items()}
            utils.invalid_param_checker(spot_condensed, params.SPOTS_KEY_MAP.values(), spot_name)
            utils.check_missing_params(params.SPOTS_KEY_MAP.values(), spot_condensed, spot_name)

    # checking validity of pulsation mode parameters
    pulsations = {key: value for key, value in x0.items() if key in params.PULSATIONS_PARAMS}
    pulsation_names = []
    for pulsation_object in pulsations.values():
        for pulsation_name, pulsation in pulsation_object.items():
            # checking for duplicate names
            if pulsation_name in pulsation_names:
                raise NameError(f'Pulsations mode name `{pulsation_name}` is duplicate.')
            pulsation_names.append(pulsation_name)

            pulsation_condensed = {key: value['value'] for key, value in pulsation.items()}
            utils.invalid_param_checker(pulsation_condensed, params.PULSATIONS_KEY_MAP.values(), pulsation_name)
            utils.check_missing_params(params.PULSATIONS_KEY_MAP.values(), pulsation_condensed, pulsation_name)


def write_ln(fit_instance, param_name, designation, write_fn, line_sep):
    """
    Auxiliary function to the fit_summary functions, produces one line in output for given parameter that is present
    in `fit_params`.

    :param fit_instance: Union[LCFit, RVFit];
    :param param_name: str; name os the parameter in `fit_params`
    :param designation: str; displayed name of the parameter
    :param write_fn: function used to write into console or to the file
    :param line_sep: str; symbols to finish the line
    :return:
    """
    params = fit_instance.fit_params
    if 'min' in params[param_name].keys() and 'max' in params[param_name].keys():
        bot = params[param_name]['min'] - params[param_name]['value']
        top = params[param_name]['max'] - params[param_name]['value']
        sig_figures = -int(np.log10(np.min(np.abs([bot, top])))//1) + 1
        bot = round(bot, sig_figures)
        top = round(top, sig_figures)
    else:
        bot, top = '', '',
        sig_figures = 10

    status = 'not recognized'
    if 'fixed' in params[param_name].keys():
        status = 'Fixed' if params[param_name]['fixed'] else 'Variable'
    elif 'constraint' in params[param_name].keys():
        status = params[param_name]['constraint']

    return write_fn(f"{designation:<35} "
                    f"{round(params[param_name]['value'], sig_figures):>20}"
                    f"{bot:>20}"
                    f"{top:>20}"
                    f"{params[param_name]['unit']:>20}    "
                    f"{status:<50}{line_sep}")

