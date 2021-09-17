import numpy as np

from . mixins import MCMCMixin
from . shared import AbstractFit

from .. params import parameters
from .. params.parameters import ParameterMeta, BinaryInitialParameters


def filter_chain(mcmc_fit_cls, **boundaries):
    """
    Filtering mcmc chain down to given parameter intervals. This function is usable in case of bimodal distribution of
    the MCMC chain.

    :param mcmc_fit_cls: MCMC fitting cls instance e.g.: (LCFitMCMC, RVFitMCMC)
    :param boundaries: Dict; dictionary of boundaries in flat format (using @)
                             e.g. {`primary@te_ff`: (5000, 6000), other parameters ...}
    :return: numpy.array; filtered flat chain
    """
    for key, boundary in boundaries.items():
        if not isinstance(boundary, (tuple, list, np.ndarray)):
            raise TypeError(f'`{key}` boundary is not tuple or list.')
        if len(boundary) != 2:
            raise TypeError(f'`{key}` has incorrect length of {len(boundary)}.')
        if key not in mcmc_fit_cls.variable_labels:
            raise NameError(f'{key} is not valid model parameter.')

        column_idx = mcmc_fit_cls.variable_labels.index(key)
        column = mcmc_fit_cls.flat_chain[:, column_idx]

        condition_mask = np.logical_and(
            column > parameters.normalize_value(boundary[0], *mcmc_fit_cls.normalization[key]),
            column < parameters.normalize_value(boundary[1], *mcmc_fit_cls.normalization[key])
        )
        if np.sum(condition_mask) == 0:
            raise ValueError(f'Boundaries for {key} yielded an empty array.')
        setattr(mcmc_fit_cls, 'flat_chain', mcmc_fit_cls.flat_chain[condition_mask, :])

    fitted_params = {key: mcmc_fit_cls.flat_result[key] for key in mcmc_fit_cls.variable_labels}
    update_solution(mcmc_fit_cls, fitted_params, percentiles=None)

    return mcmc_fit_cls.flat_chain


def load_chain(mcmc_fit_cls, fit_id, discard=0, percentiles=None):
    """
    Function loads MCMC chain along with auxiliary data from json file created after each MCMC run.

    :param percentiles: List; percentile intervals used to generate confidence intervals, provided in form:
                              [percentile for lower bound of confidence interval, percentile of the centre,
                              percentile for the upper bound of confidence interval]
    :param mcmc_fit_cls: MCMC fitting cls instance e.g.: (LCFitMCMC, RVFitMCMC)
    :param discard: int; Discard the first discard steps in the chain as burn-in. (default: 0)
    :param fit_id: str; chain identificator or filename containing the chain
    :return: Tuple[numpy.ndarray, list, dict]; flattened mcmc chain, labels of variables in `flat_chain` columns,
                                               {var_name: (min_boundary, max_boundary), ...} dictionary of
                                               boundaries defined by user for each variable needed
                                               to reconstruct real values from normalized `flat_chain` array
    """
    data = MCMCMixin.load_flat_chain(fit_id=fit_id)

    mcmc_fit_cls.flat_chain = np.array(data['flat_chain'])[discard:, :]
    mcmc_fit_cls.variable_labels = data['fitable_parameters']
    mcmc_fit_cls.normalization = data['normalization']

    update_solution(mcmc_fit_cls, data['fitable'], percentiles)

    return mcmc_fit_cls.flat_chain, mcmc_fit_cls.variable_labels, mcmc_fit_cls.normalization


def update_solution(mcmc_fit_cls, fitted_params, percentiles):
    """
    Updating solutions based on the distribution of to MCMC chain.

    :param mcmc_fit_cls: MCMC fitting cls instance e.g.: (LCFitMCMC, RVFitMCMC)
    :param fitted_params: Dict; only variable part of flat_result
    :param percentiles: List; percentiles used for evaluation of confidence intervals
    :return: Tuple;
    """
    fitable = {key: ParameterMeta(**val) for key, val in fitted_params.items()}

    # reproducing results from chain
    flat_result_update = MCMCMixin.resolve_mcmc_result(mcmc_fit_cls.flat_chain, fitable,
                                                       mcmc_fit_cls.normalization, percentiles=percentiles)

    if mcmc_fit_cls.result is not None:
        mcmc_fit_cls.flat_result.update(flat_result_update)

        # evaluating constraints
        fit_params = parameters.serialize_result(mcmc_fit_cls.flat_result)
        constrained = BinaryInitialParameters(**fit_params).get_constrained()
        mcmc_fit_cls.flat_result = AbstractFit.eval_constrained_results(mcmc_fit_cls.flat_result, constrained)

        mcmc_fit_cls.result = parameters.serialize_result(mcmc_fit_cls.flat_result)

    else:
        msg = 'Load fit parameters before loading the chain. For eg. UPDATE THIS.'
        raise ValueError(msg)


def write_ln(write_fn, designation, value, bot, top, unit, status, line_sep, precision=8):
    val = round(value, precision) if type(value) is not str else value
    return write_fn(f"{designation:<35} "
                    f"{val:>20}"
                    f"{bot:>20}"
                    f"{top:>20}"
                    f"{unit:>20}    "
                    f"{status:<50}{line_sep}")


def write_param_ln(fit_params, param_id, designation, write_fn, line_sep, precision=8):
    """
    Auxiliary function to the fit_summary functions, produces one
    line in output for given parameter that is present in `fit_params`.

    :param precision: int;
    :param fit_params: Dict;
    :param param_id: str; name os the parameter in `fit_params`
    :param designation: str; displayed name of the parameter
    :param write_fn: function used to write into console or to the file
    :param line_sep: str; symbols to finish the line
    :return:
    """

    if 'confidence_interval' in fit_params[param_id]:
        bot = fit_params[param_id]['value'] - fit_params[param_id]['confidence_interval']['min']
        top = fit_params[param_id]['confidence_interval']['max'] - fit_params[param_id]['value']

        aux = np.abs([bot, top])
        aux[aux == 0] = 1e6
        sig_figures = -int(np.log10(np.min(aux))//1) + 1

        bot = round(bot, sig_figures)
        top = round(top, sig_figures)
    else:
        bot, top = '-', '-',
        sig_figures = precision

    status = 'Not recognized'
    if 'fixed' in fit_params[param_id]:
        status = 'Fixed' if fit_params[param_id]['fixed'] else 'Variable'

    elif 'constraint' in fit_params[param_id].keys():
        status = fit_params[param_id]['constraint']
    elif param_id in ['r_squared']:
        status = 'Derived'

    unit = str(fit_params[param_id]['unit']) if 'unit' in fit_params[param_id].keys() else '-'
    args = write_fn, designation, round(fit_params[param_id]['value'], sig_figures), bot, top, unit, status, line_sep
    return write_ln(*args)


def write_propagated_ln(values, fit_params, param_id, designation, write_fn, line_sep, unit):
    """
    Auxiliary function to the fit_summary functions, produces one
    line in output for given parameter that is present in `fit_params`.

    :param values:
    :param fit_params: Dict;
    :param param_id: str; name os the parameter in `fit_params`
    :param designation: str; displayed name of the parameter
    :param write_fn: function used to write into console or to the file
    :param line_sep: str; symbols to finish the line
    :return:
    """
    # if parameter does not exists in given fitting mode, the line in summary is omitted
    if np.isnan(values).any():
        return

    aux = np.abs([values[1], values[2]])
    aux[aux <= 1e-15] = 1e-15
    sig_figures = -int(np.log10(np.min(aux))//1) + 1

    values = np.round(values, sig_figures)

    if param_id not in fit_params.keys():
        status = 'Derived'
    elif 'fixed' in fit_params[param_id]:
        status = 'Fixed' if fit_params[param_id]['fixed'] else 'Variable'

    elif 'constraint' in fit_params[param_id].keys():
        status = fit_params[param_id]['constraint']
    elif param_id in ['r_squared']:
        status = 'Derived'
    else:
        status = 'Unknown'

    return write_ln(write_fn, designation, values[0], values[1], values[2], unit, status, line_sep)
