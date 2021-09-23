import numpy as np

from ... const import PI


# _______MCMC_method_______
def s_squared(y_errs, ln_f):
    """
    Calculates error component of the likelihood function derived from the errors.

    :param y_errs: Dict[str, numpy.array]; errors of observables
    :param ln_f: float; error underestimation factor
    :return: Dict[str, float]; s2 parameter
    """
    return {key: np.power(y_errs[key], 2) + np.power(values, 2) * np.exp(2 * ln_f)
            for key, values in y_errs.items()}


def likelihood_fn(y_data, y_errs, synthetic, ln_f):
    """
    Calculates value of likelihood function for observational data being drawn from distribution around synthetic
    model.

    :param y_data: Dict[str, numpy.array]; observables
    :param y_errs: Dict[str, numpy.array]; errors of observables
    :param ln_f: float; marginalization parameters (currently supported single parameter for error penalization)
    :param synthetic: Dict[str, numpy.array]; synthetic observables
    :return: float; likelihood value
    """
    sigma2 = s_squared(y_errs, ln_f)

    lh = - 0.5 * (np.sum(
        [np.sum((np.power((y_data[key] - synthetic[key]), 2) / sigma2[key]) + np.log(2.0 * PI * sigma2[key]))
         for key, value in synthetic.items()])
    )
    return lh


# ________Leat_Squares_method________
def wssr(y_data, y_err, synthetic):
    """
    Return error weighted sum of squared residuals (wssr) for given set of observational and synthetic data.

    :param y_data: Dict[str, numpy.array]; observables
    :param y_err: Dict[str, numpy.array]; errors of observables
    :param synthetic: Dict[str, numpy.array]; synthetic observables
    :return: float; wssr
    """
    return np.sum([np.sum(np.power((synthetic[item] - y_data[item]) / y_err[item], 2))for item in synthetic])


