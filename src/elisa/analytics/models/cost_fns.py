import numpy as np


def likelihood_fn(observed, synthetic, errors, log_underestimation):
    """
    Returns likelihood function taking into account underestimation of errors.

    :param observed: Dict;
    :param synthetic: Dict;
    :param errors: Dict;
    :param log_underestimation: float;
    :return: float;
    """
    sigma = {np.power(error, 2) + np.exp(2*log_underestimation) * np.power(observed[item], 2)
             for item, error in errors.items()}
    return - 0.5 * (
        np.sum([
            np.sum(np.power((observed[item] - synthetic[item]) / sigma[item], 2)) + np.log(sigma[item])
            for item, value in synthetic.items()])
    )
