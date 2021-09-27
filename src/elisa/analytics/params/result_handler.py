import json

from elisa.analytics.binary_fit.shared import eval_constraint_in_dict
from elisa.analytics.params import parameters


class FitResultHandler(object):
    """
    Handling of fit results in standard JSON format.
    """
    def __init__(self):
        self.result = None
        self.flat_result = None

    def get_result(self):
        """
        Returns model parameters in standard dict (JSON) format.

        :return: Dict; model parameters in a standardized format
        """
        return self.result

    def load_result(self, path):
        """
        Function loads a JSON file containing model parameters and stores it as an attribute of AnalyticsTask fitting
        instance. This is useful if you want to examine already calculated results using functionality provided by the
        AnalyticsTask instances (e.g: LCBinaryAnalyticsTask, RVBinaryAnalyticsTask, etc.).

        :param path: str; location of a JSON file with parameters
        """
        with open(path, 'r') as f:
            loaded_result = json.load(f)
        self.set_result(loaded_result)

    def save_result(self, path):
        """
        Save result as JSON file.

        :param path: str; path to file
        """
        if self.result is None:
            raise IOError("No result to store.")

        with open(path, 'w') as f:
            json.dump(self.result, f, separators=(',', ': '), indent=4)

    def set_result(self, result):
        """
        Set model parameters in dictionary (JSON format) as an attribute of AnalyticsTask fitting instance. This is
        useful f you want to examine already calculated results using functionality provided by the AnalyticsTask
        instances (e.g: LCBinaryAnalyticsTask, RVBinaryAnalyticsTask, etc.).

        :param result: Dict; model parameters in JSON format
        """
        result = eval_constraint_in_dict(result)
        self.result = result
        self.flat_result = parameters.deserialize_result(self.result)
