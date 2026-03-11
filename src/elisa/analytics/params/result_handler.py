from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from elisa.analytics.binary_fit.shared import eval_constraint_in_dict
from elisa.analytics.params import parameters

if TYPE_CHECKING:
    from typing import Any


class FitResultHandler:
    """Handler for fit results in standard JSON format.

    This class manages the storage, retrieval, and manipulation of fitting results
    in JSON format for binary system fitting tasks.
    """

    def __init__(self) -> None:
        """Initialize the FitResultHandler with empty result and flat result attributes."""
        self.result: dict[str, Any] | None = None
        self.flat_result: dict[str, Any] | None = None

    def get_result(self) -> dict[str, Any] | None:
        """Get model parameters in standard dictionary (JSON) format.

        :returns: Model parameters in a standardized format as a dictionary,
                  or None if no result has been set.
        :rtype: dict[str, Any] | None
        """
        return self.result

    def load_result(self, path: str | Path, *, autofill_sma: bool = False) -> None:
        """Load a JSON file containing model parameters and store it in this instance.

        This is useful for examining already calculated results using functionality
        provided by AnalyticsTask instances (e.g., LCBinaryAnalyticsTask, RVBinaryAnalyticsTask).

        :param path: Location of a JSON file with model parameters.
        :type path: str | Path
        :param autofill_sma: If True, the semi-major axis will be autofilled to fitting
                             parameters if absent. Defaults to False.
        :type autofill_sma: bool
        """
        path_obj = Path(path) if isinstance(path, str) else path
        loaded_result = json.loads(path_obj.read_text(encoding="utf-8"))
        self.set_result(loaded_result, autofill_sma=autofill_sma)

    def save_result(self, path: str | Path) -> None:
        """Save result as a JSON file.

        :param path: Path to file where result will be saved.
        :type path: str | Path
        :raises OSError: If no result has been set.
        """
        if self.result is None:
            error_msg = "No result to store."
            raise OSError(error_msg)

        path_obj = Path(path) if isinstance(path, str) else path
        path_obj.write_text(
            json.dumps(self.result, separators=(",", ": "), indent=4),
            encoding="utf-8",
        )

    def set_result(self, result: dict[str, Any], *, autofill_sma: bool = False) -> None:
        """Set model parameters in dictionary (JSON format) as an attribute of this instance.

        This is useful for examining already calculated results using functionality
        provided by AnalyticsTask instances (e.g., LCBinaryAnalyticsTask, RVBinaryAnalyticsTask).

        :param result: Model parameters in JSON format.
        :type result: dict[str, Any]
        :param autofill_sma: If True, the function will try to autofill the semi-major axis
                             to fitting parameters if absent. Defaults to False.
        :type autofill_sma: bool
        """
        result = eval_constraint_in_dict(result)
        result = parameters.extend_result_with_sma(result) if autofill_sma else result
        self.result = result
        self.flat_result = parameters.deserialize_result(self.result)
