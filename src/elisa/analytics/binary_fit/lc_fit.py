from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from elisa.analytics.binary_fit import io_tools, least_squares, mcmc
from elisa.analytics.binary_fit.summary import (
    fit_lc_summary_with_error_propagation,
    simple_lc_fit_summary,
)
from elisa.analytics.params import parameters
from elisa.analytics.params.result_handler import FitResultHandler
from elisa.logger import getLogger

if TYPE_CHECKING:
    from elisa.analytics.binary_fit.mcmc import LightCurveFit
    from elisa.analytics.params.parameters import BinaryInitialParameters
    from elisa.types import Float
else:
    Float = float

logger = getLogger("analytics.binary_fit.lc_fit")

DASH_N = 126


class LCFit(FitResultHandler, ABC):
    """Provide common methods used during a light curve fit.

    This class defines the interface for light curve fitting with support for
    different morphologies and fitting methods.

    :param morphology: str
        Morphology of the binary system ('detached' or 'over-contact').
    :param atmosphere_model: dict | None
        Atmosphere model parameters. If None, default atmosphere model is used.
    :param limb_darkening_coefficients: dict | None
        Limb darkening coefficients. If None, default values are used.
    """

    def __init__(
        self,
        morphology: str,
        atmosphere_model: dict | None,
        limb_darkening_coefficients: dict | None,
    ) -> None:
        super().__init__()
        self.morphology: str = morphology
        self.fit_method_instance: LCFitLeastSquares | LCFitMCMC | None = None
        self.atmosphere_model: dict | None = atmosphere_model
        self.limb_darkening_coefficients: dict | None = limb_darkening_coefficients

    def coefficient_of_determination(
        self,
        model_parameters: dict,
        data: dict[str, Any],
        discretization: Float,
        interp_treshold: int,
    ) -> Float:
        """Return R^2 for given model parameters and observed data.

        The coefficient of determination measures how well the model parameters
        fit the observed light curve data across all passbands. A value of 1.0
        represents a perfect fit to the observations.

        :param model_parameters: dict
            Set of model parameters in JSON format.
        :param data: dict[str, Any]
            Observational data in each passband. Keys are filter names,
            values are LCData instances.
        :param discretization: Float
            Discretization factor for the primary component. Controls the
            number of surface elements used for calculations.
        :param interp_treshold: int
            Number of observation points above which the synthetic curves
            will be calculated using `interp_treshold` equally spaced points
            that will be subsequently interpolated to the desired times of
            observation. This improves computational efficiency for large
            datasets.
        :return: Float
            Coefficient of determination (1.0 means a perfect fit to
            the observations).
        """
        b_parameters = parameters.BinaryInitialParameters(**model_parameters)
        b_parameters.validate_lc_parameters(morphology=self.morphology)
        args = model_parameters, data, discretization, interp_treshold
        return self.fit_method_instance.coefficient_of_determination(*args)

    @abstractmethod
    def resolve_fit_cls(self, morphology: str) -> type:
        """Return the fitting class suitable for the model based on its morphology.

        :param morphology: str
            Morphology of the binary system.
        :return: type
            Fitting class (DetachedLightCurveFit or OvercontactLightCurveFit).
        """
        ...


class LCFitMCMC(LCFit):
    """Perform light curve fitting using the MCMC method.

    This class wraps MCMC sampling functionality for fitting light curves
    with support for error propagation and chain analysis.

    :param morphology: str
        Morphology of the binary system ('detached' or 'over-contact').
    :param atmosphere_model: dict | None
        Atmosphere model parameters.
    :param limb_darkening_coefficients: dict | None
        Limb darkening coefficients.
    """

    def __init__(
        self,
        morphology: str,
        atmosphere_model: dict | None,
        limb_darkening_coefficients: dict | None,
    ) -> None:
        super().__init__(morphology, atmosphere_model, limb_darkening_coefficients)
        self.fit_method_instance: LightCurveFit = self.resolve_fit_cls(morphology)()
        self.fit_method_instance.atmosphere_model = atmosphere_model
        self.fit_method_instance.limb_darkening_coefficients = limb_darkening_coefficients

        self.flat_chain: Any = None
        self.flat_chain_path: Any = None
        self.normalization: Any = None
        self.variable_labels: list[str] | None = None

    def filter_chain(self, **boundaries: dict) -> Any:
        """Filter MCMC chain down to given parameter intervals.

        This function is useful in case of bimodal distribution of the MCMC chain.
        It allows selecting a subset of chain samples based on parameter boundaries.

        :param boundaries: dict
            Dictionary of boundaries, e.g., ``{'primary@te_ff': (5000, 6000), ...}``.
            Keys are parameter names in dotted notation, values are tuples
            (min_bound, max_bound).
        :return: Any
            Filtered flat chain.
        """
        return io_tools.filter_chain(self, **boundaries)

    def fit(
        self,
        x0: BinaryInitialParameters,
        data: dict[str, Any],
        **kwargs: Any,
    ) -> dict:
        """Perform MCMC sampling on the light curve fit.

        :param x0: BinaryInitialParameters
            Initial information about the model parameters including status
            (fixed, variable, constrained), bounds (prior distribution),
            and initial values.
        :param data: dict[str, Any]
            Observational data (light curves in multiple filters). Keys are
            filter names, values are LCData instances.
        :param kwargs: Any
            Additional arguments passed to the fitting method. See
            AnalyticsTask.fit kwargs for MCMC or mcmc.LightCurveFit.fit
            for further information.
        :return: dict
            Optimized model parameters in JSON format.
        """
        x0.validate_lc_parameters(morphology=self.morphology)
        self.result = self.fit_method_instance.fit(data=data, x0=x0, **kwargs)
        self.flat_result = self.fit_method_instance.flat_result

        # noinspection PyArgumentList
        self.flat_chain = self.fit_method_instance.last_sampler.get_chain(flat=True)
        self.flat_chain_path = self.fit_method_instance.flat_chain_path
        self.normalization = self.fit_method_instance.normalization
        self.variable_labels = list(self.fit_method_instance.fitable.keys())

        logger.info("Fitting and processing of results finished successfully.")
        self.fit_summary()
        return self.result

    def fit_summary(
        self,
        filename: str | None = None,
        *,
        propagate_errors: bool = False,
        percentiles: list[int] | None = None,
        dimensionless_radii: bool = True,
    ) -> dict | None:
        """Produce detailed summary of the light curve fitting task.

        This method generates a comprehensive summary with the possibility
        to propagate uncertainties of the fitted binary model parameters
        if MCMC was used and `propagate_errors` is True.

        :param filename: str | None
            Path to store the summary. If None, summary is not saved.
        :param propagate_errors: bool
            If True, propagate errors of fitted parameters to the rest of
            the eclipsing binary parameters (takes a while to calculate).
            Default is False.
        :param percentiles: list[int] | None
            Percentiles used to evaluate confidence intervals from the
            posterior distribution of eclipsing binary parameters in the
            MCMC chain. Used only when `propagate_errors` is True.
            Default is [16, 50, 84] (1-sigma confidence interval).
        :param dimensionless_radii: bool
            If True (default), radii are provided in semi-major axis units.
            If False, radii are provided in solar radii.
        :return: dict | None
            Resulting parameters if errors are propagated, otherwise None.
        """
        if percentiles is None:
            percentiles = [16, 50, 84]
        if not propagate_errors:
            simple_lc_fit_summary(self, filename, dimensionless_radii=dimensionless_radii)
            return None

        fit_lc_summary_with_error_propagation(
            self,
            filename,
            percentiles,
            dimensionless_radii=dimensionless_radii,
        )
        return self.result

    def load_chain(
        self,
        filename: str,
        discard: int = 0,
        percentiles: list[int] | None = None,
    ) -> Any:
        """Load MCMC chain along with auxiliary data from a JSON file.

        Load chain from a JSON file created after each MCMC run,
        including the flattened chain, variable labels, and boundaries.

        :param filename: str
            Chain identifier or filename (ending with .json) containing
            the chain data.
        :param discard: int
            Discard the first `discard` steps in the chain as part of the
            thermalization phase (burn-in). Default is 0.
        :param percentiles: list[int] | None
            Percentile intervals used to generate confidence intervals.
            Should be provided in form [lower_percentile, center_percentile,
            upper_percentile], e.g., [16, 50, 84] for 1-sigma interval.
        :return: Any
            Tuple containing flattened MCMC chain (numpy.ndarray),
            labels of variables in `flat_chain` columns (list),
            and boundaries dictionary (dict) of form
            {var_name: (min_boundary, max_boundary), ...} needed
            to reconstruct real values from normalized `flat_chain` array.
        """
        return io_tools.load_chain(self, filename, discard, percentiles)

    def resolve_fit_cls(self, morphology: str) -> type:
        """Return MCMC fitting class suited for the model based on morphology.

        :param morphology: str
            Morphology of the binary system: 'detached' or 'over-contact'.
        :return: type
            Fitting class (mcmc.DetachedLightCurveFit or
            mcmc.OvercontactLightCurveFit).
        """
        _cls = {
            "detached": mcmc.DetachedLightCurveFit,
            "over-contact": mcmc.OvercontactLightCurveFit,
            "overcontact": mcmc.OvercontactLightCurveFit,
        }
        return _cls[morphology]


class LCFitLeastSquares(LCFit):
    """Perform light curve fitting using the Least-Squares method.

    This class wraps least-squares optimization functionality for fitting
    light curves with support for bounds and constraints.

    :param morphology: str
        Morphology of the binary system ('detached' or 'over-contact').
    :param atmosphere_model: dict | None
        Atmosphere model parameters.
    :param limb_darkening_coefficients: dict | None
        Limb darkening coefficients.
    """

    def __init__(
        self,
        morphology: str,
        atmosphere_model: dict | None,
        limb_darkening_coefficients: dict | None,
    ) -> None:
        super().__init__(morphology, atmosphere_model, limb_darkening_coefficients)
        self.fit_method_instance = self.resolve_fit_cls(morphology)()
        self.fit_method_instance.atmosphere_model = atmosphere_model
        self.fit_method_instance.limb_darkening_coefficients = limb_darkening_coefficients

    def fit(
        self,
        x0: BinaryInitialParameters,
        data: dict[str, Any],
        **kwargs: Any,
    ) -> dict:
        """Perform Least-Squares optimization on the light curve fit.

        :param x0: BinaryInitialParameters
            Initial information about the model parameters including status
            (fixed, variable, constrained), bounds (prior distribution),
            and initial values.
        :param data: dict[str, Any]
            Observational data (light curves in multiple filters). Keys are
            filter names, values are LCData instances.
        :param kwargs: Any
            Additional arguments passed to the fitting method. See
            AnalyticsTask.fit kwargs for Least-Squares or
            least_squares.LightCurveFit.fit for further information.
        :return: dict
            Optimized model parameters in JSON format.
        """
        x0.validate_lc_parameters(morphology=self.morphology)
        self.result = self.fit_method_instance.fit(data=data, x0=x0, **kwargs)
        self.flat_result = self.fit_method_instance.flat_result
        logger.info("Fitting and processing of results finished successfully.")
        self.fit_summary()
        return self.result

    def fit_summary(
        self,
        path: str | None = None,
    ) -> None:
        """Produce detailed summary of the light curve fitting task.

        :param path: str | None
            Path to store the summary. If None, summary is not saved.
        :return: None
        """
        simple_lc_fit_summary(self, path)

    def resolve_fit_cls(self, morphology: str) -> type:
        """Return Least-Squares fitting class suited for the model based on morphology.

        :param morphology: str
            Morphology of the binary system: 'detached' or 'over-contact'.
        :return: type
            Fitting class (least_squares.DetachedLightCurveFit or
            least_squares.OvercontactLightCurveFit).
        """
        _cls = {
            "detached": least_squares.DetachedLightCurveFit,
            "over-contact": least_squares.OvercontactLightCurveFit,
            "overcontact": least_squares.OvercontactLightCurveFit,
        }
        return _cls[morphology]
