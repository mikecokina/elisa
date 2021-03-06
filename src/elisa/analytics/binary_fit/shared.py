import numpy as np

from typing import Iterable, Dict
from scipy import interpolate
from abc import ABCMeta, abstractmethod

from .. params import parameters
from .. params.parameters import BinaryInitialParameters
from .. tools.utils import (
    radialcurves_mean_error,
    lightcurves_mean_error,
    time_layer_resolver
)
from .. models import lc as lc_model
from .. models import rv as rv_model
from ... utils import is_empty
from ... import settings
from ... observer.observer import Observer
from ... observer.utils import normalize_light_curve
from ... binary_system.system import BinarySystem
from ... binary_system.curves.community import RadialVelocitySystem

from elisa.const import PI

from ... logger import getLogger

logger = getLogger("analytics.binary_fit.shared")


class AbstractFit(metaclass=ABCMeta):
    MEAN_ERROR_FN = None

    __slots__ = ['fixed', 'constrained', 'fitable', 'normalized', 'observer', 'x_data', 'y_data', 'num_of_points'
                 'y_err', 'x_data_reduced', 'x_data_reducer', 'initial_vector', 'normalization', 'flat_result']

    def set_up(self, x0: BinaryInitialParameters, data: Dict, passband: Iterable = None, **kwargs):
        setattr(self, 'fixed', x0.get_fixed(jsonify=False))
        setattr(self, 'constrained', x0.get_constrained(jsonify=False))
        setattr(self, 'fitable', x0.get_fitable(jsonify=False))
        setattr(self, 'normalization', x0.get_normalization_map())

        observer = Observer(passband='bolometric' if passband is None else passband, system=None)
        setattr(self, 'observer', observer)
        setattr(self.observer, 'system_cls', kwargs.get('observer_system_cls'))

        setattr(self, 'x_data', {key: val.x_data for key, val in data.items()})
        setattr(self, 'y_data', {key: val.y_data for key, val in data.items()})
        setattr(self, 'num_of_points', {key: np.shape(val.y_data)[0] for key, val in data.items()})

        err = {key: abs(self.__class__.MEAN_ERROR_FN(val)) if data[key].y_err is None else data[key].y_err
               for key, val in self.y_data.items()}
        setattr(self, 'y_err', err)

        x_data_reduced, x_data_reducer = parameters.xs_reducer({key: val.x_data for key, val in data.items()})
        setattr(self, 'x_data_reduced', x_data_reduced)
        setattr(self, 'x_data_reducer', x_data_reducer)

        setattr(self, 'initial_vector', [val.value for val in self.fitable.values()])
        setattr(self, 'flat_result', dict())

        setattr(self, 'error_penalization', self.calculate_error_penalization())

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @staticmethod
    def eval_constrained_results(result_dict: Dict[str, Dict], constraints: Dict[str, 'InitialParameter']):
        """
        Function adds constrained parameters into the resulting dictionary.

        :param constraints: Dict; contains constrained parameters
        :param result_dict: Dict; {'name': {'value': value, 'unit': unit, ...}
        :return: Dict; {'name': {'value': value, 'unit': unit, ...}
        """
        if is_empty(constraints):
            return result_dict

        res_val_dict = {key: val['value'] for key, val in result_dict.items()}
        constrained_values = parameters.constraints_evaluator(res_val_dict, constraints)
        result_dict.update(
            {
                key: {
                    'value': val,
                    'constraint': constraints[key].constraint,
                    'unit': constraints[key].to_dict()['unit']
                } for key, val in constrained_values.items()
            })
        return result_dict

    def calculate_error_penalization(self):
        """
        Calculates constant component to the likelihood function derived from the errors

        :return: np.float;
        """
        return np.sum([np.sum(np.log(2 * PI * np.power(value, 2))) for value in self.y_err.values()])


class AbstractRVFit(AbstractFit):
    MEAN_ERROR_FN = radialcurves_mean_error

    __slots__ = ['fixed', 'constrained', 'fitable', 'normalized', 'observer', "discretization",
                 'interp_treshold', 'data', 'y_err', 'num_of_points', 'x_data_reduced', 'x_data_reducer',
                 'initial_vector', 'normalization', 'x_data', 'y_data']

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    def set_up(self, x0: BinaryInitialParameters, data: Dict, **kwargs):
        super().set_up(x0, data, passband=None, **kwargs)

    def coefficient_of_determination(self, model_parameters, data, discretization, interp_treshold):
        """
        Function returns R^2 for given model parameters and observed data.

        :param model_parameters: Dict; serialized form
        :param data: DataSet; observational data
        :param discretization: float;
        :param interp_treshold: int;
        :return: float;
        """
        self.set_up(parameters.BinaryInitialParameters(**model_parameters), data,
                    observer_system_cls=RadialVelocitySystem)
        r_squared_args = self.x_data_reduced, self.y_data, self.x_data_reducer, self.observer.system_cls
        flat_result = parameters.deserialize_result(model_parameters)
        r_dict = {key: value['value'] for key, value in flat_result.items()}

        logger.info("Evaluating light curve for calculation of R^2.")
        r_squared_result = rv_r_squared(rv_model.central_rv_synthetic, *r_squared_args, **r_dict)
        logger.info("Calculation of R^2 finished.")
        return r_squared_result


class AbstractLCFit(AbstractFit):
    MEAN_ERROR_FN = lightcurves_mean_error

    __slots__ = ['fixed', 'constrained', 'fitable', 'normalized', 'observer', "discretization",
                 'interp_treshold', 'data', 'y_err', 'num_of_points', 'x_data_reduced', 'x_data_reducer',
                 'initial_vector', 'normalization', 'x_data', 'y_data']

    def set_up(self, x0: BinaryInitialParameters, data: Dict, passband: Iterable = None, **kwargs):
        super().set_up(x0, data, passband, observer_system_cls=kwargs.get('observer_system_cls'))
        setattr(self, 'discretization', kwargs.pop('discretization'))
        setattr(self, 'interp_treshold', kwargs.pop('interp_treshold'))
        fit_xs = self.generate_sample_phases(kwargs.pop('samples'))
        setattr(self, 'fit_xs', fit_xs)
        self.normalize_data(kind='average')

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    def normalize_data(self, kind='global_maximum', top_fraction_to_average=0.1):
        y_data, y_err = normalize_light_curve(self.y_data, self.y_err, kind, top_fraction_to_average)
        setattr(self, 'y_data', y_data)
        setattr(self, 'y_err', y_err)

    def generate_sample_phases(self, samples):
        kwargs = parameters.prepare_properties_set(self.initial_vector, self.fitable.keys(), self.constrained,
                                                   self.fixed)
        phases, kwargs = time_layer_resolver(self.x_data_reduced, pop=False, **kwargs)
        if np.shape(phases)[0] < self.interp_treshold:
            return None

        if samples is 'uniform':
            diff = 1.0 / self.interp_treshold
            return np.linspace(0.0 - diff, 1.0 + diff, num=self.interp_treshold + 2)
        elif samples is 'adaptive':
            logger.info('Generating equidistant samples along the light curve using adaptive sampling method')
            return self.adaptive_sampling()
        elif isinstance(samples, (list, np.ndarray)):
            return np.sort(samples)
        else:
            raise ValueError(f'Parameter `samples` has to be either string with values `uniform` or `adaptive` or '
                             f'array of phases in (0, 1) interval')

    def adaptive_sampling(self):
        """
        Function generates sampling equidistantly along the curve defined by the initial vector.
        :return: np.array;
        """
        n = 3 * settings.MAX_CURVE_DATA_POINTS
        diff = 1.0 / n
        x = np.linspace(0.0 - diff, 1.0 + diff, num=n)

        kwargs = parameters.prepare_properties_set(self.initial_vector, self.fitable.keys(), self.constrained,
                                                   self.fixed)
        observer = Observer(passband='bolometric', system=None)
        setattr(observer, 'system_cls', getattr(self.observer, 'system_cls'))
        try:
            synthetic = lc_model.synthetic_binary(x, self.discretization, observer, **kwargs)
            synthetic, _ = normalize_light_curve(synthetic, kind='average')
        except Exception as e:
            raise RuntimeError('Your initial parameters are invalid and phase sampling could not be generated.')

        curve = np.column_stack((x, synthetic['bolometric']))
        lengths = np.sqrt(np.sum(np.diff(curve, axis=0) ** 2, axis=1))
        crv_lengths = np.cumsum(np.concatenate(([0, ], lengths)))
        segments = np.linspace(0, crv_lengths[-1], num=self.interp_treshold)

        return np.interp(segments, crv_lengths, x)

    def coefficient_of_determination(self, model_parameters, data, discretization, interp_treshold):
        """
        Function returns R^2 for given model parameters and observed data.

        :param model_parameters: Dict; serialized form
        :param data: DataSet; observational data
        :param discretization: float;
        :param interp_treshold: int;
        :return: float;
        """
        self.set_up(x0=parameters.BinaryInitialParameters(**model_parameters), data=data, passband=data.keys(),
                    discretization=discretization, interp_treshold=interp_treshold, samples='uniform',
                    observer_system_cls=BinarySystem)

        r_squared_args = (self.x_data_reduced, self.y_data, self.observer.passband, discretization,
                          self.x_data_reducer, 1.0 / self.interp_treshold, self.interp_treshold,
                          self.observer.system_cls)
        flat_result = parameters.deserialize_result(model_parameters)
        r_dict = {key: value['value'] for key, value in flat_result.items()}
        logger.info("Evaluating light curve for calculation of R^2.")
        r_squared_result = lc_r_squared(lc_model.synthetic_binary, *r_squared_args, **r_dict)
        logger.info("Calculation of R^2 finished.")
        return r_squared_result


def lc_r_squared(synthetic, *args, **x):
    """
    Compute R^2 (coefficient of determination).

    :param synthetic: callable; synthetic method
    :param args: Tuple;
    :**args**:
        * **x_data_reduced** * -- numpy.array; phases
        * **y_data** * -- numpy.array; supplied fluxes (lets say fluxes from observation) normalized to max value
        * **period** * -- float;
        * **passband** * -- Union[str, List[str]];
        * **discretization** * -- flaot;
        * **diff** * -- flaot; auxiliary parameter for interpolation defining spacing between evaluation phases if
                               interpolation is active
        * **interp_treshold** * - int; number of points
    :param x: Dict;
    :** x options**: kwargs of current parameters to compute binary system
    :return: float;
    """
    x_data_reduced, y_data, passband, discretization, x_data_reducer, diff, interp_treshold, cls = args

    x_data_reduced, kwargs = time_layer_resolver(x_data_reduced, pop=False, **x)
    fit_xs = np.linspace(np.min(x_data_reduced) - diff, np.max(x_data_reduced) + diff, num=interp_treshold + 2) \
        if np.shape(x_data_reduced)[0] > interp_treshold else x_data_reduced

    observer = Observer(passband=passband, system=None)
    observer._system_cls = cls
    synthetic = synthetic(fit_xs, discretization, observer, **x)

    if np.shape(fit_xs) != np.shape(x_data_reduced):
        synthetic = {
            fltr: interpolate.interp1d(fit_xs, curve, kind='cubic')(x_data_reduced)
            for fltr, curve in synthetic.items()
        }
    synthetic = {band: synthetic[band][x_data_reducer[band]] for band in synthetic}
    synthetic, _ = normalize_light_curve(synthetic, kind='average')

    return r_squared(synthetic, y_data)


def rv_r_squared(synthetic, *args, **x):
    """
    Compute R^2 (coefficient of determination).

    :param synthetic: callable; synthetic method
    :param args: Tuple;
    :**args**:
        * **x_data_reduced** * -- numpy.array; phases
        * **y_data** * -- numpy.array; supplied fluxes (lets say fluxes from observation) normalized to max value
        * **period** * -- float;
        * **on_normalized** * -- bool;
    :param x: Dict;synth_phases
    :** x options**: kwargs of current parameters to compute radial velocities curve
    :return: float;
    """
    x_data_reduced, y_data, x_data_reducer, cls = args

    observer = Observer(passband='bolometric', system=None)
    observer._system_cls = cls
    synthetic = synthetic(x_data_reduced, observer, **x)
    synthetic = {comp: synthetic[comp][x_data_reducer[comp]] for comp in synthetic}

    return r_squared(synthetic, y_data)


def r_squared(synthetic, observed):
    variability = np.sum([np.sum(np.power(observed[item] - np.mean(observed[item]), 2)) for item in observed])
    residual = np.sum([np.sum(np.power(synthetic[item] - observed[item], 2)) for item in observed])

    return 1 - (residual / variability)


def extend_observations_to_desired_interval(start_phase, stop_phase, x_data, y_data, y_err):
    """
    Extending observations to desired phase interval.

    :param start_phase: float;
    :param stop_phase: float;
    :param x_data: Dict;
    :param y_data: Dict;
    :param y_err: Dict;
    :return: Tuple;
    """
    for item, curve in x_data.items():
        phases_extended = np.concatenate((x_data[item] - 1.0, x_data[item], x_data[item] + 1.0))
        phases_extended_filter = np.logical_and(start_phase < phases_extended, phases_extended < stop_phase)
        x_data[item] = phases_extended[phases_extended_filter]

        y_data[item] = np.tile(y_data[item], 3)[phases_extended_filter]
        if y_err[item] is not None:
            y_err[item] = np.tile(y_err[item], 3)[phases_extended_filter]

    return x_data, y_data, y_err


def check_for_boundary_surface_potentials(result_dict, morphology=None):
    """
    Function checks if surface potential are within errors below critical potentials (which would break BinarySystem
    initialization). If surface potential are within errors they are snapped to critical values.

    :param result_dict: Dict; flat dict of fit results
    :param morphology: str; expected morphology
    :return: Dict; corrected flat dict of fit results
    """
    if "primary@surface_potential" not in result_dict.keys() or "secondary@surface_potential" not in result_dict.keys():
        return result_dict

    for component in settings.BINARY_COUNTERPARTS:
        pot = result_dict[f"{component}@surface_potential"]
        if "fixed" not in pot.keys() or "value" not in pot.keys():
            continue

        sigma = pot["value"] - pot["confidence_interval"]["min"] if "confidence_interval" in pot.keys() else 0.001

        synchronicity = result_dict[f"{component}@synchronicity"]["value"] \
            if f"{component}@synchronicity" in result_dict.keys() else 1.0

        mass_ratio = result_dict["system@mass_ratio"]["value"] \
            if "system@mass_ratio" in result_dict.keys() \
            else result_dict["secondary@mass"]["value"]/result_dict["primary@mass"]["value"]

        periastron_distance = 1 - result_dict["system@eccentricity"]["value"]

        l1 = \
            BinarySystem.critical_potential_static(
                component=component,
                components_distance=periastron_distance,
                mass_ratio=mass_ratio,
                synchronicity=synchronicity
            )

        # if resulting potential is too close critical potentials (within errors), it will snap potential to critical
        # to avoid problems
        if 5 * sigma >= l1 - pot["value"] >= 0.0:
            pot["value"] = l1 + 1e-5 * sigma

        # test for over-contact overflow trough L2 point
        l2 = BinarySystem.libration_potentials_static(periastron_distance, mass_ratio)[2]
        if 5 * sigma >= l2 - pot["value"] >= 0.0:
            pot["value"] = l2 - 1e-5 * sigma

    if morphology == 'over-contact':
        result_dict['secondary@surface_potential']['value'] = result_dict['primary@surface_potential']['value']

    return result_dict


def eval_constraint_in_dict(input_dict):
    """
    Evaluates constraints defined bt the user in fit json.

    :param input_dict: Dict;
    :return: Dict; dict with constraints that have values assigned
    """
    input_dict1 = parameters.deserialize_result(input_dict)
    result_dict = {key: val for key, val in input_dict1.items() if 'fixed' in val}

    reduced_dict = {key: val['value'] for key, val in result_dict.items()}

    b_parameters = BinaryInitialParameters(**input_dict)
    constraints = b_parameters.get_constrained(jsonify=False)

    constrained_values = parameters.constraints_evaluator(reduced_dict, constraints)
    result_dict.update(
        {
            key: {
                'value': constrained_values[key],
                'constraint': constraints[key].constraint,
                'unit': constraints[key].unit
            } for key, val in constrained_values.items()
        })

    result_dict = parameters.serialize_result(result_dict)
    return result_dict
