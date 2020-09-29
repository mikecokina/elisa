import json
import re
from copy import deepcopy

import numpy as np

from collections import Iterable
from typing import Dict
from jsonschema import ValidationError

from .. params import conf
from .. params import bonds
from .. params.bonds import (
    ALLOWED_CONSTRAINT_METHODS,
    ALLOWED_CONSTRAINT_CHARS,
    TRANSFORM_TO_METHODS)
from .. params.transform import (
    BinaryInitialProperties,
    StarInitialProperties,
    SpotInitialProperties,
    PulsationModeInitialProperties
)
from ... import units as u
from ... import utils
from ... import settings
from ... base.error import InitialParamsError
from ... utils import is_empty


def deflate_phenomena(flatten):
    result = dict()
    for phenom_uid, phenom_meta in flatten.items():
        _, _, label, param = str(phenom_uid).split(conf.PARAM_PARSER)
        if label not in result:
            result[label] = dict(label=label)
        result[label][param] = phenom_meta
    return result


def deserialize_result(result_dict: Dict) -> Dict:
    """
    Function converts dictionary of results in user format to flat format.

    :param result_dict:
    :return:
    """
    data = {}

    if 'r_quared' in result_dict:
        data['r_squared'] = result_dict['r_squared']

    for system_slot in BinaryInitialParameters.__slots__:

        if system_slot in result_dict['system']:
            system_prop = result_dict['system'][system_slot]
            data.update({f'system@{system_slot}': system_prop})

        if system_slot in ['primary', 'secondary'] and system_slot in result_dict:
            component_prop = result_dict[system_slot]

            for component_slot in StarInitialParameters.__slots__:
                if component_slot not in ['spots', 'pulsations'] and component_slot in result_dict[system_slot]:
                    data.update({f'{system_slot}@{component_slot}': component_prop[component_slot]})

                elif component_slot in ['spots', 'pulsations'] and component_slot in result_dict[system_slot]:
                    for phenom in component_prop[component_slot]:
                        for phenom_key, phenom_value in phenom.items():
                            if phenom_key not in ['label']:
                                data.update({
                                    f'{system_slot}@{component_slot[:-1]}@{phenom["label"]}@{phenom_key}': phenom_value
                                })
    return data


def serialize_result(result_dict: Dict) -> Dict:
    """
    Function converts dictionary of results of parameter labels back to user format.

    :param result_dict: Dict; result dict
    :return: Dict; user formatted dict
    """

    ret_dict = dict()

    for param, value in result_dict.items():
        if param == 'r_squared':
            ret_dict['r_squared'] = result_dict['r_squared']
            continue

        identificators = param.split(conf.PARAM_PARSER)

        if identificators[0] not in ret_dict:
            ret_dict[identificators[0]] = dict()

        if str(identificators[1]) in conf.COMPOSITE_FLAT_PARAMS:
            identificators[1] = f'{identificators[1]}s'

        if identificators[1] not in ret_dict[identificators[0]]:
            ret_dict[identificators[0]][identificators[1]] = dict()

        if str(identificators[0]).startswith('system'):
            ret_dict['system'].update({
                identificators[1]: value
            })
            continue

        if str(identificators[1]) not in ['spots', 'pulsations']:
            ret_dict[identificators[0]].update({
                identificators[1]: value
            })
        else:
            if str(identificators[2]) not in ret_dict[identificators[0]][identificators[1]]:
                ret_dict[identificators[0]][identificators[1]][identificators[2]] = dict()

            ret_dict[identificators[0]][identificators[1]][identificators[2]].update({
                "label": identificators[2],
                identificators[3]: value
            })

    # renormalize spots and pulastions if presented
    for component in settings.BINARY_COUNTERPARTS:
        if component in ret_dict:
            for phenomena in ['spots', 'pulsations']:
                if phenomena in ret_dict[component]:
                    ret_dict[component][phenomena] = [value for value in ret_dict[component][phenomena].values()]

    return ret_dict


def check_initial_param_validity(x0: Dict[str, 'InitialParameter'], all_fit_params, mandatory_fit_params):
    """
    Checking if initial parameters system and composite (spots and pulsations) are containing all necessary values and
    no invalid ones.

    :param x0: Dict[str, 'InitialParameter']; dictionary of initial parameters
    :param all_fit_params: list; list of all valid system parameters (spot and pulsation parameters excluded)
    :param mandatory_fit_params: list; list of mandatory system parameters (spot and pulsation parameters excluded)
    :return:
    """
    param_names = {key: val.value for key, val in x0.items() if not re.search(r"|".join(conf.COMPOSITE_FLAT_PARAMS), key)}
    utils.invalid_param_checker(param_names, all_fit_params, 'x0')
    utils.check_missing_params(mandatory_fit_params, param_names, 'x0')

    # checking validity of spot parameters and pulsations parameters
    params_map = {'spot': conf.SPOTS_PARAMETERS, 'pulsation': conf.PULSATIONS_PARAMETERS}
    for phenom in params_map:

        _ = {key: value for key, value in x0.items() if re.search(phenom, key)}
        _ = deflate_phenomena(_)

        for label, meta in _.items():
            meta = meta.copy()
            meta.pop('label', None)

            utils.invalid_param_checker(meta.keys(), params_map[phenom], label)
            utils.check_missing_params(params_map[phenom], meta, label)


def xs_reducer(xs):
    """
    Convert phases `xs` to single list and inverse map related to given passband (in case of light curves)
    or component (in case of radial velocities).

    :param xs: Dict[str, numpy.array]; phases defined for each passband or ceomponent::

        {<passband>: <phases>} for light curves
        {<component>: <phases>} for radial velocities

    :return: Tuple; (numpy.array, Dict[str, List[int]]);
    """
    # this most likely cannot work corretly in python < 3.6, since dicts are not ordered in lower python versions
    x = np.hstack(list(xs.values())).flatten()
    y = np.arange(len(x)).tolist()
    reverse_dict = dict()
    for xs_key, phases in xs.items():
        reverse_dict[xs_key] = y[:len(phases)]
        del (y[:len(phases)])

    xs_reduced, inverse = np.unique(x, return_inverse=True)
    reverse = {band: inverse[indices] for band, indices in reverse_dict.items()}
    return xs_reduced, reverse


def renormalize_value(val, _min, _max):
    """
    Renormalize value `val` to value from interval specific for given parameter defined my `_min` and `_max`. Inverse
    function to `normalize_value`.

    :param val: float;
    :param _min: float;
    :param _max: float;
    :return: float;
    """
    return (val * (_max - _min)) + _min


def normalize_value(val, _min, _max):
    """
    Normalize value `val` to value from interval (0, 1) based on `_min` and `_max`.

    :param val: float;
    :param _min: float;
    :param _max: float;
    :return: float;
    """
    return (val - _min) / (_max - _min)


def vector_renormalizer(vector, properties, normalization):
    """
    Renormalize values from `x` to their native form.

    :param vector: List[float]; iterable of normalized parameter values
    :param properties: Iterable[str]; related parameter names from `x`
    :param normalization: Dict[str, Tuple[float, float]]; normalization map
    :return: List[float];
    """
    return [renormalize_value(value, *normalization[prop]) for value, prop in zip(vector, properties)]


def vector_normalizer(vector, properties, normalization):
    """
    Normalize values from `x` to value between (0, 1).

    :param vector: List[float]; iterable of normalized parameter values
    :param properties: Iterable[str]; related parameter names from `x`
    :param normalization: Dict[str, Tuple[float, float]]; normalization map
    :return: List[float];
    """
    return [normalize_value(value, *normalization[prop]) for value, prop in zip(vector, properties)]


def prepare_properties_set(xn, properties, constrained, fixed):
    """
    This will prepare final kwargs for synthetic model evaluation.

    :param xn: numpy.array; initial vector
    :param properties: list; variable labels
    :param constrained: dict;
    :param fixed: dict;
    :return: Dict[str, float];
    """
    kwargs = dict(zip(properties, xn))
    kwargs.update(constraints_evaluator(kwargs, constrained))
    fixed = {key: val.value if isinstance(val, InitialParameter) else val for key, val in fixed.items()}
    kwargs.update(fixed)
    return kwargs


def constraints_evaluator(substitution: Dict, constrained: Dict) -> Dict:
    """
    Substitute variables in constraint with values and evaluate to number.

    :param substitution: Dict[str, Union[InitialParameter, float]]; non-fixed values
                                                                   (xn vector in dict form {label: param})
    :param constrained: Dict[str, Union[InitialParameter, float]]; values estimated as constraintes
                                                                   in form {label: InitialParameter}
    :return: Dict[str, float];
    """
    if is_empty(constrained):
        return dict()

    if isinstance(list(substitution.values())[-1], InitialParameter):
        substitution = {key: val.value for key, val in substitution.items()}

    if isinstance(list(constrained.values())[-1], InitialParameter):
        constrained = {key: val.constraint for key, val in constrained.items()}
    
    numpy_methods = [f'bonds.{method}' for method in TRANSFORM_TO_METHODS]
    allowed_methods = bonds.ALLOWED_CONSTRAINT_METHODS
    constrained = constrained.copy()

    subst = {key: utils.str_repalce(val, substitution.keys(), substitution.values())
             for key, val in constrained.items()}
    numpy_callable = {key: utils.str_repalce(val, allowed_methods, numpy_methods) for key, val in subst.items()}
    try:
        evaluated = {key:  eval(val) for key, val in numpy_callable.items()}
    except Exception as e:
        raise InitialParamsError(f'Invalid syntax or value in constraint, {str(e)}.')
    return evaluated


class ParameterMeta(object):
    def __init__(self, **kwargs):
        self.unit = kwargs.get("unit")
        self.param = kwargs.get("param")
        self.property = self.param
        self.fixed = kwargs.get("fixed")
        self.constraint = kwargs.get("constraint")
        self.value = kwargs.get("value")
        self.min = kwargs.get("min")
        self.max = kwargs.get("max")

    def to_dict(self):
        return dict(
            **{
                "value": self.value,
                "param": self.param,
                "min": self.min,
                "max": self.max,
                "unit": str(self.unit) if self.unit is not None else None,
                "fixed": self.fixed,
                "constraint": self.constraint
            }
        )


class InitialParameter(object):
    DEFAULT = {
        "param": None,
        "value": None,
        "fixed": None,
        "constraint": None,
        "min": None,
        "max": None,
        "unit": None
    }

    def __init__(self, transform_cls, **kwargs):
        self.unit = kwargs.get("unit")
        self.param = kwargs.get("param")
        self.property = self.param
        self.fixed = kwargs.get("fixed")
        self.constraint = kwargs.get("constraint")
        self.value = None
        self.min = None
        self.max = None

        # units transformaton
        self.unit = u.Unit(self.unit) if self.unit is not None else self.unit
        if self.unit is not None and self.unit is not u.dimensionless_unscaled and self.constraint is None:
            kwargs.update({
                "value": kwargs.get("value") * self.unit,
                "min": kwargs.get("min") * self.unit,
                "max": kwargs.get("max") * self.unit
            })

        if self.constraint is None:
            self.value = transform_cls.transform_input(**{self.param: kwargs.get("value")})[self.param]
            self.min = transform_cls.transform_input(**{self.param: kwargs.get("min")})[self.param]
            self.max = transform_cls.transform_input(**{self.param: kwargs.get("max")})[self.param]

        if self.fixed:
            self.min, self.max = None, None

        self.unit = conf.DEFAULT_FLOAT_UNITS[self.property]

    def copy(self):
        return deepcopy(self)

    def to_dict(self):
        return dict(
            **{
                "value": self.value,
                "param": self.param,
                "min": self.min,
                "max": self.max,
                "unit": str(self.unit) if self.unit is not None else None
            },
            **{"fixed": self.fixed} if self.fixed is not None else {},
            **{"constraint": self.constraint} if self.constraint is not None else {},
        )

    def __repr__(self):
        return json.dumps({
            "param": self.param,
            "value": self.value,
            "fixed": self.fixed,
            "constraint": self.constraint,
            "min": self.min,
            "max": self.max,
            "unit": str(self.unit) if self.unit is not None else None
        }, indent=4)

    __str__ = __repr__


class InitialParameters(object):
    TRANSFORM_PROPERTIES_CLS = None
    DEFAULT_NORMALIZATION = None

    def validity_check(self):
        for slot in self.__slots__:
            if not hasattr(self, str(slot)):
                continue
            prop = getattr(self, str(slot))

            if not isinstance(prop, InitialParameter):
                continue

            if prop.constraint is None and not prop.fixed:
                if not (prop.min <= prop.value <= prop.max):
                    raise InitialParamsError(f'Initial parameters in parameter `{prop.param}` are not valid. '
                                             f'Invalid bounds: {prop.min} <= {prop.value} <= {prop.max}')
            if prop.fixed is not None and prop.constraint is not None:
                raise InitialParamsError(f'It is not allowed for `{prop.param}` to contain '
                                         f'`fixed` and `constraint` parameter.')

    def init_parameter(self, parameter: str, items: Dict) -> InitialParameter:
        _kwarg = InitialParameter.DEFAULT.copy()
        _kwarg.update(dict(param=parameter, **items))
        _kwarg.update(dict(min=items.get("min", self.DEFAULT_NORMALIZATION[parameter][0]),
                           max=items.get("max", self.DEFAULT_NORMALIZATION[parameter][1])
                           )
                      )
        return InitialParameter(transform_cls=self.__class__.TRANSFORM_PROPERTIES_CLS, **_kwarg)


class SpotInitialParameters(InitialParameters):
    __slots__ = ["longitude", "latitude", "angular_radius", "temperature_factor", "label"]

    TRANSFORM_PROPERTIES_CLS = SpotInitialProperties
    DEFAULT_NORMALIZATION = conf.DEFAULT_NORMALIZATION_SPOT

    def __init__(self, **kwargs):
        self.label = f'spot{conf.PARAM_PARSER}{kwargs.pop("label")}'
        for parameter, items in kwargs.items():
            value = self.init_parameter(parameter, items)
            setattr(self, parameter, value)
        self.validity_check()


class PulsationInitialParameters(InitialParameters):
    __slots__ = ["l", "m", "amplitude", "frequency", "start_phase", "mode_axis_theta", "mode_axis_phi", "label"]

    TRANSFORM_PROPERTIES_CLS = PulsationModeInitialProperties
    DEFAULT_NORMALIZATION = conf.DEFAULT_NORMALIZATION_PULSATION

    def __init__(self, **kwargs):
        self.label = f'pulsation{conf.PARAM_PARSER}{kwargs.pop("label")}'
        self.mode_axis_phi = 0
        self.mode_axis_theta = 0

        for parameter, items in kwargs.items():
            value = self.init_parameter(parameter, items)
            setattr(self, parameter, value)
        self.validity_check()


class StarInitialParameters(InitialParameters):
    __slots__ = ["spots", "pulsations", "t_eff", "metallicity", "surface_potential",
                 "albedo", "gravity_darkening", "synchronicity", "mass"]

    TRANSFORM_PROPERTIES_CLS = StarInitialProperties
    DEFAULT_NORMALIZATION = conf.DEFAULT_NORMALIZATION_STAR

    def __init__(self, **kwargs):
        self.label = None
        spots = kwargs.pop('spots', None)
        pulsations = kwargs.pop('pulsations', None)

        for parameter, items in kwargs.items():
            value = self.init_parameter(parameter, items)
            setattr(self, parameter, value)

        if not is_empty(spots):
            spots = [SpotInitialParameters(**spot) for spot in spots]
            self.spots = spots

        if not is_empty(pulsations):
            pulsations = [PulsationInitialParameters(**pulsation) for pulsation in pulsations]
            self.pulsations = pulsations

        self.validity_check()


class BinaryInitialParameters(InitialParameters):
    __slots__ = ["primary", "secondary", "eccentricity", "argument_of_periastron",
                 "inclination", "gamma", "period", "mass_ratio", "asini", "semi_major_axis",
                 "additional_light", "phase_shift", "primary_minimum_time"]

    TRANSFORM_PROPERTIES_CLS = BinaryInitialProperties
    DEFAULT_NORMALIZATION = conf.DEFAULT_NORMALIZATION_SYSTEM

    def __init__(self, **kwargs):
        self._primary = kwargs.pop('primary', None)
        self._secondary = kwargs.pop('secondary', None)
        system = kwargs.pop('system')

        for parameter, items in system.items():
            value = self.init_parameter(parameter, items)
            setattr(self, parameter, value)

        for component in settings.BINARY_COUNTERPARTS:
            props = getattr(self, f'_{component}')
            if is_empty(props):
                continue
            setattr(self, component, StarInitialParameters(**props))
            setattr(getattr(self, component), 'label', component)

        self.validity_check()
        self.unique_labels_validation()
        self._data = self.serialize_flat_set()
        self.validate_data()

    @property
    def data(self):
        return self._data

    def __getitem__(self, item):
        iterable = True
        if not isinstance(item, Iterable):
            iterable = True
            item = [item]
        data = [self.data[_item] for _item in item]
        return data if iterable else data[-1]

    def to_flat_json(self):
        return [{key: val.to_dict()} for key, val in self.data.items()]

    def unique_labels_validation(self):
        def _test(_what):
            for component in settings.BINARY_COUNTERPARTS:
                if hasattr(self, component):
                    inst = getattr(self, component)
                    if hasattr(inst, _what):
                        phenom = getattr(inst, _what)
                        _all = [_phenom.label for _phenom in phenom]
                        if len(_all) != len(set(_all)):
                            raise InitialParamsError(f'It is not allowed to have multiple {_what} with same label.')
        _test('spots')
        _test('pulsations')

    def serialize_flat_set(self):
        data = {}

        for system_slot in self.__slots__:

            if not hasattr(self, str(system_slot)):
                continue
            system_prop = getattr(self, str(system_slot))
            if isinstance(system_prop, InitialParameter):
                data.update({f'system{conf.PARAM_PARSER}{system_slot}': system_prop})
            elif isinstance(system_prop, StarInitialParameters):

                for component_slot in system_prop.__slots__:

                    if not hasattr(system_prop, str(component_slot)):
                        continue
                    component_prop = getattr(system_prop, str(component_slot))
                    if isinstance(component_prop, InitialParameter):

                        data.update({f'{getattr(system_prop, "label")}'
                                     f'{conf.PARAM_PARSER}{component_slot}': component_prop})

                    elif isinstance(component_prop, list):
                        for phenomena_prop in component_prop:
                            if isinstance(phenomena_prop, (PulsationInitialParameters, SpotInitialParameters)):

                                for phenomena_slot in phenomena_prop.__slots__:
                                    if not hasattr(phenomena_prop, str(phenomena_slot)):
                                        continue
                                    prop = getattr(phenomena_prop, str(phenomena_slot))

                                    if isinstance(prop, InitialParameter):
                                        data.update({f'{getattr(system_prop,"label")}'
                                                     f'{conf.PARAM_PARSER}{getattr(phenomena_prop,"label")}'
                                                     f'{conf.PARAM_PARSER}{phenomena_slot}': prop})

        return data

    def constraint_validator(self):
        """
        Validate constraints. Make sure there is no harmful code.
        Allowed methods used in constraints::

            'arcsin', 'arccos', 'arctan', 'log', 'sin', 'cos', 'tan', 'exp', 'degrees', 'radians'

        Allowed characters used in constraints::

            '(', ')', '+', '-', '*', '/', '.'

        :raise: elisa.base.error.ValidationError;
        """
        constrained = self.get_constrained(jsonify=False)
        substitution = self.get_substitution_dict()
        try:
            subst = {param:  utils.str_repalce(
                             utils.str_repalce(constraint.constraint, substitution.keys(), substitution.values()),
                             ALLOWED_CONSTRAINT_METHODS, [''] * len(ALLOWED_CONSTRAINT_METHODS)
                         ).replace(' ', '')
                     for param, constraint in constrained.items()}
        except KeyError:
            msg = f'It seems your constraint contain variable that cannot be resolved. ' \
                f'Make sure that linked constraint variable is not fixed or check for typos in variable name in ' \
                f'constraint expression.'
            raise ValidationError(msg)

        for key, val in subst.items():
            if not np.all(np.isin(list(val), ALLOWED_CONSTRAINT_CHARS)):
                msg = f'Constraint {key} contain forbidden characters. Allowed: {ALLOWED_CONSTRAINT_CHARS}'
                raise ValidationError(msg)

    def validate_data(self):
        # validate that at least one parameter is not fixed
        if len(self.get_fitable(jsonify=False)) == 0:
            raise ValidationError('There are no variable parameters to fit.')

        # constraint validation
        self.constraint_validator()

    def adjust_overcontact_potential(self, morphology):
        if self.is_overcontact(morphology):
            update_surfcae_potential = self.primary.surface_potential.copy()
            update_surfcae_potential.fixed = None
            update_surfcae_potential.constraint = 'primary@surface_potential'
            self.secondary.surface_potential = update_surfcae_potential
            self.data['secondary@surface_potential'] = update_surfcae_potential

    def validate_lc_parameters(self, morphology):
        """
        Validate parameters for light curve fitting.
        """
        mandatory_fit_params = ['system@eccentricity', 'system@argument_of_periastron',
                                'system@period', 'system@inclination', 'system@period'] + \
                               [f'{component}@{param}'
                                for param in ['t_eff', 'surface_potential', 'gravity_darkening', 'albedo']
                                for component in settings.BINARY_COUNTERPARTS]

        optional_fit_params = ['system@semi_major_axis', 'system@primary_minimum_time', 'system@phase_shift',
                               'system@asini', 'system@mass_ratio', 'system@additional_light'] + \
                              [f'{component}@{param}'
                               for param in ['mass', 'synchronicity', 'metallicity', 'spots', 'pulsations']
                               for component in settings.BINARY_COUNTERPARTS]

        all_fit_params = mandatory_fit_params + optional_fit_params
        utils.check_missing_kwargs(mandatory_fit_params, self.data, instance_of=self.__class__)
        check_initial_param_validity(self.data, all_fit_params, mandatory_fit_params)

        is_oc = self.is_overcontact(morphology)
        are_same = self.data['primary@surface_potential'].value == self.data['secondary@surface_potential'].value

        is_fixed_omega_1 = self.data['primary@surface_potential'].fixed or False
        is_fixed_omega_2 = self.data['secondary@surface_potential'].fixed or False

        any_fixed = is_fixed_omega_1 | is_fixed_omega_2
        all_fixed = is_fixed_omega_1 & is_fixed_omega_2

        if is_oc and all_fixed and are_same:
            return
        if is_oc and all_fixed and not are_same:
            msg = 'Different potential in over-contact morphology with all fixed (pontetial) value are not allowed.'
            raise ValidationError(msg)
        if is_oc and any_fixed:
            msg = 'Just one fixed potential in over-contact morphology is not allowed.'
            raise ValidationError(msg)

        # adjust constraint for secondary potential
        self.adjust_overcontact_potential(morphology)

    def validate_rv_parameters(self):
        """
        Validate parameters for radial velocities curve fitting.
        """
        mandatory_fit_params = ['system@eccentricity', 'system@argument_of_periastron', 'system@gamma']
        optional_fit_params = ['system@period', 'system@primary_minimum_time', 'primary@mass', 'secondary@mass',
                               'system@inclination', 'system@asini', 'system@mass_ratio']
        all_fit_params = mandatory_fit_params + optional_fit_params
        utils.check_missing_kwargs(mandatory_fit_params, self.data, instance_of=self.__class__)
        check_initial_param_validity(self.data, all_fit_params, mandatory_fit_params)

        # validate consistency of parameters (system has to be definable)
        has_primary_minimum_time, has_period = 'system@primary_minimum_time' in self.data, 'system@period' in self.data
        if has_primary_minimum_time:
            if not (has_primary_minimum_time and has_period):
                raise ValidationError("Input requires both, period and primary minimum time.")
        else:
            if not has_period:
                raise ValidationError("Input requires at least period.")

    def _get_kind_of(self, kind_of, jsonify=False):
        if jsonify:
            return {key: val.to_dict() for key, val in self.data.items() if getattr(val, kind_of)}
        return {key: val for key, val in self.data.items() if getattr(val, kind_of)}

    def get_fixed(self, jsonify=False):
        """
        Transform native form to `fixed` parameters only: {key: InitialParam, ...}
        :return: Dict[str, InitialParam];
        """
        return self._get_kind_of(kind_of="fixed", jsonify=jsonify)

    def get_constrained(self, jsonify=False):
        """
        Transform native form to `constrained` parameters only: {key: InitialParam, ...}
        :return: Dict[str, InitialParam];
        """
        return self._get_kind_of(kind_of="constraint", jsonify=jsonify)

    def get_fitable(self, jsonify=False):
        """
        Transform native form to `fitable` parameters only: {key: InitialParam, ...}
        :return: Dict[str, InitialParam];
        """
        if jsonify:
            return {key: val.to_dict() for key, val in self.data.items() if not val.constraint and not val.fixed}
        return {key: val for key, val in self.data.items() if not val.constraint and not val.fixed}

    def get_substitution_dict(self):
        return {key: val.value for key, val in self.data.items() if not val.constraint}

    def get_normalization_map(self):
        """
        Return normalization boundaries for given parmeter.
        :return: Tuple[float, float];
        """
        return {key: (val.min, val.max) for key, val in self.data.items()}

    @staticmethod
    def is_overcontact(morphology):
        """
        Is string equal to `over-contact`?
        """
        return morphology in ['over-contact']
