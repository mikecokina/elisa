import re

from .. params.parameters import deflate_phenomena
from ... import const


def serialize_system_kwargs(**kwargs):
    """
    The function is used to extract the system-related parameters used in synthetic model functions during the fit.
    Those parameters are then returned as `system` component of the input JSON used to initialize the BinarySystem
    instance.

    :param kwargs: Dict[str, float]; model parameters in flat format
    :return: Dict[str, float]; `system` model parameters
    """
    return dict(
        argument_of_periastron=kwargs.get('system@argument_of_periastron', const.HALF_PI),
        gamma=kwargs.get('system@gamma', 0.0),
        period=kwargs["system@period"],
        eccentricity=kwargs.get('system@eccentricity', 0.0),
        primary_minimum_time=0.0,
        **{"inclination": kwargs["system@inclination"]} if kwargs.get("system@inclination") else {},
        **{"semi_major_axis": kwargs["system@semi_major_axis"]} if kwargs.get("system@semi_major_axis") else {},
        **{"mass_ratio": kwargs["system@mass_ratio"]} if kwargs.get("system@mass_ratio") else {},
        **{"asini": kwargs["system@asini"]} if kwargs.get("system@asini") else {},
        **{"additional_light": kwargs["system@additional_light"]} if kwargs.get("system@additional_light") else {},
        **{"phase_shift": kwargs["system@phase_shift"]} if kwargs.get("system@phase_shift") else {}
    )


def _serialize_star_kwargs(component, **kwargs):
    """
    General function for extraction of component-related parameters used in synthetic model functions during the fit.
    Compoenet related parameters are then returned as `primary` or `secondary` component of the input JSON used to
    initialize the BinarySystem instance.

    :param component: str; `primary` or `secondary`
    :param kwargs: Dict; model parameters in flat format
    :return: Dict; component-related model parameters
    """
    prefix = lambda prop: f'{component}@{prop}'
    params_tree = {"spots": dict(), "pulsations": dict()}

    for phenom in params_tree.keys():

        _ = {key: value for key, value in kwargs.items()
             if re.search(rf"^(?=.*\b{phenom[:-1]}\b)(?=.*\b{component}\b).*$", key)}
        _ = deflate_phenomena(_)
        params_tree[phenom] = [{key: val for key, val in value.items() if key not in ['label']} for value in _.values()]

    spots = params_tree['spots'] or dict()
    pulsations = params_tree['pulsations'] or dict()

    return dict(
        surface_potential=kwargs[prefix('surface_potential')],
        synchronicity=kwargs.get(prefix('synchronicity'), 1.0),
        t_eff=kwargs[prefix('t_eff')],
        **{"gravity_darkening": kwargs[prefix("gravity_darkening")]} if kwargs.get(prefix("gravity_darkening")) else {},
        **{"albedo": kwargs[prefix("albedo")]} if kwargs.get(prefix("albedo")) else {},
        **{"metallicity": kwargs[prefix("metallicity")]} if kwargs.get(prefix("metallicity")) else {},
        **{"mass": kwargs[prefix("mass")]} if kwargs.get(prefix("mass")) else {},
        **{"discretization_factor": kwargs[prefix("discretization_factor")]}
        if kwargs.get(prefix("discretization_factor")) else {},
        **{"spots": spots},
        **{"pulsations": pulsations},
    )


def serialize_primary_kwargs(**kwargs):
    """Parameter extractor for the primary component."""
    return _serialize_star_kwargs(component='primary', **kwargs)


def serialize_secondary_kwargs(**kwargs):
    """Parameter extractor for the secondary component."""
    return _serialize_star_kwargs(component='secondary', **kwargs)
