import re

from .. params.parameters import deflate_phenomena
from ... import const


def serialize_system_kwargs(**kwargs):
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
    Serialize `xn` input like kwargs to Star kwargs (truncate p__ or s__).

    :param component: str; `primary` or `secondary`
    :param kwargs: Dict;
    :return: Dict;
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
        gravity_darkening=kwargs.get(prefix('gravity_darkening'), 1.0),
        albedo=kwargs.get(prefix('albedo'), 1.0),
        metallicity=kwargs.get(prefix('metallicity'), 1.0),
        **{"mass": kwargs[prefix("mass")]} if kwargs.get(prefix("mass")) else {},
        **{"discretization_factor": kwargs[prefix("discretization_factor")]}
        if kwargs.get(prefix("discretization_factor")) else {},
        **{"spots": spots},
        **{"pulsations": pulsations},
    )


def serialize_primary_kwargs(**kwargs):
    return _serialize_star_kwargs(component='primary', **kwargs)


def serialize_secondary_kwargs(**kwargs):
    return _serialize_star_kwargs(component='secondary', **kwargs)
