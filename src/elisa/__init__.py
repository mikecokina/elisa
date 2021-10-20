__version__ = '0.5'

import json
import os.path as op
from . conf.settings import settings
from . binary_system.system import BinarySystem
from . single_system.system import SingleSystem
from . base.star import Star
from . observer.observer import Observer
from . analytics.dataset.base import LCData, RVData
from . import units


def get_default_binary_definition():
    _path = op.join(op.dirname(__file__), "data", "default_binary_system.json")
    with open(_path, "r") as f:
        return json.loads(f.read())


def _default_binary():
    return BinarySystem.from_json(data=get_default_binary_definition())


def _default_observer():
    return Observer(passband=["Generic.Bessell.U", "Generic.Bessell.V", "Generic.Bessell.R"], system=_default_binary())


def _bolometric_default_observer():
    return Observer(passband=["bolometric"], system=_default_binary())


default_binary = _default_binary
default_observer = _default_observer
bolometric_default_observer = _bolometric_default_observer

# prepare units as u for simpler import
u = units
