__version__ = '0.4'

import json
import os.path as op
from . conf.settings import settings
from . binary_system.system import BinarySystem
from . import units


def _default_binary():
    _path = op.join(op.dirname(__file__), "data", "default_binary_system.json")
    with open(_path, "r") as f:
        model_definition = json.loads(f.read())
    return BinarySystem.from_json(data=model_definition)


default_binary = _default_binary

# prepare units as u for simpler import
u = units
