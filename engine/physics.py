import logging
from engine import utils


class Physics:
    KWARGS = []

    def __init__(self, **kwargs):
        utils.is_property(kwargs)