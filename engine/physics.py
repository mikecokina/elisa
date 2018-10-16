import logging
from engine import utils

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : [%(levelname)s] : %(name)s : %(message)s')

class Physics:
    KWARGS = []

    def __init__(self, **kwargs):
        utils.is_property(kwargs)