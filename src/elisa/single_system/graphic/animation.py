from elisa.logger import getLogger

logger = getLogger(__name__)


class Animation(object):
    def __init__(self, instance):
        self._self = instance
