from elisa import logger


class Animation(object):
    def __init__(self, instance):
        self._logger = logger.getLogger(name=self.__class__.__name__)
        self._self = instance
