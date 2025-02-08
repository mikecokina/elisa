import logging
from . import settings

settings.set_up_logging()


# noinspection PyPep8Naming
def getLogger(name, suppress=False):
    if settings.SUPPRESS_LOGGER is not None:
        suppress = settings.SUPPRESS_LOGGER

    return logging.getLogger(name=name) if not suppress else Logger(name)


# noinspection PyPep8Naming
def getPersistentLogger(name):
    return logging.getLogger(name=name)


class Logger(object):
    # noinspection PyUnusedLocal
    def __init__(self, name, *args, **kwargs):
        self.name = name

    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def warn(self, *args, **kwargs):
        pass
