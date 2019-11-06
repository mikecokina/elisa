import logging
from elisa.conf import config


def getLogger(name, suppress=False):
    if config.SUPPRESS_LOGGER is not None:
        suppress = config.SUPPRESS_LOGGER

    return logging.getLogger(name=name) if not suppress else Logger(name)


class Logger(object):
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
