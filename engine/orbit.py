class Orbit(object):

    KWARGS = ['period', 'inclination', 'eccentricity', 'periastron']

    def __init__(self, name=None, **kwargs):
        pass

    @classmethod
    def is_property(cls, kwargs):
        is_not = ['`{}`'.format(k) for k in kwargs if k not in cls.KWARGS]
        if is_not:
            raise AttributeError('Arguments {} are not valid {} properties.'.format(', '.join(is_not), cls.__name__))

