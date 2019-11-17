class ElisaError(Exception):
    pass


class MaxIterationError(ElisaError):
    pass


class InitialParamsError(ElisaError, ValueError):
    pass


class AtmosphereError(ElisaError, ValueError):
    pass


class TemperatureError(ElisaError, ValueError):
    pass


class LimbDarkeningError(ElisaError, ValueError):
    pass


class MetallicityError(ElisaError, ValueError):
    pass


class GravityError(ElisaError, ValueError):
    pass


class MorphologyError(ElisaError):
    pass


class HitSolutionBubble(ElisaError):
    def __init__(self, *args, **kwargs):
        self.solution = kwargs.get('solution')
        super().__init__(*args)
