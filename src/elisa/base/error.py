from jsonschema import ValidationError


class YouHaveNoIdeaError(ValidationError):
    pass


class ElisaError(Exception):
    pass


class MaxIterationError(ElisaError):
    pass


class InitialParamsError(ElisaError):
    pass


class AtmosphereError(ElisaError):
    pass


class TemperatureError(ElisaError):
    pass


class LimbDarkeningError(ElisaError):
    pass


class MetallicityError(ElisaError):
    pass


class GravityError(ElisaError):
    pass


class MorphologyError(ElisaError):
    pass


class SpotError(ElisaError):
    pass


class SolutionBubbleException(ElisaError):
    def __init__(self, *args, **kwargs):
        self.solution = kwargs.get('solution')
        super().__init__(*args)
