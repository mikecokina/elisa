import numpy as np

from elisa.base.container import (
    StarContainer,
    PositionContainer
)
from elisa.single_system.surface import (
    mesh,
    faces,
)
from elisa.logger import getLogger

logger = getLogger("single-system-container-module")


class SystemContainer(PositionContainer):
    def __init__(self, star: StarContainer, **properties):
        self.star = star

        # placeholder (set in loop below)
        self.inclination = np.nan

        for key, val in properties.items():
            setattr(self, key, val)

    def build(self, *args, **kwargs):
        pass

    def build_mesh(self):
        return mesh.build_mesh(self)

    def build_faces(self):
        return faces.build_faces(self)

