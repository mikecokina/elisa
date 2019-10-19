from elisa import logger
from elisa.base.container import StarContainer
from elisa.binary_system.surface import mesh
from elisa.conf import config
from elisa.binary_system import utils as bsutils

config.set_up_logging()
__logger__ = logger.getLogger("binary-system-container-module")


class OrbitalPositionContainer(object):
    def __init__(self, primary: StarContainer, secondary, position, **properties):
        self.primary = primary
        self.secondary = secondary
        self.position = position

        for key, val in properties.items():
            setattr(self, key, val)

    def build(self, components_distance, component="all", **kwargs):
        pass

    def build_mesh(self, components_distance, component="all", **kwargs):
        """
        Build points of surface for primary or/and secondary component. Mesh is evaluated with spots.
        :param self: BinarySystem; instance
        :param component: str or empty
        :param components_distance: float
        :return:
        """
        components = bsutils.component_to_list(component)

        for component in components:
            start_container = getattr(self, component)
            # in case of spoted surface, symmetry is not used
            a, b, c, d = mesh.mesh_over_contact(self, component=component, symmetry_output=True) \
                if getattr(self, 'morphology') == 'over-contact' \
                else mesh.mesh_detached(self, components_distance, component, symmetry_output=True)

            start_container.points = a
            start_container.point_symmetry_vector = b
            start_container.base_symmetry_points_number = c
            start_container.inverse_point_symmetry_matrix = d

        mesh.add_spots_to_mesh(self, components_distance, component="all")
