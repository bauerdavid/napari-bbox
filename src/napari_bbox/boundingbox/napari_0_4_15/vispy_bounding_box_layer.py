# A copy of napari._vispy.layers.shapes
import numpy as np
from napari._vispy.layers.base import VispyBaseLayer
from napari._vispy.utils.text import update_text
from napari.utils.events import disconnect_events
from napari.settings import get_settings
from vispy.scene import Compound, Mesh, Line, Markers, Text

from ..._utils import NAPARI_VERSION
from ..._helper_functions import layer_ndisplay


class VispyBoundingBoxLayer(VispyBaseLayer):
    def __init__(self, layer):
        # Create a compound visual with the following four subvisuals:
        # Markers: corresponding to the vertices of the interaction box or the
        # bounding boxes that are used for highlights.
        # Lines: The lines of the interaction box used for highlights.
        # Mesh: The mesh of the outlines for each bounding box used for highlights.
        # Mesh: The actual meshes of the bounding box faces and edges
        node = Compound([Mesh(), Mesh(), Line(), Markers(), Text()])

        super().__init__(layer, node)

        self.layer.events.edge_width.connect(self._on_data_change)
        self.layer.events.edge_color.connect(self._on_data_change)
        self.layer.events.face_color.connect(self._on_data_change)
        self.layer.text.events.connect(self._on_text_change)
        self.layer.events.highlight.connect(self._on_highlight_change)

        # TODO: move to overlays
        self.node._subvisuals[3].symbol = 'square'
        self.node._subvisuals[3].scaling = False

        self.reset()
        self._on_data_change()
        self._on_highlight_change()

    def _on_data_change(self, event=None):
        faces = self.layer._data_view._mesh.displayed_triangles
        colors = self.layer._data_view._mesh.displayed_triangles_colors
        vertices = self.layer._data_view._mesh.vertices

        # Note that the indices of the vertices need to be reversed to
        # go from numpy style to xyz
        if vertices is not None:
            vertices = vertices[:, ::-1]

        if len(vertices) == 0 or len(faces) == 0:
            vertices = np.zeros((3, layer_ndisplay(self.layer)))
            faces = np.array([[0, 1, 2]])
            colors = np.array([[0, 0, 0, 0]])

        if (
            len(self.layer.data)
            and layer_ndisplay(self.layer) == 3
            and self.layer.ndim == 2
        ):
            vertices = np.pad(vertices, ((0, 0), (0, 1)), mode='constant')

        self.node._subvisuals[0].set_data(
            vertices=vertices, faces=faces, face_colors=colors
        )

        # Call to update order of translation values with new dims:
        self._on_matrix_change()
        self._update_text(update_node=False)
        self.node.update()

    def _on_highlight_change(self, event=None):
        settings = get_settings()
        self.layer._highlight_width = settings.appearance.highlight_thickness

        # Compute the vertices and faces of any bounding box outlines
        vertices, faces = self.layer._outline_bounding_boxes()
        ndisplay = layer_ndisplay(self.layer)
        if vertices is None or len(vertices) == 0 or len(faces) == 0:
            vertices = np.zeros((3, ndisplay))
            faces = np.array([[0, 1, 2]])

        self.node._subvisuals[1].set_data(
            vertices=vertices,
            faces=faces,
            color=self.layer._highlight_color,
        )

        # Compute the location and properties of the vertices and box that
        # need to get rendered
        (
            vertices,
            face_color,
            edge_color,
            pos,
            width,
        ) = self.layer._compute_vertices_and_box()

        width = settings.appearance.highlight_thickness

        if vertices is None or len(vertices) == 0:
            vertices = np.zeros((1, ndisplay))
            size = 0
        else:
            size = self.layer._vertex_size

        self.node._subvisuals[3].set_data(
            vertices,
            size=size,
            face_color=face_color,
            edge_color=edge_color,
            edge_width=width,
            symbol='square',
            **({"scaling": False} if NAPARI_VERSION <= "0.11.0" else {}),
        )

        if pos is None or len(pos) == 0:
            pos = np.zeros((1, ndisplay))
            width = 0

        self.node._subvisuals[2].set_data(
            pos=pos, color=edge_color, width=width
        )

    def _update_text(self, *, update_node=True):
        """Function to update the text node properties

        Parameters
        ----------
        update_node : bool
            If true, update the node after setting the properties
        """
        update_text(node=self._get_text_node(), layer=self.layer)
        if update_node:
            self.node.update()

    def _on_text_change(self, event=None):
        if event is not None:
            if event.type == 'blending':
                self._on_blending_change(event)
                return
            if event.type == 'values':
                return
        self._update_text()

    def _get_text_node(self):
        """Function to get the text node from the Compound visual"""
        text_node = self.node._subvisuals[-1]
        return text_node

    def _on_blending_change(self, event=None):
        """Function to set the blending mode"""
        self.node.set_gl_state(self.layer.blending)

        text_node = self._get_text_node()
        text_node.set_gl_state(self.layer.text.blending)
        self.node.update()

    def reset(self):
        super().reset()
        self._on_highlight_change()
        self._on_blending_change()

    def close(self):
        """Vispy visual is closing."""
        disconnect_events(self.layer.text.events, self)
        super().close()

from napari._vispy.utils.visual import layer_to_visual

def register_layer_visual(layer_type):
    layer_to_visual[layer_type] = VispyBoundingBoxLayer