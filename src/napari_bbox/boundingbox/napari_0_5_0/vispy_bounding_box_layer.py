# A copy of napari._vispy.layers.shapes
from ..napari_0_4_15.vispy_bounding_box_layer import VispyBoundingBoxLayer
import numpy as np

from napari._vispy.layers.base import VispyBaseLayer
from napari._vispy.utils.gl import BLENDING_MODES
from napari._vispy.utils.text import update_text
from .vispy_bounding_box_visual import BoundingBoxVisual
from napari.settings import get_settings
from napari.utils.events import disconnect_events

from ..._helper_functions import layer_ndisplay


class VispyBoundingBoxLayer(VispyBoundingBoxLayer):
    def __init__(self, layer) -> None:
        node = BoundingBoxVisual()
        VispyBaseLayer.__init__(self, layer, node)

        self.layer.events.edge_width.connect(self._on_data_change)
        self.layer.events.edge_color.connect(self._on_data_change)
        self.layer.events.face_color.connect(self._on_data_change)
        self.layer.text.events.connect(self._on_text_change)
        self.layer.events.highlight.connect(self._on_highlight_change)

        # TODO: move to overlays
        self.node.highlight_vertices.symbol = 'square'
        self.node.highlight_vertices.scaling = False

        self.reset()
        self._on_data_change()

    def _on_highlight_change(self, event=None):
        settings = get_settings()
        self.layer._highlight_width = settings.appearance.highlight.highlight_thickness
        self.layer._highlight_color = settings.appearance.highlight.highlight_color

        # Compute the vertices and faces of any bounding box outlines
        vertices, faces = self.layer._outline_bounding_boxes()

        ndisplay = layer_ndisplay(self.layer)
        if vertices is None or len(vertices) == 0 or len(faces) == 0:
            vertices = np.zeros((3, ndisplay))
            faces = np.array([[0, 1, 2]])

        self.node.bounding_box_highlights.set_data(
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

        width = settings.appearance.highlight.highlight_thickness

        if vertices is None or len(vertices) == 0:
            vertices = np.zeros((1, ndisplay))
            size = 0
        else:
            size = self.layer._vertex_size

        self.node.highlight_vertices.set_data(
            vertices,
            size=size,
            face_color=face_color,
            edge_color=edge_color,
            edge_width=width,
        )

        if pos is None or len(pos) == 0:
            pos = np.zeros((1, ndisplay))
            width = 0

        self.node.highlight_lines.set_data(
            pos=pos, color=edge_color, width=width
        )

    def _get_text_node(self):
        """Function to get the text node from the Compound visual"""
        return self.node.text

    def _on_blending_change(self, event=None):
        """Function to set the blending mode"""
        shapes_blending_kwargs = BLENDING_MODES[self.layer.blending]
        self.node.set_gl_state(**shapes_blending_kwargs)

        text_node = self._get_text_node()
        text_blending_kwargs = BLENDING_MODES[self.layer.text.blending]
        text_node.set_gl_state(**text_blending_kwargs)
        self.node.update()

from napari._vispy.utils.visual import layer_to_visual

def register_layer_visual(layer_type):
    layer_to_visual[layer_type] = VispyBoundingBoxLayer