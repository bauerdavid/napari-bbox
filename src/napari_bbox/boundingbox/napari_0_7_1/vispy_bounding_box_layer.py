# A copy of napari._vispy.layers.shapes
from ..napari_0_5_0.vispy_bounding_box_layer import VispyBoundingBoxLayer
from napari._vispy.layers.base import VispyBaseLayer
from .vispy_bounding_box_visual import BoundingBoxVisual


class VispyBoundingBoxLayer(VispyBoundingBoxLayer):
    def __init__(self, layer, font_info) -> None:
        node = BoundingBoxVisual(font_info=font_info)
        VispyBaseLayer.__init__(self, layer, node, font_info=font_info)

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


from napari._vispy.utils.visual import layer_to_visual

def register_layer_visual(layer_type):
    layer_to_visual[layer_type] = VispyBoundingBoxLayer
