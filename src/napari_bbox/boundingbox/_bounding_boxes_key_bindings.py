# A copy of napari.layers.shapes._shapes_key_bindings
import napari
from napari.utils.translations import trans
from napari.utils.action_manager import action_manager
from ._bounding_box_constants import Mode
from .bounding_boxes import BoundingBoxLayer
from packaging import version

if version.parse(napari.__version__) <= version.parse("0.4.16"):
    from napari.layers.utils.layer_utils import register_layer_action
    def register_bounding_box_action(description):
        return register_layer_action(BoundingBoxLayer, description)

    def register_bounding_box_mode_action(description, *args):
        return register_bounding_box_action(description)
else:
    from napari.layers.utils.layer_utils import register_layer_action, register_layer_attr_action
    def register_bounding_box_action(description, repeatable: bool = False):
        return register_layer_action(BoundingBoxLayer, description, repeatable)

    def register_bounding_box_mode_action(description):
        return register_layer_attr_action(BoundingBoxLayer, description, "mode")


@register_bounding_box_mode_action(trans._('Select bounding boxes'))
def activate_bb_select_mode(layer):
    """Activate bounding box selection tool."""
    layer.mode = Mode.SELECT

@register_bounding_box_mode_action(trans._('Pan/Zoom'))
def activate_bb_pan_zoom_mode(layer):
    """Activate pan and zoom mode."""
    layer.mode = Mode.PAN_ZOOM


@register_bounding_box_mode_action(trans._('Add bounding box'))
def activate_add_bb_mode(layer):
    """Activate add bounding box tool."""
    layer.mode = Mode.ADD_BOUNDING_BOX



@BoundingBoxLayer.bind_key('Space')
def hold_to_pan_zoom(layer: BoundingBoxLayer):
    """Hold to pan and zoom in the viewer."""
    if layer._mode != Mode.PAN_ZOOM:
        # on key press
        prev_mode = layer.mode
        layer.mode = Mode.PAN_ZOOM

        yield

        # on key release
        layer.mode = prev_mode

@BoundingBoxLayer.bind_key("Delete")
def delete_bbox(layer: BoundingBoxLayer):
    layer.remove_selected()

action_manager.bind_shortcut("napari:activate_bb_select_mode", "2")
action_manager.bind_shortcut("napari:activate_bb_pan_zoom_mode", "3")
action_manager.bind_shortcut("napari:activate_add_bb_mode", "4")