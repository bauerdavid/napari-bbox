# A copy of napari.layers.shapes._shapes_key_bindings
import numpy as np
import napari
from ._bounding_box_mouse_bindings import (
    _move_active_element_under_cursor,
)
from napari.utils.action_manager import action_manager

from napari.utils.translations import trans
from ._bounding_box_constants import Box, Mode
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
    
@BoundingBoxLayer.bind_key('Shift')
def hold_to_lock_aspect_ratio(layer: BoundingBoxLayer):
    """Hold to lock aspect ratio when resizing a bounding box."""
    # on key press
    layer._fixed_aspect = True
    box = layer._selected_box
    if box is not None:
        size = box[Box.BOTTOM_RIGHT] - box[Box.TOP_LEFT]
        if not np.any(size == np.zeros(2)):
            layer._aspect_ratio = abs(size[1] / size[0])
        else:
            layer._aspect_ratio = 1
    else:
        layer._aspect_ratio = 1
    if layer._is_moving:
        assert layer._moving_coordinates is not None, layer
        _move_active_element_under_cursor(layer, layer._moving_coordinates)

    yield

    # on key release
    layer._fixed_aspect = False


@register_bounding_box_mode_action(trans._('Transform'))
def activate_transform_mode(layer):
    layer.mode = Mode.TRANSFORM


@register_bounding_box_mode_action(trans._('Pan/zoom'))
def activate_bb_pan_zoom_mode(layer: BoundingBoxLayer):
    """Activate pan and zoom mode."""
    layer.mode = Mode.PAN_ZOOM


@register_bounding_box_mode_action(trans._('Add bounding boxes'))
def activate_add_bb_mode(layer: BoundingBoxLayer):
    """Activate add bounding box tool."""
    layer.mode = Mode.ADD_BOUNDING_BOX


@register_bounding_box_mode_action(trans._('Select bounding boxes'))
def activate_select_mode(layer: BoundingBoxLayer):
    """Activate bounding box selection tool."""
    layer.mode = Mode.SELECT

bounding_box_fun_to_mode = [
    (activate_bb_pan_zoom_mode, Mode.PAN_ZOOM),
    (activate_transform_mode, Mode.TRANSFORM),
    (activate_add_bb_mode, Mode.ADD_BOUNDING_BOX),
    (activate_select_mode, Mode.SELECT)
]


@register_bounding_box_action(trans._('Copy any selected bounding boxes'))
def copy_selected_bounding_boxes(layer: BoundingBoxLayer):
    """Copy any selected bounding boxes."""
    if layer._mode == Mode.SELECT:
        layer._copy_data()


@register_bounding_box_action(trans._('Paste any copied bounding_boxes'))
def paste_bounding_box(layer: BoundingBoxLayer):
    """Paste any copied bounding_boxes."""
    if layer._mode == Mode.SELECT:
        layer._paste_data()


@register_bounding_box_action(trans._('Select all bounding boxes in the current view slice'))
def select_all_bounding_boxes(layer: BoundingBoxLayer):
    """Select all bounding boxes in the current view slice."""
    if layer._mode == Mode.SELECT:
        layer.selected_data = set(np.nonzero(layer._data_view._displayed)[0])
        layer._set_highlight()


@register_bounding_box_action(trans._('Delete any selected bounding boxes'))
def delete_selected_bounding_boxes(layer: BoundingBoxLayer):
    """."""

    if not layer._is_creating:
        layer.remove_selected()


@register_bounding_box_action(
    trans._(
        'Finish any drawing.'
    ),
)
def finish_drawing_bounding_box(layer: BoundingBoxLayer):
    """Finish any drawing."""
    layer._finish_drawing()

action_manager.bind_shortcut("napari:activate_select_mode", "2")
action_manager.bind_shortcut("napari:activate_pan_zoom_mode", "3")
action_manager.bind_shortcut("napari:activate_add_bb_mode", "4")