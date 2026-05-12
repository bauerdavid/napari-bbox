# A copy of napari.layers.shapes._bounding_box_key_bindings
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from app_model.types import KeyCode

from ..napari_0_4_18._bounding_box_constants import Box, Mode
from ._bounding_box_mouse_bindings import (
    _move_active_element_under_cursor,
)

from napari.utils.action_manager import action_manager

from .bounding_boxes import BoundingBoxLayer
from napari.layers.utils.layer_utils import (
    register_layer_action,
    register_layer_attr_action,
)
from napari.utils.notifications import show_info
from napari.utils.translations import trans

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

@BoundingBoxLayer.bind_key(KeyCode.Shift, overwrite=True)
def hold_to_lock_aspect_ratio(layer: BoundingBoxLayer) -> Generator[None, None, None]:
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


def register_bounding_box_action(
    description: str, repeatable: bool = False
) -> Callable[[Callable], Callable]:
    return register_layer_action(BoundingBoxLayer, description, repeatable)


def register_bounding_box_mode_action(
    description: str,
) -> Callable[[Callable], Callable]:
    return register_layer_attr_action(BoundingBoxLayer, description, 'mode')


@register_bounding_box_mode_action(trans._('Transform'))
def activate_transform_mode(layer: BoundingBoxLayer) -> None:
    layer.mode = Mode.TRANSFORM


@register_bounding_box_mode_action(trans._('Move camera'))
def activate_bb_pan_zoom_mode(layer: BoundingBoxLayer) -> None:
    layer.mode = Mode.PAN_ZOOM


@register_bounding_box_mode_action(trans._('Add bounding boxes'))
def activate_add_bb_mode(layer: BoundingBoxLayer) -> None:
    """Activate add bounding box tool."""
    layer.mode = Mode.ADD_BOUNDING_BOX


@register_bounding_box_mode_action(trans._('Select bounding boxes'))
def activate_bb_select_mode(layer: BoundingBoxLayer) -> None:
    """Activate bounding box selection tool."""
    layer.mode = Mode.SELECT

bounding_box_fun_to_mode = [
    (activate_bb_pan_zoom_mode, Mode.PAN_ZOOM),
    (activate_transform_mode, Mode.TRANSFORM),
    (activate_add_bb_mode, Mode.ADD_BOUNDING_BOX),
    (activate_bb_select_mode, Mode.SELECT),
]

@register_bounding_box_action(trans._('Copy any selected bounding boxes'))
def copy_selected_bounding_boxes(layer: BoundingBoxLayer) -> None:
    """Copy any selected bounding boxes."""
    if layer._mode == Mode.SELECT:
        layer._copy_data()


@register_bounding_box_action(trans._('Paste any copied bounding boxes'))
def paste_bounding_box(layer: BoundingBoxLayer) -> None:
    """Paste any copied bounding boxes."""
    if layer._mode == Mode.SELECT:
        layer._paste_data()


@register_bounding_box_action(
    trans._('Select/Deselect all bounding boxes in the current view slice')
)
def select_bounding_box_in_slice(layer: BoundingBoxLayer) -> None:
    """Select/Deselect all bounding boxes in the current view slice."""
    new_selected = set(np.nonzero(layer._data_view._displayed)[0])

    if new_selected.issubset(layer.selected_data):
        # If all visible bounding boxes are already selected, deselect them
        layer.selected_data = layer.selected_data - new_selected
    else:
        # If not all visible bounding boxes are selected, select them and replace
        # any other selection.
        layer.selected_data = new_selected
        if new_selected:
            show_info(
                trans._(
                    'Selected {n_new} bounding boxes in this slice.',
                    n_new=len(new_selected),
                    deferred=True,
                )
            )
    layer._set_highlight()


@register_bounding_box_action(trans._('Delete any selected bounding boxes'))
def delete_selected_bounding_boxes(layer: BoundingBoxLayer) -> None:
    """."""

    if not layer._is_creating:
        layer.remove_selected()

@register_bounding_box_action(
    trans._(
        'Finish any drawing.'
    ),
)
def finish_drawing_bounding_box(layer: BoundingBoxLayer) -> None:
    """Finish any drawing."""
    layer._finish_drawing()

# [NOTE] unbid first
action_manager.unbind_shortcut("napari:activate_bb_select_mode")
action_manager.unbind_shortcut("napari:activate_bb_pan_zoom_mode")
action_manager.unbind_shortcut("napari:activate_add_bb_mode")

action_manager.bind_shortcut("napari:activate_bb_select_mode", "2")
action_manager.bind_shortcut("napari:activate_bb_pan_zoom_mode", "4")
action_manager.bind_shortcut("napari:activate_add_bb_mode", "3")