# A copy of napari.layers.shapes._shapes_mouse_bindings
from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

import numpy as np
from psygnal.containers import Selection

from napari.layers.base._base_constants import ActionType

from ._bounding_box_constants import Box, Mode

from napari.layers.shapes._shapes_models import (
    Ellipse,
    Line,
    Path,
    Polygon,
    Rectangle,
)
from napari.layers.shapes._shapes_utils import point_to_lines
from napari.settings import get_settings

if TYPE_CHECKING:
    from collections.abc import Generator

    import numpy.typing as npt
    from vispy.app.canvas import MouseEvent

    from napari._vispy.mouse_event import NapariMouseEvent
    from .bounding_boxes import BoundingBoxLayer


def highlight(layer: BoundingBoxLayer, event: MouseEvent) -> None:
    """Render highlights of bounding boxes.

    Highlight hovered bounding boxes, including boundaries, vertices, interaction boxes, and drag
    selection box when appropriate.

    Parameters
    ----------
    layer : BoundingBoxLayer
        Napari-bbox bounding box layer
    event : MouseEvent
        A proxy read only wrapper around a vispy mouse event. Though not used here it is passed as argument by the
        bounding box layer mouse move callbacks.

    Returns
    -------
    None
    """
    layer._set_highlight()


def select(layer: BoundingBoxLayer, event: MouseEvent) -> Generator[None, None, None]:
    """Select bounding boxes or vertices either in select or direct select mode.

    Once selected bounding boxes can be moved or resized, and vertices can be moved
    depending on the mode. Holding shift when resizing a bounding box will preserve
    the aspect ratio.

    Parameters
    ----------
    layer : BoudingBoxLayer
        Napari-bbox bounding box layer
    event : MouseEvent
        A proxy read only wrapper around a vispy mouse event.
    """
    shift = 'Shift' in event.modifiers
    # on press
    value = layer.get_value(event.position, world=True)
    layer._moving_value = copy(value)
    bounding_box_under_cursoe, vertex_under_cursor = value
    if vertex_under_cursor is None or layer.mode == Mode.SELECT:
        if shift and bounding_box_under_cursoe is not None:
            if bounding_box_under_cursoe in layer.selected_data:
                layer.selected_data.remove(bounding_box_under_cursoe)
            else:
                if len(layer.selected_data):
                    # one or more bounding boxes already selected
                    layer.selected_data.add(bounding_box_under_cursoe)
                else:
                    # first bounding box being selected
                    layer.selected_data = {bounding_box_under_cursoe}
        elif bounding_box_under_cursoe is not None:
            if bounding_box_under_cursoe not in layer.selected_data:
                layer.selected_data = {bounding_box_under_cursoe}
        else:
            layer.selected_data = set()
    layer._set_highlight()

    # we don't update the thumbnail unless a bounding box has been moved
    update_thumbnail = False

    # Set _drag_start value here to prevent an offset when mouse_move happens
    # https://github.com/napari/napari/pull/4999
    _set_drag_start(layer, layer.world_to_data(event.position))
    yield

    # on move
    while event.type == 'mouse_move':
        coordinates = layer.world_to_data(event.position)
        # ToDo: Need to pass moving_coordinates to allow fixed aspect ratio
        # keybinding to work, this should be dropped
        layer._moving_coordinates = coordinates
        # Drag any selected bounding boxes
        if len(layer.selected_data) == 0:
            _drag_selection_box(layer, coordinates)
        else:
            _move_active_element_under_cursor(layer, coordinates) # [NOTE] new adddition - before used _move

        # if a bounding box is being moved, update the thumbnail
        if layer._is_moving:
            update_thumbnail = True
        yield

    # only emit data once dragging has finished
    if layer._is_moving:
        vertex_indices = tuple(
            tuple(
                vertex_index
                for vertex_index, coord in enumerate(layer.data[i])
            )
            for i in layer.selected_data
        )
        layer.events.data(
            value=layer.data,
            action=ActionType.CHANGED,
            data_indices=tuple(layer.selected_data),
            vertex_indices=vertex_indices,
        )

    # on release
    shift = 'Shift' in event.modifiers
    if not layer._is_moving and not layer._is_selecting and not shift:
        if bounding_box_under_cursoe is not None:
            layer.selected_data = {bounding_box_under_cursoe}
        else:
            layer.selected_data = set()
    elif layer._is_selecting:
        layer.selected_data = layer._data_view.bounding_boxes_in_box(layer._drag_box)
        layer._is_selecting = False
        layer._set_highlight()

    layer._is_moving = False
    layer._drag_start = None
    layer._drag_box = None
    layer._fixed_vertex = None
    layer._moving_value = (None, None)
    layer._set_highlight()

    if update_thumbnail:
        layer._update_thumbnail()


def add_bounding_box(
    layer: BoundingBoxLayer, event: NapariMouseEvent
) -> Generator[None, None, None]:
    """Add a bounding box to the bounding box layer.

    Parameters
    ----------
    layer : BoudingBoxLayer
        Napari-bbox bounding box layer
    event : NapariMouseEvent
        A proxy read only wrapper around a vispy mouse event.
    """
    size = layer._vertex_size * layer.scale_factor / 4
    coordinates = layer.world_to_data(event.position)
    corner = np.array(coordinates)

    # [NOTE] 2ⁿ vertices (all corners of an N-D box)
    # 2D → 4 corners / 3D → 8 corners
    # want to calculate all offset based vertex size and scale factor
    # [TODO] check [CLANKER] optimisation → generate binary representation of a number as a list of integers
    # 2D → [0,0], [1,0], [0,1], [1,1]
    # 3D → [0,0,0], [1,0,0], [0,1,0], [1,1,0], ... [1,1,1]
    ndim = layer.ndim
    offsets = np.array([
        [(i >> d) & 1 for d in range(ndim)]
        for i in range(2**ndim)
    ], dtype=float)

    # scale offsets
    offsets *= size
    data = corner + offsets

    # (1, N, D)
    data = data[np.newaxis]

    yield from _add_bounding_box(
        layer, event, data=data
    )


def _add_bounding_box(
    layer: BoundingBoxLayer, event: NapariMouseEvent, data: npt.NDArray
) -> Generator[None, None, None]:
    """Helper function for adding a bounding box.

    Parameters
    ----------
    layer : BoundingBoxLayer
        Napari-bbox bounding box layer
    event : NapariMouseEvent
        A proxy read only wrapper around a vispy mouse event.
    data : np.ndarray
        Array containing the initial datapoints of the bounding box in image data space.
    """
    # on press
    layer._is_creating = True
    # reset layer._aspect_ratio for a new bounding box
    layer._aspect_ratio = 1
    # Start drawing rectangle / ellipse / line
    layer.add(data, gui=True)
    layer.selected_data = {layer.nbounding_boxes - 1}
    layer._value = (layer.nbounding_boxes - 1, 4)
    layer._moving_value = copy(layer._value)
    layer.refresh()
    yield

    # [NOTE] bounding box plugin specific
    data = layer.data[layer.nbounding_boxes - 1]
    # on move
    const_set = False
    while event.type == 'mouse_move':
        # Drag any selected bounding boxes
        coordinates = layer.world_to_data(event.position)
        layer._moving_coordinates = coordinates
        _move_active_element_under_cursor(layer, coordinates) # [NOTE] new adddition - before used _move
        # [NOTE] bounding box plugin specific
        min = data.min(0)
        max = data.max(0)
        size = max-min
        visible_size = size[layer._slice_input.displayed]
        max[layer._slice_input.displayed] = np.nan
        min[layer._slice_input.displayed] = np.nan
        if layer.size_mode == "average":
            data[:] = np.where(data == max, coordinates+visible_size.mean()/2*layer.size_multiplier, data)
            data[:] = np.where(data == min, coordinates-visible_size.mean()/2*layer.size_multiplier, data)
        elif not const_set and layer.size_mode == "constant":
            data[:] = np.where(data == max, np.asarray(coordinates) + layer.size_constant/2, data)
            data[:] = np.where(data == min, np.asarray(coordinates) - layer.size_constant/2, data)
            const_set = True
        yield

    # on release
    layer._finish_drawing()

# [NOTE] for completeness renaming this but not used in plugin
def finish_drawing_bounding_box(layer: BoundingBoxLayer, event: MouseEvent) -> None:
    """Finish drawing of bounding box.

    Calls the finish drawing method of the bounding boxes layer which resets all the properties used for bounding box drawing
    and deletes the bounding box if the number of vertices do not meet the threshold of 3.

    Parameters
    ----------
    layer : BoundingBoxLayer
        Napari bounding boxes layer
    event : MouseEvent
        A proxy read only wrapper around a vispy mouse event. Not used here, but passed as argument due to being a
        double click callback of the bounding boxes layer.
    """
    layer._finish_drawing()


def _drag_selection_box(layer: BoundingBoxLayer, coordinates: tuple[float, ...]) -> None:
    """Drag a selection box.

    Parameters
    ----------
    layer : BoundingBoxLayer
        Napari-bbox bounding box layer.
    coordinates : Tuple[float, ...]
        The current position of the cursor during the mouse move event in image data space.
    """
    # If something selected return
    if len(layer.selected_data) > 0:
        return

    coord = [coordinates[i] for i in layer._slice_input.displayed]

    # Create or extend a selection box
    layer._is_selecting = True
    if layer._drag_start is None:
        layer._drag_start = coord
    layer._drag_box = np.array([layer._drag_start, coord])
    layer._set_highlight()

# [NOTE] maybe rephrase the description there 
def _set_drag_start(
    layer: BoundingBoxLayer, coordinates: tuple[float, ...]
) -> list[float]:
    """Indicate where in data space a drag event started.

    Sets the coordinates relative to the center of the napari internal bounding box of a bounding box and returns the position
    of where a drag event of a bounding box started.

    Parameters
    ----------
    layer : BoundingBoxLayer
        Napari-bbox bounding box layer.
    coordinates : Tuple[float, ...]
        The position in image data space where dragging started.

    Returns
    -------
    coord: List[float, ...]
        The coordinates of where a bounding box drag event started.
    """
    coord = [coordinates[i] for i in layer._slice_input.displayed]
    if layer._drag_start is None and len(layer.selected_data) > 0:
        center = layer._selected_box[Box.CENTER]
        layer._drag_start = coord - center
    return coord


def _move_selected_layer(
    layer: BoundingBoxLayer, coordinates: tuple[float, ...], vertex: int | None
) -> None:
    if layer._mode == Mode.SELECT and not layer._is_moving:
        vertex_indices = tuple(
            tuple(
                vertex_index
                for vertex_index, coord in enumerate(layer.data[i])
            )
            for i in layer.selected_data
        )
        layer.events.data(
            value=layer.data,
            action=ActionType.CHANGING,
            data_indices=tuple(layer.selected_data),
            vertex_indices=vertex_indices,
        )
    coord = _set_drag_start(layer, coordinates)
    layer._moving_coordinates = coordinates
    layer._is_moving = True
    if vertex is None:
        # Check where dragging box from to move whole object
        center = layer._selected_box[Box.CENTER]
        shift = coord - center - layer._drag_start
        for index in layer.selected_data:
            layer._data_view.shift(index, shift)
        layer._selected_box = layer._selected_box + shift
        layer.refresh()
    else: # [NOTE] ignoring elif vertex < Box.LEN because no rotation handle on Box allowed → bounding boxes are kept axis aligned
        # Corner / edge vertex is being dragged so resize object
        # Also applies while drawing line, rectangle, ellipse
        box = layer._selected_box
        if layer._fixed_vertex is None:
            layer._fixed_index = (vertex + 4) % Box.LEN
            layer._fixed_vertex = box[layer._fixed_index]

        fixed = layer._fixed_vertex
        new = list(coord)

        box_center = box[Box.CENTER]
        if layer._fixed_aspect and layer._fixed_index % 2 == 0:
            new = (box[vertex] - box_center) / np.linalg.norm(
                box[vertex] - box_center
            ) * np.linalg.norm(new - box_center) + box_center

        if layer._fixed_index % 2 == 0:
            # corner selected
            drag_scale = ((new - fixed)) / (
                (box[vertex] - fixed)
            )
        elif layer._fixed_index % 4 == 3:
            # top or bottom selected
            drag_scale = np.array(
                [
                    ((new - fixed))[0]
                    / ((box[vertex] - fixed))[0],
                    1,
                ]
            )
        else:
            # left or right selected
            drag_scale = np.array(
                [
                    1,
                    ((new - fixed))[1]
                    / ((box[vertex] - fixed))[1],
                ]
            )

        # prevent box from shrinking below a threshold size
        size = (np.linalg.norm(box[Box.TOP_LEFT] - box_center),)
        # [TODO] check if necessary 
        threshold = (
            layer._vertex_size * layer.scale_factor / layer.scale[-1] / 2
        )
        if np.linalg.norm(size * drag_scale) < threshold:
            drag_scale[:] = 1

        # check orientation of box
        # [NOTE] in bbox plugin only keeping axis aligned bounding boxes
        for index in layer.selected_data:
            layer._data_view.scale(
                index, drag_scale, center=layer._fixed_vertex
            )
        layer._scale_box(drag_scale, center=layer._fixed_vertex)
        
        layer.refresh()


def _move_active_element_under_cursor(
    layer: BoundingBoxLayer, coordinates: tuple[float, ...]
) -> None:
    """Moves object at given mouse position and set of indices.

    Parameters
    ----------
    layer : BoundingBoxLayer
        Napari-bbox bounding box layer.
    coordinates : Tuple[float, ...]
        Position of mouse cursor in data coordinates.
    """
    # If nothing selected return
    if len(layer.selected_data) == 0:
        return

    vertex = layer._moving_value[1]

    if not layer._mode in (
        [Mode.SELECT, Mode.ADD_BOUNDING_BOX]
    ):
        return

    _move_selected_layer(layer, coordinates, vertex)