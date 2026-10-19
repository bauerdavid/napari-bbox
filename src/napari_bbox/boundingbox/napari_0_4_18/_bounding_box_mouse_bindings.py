# A copy of napari.layers.shapes._shapes_mouse_bindings
from __future__ import annotations
from ..napari_0_4_15._bounding_box_mouse_bindings import (
    highlight,
    add_bounding_box,
    _drag_selection_box,
    _move
)
from ..napari_0_4_15._bounding_box_mouse_bindings import _drag_selection_box

from copy import copy
from typing import TYPE_CHECKING

import numpy as np

from napari.layers.base._base_constants import ActionType
from napari.layers.points._points_utils import _points_in_box_3d

from . import _bounding_box_3d as bbox_3d
from ..napari_0_4_15._bounding_box_constants import Box, Mode
from ..._utils import NAPARI_VERSION

if TYPE_CHECKING:
    from typing import List, Tuple

    import numpy.typing as npt
    from vispy.app.canvas import MouseEvent


def select(layer, event) -> None:
    """Select bounding boxes or vertices either in select mode.

    Once selected bounding boxes can be moved or resized, and vertices can be moved
    depending on the mode. Holding shift when resizing a bounding box will preserve
    the aspect ratio.

    Parameters
    ----------
    layer: BoundingBoxLayer
        Bounding box layer
    event: MouseEvent
        A proxy read only wrapper around a vispy mouse event.
    """
    if layer._is_3d_editable:
        yield from _select_3d(layer, event)
        return

    shift = 'Shift' in event.modifiers
    # on press
    value = layer.get_value(event.position, world=True) or (None, None)
    layer._moving_value = copy(value)
    bounding_box_under_cursor, vertex_under_cursor = value
    if vertex_under_cursor is None:
        if shift and bounding_box_under_cursor is not None:
            if bounding_box_under_cursor in layer.selected_data:
                layer.selected_data.remove(bounding_box_under_cursor)
            else:
                if len(layer.selected_data):
                    # one or more shapes already selected
                    layer.selected_data.add(bounding_box_under_cursor)
                else:
                    # first shape being selected
                    layer.selected_data = {bounding_box_under_cursor}
        elif bounding_box_under_cursor is not None:
            if bounding_box_under_cursor not in layer.selected_data:
                layer.selected_data = {bounding_box_under_cursor}
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
            _move_active_element_under_cursor(layer, coordinates)

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
        if NAPARI_VERSION >= "0.4.19":
            layer.events.data(
                value=layer.data,
                action=ActionType.CHANGED,
                data_indices=tuple(layer.selected_data),
                vertex_indices=vertex_indices,
            )
        else:
            layer.events.data(
                value=layer.data,
                action=ActionType.CHANGE.value,
                data_indices=tuple(layer.selected_data),
                vertex_indices=vertex_indices,
            )

    # on release
    shift = 'Shift' in event.modifiers
    if not layer._is_moving and not layer._is_selecting and not shift:
        if bounding_box_under_cursor is not None:
            layer.selected_data = {bounding_box_under_cursor}
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


def _add_bounding_box(
    layer, event: MouseEvent, data: npt.NDArray
) -> None:
    """Helper function for adding a bounding box.

    Parameters
    ----------
    layer: BoundingBoxLayer
        Bounding box layer
    event: MouseEvent
        A proxy read only wrapper around a vispy mouse event.
    data: np.NDarray
        Array containing the initial datapoints of the bounding box in image data space.
    """
    # on press
    # Start drawing rectangle / ellipse / line
    layer.add(data)
    layer.selected_data = {layer.nbounding_boxes - 1}
    layer._value = (layer.nbounding_boxes - 1, 4)
    layer._moving_value = copy(layer._value)
    layer.refresh()
    yield

    data = layer.data[layer.nbounding_boxes - 1]
    # on move
    const_set = False
    while event.type == 'mouse_move':
        # Drag any selected bounding boxes
        coordinates = layer.world_to_data(event.position)
        layer._moving_coordinates = coordinates
        _move_active_element_under_cursor(layer, coordinates) #TODO check
        min = data.min(0)
        max = data.max(0)
        size = max - min
        visible_size = size[layer._slice_input.displayed]
        max[layer._slice_input.displayed] = np.nan
        min[layer._slice_input.displayed] = np.nan
        if layer.size_mode == "average":
            data[:] = np.where(data == max, coordinates + visible_size.mean() / 2 * layer.size_multiplier, data)
            data[:] = np.where(data == min, coordinates - visible_size.mean() / 2 * layer.size_multiplier, data)
        elif not const_set and layer.size_mode == "constant":
            data[:] = np.where(data == max, np.asarray(coordinates) + layer.size_constant / 2, data)
            data[:] = np.where(data == min, np.asarray(coordinates) - layer.size_constant / 2, data)
            const_set = True
        yield
    layer._clear_extent()
    layer.events.data(value=layer.data, data_indices=(-1,), vertex_indices=((),))
    layer.events.extent()
    # on release
    layer._finish_drawing()


def finish_drawing_bounding_box(layer, event: MouseEvent) -> None:
    """Finish drawing of bounding box.

    Calls the finish drawing method of the bounding box layer which resets all the properties used for bounding box drawing.

    Parameters
    ----------
    layer: BoundingBoxLayer
        Bounding box layer
    event: MouseEvent
        A proxy read only wrapper around a vispy mouse event. Not used here, but passed as argument due to being a
        double click callback of the shapes layer.
    """
    layer._finish_drawing()


def _set_drag_start(
    layer, coordinates: Tuple[float, ...]
) -> List[float, ...]:
    """Indicate where in data space a drag event started.

    Sets the coordinates relative to the center of the bounding box and returns the position
    of where a drag event of a shape started.

    Parameters
    ----------
    layer: BoundingBoxLayer
        The bounding box.
    coordinates: Tuple[float, ...]
        The position in image data space where dragging started.

    Returns
    -------
    coord: List[float, ...]
        The coordinates of where a shape drag event started.
    """
    coord = [coordinates[i] for i in layer._slice_input.displayed]
    if layer._drag_start is None and len(layer.selected_data) > 0:
        center = layer._selected_box[Box.CENTER]
        layer._drag_start = coord - center
    return coord


def _move_active_element_under_cursor(
    layer, coordinates: Tuple[float, ...]
) -> None:
    """Moves object at given mouse position and set of indices.

    Parameters
    ----------
    layer : BoundingBoxLayer
        Bounding box layer.
    coordinates : Tuple[float, ...]
        Position of mouse cursor in data coordinates.
    """
    # TODO check
    # If nothing selected return
    if len(layer.selected_data) == 0:
        return

    vertex = layer._moving_value[1]

    if not layer._mode in (
        [Mode.SELECT, Mode.ADD_BOUNDING_BOX]
    ):
        return
    if NAPARI_VERSION >= "0.4.19" and layer._mode == Mode.SELECT and not layer._is_moving:
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
    else:
        # Corner / edge vertex is being dragged so resize object
        box = layer._selected_box
        if layer._fixed_vertex is None:
            layer._fixed_index = (vertex + 4) % Box.LEN
            layer._fixed_vertex = box[layer._fixed_index]

        fixed = layer._fixed_vertex
        new = list(coord)

        box_center = box[Box.CENTER]
        if layer._fixed_aspect and layer._fixed_index % 2 == 0:
            # corner
            new = (box[vertex] - box_center) / np.linalg.norm(
                box[vertex] - box_center
            ) * np.linalg.norm(new - box_center) + box_center

        if layer._fixed_index % 2 == 0:
            # corner selected
            drag_scale = ( (new - fixed)) / (
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
        threshold = (
            layer._vertex_size * layer.scale_factor / layer.scale[-1] / 2
        )
        if np.linalg.norm(size * drag_scale) < threshold:
            drag_scale[:] = 1
        # on vertical/horizontal drags we get scale of 0
        # when we actually simply don't want to scale
        drag_scale[drag_scale == 0] = 1

        # check orientation of box
        for index in layer.selected_data:
            layer._data_view.scale(
                index, drag_scale, center=layer._fixed_vertex
            )
        layer._scale_box(drag_scale, center=layer._fixed_vertex)

        layer.refresh()





def _emit_data_event(layer, indices, changing: bool) -> None:
    """Emit a data event for boxes that are being or have been edited."""
    indices = tuple(indices)
    vertex_indices = tuple(
        tuple(range(len(layer.data[i]))) for i in indices
    )
    if NAPARI_VERSION >= "0.4.19":
        action = ActionType.CHANGING if changing else ActionType.CHANGED
    elif changing:
        return
    else:
        action = ActionType.CHANGE.value
    layer.events.data(
        value=layer.data,
        action=action,
        data_indices=indices,
        vertex_indices=vertex_indices,
    )


def _select_3d(layer, event):
    """Select, move and resize bounding boxes in a 3D view.

    Follows the selection behavior of the napari points layer:

    * Clicking a box selects it, holding Shift or Control toggles it.
    * Dragging a selected box moves all selected boxes in the plane parallel
      to the canvas.
    * Dragging a face handle of a single selected box moves that face along
      its axis.
    * Dragging on empty space draws a selection box. Boxes whose center falls
      inside it get selected.
    """
    dims = list(layer._slice_input.displayed)
    modify_selection = (
        'Shift' in event.modifiers or 'Control' in event.modifiers
    )

    # on press
    value = layer.get_value(
        event.position,
        view_direction=event.view_direction,
        dims_displayed=event.dims_displayed,
        world=True,
    ) or (None, None)
    index, handle = value
    if handle is None:
        if modify_selection:
            if index is not None:
                layer.selected_data = layer.selected_data ^ {index}
        elif index is not None:
            if index not in layer.selected_data:
                layer.selected_data = {index}
        else:
            layer.selected_data = set()
    layer._moving_value = copy(value)
    layer._set_highlight()

    start_position = np.asarray(event.position)
    start_data = layer.world_to_data(start_position)[dims]
    edited = sorted(layer.selected_data) if handle is None else [index]
    start_los, start_his = layer._box_bounds_3d(edited)
    if handle is not None:
        handle_line = layer._handle_axis_line_3d(index, handle)
        handle_start = layer._handle_axis_parameter_3d(
            handle_line, start_position, event.view_direction
        )
    layer._select_drag_active = True
    yield

    # on move
    is_moving = False
    handed_to_camera = False
    while event.type == 'mouse_move':
        if handed_to_camera or layer._mode != Mode.SELECT:
            # Space was pressed during the drag, the camera rotates/pans the
            # view for the rest of it (see VispyBoundingBoxLayer)
            if not handed_to_camera:
                handed_to_camera = True
                layer._is_selecting = False
                layer._drag_box = None
                layer._set_highlight()
        elif handle is not None:
            # Resize: move the grabbed face along its axis
            # The face follows the point of its axis closest to the ray under
            # the cursor, so the grabbed point stays under the cursor
            axis = handle // 2
            current = layer._handle_axis_parameter_3d(
                handle_line, event.position, event.view_direction
            )
            if current is None or handle_start is None:
                yield
                continue
            distance = current - handle_start
            if not is_moving:
                _emit_data_event(layer, edited, changing=True)
            is_moving = layer._is_moving = True
            lo, hi = bbox_3d.resize_bounds(
                start_los[0],
                start_his[0],
                handle,
                distance,
                layer._min_box_size_3d(axis),
            )
            layer._set_box_bounds_3d(index, lo, hi)
            layer.refresh()
            layer._set_highlight(force=True)
        elif index is not None and not modify_selection and edited:
            # Move: shift selected boxes in the plane parallel to the canvas
            shift = layer.world_to_data(np.asarray(event.position))[dims] - start_data
            if not is_moving:
                _emit_data_event(layer, edited, changing=True)
            is_moving = layer._is_moving = True
            for i, lo, hi in zip(edited, start_los, start_his):
                layer._set_box_bounds_3d(i, lo + shift, hi + shift)
            layer.refresh()
            layer._set_highlight(force=True)
        else:
            # Draw a selection box. A modified click on a box only drags a
            # selection box, so undo its selection toggle
            if modify_selection and index is not None and not layer._is_selecting:
                layer.selected_data = layer.selected_data ^ {index}
            layer._is_selecting = True
            coordinates = layer.world_to_data(np.asarray(event.position))[dims]
            layer._drag_box = np.array([start_data, coordinates])
            layer._drag_normal = layer._world_to_displayed_data_ray(
                event.view_direction, dims
            )
            layer._drag_up = layer._world_to_displayed_data_ray(
                event.up_direction, dims
            )
            layer._set_highlight()
        yield

    # on release
    if is_moving:
        _emit_data_event(layer, edited, changing=False)
        if hasattr(layer, '_clear_extent'):
            layer._clear_extent()
        layer.events.extent()
    elif handed_to_camera:
        pass
    elif layer._is_selecting:
        layer.selected_data = _bounding_boxes_in_drag_box_3d(
            layer, modify_selection
        )
    elif not modify_selection and handle is None:
        layer.selected_data = set() if index is None else {index}

    layer._select_drag_active = False
    layer._is_moving = False
    layer._is_selecting = False
    layer._drag_box = None
    layer._drag_normal = None
    layer._drag_up = None
    layer._moving_value = (None, None)
    layer._set_highlight(force=True)

    if is_moving:
        layer._update_thumbnail()


def _bounding_boxes_in_drag_box_3d(layer, modify_selection: bool) -> set:
    """Selection after dragging a selection box in 3D."""
    displayed = np.nonzero(layer._data_view._displayed)[0]
    selection = set()
    if len(displayed) > 0:
        los, his = layer._box_bounds_3d(displayed)
        inside = _points_in_box_3d(
            layer._drag_box,
            (los + his) / 2,
            np.zeros(len(displayed)),
            layer._drag_normal,
            layer._drag_up,
        )
        selection = {int(displayed[i]) for i in inside}
    if modify_selection:
        return set(layer.selected_data) ^ selection
    return selection
