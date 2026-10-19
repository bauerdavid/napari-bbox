from dataclasses import dataclass, field

import numpy as np
import pytest
from napari.components import Dims
from napari.utils._proxies import ReadOnlyWrapper
from napari.utils.interactions import (
    mouse_move_callbacks,
    mouse_press_callbacks,
    mouse_release_callbacks,
)

from napari_bbox import BoundingBoxLayer
from napari_bbox._utils import NAPARI_VERSION
from napari_bbox.boundingbox.napari_0_4_18 import _bounding_box_3d as bbox_3d

pytestmark = pytest.mark.skipif(
    NAPARI_VERSION < "0.4.18", reason="3D editing requires napari>=0.4.18"
)

VIEW_DIRECTION = (1.0, 0.0, 0.0)
UP_DIRECTION = (0.0, -1.0, 0.0)


@dataclass
class Event:
    """Mouse event as passed to layer mouse callbacks."""

    type: str
    is_dragging: bool = False
    modifiers: list = field(default_factory=list)
    position: tuple = (0, 0, 0)
    view_direction: tuple = VIEW_DIRECTION
    up_direction: tuple = UP_DIRECTION
    dims_displayed: list = field(default_factory=lambda: [0, 1, 2])


def _bounds(lo, hi):
    return np.array([lo, hi], dtype=float)


def _layer_3d(*boxes):
    layer = BoundingBoxLayer([_bounds(lo, hi) for lo, hi in boxes])
    layer._slice_dims(Dims(ndim=layer.ndim, ndisplay=3))
    layer.mode = 'select'
    return layer


def _event(event_type, position, modifiers=(), view_direction=VIEW_DIRECTION):
    return ReadOnlyWrapper(
        Event(
            type=event_type,
            is_dragging=event_type == 'mouse_move',
            modifiers=list(modifiers),
            position=tuple(float(p) for p in position),
            view_direction=tuple(view_direction),
            up_direction=UP_DIRECTION,
            dims_displayed=[0, 1, 2],
        )
    )


def _drag(layer, path, modifiers=(), view_direction=VIEW_DIRECTION):
    """Press at the first position, move through the rest, then release."""
    kwargs = {'modifiers': modifiers, 'view_direction': view_direction}
    mouse_press_callbacks(layer, _event('mouse_press', path[0], **kwargs))
    for position in path[1:]:
        mouse_move_callbacks(layer, _event('mouse_move', position, **kwargs))
    mouse_release_callbacks(layer, _event('mouse_release', path[-1], **kwargs))


def _box(layer, index):
    return bbox_3d.box_bounds(layer.data[index], [0, 1, 2])


def test_ray_box_intersection():
    lo, hi = np.zeros(3), np.ones(3) * 10
    hit = bbox_3d.ray_box_intersection(
        np.array([-5.0, 5, 5]), np.array([1.0, 0, 0]), lo, hi
    )
    assert hit == pytest.approx((5, 15))
    assert (
        bbox_3d.ray_box_intersection(
            np.array([-5.0, 20, 5]), np.array([1.0, 0, 0]), lo, hi
        )
        is None
    )


def test_pick_box_prefers_nested_box():
    los = np.array([[0.0, 0, 0], [40, 40, 40]])
    his = np.array([[100.0, 100, 100], [60, 60, 60]])
    origin, direction = np.array([-10.0, 50, 50]), np.array([1.0, 0, 0])
    assert bbox_3d.pick_box(origin, direction, los, his) == 1
    origin = np.array([-10.0, 10, 10])
    assert bbox_3d.pick_box(origin, direction, los, his) == 0


def test_visible_handles_hide_axis_facing_camera():
    assert list(bbox_3d.visible_handles(np.array([1.0, 0, 0]))) == [2, 3, 4, 5]
    assert list(bbox_3d.visible_handles(None)) == list(range(6))


def test_set_box_bounds_keeps_hidden_dims():
    data = bbox_3d.find_bbox_corners(_bounds([1, 0, 0, 0], [2, 10, 10, 10]))
    new = bbox_3d.set_box_bounds(data, [1, 2, 3], [5, 5, 5], [6, 7, 8])
    lo, hi = bbox_3d.box_bounds(new, [0, 1, 2, 3])
    np.testing.assert_allclose(lo, [1, 5, 5, 5])
    np.testing.assert_allclose(hi, [2, 6, 7, 8])


def test_resize_bounds_does_not_cross_faces():
    lo, hi = np.zeros(3), np.ones(3) * 10
    new_lo, new_hi = bbox_3d.resize_bounds(lo, hi, 5, -50, 1)
    np.testing.assert_allclose(new_hi, [10, 10, 1])
    np.testing.assert_allclose(new_lo, lo)


def test_click_selects_box():
    layer = _layer_3d(([0, 0, 0], [100, 100, 100]), ([0, 200, 200], [100, 300, 300]))
    _drag(layer, [(50, 250, 250)])
    assert layer.selected_data == {1}
    _drag(layer, [(50, 150, 150)])
    assert layer.selected_data == set()


def test_modified_click_toggles_selection():
    layer = _layer_3d(([0, 0, 0], [100, 100, 100]), ([0, 200, 200], [100, 300, 300]))
    _drag(layer, [(50, 50, 50)])
    _drag(layer, [(50, 250, 250)], modifiers=['Shift'])
    assert layer.selected_data == {0, 1}
    _drag(layer, [(50, 50, 50)], modifiers=['Control'])
    assert layer.selected_data == {1}


def test_click_selects_nested_box():
    layer = _layer_3d(([0, 0, 0], [100, 100, 100]), ([40, 40, 40], [60, 60, 60]))
    _drag(layer, [(50, 50, 50)])
    assert layer.selected_data == {1}


def test_drag_moves_selected_boxes_parallel_to_canvas():
    layer = _layer_3d(([0, 0, 0], [100, 100, 100]), ([0, 200, 200], [100, 300, 300]))
    events = []
    layer.events.data.connect(lambda e: events.append(e.action))
    _drag(layer, [(50, 50, 50)])
    _drag(layer, [(50, 250, 250)], modifiers=['Shift'])
    _drag(layer, [(50, 250, 250), (50, 255, 260), (50, 260, 270)])

    lo, hi = _box(layer, 0)
    np.testing.assert_allclose(lo, [0, 10, 20])
    np.testing.assert_allclose(hi, [100, 110, 120])
    lo, hi = _box(layer, 1)
    np.testing.assert_allclose(lo, [0, 210, 220])
    assert layer.selected_data == {0, 1}
    assert str(events[-1]) == 'changed'


def test_drag_face_handle_resizes_box():
    layer = _layer_3d(([0, 0, 0], [100, 100, 100]))
    _drag(layer, [(50, 50, 50)])
    # The handle of the upper face along the last axis is at (50, 50, 100)
    _drag(layer, [(50, 50, 100), (50, 50, 115), (50, 50, 130)])

    lo, hi = _box(layer, 0)
    np.testing.assert_allclose(lo, [0, 0, 0])
    np.testing.assert_allclose(hi, [100, 100, 130])
    assert layer.selected_data == {0}


def test_handle_facing_camera_is_not_picked():
    layer = _layer_3d(([0, 0, 0], [100, 100, 100]))
    _drag(layer, [(50, 50, 50)])
    # The handles on the first axis face the camera, so this drags the box
    _drag(layer, [(50, 50, 50), (50, 60, 50)])
    lo, hi = _box(layer, 0)
    np.testing.assert_allclose(lo, [0, 10, 0])
    np.testing.assert_allclose(hi, [100, 110, 100])


def test_drag_box_selects_boxes_with_center_inside():
    layer = _layer_3d(
        ([0, 0, 0], [100, 100, 100]),
        ([0, 200, 200], [100, 300, 300]),
        ([0, 500, 500], [100, 600, 600]),
    )
    _drag(layer, [(50, 150, 150), (50, 400, 400)])
    assert layer.selected_data == {1}
    _drag(layer, [(50, -50, -50), (50, 400, 400)], modifiers=['Shift'])
    assert layer.selected_data == {0}


def test_add_mode_is_disabled_in_3d():
    layer = _layer_3d(([0, 0, 0], [100, 100, 100]))
    layer.mode = 'add_bounding_box'
    assert layer.mode == 'select'

    layer._slice_dims(Dims(ndim=3, ndisplay=2))
    layer.mode = 'add_bounding_box'
    assert layer.mode == 'add_bounding_box'
    layer._slice_dims(Dims(ndim=3, ndisplay=3))
    assert layer.mode == 'select'


def test_selection_survives_holding_space():
    layer = _layer_3d(([0, 0, 0], [100, 100, 100]))
    _drag(layer, [(50, 50, 50)])
    layer.mode = 'pan_zoom'
    layer.mode = 'select'
    assert layer.selected_data == {0}
    layer.mode = 'transform'
    assert layer.selected_data == set()


def test_space_pressed_mid_drag_stops_editing():
    layer = _layer_3d(([0, 0, 0], [100, 100, 100]))
    _drag(layer, [(50, 50, 50)])
    mouse_press_callbacks(layer, _event('mouse_press', (50, 50, 50)))
    assert layer._select_drag_active
    mouse_move_callbacks(layer, _event('mouse_move', (50, 60, 50)))
    layer.mode = 'pan_zoom'  # holding Space
    mouse_move_callbacks(layer, _event('mouse_move', (50, 90, 50)))
    mouse_release_callbacks(layer, _event('mouse_release', (50, 90, 50)))
    layer.mode = 'select'

    lo, _ = _box(layer, 0)
    # only the move before Space was pressed was applied
    np.testing.assert_allclose(lo, [0, 10, 0])
    assert layer.selected_data == {0}
    assert not layer._select_drag_active


def test_axis_parameter_under_ray():
    point, axis = np.array([0.0, 0, 0]), np.array([0.0, 0, 2])
    direction = np.array([1.0, 0, 1]) / np.sqrt(2)
    # the ray passes through the point at s = 5 (z = 10)
    s = bbox_3d.axis_parameter_under_ray(point, axis, np.array([0.0, 0, 10]), direction)
    assert s == pytest.approx(5)
    assert bbox_3d.axis_parameter_under_ray(point, axis, point, np.array([0.0, 0, 1])) is None


@pytest.mark.parametrize('scale', [(1, 1, 1), (2, 1, 0.5)])
def test_face_handle_stays_under_cursor_in_oblique_view(scale):
    layer = _layer_3d(([0, 0, 0], [100, 100, 100]))
    layer.scale = scale
    # the dragged face's axis is oblique to the view direction (63 degrees),
    # so the face would lag behind the cursor with a plain drag projection
    view = np.array([1.0, 0, 0.5]) / np.linalg.norm([1.0, 0, 0.5])
    world = lambda p: np.asarray(p, float) * scale
    _drag(layer, [world((50, 50, 50))], view_direction=view)
    assert layer.selected_data == {0}
    # rays through the handle of the upper face along the last axis, as it
    # moves from 100 to 130
    _drag(
        layer,
        [world((50, 50, 100)), world((50, 50, 115)), world((50, 50, 130))],
        view_direction=view,
    )
    lo, hi = _box(layer, 0)
    np.testing.assert_allclose(hi, [100, 100, 130])
    np.testing.assert_allclose(lo, [0, 0, 0])


def test_handle_on_lower_data_edge_can_be_dragged_edge_on():
    layer = _layer_3d(([0, 0, 0], [100, 100, 100]))
    _drag(layer, [(50, 50, 50)])
    # The handle of the lower face along the last axis lies on the edge of the
    # layer extent. Pressing just outside the extent, but on the handle, must
    # still grab it
    _drag(layer, [(50, 50, -2), (50, 50, -17), (50, 50, -32)])
    lo, hi = _box(layer, 0)
    np.testing.assert_allclose(lo, [0, 0, -30])
    np.testing.assert_allclose(hi, [100, 100, 100])
