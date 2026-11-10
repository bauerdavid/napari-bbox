import numpy as np
import pytest
from napari.utils._proxies import ReadOnlyWrapper
from napari.utils.interactions import (
    mouse_move_callbacks,
    mouse_press_callbacks,
    mouse_release_callbacks,
)

from napari_bbox import BoundingBoxLayer
from napari_bbox._utils import NAPARI_VERSION

# has to happen before importing modules that need napari>=0.4.18
if NAPARI_VERSION < "0.4.18":
    pytest.skip(
        "undo and redo require napari>=0.4.18", allow_module_level=True
    )

from napari_bbox._tests.test_3d_editing import Event, _slice_dims  # noqa: E402
from napari_bbox.boundingbox import _undo_redo  # noqa: E402


def _square(lo, hi):
    (y0, x0), (y1, x1) = lo, hi
    return np.array([[y0, x0], [y0, x1], [y1, x0], [y1, x1]], dtype=float)


def _event(event_type, position, modifiers=()):
    return ReadOnlyWrapper(
        Event(
            type=event_type,
            is_dragging=event_type == 'mouse_move',
            modifiers=list(modifiers),
            position=tuple(float(p) for p in position),
            view_direction=None,
            up_direction=None,
            dims_displayed=list(range(len(position))),
        )
    )


def _drag(layer, path, modifiers=()):
    """Press at the first position, move through the rest, then release."""
    mouse_press_callbacks(layer, _event('mouse_press', path[0], modifiers))
    for position in path[1:]:
        mouse_move_callbacks(layer, _event('mouse_move', position, modifiers))
    mouse_release_callbacks(layer, _event('mouse_release', path[-1], modifiers))


def _label_new_boxes(layer, event):
    """What napari-nD-annotator does to a box right after it was added."""
    n_boxes = len(layer.data)
    yield
    if len(layer.data) > n_boxes:
        column = layer.features.columns.get_loc('label')
        layer.features.iloc[-1, column] = layer.features['label'].max() + 1
        layer.refresh_text()
    while event.type == 'mouse_move':
        yield
    layer.data = layer.data


def _layer(*boxes):
    """A 2D layer in select mode, with a label for each box."""
    layer = BoundingBoxLayer([_square(*box) for box in boxes], ndim=2)
    layer.features = {'label': np.arange(1, len(boxes) + 1)}
    layer.text = {'text': '{label:d}'}
    layer.mouse_drag_callbacks.append(_label_new_boxes)
    layer.mode = 'select'
    return layer


def _state(layer):
    return {
        'data': [np.array(d) for d in layer.data],
        'features': layer.features.copy(),
        'edge_color': np.array(layer.edge_color),
        'face_color': np.array(layer.face_color),
        'edge_width': list(layer.edge_width),
        'z_index': list(layer.z_index),
    }


def _assert_state(layer, expected):
    state = _state(layer)
    assert len(state['data']) == len(expected['data'])
    for box, expected_box in zip(state['data'], expected['data']):
        np.testing.assert_array_equal(box, expected_box)
    assert state['features'].equals(expected['features'])
    for key in ('edge_color', 'face_color'):
        np.testing.assert_array_equal(state[key], expected[key])
    assert state['edge_width'] == expected['edge_width']
    assert state['z_index'] == expected['z_index']
    # the layer stays consistent
    n_boxes = len(layer.data)
    assert len(layer.features) == n_boxes
    assert layer.nbounding_boxes == n_boxes
    assert len(layer._indices_view) == n_boxes
    assert all(i < n_boxes for i in layer.selected_data)


def _round_trip(layer, edit):
    """Edit the layer, undo, redo and undo again, checking the state each time."""
    before = _state(layer)
    edit()
    after = _state(layer)
    assert layer.can_undo
    assert layer.undo()
    _assert_state(layer, before)
    assert layer.can_redo
    assert layer.redo()
    _assert_state(layer, after)
    assert layer.undo()
    _assert_state(layer, before)
    return before, after


BOX_A = ((10, 10), (30, 30))
BOX_B = ((50, 50), (80, 90))


def test_nothing_to_undo():
    layer = _layer(BOX_A)
    assert not layer.can_undo and not layer.can_redo
    assert not layer.undo()
    assert not layer.redo()
    assert len(layer.data) == 1


def test_adding_a_box_by_dragging(request):
    layer = _layer(BOX_A)
    layer.mode = 'add_bounding_box'

    before, after = _round_trip(
        layer, lambda: _drag(layer, [(60, 60), (65, 66), (70, 75)])
    )
    assert len(after['data']) == 2
    # the label the plugin gave the box comes back with it
    assert list(after['features']['label']) == [1, 2]
    assert list(before['features']['label']) == [1]
    layer.redo()
    assert list(layer.features['label']) == [1, 2]
    assert layer.can_undo and not layer.can_redo


def test_moving_a_box_is_one_step():
    layer = _layer(BOX_A, BOX_B)
    assert not layer.can_undo
    _drag(layer, [(20, 20)])  # selects the box
    assert layer.selected_data == {0}
    assert not layer.can_undo
    path = [(20, 20)] + [(20 + i, 20 + 2 * i) for i in range(1, 8)]
    before, after = _round_trip(layer, lambda: _drag(layer, path))
    np.testing.assert_allclose(after['data'][0].min(0), before['data'][0].min(0) + [7, 14])
    # one step for the whole drag
    layer.redo()
    assert len(layer._undo_stack) == 1
    assert layer.undo()
    assert not layer.can_undo


def test_selection_and_clicks_are_not_steps():
    layer = _layer(BOX_A, BOX_B)
    _drag(layer, [(20, 20)])
    _drag(layer, [(60, 60)])
    _drag(layer, [(200, 200)])
    _drag(layer, [(0, 0), (150, 150)])  # a selection box
    layer.selected_data = {1}
    assert not layer.can_undo


def test_resizing_a_box():
    layer = _layer(BOX_A)
    _drag(layer, [(20, 20)])
    handle = tuple(layer._selected_box[2])  # bottom right corner

    def resize():
        _drag(layer, [handle, (handle[0] + 5, handle[1] + 5), (handle[0] + 12, handle[1] + 20)])

    before, after = _round_trip(layer, resize)
    assert not np.array_equal(before['data'][0], after['data'][0])
    assert after['data'][0].max() > before['data'][0].max()
    layer.redo()
    assert len(layer._undo_stack) == 1


def test_deleting_boxes():
    layer = _layer(BOX_A, BOX_B, ((100, 100), (120, 120)))
    layer.selected_data = {0, 2}
    _round_trip(layer, layer.remove_selected)
    assert list(layer.features['label']) == [1, 2, 3]


def test_undoing_the_creation_of_the_only_box_and_the_deletion_of_the_last_one():
    layer = BoundingBoxLayer(ndim=2)
    layer.mode = 'add_bounding_box'
    _drag(layer, [(60, 60), (70, 75)])
    assert len(layer.data) == 1
    assert layer.undo()
    assert len(layer.data) == 0
    assert len(layer.features) == 0
    assert len(layer._indices_view) == 0
    assert len(layer._view_text) == 0
    assert layer.redo()
    assert len(layer.data) == 1

    layer.mode = 'select'
    layer.selected_data = {0}
    layer.remove_selected()
    assert len(layer.data) == 0
    assert layer.undo()
    assert len(layer.data) == 1
    assert len(layer.features) == 1
    assert layer.undo()
    assert len(layer.data) == 0
    assert not layer.can_undo
    assert layer.redo() and layer.redo()
    assert len(layer.data) == 0


def test_undoing_the_first_of_several_boxes():
    layer = _layer(BOX_A, BOX_B)
    layer.selected_data = {0}
    layer.remove_selected()
    layer.selected_data = {0}
    layer.remove_selected()
    assert len(layer.data) == 0
    layer.undo()
    layer.undo()
    assert len(layer.data) == 2
    assert list(layer.features['label']) == [1, 2]
    np.testing.assert_array_equal(layer.data[0], _square(*BOX_A))


def test_a_new_edit_clears_the_redo_stack():
    layer = _layer(BOX_A, BOX_B)
    layer.selected_data = {0}
    layer.remove_selected()
    layer.undo()
    assert layer.can_redo
    layer.selected_data = {1}
    layer.remove_selected()
    assert not layer.can_redo
    assert not layer.redo()
    assert len(layer.data) == 1


def test_history_is_bounded(monkeypatch):
    monkeypatch.setattr(_undo_redo, 'MAX_HISTORY', 3)
    layer = _layer(*[((10 * i, 10 * i), (10 * i + 5, 10 * i + 5)) for i in range(6)])
    for _ in range(5):
        layer.selected_data = {0}
        layer.remove_selected()
    assert len(layer.data) == 1
    steps = 0
    while layer.undo():
        steps += 1
    assert steps == 3
    assert len(layer.data) == 4
    steps = 0
    while layer.redo():
        steps += 1
    assert steps == 3


def test_default_history_size():
    assert _undo_redo.MAX_HISTORY == 50


def test_colors_widths_and_z_index_are_restored():
    layer = _layer(BOX_A, BOX_B)
    layer.edge_color = np.array([[1, 0, 0, 1], [0, 0, 1, 1]], dtype=float)
    layer.edge_width = [3, 4]
    before = _state(layer)
    layer.selected_data = {1}
    layer.remove_selected()
    layer.undo()
    _assert_state(layer, before)
    layer.selected_data = {0}
    _round_trip(layer, layer.move_to_front)


def test_paste_is_an_edit():
    layer = _layer(BOX_A, BOX_B)
    layer.selected_data = {0}
    layer._copy_data()
    _round_trip(layer, layer._paste_data)


def test_adding_from_code_is_an_edit():
    layer = _layer(BOX_A)
    _round_trip(layer, lambda: layer.add(_square(*BOX_B)))


def test_programmatic_data_change_is_not_recorded_and_resets_the_history():
    layer = _layer(BOX_A, BOX_B)
    layer.selected_data = {0}
    layer.remove_selected()
    assert layer.can_undo
    layer.data = [_square((0, 0), (5, 5))]
    assert len(layer.data) == 1
    # nothing recorded can be restored over the new data
    assert not layer.can_undo and not layer.can_redo
    assert not layer.undo()
    np.testing.assert_array_equal(layer.data[0], _square((0, 0), (5, 5)))


def test_setting_the_same_data_keeps_the_history():
    layer = _layer(BOX_A, BOX_B)
    layer.selected_data = {0}
    layer.remove_selected()
    layer.undo()
    layer.data = layer.data
    assert layer.can_redo
    layer.redo()
    layer.data = layer.data
    assert layer.can_undo
    assert len(layer.data) == 1


def test_programmatic_features_change_resets_the_history():
    layer = _layer(BOX_A, BOX_B)
    layer.selected_data = {0}
    layer.remove_selected()
    layer.features = {'label': [7]}
    assert not layer.can_undo
    layer.add(_square(*BOX_A))
    layer.undo()
    assert list(layer.features['label']) == [7]


def test_clear_history():
    layer = _layer(BOX_A, BOX_B)
    layer.selected_data = {0}
    layer.remove_selected()
    layer.undo()
    layer.clear_history()
    assert not layer.can_undo and not layer.can_redo


def test_undo_during_a_drag_is_ignored():
    layer = _layer(BOX_A, BOX_B)
    layer.selected_data = {1}
    layer.remove_selected()
    assert layer.can_undo
    layer.selected_data = {0}
    mouse_press_callbacks(layer, _event('mouse_press', (20, 20)))
    mouse_move_callbacks(layer, _event('mouse_move', (25, 25)))
    moved = layer.data[0].copy()
    assert not layer.undo()
    assert not layer.redo()
    np.testing.assert_array_equal(layer.data[0], moved)
    mouse_release_callbacks(layer, _event('mouse_release', (25, 25)))
    assert layer.undo()
    np.testing.assert_array_equal(layer.data[0], _square(*BOX_A))


# napari<0.4.19 only knows the "completed" action names and no "-ing" ones
if NAPARI_VERSION >= "0.4.19":
    ADDED, REMOVED = 'added', 'removed'
    CHANGE_EVENTS = [('changing', (0,)), ('changed', (0,))]
else:
    ADDED, REMOVED = 'add', 'remove'
    CHANGE_EVENTS = [('change', (0,))]


def test_undo_emits_completed_data_events():
    layer = _layer(BOX_A, BOX_B)
    layer.selected_data = {1}
    layer.remove_selected()
    events = []
    layer.events.data.connect(lambda e: events.append((str(getattr(e, 'action', None)), getattr(e, 'data_indices', None))))
    layer.undo()
    assert events[-1] == (ADDED, (1,))
    layer.redo()
    assert events[-1] == (REMOVED, (1,))
    _drag(layer, [(20, 20)])
    _drag(layer, [(20, 20), (25, 25), (30, 30)])
    del events[:]
    layer.undo()
    # an undo is never left looking like an edit in progress
    assert events == CHANGE_EVENTS
    assert not layer._is_moving and not layer._is_creating


def test_undo_emits_set_data_for_the_visuals():
    layer = _layer(BOX_A, BOX_B)
    layer.selected_data = {1}
    layer.remove_selected()
    calls = []
    layer.events.set_data.connect(lambda e: calls.append(len(layer.data)))
    layer.undo()
    assert calls and calls[-1] == 2


def test_undo_clears_the_selection():
    layer = _layer(BOX_A, BOX_B)
    _drag(layer, [(60, 60), (65, 65), (70, 70)])
    assert layer.selected_data == {1}
    layer.undo()
    assert layer.selected_data == set()
    assert layer._selected_box is None


def _layer_3d(*boxes):
    layer = BoundingBoxLayer([np.array(box, dtype=float) for box in boxes])
    _slice_dims(layer, ndisplay=3)
    layer.mode = 'select'
    return layer


def _drag_3d(layer, path):
    from napari_bbox._tests.test_3d_editing import _drag

    _drag(layer, path)


@pytest.mark.skipif(
    NAPARI_VERSION < "0.5.0", reason="3D editing is not verified on napari<0.5"
)
def test_editing_in_3d():
    layer = _layer_3d(
        [[0, 0, 0], [100, 100, 100]],
        [[0, 200, 200], [100, 300, 300]],
    )
    _drag_3d(layer, [(50, 50, 50)])
    assert layer.selected_data == {0}
    assert not layer.can_undo

    def move():
        _drag_3d(layer, [(50, 50, 50), (50, 55, 60), (50, 60, 70)])

    before, after = _round_trip(layer, move)
    np.testing.assert_allclose(after['data'][0].min(0), [0, 10, 20])
    layer.redo()
    assert len(layer._undo_stack) == 1

    def resize():
        # the handle of the upper face along the last axis
        _drag_3d(layer, [(50, 60, 120), (50, 60, 135), (50, 60, 150)])

    before, after = _round_trip(layer, resize)
    assert before['data'][0][:, 2].max() == 120
    assert after['data'][0][:, 2].max() == 150

    layer.redo()
    layer.selected_data = {1}
    layer.remove_selected()
    assert len(layer.data) == 1
    layer.undo()
    assert len(layer.data) == 2
    assert len(layer._undo_stack) == 2


@pytest.fixture
def viewer():
    import napari

    viewer = napari.Viewer(show=False)
    viewer.add_image(np.zeros((100, 100)))
    yield viewer
    viewer.close()


def _press(viewer, key):
    handler = viewer.window._qt_viewer._key_map_handler
    handler.press_key(key)
    handler.release_key(key)


def test_key_bindings(viewer):
    layer = BoundingBoxLayer([_square(*BOX_A), _square(*BOX_B)], ndim=2)
    # the displayed text has to follow the boxes
    layer.features = {'label': [1, 2]}
    layer.text = {'text': '{label:d}'}
    viewer.layers.append(layer)
    viewer.layers.selection.active = layer
    layer.selected_data = {0}
    layer.remove_selected()
    assert len(layer.data) == 1

    _press(viewer, 'Control-Z')
    assert len(layer.data) == 2
    assert list(layer._view_text) == ['1', '2']
    _press(viewer, 'Control-Shift-Z')
    assert len(layer.data) == 1
    assert list(layer._view_text) == ['2']
    _press(viewer, 'Control-Z')
    assert len(layer.data) == 2
    _press(viewer, 'Control-Y')
    assert len(layer.data) == 1


def test_key_bindings_in_3d(viewer):
    viewer.layers.clear()
    viewer.add_image(np.zeros((20, 100, 100)))
    layer = BoundingBoxLayer(
        [np.array([[1, 10, 10], [5, 30, 30]], dtype=float), np.array([[2, 50, 50], [6, 80, 90]], dtype=float)]
    )
    viewer.layers.append(layer)
    viewer.dims.ndisplay = 3
    viewer.layers.selection.active = layer
    layer.selected_data = {1}
    layer.remove_selected()
    assert len(layer.data) == 1
    _press(viewer, 'Control-Z')
    assert len(layer.data) == 2
    _press(viewer, 'Control-Shift-Z')
    assert len(layer.data) == 1


def test_key_bindings_only_act_on_the_active_layer(viewer):
    layer = BoundingBoxLayer([_square(*BOX_A), _square(*BOX_B)], ndim=2)
    viewer.layers.append(layer)
    layer.selected_data = {0}
    layer.remove_selected()
    viewer.layers.selection.active = viewer.layers[0]
    _press(viewer, 'Control-Z')
    assert len(layer.data) == 1
