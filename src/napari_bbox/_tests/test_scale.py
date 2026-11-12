"""Bounding boxes drawn in a layer that has a scale."""
from dataclasses import dataclass, field

import numpy as np
import pytest
from napari.utils._proxies import ReadOnlyWrapper
from napari.utils.interactions import (
    mouse_move_callbacks,
    mouse_press_callbacks,
    mouse_release_callbacks,
)

from napari_bbox import BoundingBoxLayer

SCALES = [(1, 1, 1), (4, 1, 1), (1, 2, 0.5), (0.25, 3, 3)]


@dataclass
class Event:
    type: str
    position: tuple
    is_dragging: bool = False
    modifiers: list = field(default_factory=list)
    view_direction: tuple = None
    up_direction: tuple = None
    dims_displayed: list = field(default_factory=lambda: [1, 2])


def _event(event_type, position):
    return ReadOnlyWrapper(
        Event(
            type=event_type,
            position=tuple(float(p) for p in position),
            is_dragging=event_type == 'mouse_move',
        )
    )


def _drag(layer, path):
    """Press at the first position, move through the rest, then release."""
    mouse_press_callbacks(layer, _event('mouse_press', path[0]))
    for position in path[1:]:
        mouse_move_callbacks(layer, _event('mouse_move', position))
    mouse_release_callbacks(layer, _event('mouse_release', path[-1]))


def _layer(scale, size_mode):
    layer = BoundingBoxLayer(ndim=3, scale=scale)
    layer.size_mode = size_mode
    layer.mode = 'add_bounding_box'
    return layer


def _world_extent(layer):
    """Size of the last box in world units along (z, y, x)."""
    corners = layer.data[-1]
    return (corners.max(0) - corners.min(0)) * np.asarray(layer.scale)


@pytest.mark.parametrize('scale', SCALES)
def test_average_size_along_hidden_dimension_is_in_world_units(scale):
    layer = _layer(scale, 'average')
    layer.size_multiplier = 2

    # draw in the displayed (y, x) plane, at z=10, in world coordinates
    _drag(layer, [(10, 20, 20), (10, 60, 100)])

    z, y, x = _world_extent(layer)
    assert (y, x) == pytest.approx((40, 80))
    # the box is as deep as the mean of its other sizes times the multiplier
    assert z == pytest.approx(60 * 2)


@pytest.mark.parametrize('scale', SCALES)
def test_constant_size_along_hidden_dimension_is_in_world_units(scale):
    layer = _layer(scale, 'constant')
    layer.size_constant = 30

    _drag(layer, [(10, 20, 20), (10, 60, 100)])

    z, y, x = _world_extent(layer)
    assert (y, x) == pytest.approx((40, 80))
    assert z == pytest.approx(30)


@pytest.mark.parametrize('scale', SCALES)
def test_box_drawn_around_the_cursor_in_the_hidden_dimension(scale):
    layer = _layer(scale, 'average')

    _drag(layer, [(10, 20, 20), (10, 60, 100)])

    z_min, z_max = layer.data[-1][:, 0].min(), layer.data[-1][:, 0].max()
    # the plane that was drawn on is the center of the box
    assert (z_min + z_max) / 2 * scale[0] == pytest.approx(10)


@pytest.mark.parametrize('scale', SCALES)
def test_new_box_is_cubic_in_world_units(scale):
    layer = _layer(scale, 'average')

    # a click without a drag
    _drag(layer, [(10, 50, 50)])

    extent = _world_extent(layer)
    assert extent == pytest.approx(np.full(3, extent[0]))
    assert extent[0] > 0


def test_unscaled_2d_layer_is_not_affected():
    layer = BoundingBoxLayer(ndim=2)
    layer.mode = 'add_bounding_box'

    mouse_press_callbacks(layer, _event_2d('mouse_press', (20, 20)))
    mouse_move_callbacks(layer, _event_2d('mouse_move', (60, 100)))
    mouse_release_callbacks(layer, _event_2d('mouse_release', (60, 100)))

    corners = layer.data[-1]
    np.testing.assert_allclose(corners.min(0), (20, 20))
    np.testing.assert_allclose(corners.max(0), (60, 100))


def _event_2d(event_type, position):
    return ReadOnlyWrapper(
        Event(
            type=event_type,
            position=tuple(float(p) for p in position),
            is_dragging=event_type == 'mouse_move',
            dims_displayed=[0, 1],
        )
    )
