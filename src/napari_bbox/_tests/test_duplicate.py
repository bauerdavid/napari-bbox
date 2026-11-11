import numpy as np
import pytest
from napari.components import LayerList
from napari.layers._layer_actions import _duplicate_layer

from napari_bbox import BoundingBoxLayer


def _box(lo, hi):
    return np.array([lo, hi], dtype=float)


def _layer():
    layer = BoundingBoxLayer(
        [_box((10, 10, 10), (30, 30, 30)), _box((40, 40, 40), (60, 60, 70))],
        name="objects",
        edge_color=["red", "blue"],
        edge_width=[1, 3],
        opacity=0.5,
        scale=(1, 2, 3),
        translate=(5, 0, 0),
        metadata={"key": "value"},
        visible=False,
    )
    layer.features = {"label": [1, 2]}
    layer.text = {"string": "{label:d}", "size": 12}
    return layer


@pytest.fixture
def layers():
    layers = LayerList()
    layers.append(_layer())
    return layers


def test_duplicate_layer(layers):
    """The "Duplicate Layer" action creates the layer from its state."""
    original = layers[0]
    layers.selection = {original}

    _duplicate_layer(layers)

    assert [layer.name for layer in layers] == ["objects", "objects copy"]
    copy = layers[1]
    assert isinstance(copy, BoundingBoxLayer)
    assert len(copy.data) == len(original.data)
    for data, copied_data in zip(original.data, copy.data):
        np.testing.assert_array_equal(data, copied_data)
    np.testing.assert_array_equal(copy.edge_color, original.edge_color)
    np.testing.assert_array_equal(copy.face_color, original.face_color)
    assert list(copy.edge_width) == list(original.edge_width)
    assert list(copy.z_index) == list(original.z_index)
    assert copy.features.equals(original.features)
    assert copy.text.size == original.text.size
    assert copy.opacity == original.opacity
    assert copy.visible == original.visible
    assert copy.metadata == original.metadata
    np.testing.assert_array_equal(copy.scale, original.scale)
    np.testing.assert_array_equal(copy.translate, original.translate)


def test_duplicate_is_independent_of_the_original(layers):
    original = layers[0]
    layers.selection = {original}
    _duplicate_layer(layers)
    copy = layers[1]

    original_data = [data.copy() for data in original.data]
    copy.selected_data = {0}
    copy.remove_selected()

    assert len(copy.data) == 1
    assert len(original.data) == 2
    for data, kept in zip(original.data, original_data):
        np.testing.assert_array_equal(data, kept)


def test_duplicate_empty_layer():
    layers = LayerList()
    layers.append(BoundingBoxLayer(ndim=3, name="empty"))
    layers.selection = {layers[0]}

    _duplicate_layer(layers)

    assert len(layers) == 2
    assert len(layers[1].data) == 0
    assert layers[1].ndim == 3
