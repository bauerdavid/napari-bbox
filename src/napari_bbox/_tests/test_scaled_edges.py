"""The edges of the boxes in a layer that has a scale.

The thickness of the edges is in world units, so it has to be the same in
every direction. The mesh of a layer with the data ``D`` and the scale ``s`` has
to be what the mesh of a layer without a scale and the data ``D * s`` is, in
world coordinates.
"""
import numpy as np
import pytest
from napari.components import Dims

from napari_bbox import BoundingBoxLayer
from napari_bbox._helper_functions import layer_dims_displayed
from napari_bbox._utils import NAPARI_VERSION

SCALES = [(4, 1, 1), (1, 2, 0.5), (0.25, 3, 3)]
BOXES = [
    np.array([[10, 20, 20], [30, 70, 80]], dtype=float),
    np.array([[40, 5, 60], [55, 40, 90]], dtype=float),
]


def _set_ndisplay(layer, ndisplay):
    """Slice ``layer`` as the viewer would, whatever napari's signature."""
    ndim = layer.ndim
    if NAPARI_VERSION < "0.5.0":
        layer._slice_dims(
            point=(0,) * ndim, ndisplay=ndisplay, order=tuple(range(ndim))
        )
    else:
        layer._slice_dims(Dims(ndim=ndim, ndisplay=ndisplay))


def _world_mesh_vertices(layer):
    """The vertices of the mesh of ``layer`` in world coordinates."""
    scale = np.asarray(layer.scale)[list(layer_dims_displayed(layer))]
    return layer._data_view._mesh.vertices * scale


def _scaled_layer(scale, ndisplay, edge_width=3):
    layer = BoundingBoxLayer(BOXES, scale=scale, edge_width=edge_width)
    _set_ndisplay(layer, ndisplay)
    return layer


def _layer_with_scaled_data(scale, ndisplay, edge_width=3):
    """What the layer with a scale should look like in world coordinates."""
    boxes = [box * np.asarray(scale) for box in BOXES]
    layer = BoundingBoxLayer(boxes, edge_width=edge_width)
    _set_ndisplay(layer, ndisplay)
    return layer


@pytest.mark.parametrize('ndisplay', [2, 3])
@pytest.mark.parametrize('scale', SCALES)
def test_edges_are_the_same_in_world_coordinates(scale, ndisplay):
    layer = _scaled_layer(scale, ndisplay)
    expected = _layer_with_scaled_data(scale, ndisplay)

    np.testing.assert_allclose(
        _world_mesh_vertices(layer), _world_mesh_vertices(expected), atol=1e-6
    )


@pytest.mark.parametrize('ndisplay', [2, 3])
@pytest.mark.parametrize('scale', SCALES)
def test_edges_follow_the_scale_of_the_layer(scale, ndisplay):
    layer = BoundingBoxLayer(BOXES, edge_width=3)
    _set_ndisplay(layer, ndisplay)

    layer.scale = scale

    expected = _layer_with_scaled_data(scale, ndisplay)
    np.testing.assert_allclose(
        _world_mesh_vertices(layer), _world_mesh_vertices(expected), atol=1e-6
    )


def test_edges_of_a_box_added_to_a_scaled_layer():
    scale = (4, 1, 1)
    layer = BoundingBoxLayer(
        BOXES[:1], scale=scale, edge_width=3
    )
    _set_ndisplay(layer, 3)

    layer.add(BOXES[1])

    expected = _layer_with_scaled_data(scale, 3)
    np.testing.assert_allclose(
        _world_mesh_vertices(layer), _world_mesh_vertices(expected), atol=1e-6
    )


def test_edges_of_a_layer_without_scale_are_not_changed():
    layer = _scaled_layer((1, 1, 1), 3)
    expected = BoundingBoxLayer(BOXES, edge_width=3)
    _set_ndisplay(expected, 3)

    np.testing.assert_allclose(
        layer._data_view._mesh.vertices, expected._data_view._mesh.vertices
    )
