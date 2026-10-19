import numpy as np
import pytest

from napari_bbox import BoundingBoxLayer


def _box(box_min, box_max):
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)])
    return np.where(corners, box_max, box_min).astype(float)


@pytest.fixture
def viewer():
    import napari
    viewer = napari.Viewer(show=False)
    viewer.add_image(np.zeros((10, 60, 60)))
    viewer.dims.set_point(0, 3)
    yield viewer
    viewer.close()


def _add_layer_with_labels(viewer, n_boxes):
    layer = BoundingBoxLayer(ndim=3)
    viewer.add_layer(layer)
    layer.data = [_box((1, 5, 5), (4, 20, 20)), _box((2, 25, 25), (6, 50, 50))][:n_boxes]
    layer.features = {"label": np.arange(1, n_boxes + 1)}
    layer.text = {"text": "{label:d}"}
    assert list(layer._indices_view) == list(range(n_boxes))
    return layer


def test_removing_the_last_bounding_box_clears_the_displayed_state(viewer):
    layer = _add_layer_with_labels(viewer, 1)
    layer.selected_data = {0}
    layer.remove_selected()
    assert len(layer.data) == 0
    assert len(layer._indices_view) == 0
    # used to fail when drawing: the text of the removed bounding box was still requested
    assert len(layer._view_text) == 0


def test_removing_a_bounding_box_keeps_the_others_displayed(viewer):
    layer = _add_layer_with_labels(viewer, 2)
    layer.selected_data = {0}
    layer.remove_selected()
    assert len(layer.data) == 1
    assert list(layer._indices_view) == [0]
    assert list(layer._view_text) == ["2"]
