__version__ = "0.0.7"
from packaging import version
from napari import layers
layers.NAMES.add("boundingboxlayer")

from napari import types
types.BoundingboxlayerData = types.NewType("BoundingboxlayerData", types.ArrayBase)


from .boundingbox import BoundingBoxLayer
from ._reader import napari_get_reader
from ._writer import write_single_bbox
from ._utils import NAPARI_VERSION
from ._widget import BoundingBoxCreator

from napari import Viewer

if NAPARI_VERSION >= version.parse("0.4.18"):

    def revert_last_dim_point_cb(viewer):
        def revert_last_dim_point(e):
            viewer.dims.set_point(range(len(e.source._last_dim_point)), e.source._last_dim_point)
            viewer.events.layers_change()
        return revert_last_dim_point

    def store_last_dim_point_cb(layer):
        def store_last_dim_point(e):
            p = tuple(round(c) for c in e.source.point)
            layer._store_last_dim_point(p)
        return store_last_dim_point

    def add_bounding_boxes(self, *args, **kwargs):
        layer = BoundingBoxLayer(*args, **kwargs)
        layer._store_last_dim_point(self.dims.point)
        store_cb = store_last_dim_point_cb(layer)
        layer.events.data.connect(revert_last_dim_point_cb(self))
        self.dims.events.current_step.connect(store_cb)

        def disconnect_all(e):
            if e.value is not layer:
                return
            self.dims.events.range.disconnect(store_cb)
            self.dims.events.current_step.disconnect(store_cb)
            self.layers.events.removed.disconnect(disconnect_all)
        self.layers.events.removed.connect(disconnect_all)
        self.layers.append(layer)
        return layer
else:
    def add_bounding_boxes(self, *args, **kwargs):
        layer = BoundingBoxLayer(*args, **kwargs)
        self.layers.append(layer)
        return layer


Viewer.add_bounding_boxes = add_bounding_boxes

from enum import Enum


class LayerType(str, Enum):
    image = "image"
    labels = "labels"
    points = "points"
    shapes = "shapes"
    surface = "surface"
    tracks = "tracks"
    vectors = "vectors"
    bounding_boxes = "bounding_boxes"


# npe2.manifest.contributions._writers.LayerType = LayerType

# This is an ugly solution to register every component correctly
from .boundingbox import register_layer_control
from .boundingbox import register_layer_visual
register_layer_control(BoundingBoxLayer)
register_layer_visual(BoundingBoxLayer)
# register_bounding_boxes_actions(BoundingBoxLayer)
import sys
layers.__dict__["Boundingboxlayer"] = BoundingBoxLayer
layers.__dict__["bounding_boxes"] = sys.modules[__name__]

__all__ = ["BoundingBoxCreator", "BoundingBoxLayer", "napari_get_reader", "write_single_bbox"]