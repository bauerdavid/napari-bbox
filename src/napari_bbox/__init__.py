__version__ = "0.0.3"

from napari import layers
layers.NAMES.add("bounding_boxes")

from napari import types
types.Bounding_BoxesData = types.NewType("Bounding_BoxesData", types.ArrayBase)


from .boundingbox import BoundingBoxLayer
from ._reader import napari_get_reader
from ._writer import write_single_bbox

from ._widget import BoundingBoxCreator

from napari import Viewer
def add_bounding_boxes(self, *args, **kwargs):
    layer = BoundingBoxLayer(*args, **kwargs)
    self.layers.append(layer)
    return layer
Viewer.add_bounding_boxes = add_bounding_boxes

import npe2.manifest.contributions._writers
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
from .boundingbox.qt_bounding_box_control import register_layer_control
from .boundingbox.vispy_bounding_box_layer import register_layer_visual
register_layer_control(BoundingBoxLayer)
register_layer_visual(BoundingBoxLayer)
# register_bounding_boxes_actions(BoundingBoxLayer)
import sys
layers.__dict__["BoundingBoxLayer"] = BoundingBoxLayer
layers.__dict__["bounding_boxes"] = sys.modules[__name__]

__all__ = ["BoundingBoxCreator", "BoundingBoxLayer", "napari_get_reader", "write_single_bbox"]