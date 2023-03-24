__version__ = "0.0.2"

from napari.layers import NAMES
NAMES.add("bounding_boxes")
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


npe2.manifest.contributions._writers.LayerType = LayerType
__all__ = ["BoundingBoxCreator", "BoundingBoxLayer", "napari_get_reader", "write_single_bbox"]