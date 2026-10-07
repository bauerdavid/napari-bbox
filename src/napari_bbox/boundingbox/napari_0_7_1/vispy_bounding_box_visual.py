# A copy of napari._vispy.visuals.shapes
from vispy.scene.visuals import Compound, Line, Markers, Mesh

from napari._vispy.visuals.clipping_planes_mixin import ClippingPlanesMixin
from napari._vispy.visuals.text import Text

from ..napari_0_5_0.vispy_bounding_box_visual import BoundingBoxVisual


class BoundingBoxVisual(BoundingBoxVisual):
    """
    Compound vispy visual for shapes visualization with
    clipping planes functionality

    Components:
        - Mesh for bounding box faces (vispy.MeshVisual)
        - Mesh for highlights (vispy.MeshVisual)
        - Lines for highlights (vispy.LineVisual)
        - Vertices for highlights (vispy.MarkersVisual)
        - Text labels (vispy.TextVisual)
    """

    def __init__(self, font_info) -> None:
        ClippingPlanesMixin.__init__(
            self,
            [
                Mesh(),
                Mesh(),
                Line(antialias=True),
                Markers(),
                Text(font_info=font_info),
            ],
            font_info=font_info,
        )
