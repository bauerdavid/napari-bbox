# A copy of napari._vispy.visuals.shapes
from vispy.scene.visuals import Compound, Line, Markers, Mesh, Text

from napari._vispy.visuals.clipping_planes_mixin import ClippingPlanesMixin


class BoundingBoxVisual(ClippingPlanesMixin, Compound):
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

    def __init__(self) -> None:
        super().__init__([Mesh(), Mesh(), Line(), Markers(), Text()])

    @property
    def bounding_box_faces(self) -> Mesh:
        """Mesh for bounding box faces"""
        return self._subvisuals[0]

    @property
    def bounding_box_highlights(self) -> Mesh:
        """Mesh for bounding box highlights"""
        return self._subvisuals[1]

    @property
    def highlight_lines(self) -> Line:
        """Lines for bounding box highlights"""
        return self._subvisuals[2]

    @property
    def highlight_vertices(self) -> Markers:
        """Vertices for bounding box highlights"""
        return self._subvisuals[3]

    @property
    def text(self) -> Text:
        """Text labels"""
        return self._subvisuals[4]
