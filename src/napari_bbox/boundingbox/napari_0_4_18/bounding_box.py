# A copy of napari.layers.shapes._shapes_models.shape
from ..napari_0_4_15.bounding_box import BoundingBox

import numpy as np


class BoundingBox(BoundingBox):
    def scale(self, scale, center=None):
        """Performs a scaling on the shape

        Parameters
        ----------
        scale : float, list
            scalar or list specifying rescaling of shape.
        center : list
            length 2 list specifying coordinate of center of scaling.
        """
        if isinstance(scale, (list, np.ndarray)):
            transform = np.array([[scale[0], 0], [0, scale[1]]])
        else:
            transform = np.array([[scale, 0], [0, scale]])
        if center is None:
            self.transform(transform)
        else:
            self.shift(-center)
            self.transform(transform)
            self.shift(center)
