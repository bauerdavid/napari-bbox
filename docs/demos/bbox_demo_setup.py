"""Sample data for the napari-bbox demos in this folder.

The demos are rendered with napari-demo-recorder, see the ``*.demo.yaml`` files.
"""
from __future__ import annotations

import numpy as np


# (y, x, radius) of the cells of the 2D image
CELLS_2D = [
    (55, 60, 10), (60, 175, 11), (110, 110, 9), (150, 50, 10),
    (165, 185, 10), (205, 100, 11), (215, 205, 9), (100, 215, 8),
]


def make_cells_2d(shape=(256, 256), seed=4):
    """A deterministic image of bright blobs ("cells")."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    image = np.zeros(shape, float)
    for cy, cx, r in CELLS_2D:
        image += rng.uniform(0.7, 1.0) * np.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (r * 0.7) ** 2)
        )
    image += rng.normal(0, 0.03, shape)
    return np.clip(image, 0, None).astype(np.float32)


def setup_2d(viewer):
    """A 2D image, to draw and edit bounding boxes on."""
    image = make_cells_2d()
    viewer.add_image(image, name="cells", colormap="magma")


# (z, y, x, radius) of the cells of the 3D volume
CELLS_3D = [
    (20, 50, 50, 12), (28, 60, 120, 14), (14, 115, 85, 11),
    (32, 120, 150, 12), (22, 90, 20, 9), (12, 25, 95, 10),
]


def make_cells_3d(shape=(48, 160, 160), seed=7):
    """A deterministic volume of bright blobs ("cells")."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.mgrid[: shape[0], : shape[1], : shape[2]]
    volume = np.zeros(shape, float)
    for cz, cy, cx, r in CELLS_3D:
        volume += rng.uniform(0.7, 1.0) * np.exp(
            -((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (r * 0.6) ** 2)
        )
    volume += rng.normal(0, 0.02, shape)
    return np.clip(volume, 0, None).astype(np.float32)


def setup_3d(viewer):
    """A 3D volume with a bounding box layer that has a box around a few cells."""
    import napari_bbox  # noqa: F401  (adds viewer.add_bounding_boxes)

    viewer.add_image(
        make_cells_3d(), name="cells", colormap="magma", rendering="attenuated_mip"
    )
    boxes = []
    for index in (0, 1, 2):
        cz, cy, cx, r = CELLS_3D[index]
        boxes.append(
            np.array([[cz - r, cy - r, cx - r], [cz + r, cy + r, cx + r]], float)
        )
    layer = viewer.add_bounding_boxes(
        boxes, name="cells boxes", edge_color="yellow", edge_width=0.6
    )
    viewer.layers.selection.active = layer
