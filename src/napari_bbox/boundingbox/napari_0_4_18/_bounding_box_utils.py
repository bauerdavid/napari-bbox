# A copy of napari.layers.shapes._shapes_utils
from __future__ import annotations
from ..napari_0_4_15._bounding_box_utils import (
    orientation,
    is_collinear,
    create_box,
    rectangle_to_box,
    find_corners,
    find_bbox_corners,
    generate_tube_meshes,
    triangulate_edge,
    path_to_mask,
    poly_to_mask,
    grid_points_in_poly,
    points_in_poly,
    get_bounding_box_ndim,
    number_of_bounding_boxes,
    validate_num_vertices
)

from typing import TYPE_CHECKING
import numpy as np
from vispy.geometry import PolygonData

from napari.layers.utils.layer_utils import segment_normal

if TYPE_CHECKING:
    import numpy.typing as npt

try:
    # see https://github.com/vispy/vispy/issues/1029
    from triangle import triangulate
except ModuleNotFoundError:
    triangulate = None


def triangulate_face(data):
    """Determines the triangulation of the face of a bounding box.

    Parameters
    ----------
    data : np.ndarray
        Nx2 array of vertices of bounding box to be triangulated

    Returns
    -------
    vertices : np.ndarray
        Mx2 array vertices of the triangles.
    triangles : np.ndarray
        Px3 array of the indices of the vertices that will form the
        triangles of the triangulation
    """

    if triangulate is not None:
        len_data = len(data)

        edges = np.empty((len_data, 2), dtype=np.uint32)
        edges[:, 0] = np.arange(len_data)
        edges[:, 1] = np.arange(1, len_data + 1)
        # connect last with first vertex
        edges[-1, 1] = 0

        res = triangulate({"vertices": data, "segments": edges}, "p")
        vertices, triangles = res['vertices'], res['triangles']
    else:
        vertices, triangles = PolygonData(vertices=data).triangulate()

    triangles = triangles.astype(np.uint32)

    return vertices, triangles


def _mirror_point(x, y):
    return 2 * y - x


def _sign_nonzero(x):
    y = np.sign(x).astype(int)
    y[y == 0] = 1
    return y


def _sign_cross(x, y):
    """sign of cross product (faster for 2d)"""
    if x.shape[1] == y.shape[1] == 2:
        return _sign_nonzero(x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0])
    if x.shape[1] == y.shape[1] == 3:
        return _sign_nonzero(np.cross(x, y))

    raise ValueError(x.shape[1], y.shape[1])


def generate_2D_edge_meshes(path, closed=False, limit=3, bevel=False):
    """Determines the triangulation of a path in 2D. The resulting `offsets`
    can be multiplied by a `width` scalar and be added to the resulting
    `centers` to generate the vertices of the triangles for the triangulation,
    i.e. `vertices = centers + width*offsets`. Using the `centers` and
    `offsets` representation thus allows for the computed triangulation to be
    independent of the line width.

    Parameters
    ----------
    path : np.ndarray
        Nx2 or Nx3 array of central coordinates of path to be triangulated
    closed : bool
        Bool which determines if the path is closed or not
    limit : float
        Miter limit which determines when to switch from a miter join to a
        bevel join
    bevel : bool
        Bool which if True causes a bevel join to always be used. If False
        a bevel join will only be used when the miter limit is exceeded

    Returns
    -------
    centers : np.ndarray
        Mx2 or Mx3 array central coordinates of path triangles.
    offsets : np.ndarray
        Mx2 or Mx3 array of the offsets to the central coordinates that need to
        be scaled by the line width and then added to the centers to
        generate the actual vertices of the triangulation
    triangles : np.ndarray
        Px3 array of the indices of the vertices that will form the
        triangles of the triangulation
    """

    path = np.asarray(path, dtype=float)

    # add first vertex to the end if closed
    if closed:
        path = np.concatenate((path, [path[0]]))

    # extend path by adding a vertex at beginning and end
    # to get the mean normals correct
    if closed:
        _ext_point1 = path[-2]
        _ext_point2 = path[1]
    else:
        _ext_point1 = _mirror_point(path[1], path[0])
        _ext_point2 = _mirror_point(path[-2], path[-1])

    full_path = np.concatenate(([_ext_point1], path, [_ext_point2]), axis=0)

    # full_normals[:-1], full_normals[1:] are normals left and right of each path vertex
    full_normals = segment_normal(full_path[:-1], full_path[1:])

    # miters per vertex are the average of left and right normals
    miters = 0.5 * (full_normals[:-1] + full_normals[1:])

    # scale miters such that their dot product with normals is 1
    _mf_dot = np.expand_dims(
        np.einsum('ij,ij->i', miters, full_normals[:-1]), -1
    )

    miters = np.divide(
        miters,
        _mf_dot,
        where=np.abs(_mf_dot) > 1e-10,
    )

    miter_lengths_squared = (miters**2).sum(axis=1)

    # miter_signs -> +1 if edges turn clockwise, -1 if anticlockwise
    # used later to discern bevel positions
    miter_signs = _sign_cross(full_normals[1:], full_normals[:-1])
    miters = 0.5 * miters

    # generate centers/offsets
    centers = np.repeat(path, 2, axis=0)
    offsets = np.repeat(miters, 2, axis=0)
    offsets[::2] *= -1

    triangles0 = np.tile(np.array([[0, 1, 3], [0, 3, 2]]), (len(path) - 1, 1))
    triangles = triangles0 + 2 * np.repeat(
        np.arange(len(path) - 1)[:, np.newaxis], 2, 0
    )

    # get vertex indices that are to be beveled
    idx_bevel = np.where(
        np.bitwise_or(bevel, miter_lengths_squared > (limit**2))
    )[0]

    if len(idx_bevel) > 0:
        # only the 'outwards sticking' offsets should be changed
        # TODO: This is not entirely true as in extreme cases both can go to infinity
        idx_offset = (miter_signs[idx_bevel] < 0).astype(int)
        idx_bevel_full = 2 * idx_bevel + idx_offset
        sign_bevel = np.expand_dims(miter_signs[idx_bevel], -1)

        # adjust offset of outer "left" vertex
        offsets[idx_bevel_full] = (
            -0.5 * full_normals[:-1][idx_bevel] * sign_bevel
        )

        # special cases for the last vertex
        _nonspecial = idx_bevel != len(path) - 1

        idx_bevel = idx_bevel[_nonspecial]
        idx_bevel_full = idx_bevel_full[_nonspecial]
        sign_bevel = sign_bevel[_nonspecial]
        idx_offset = idx_offset[_nonspecial]

        # create new "right" bevel vertices to be added later
        centers_bevel = path[idx_bevel]
        offsets_bevel = -0.5 * full_normals[1:][idx_bevel] * sign_bevel

        n_centers = len(centers)
        # change vertices of triangles to the newly added right vertices
        triangles[2 * idx_bevel, idx_offset] = len(centers) + np.arange(
            len(idx_bevel)
        )
        triangles[
            2 * idx_bevel + (1 - idx_offset), idx_offset
        ] = n_centers + np.arange(len(idx_bevel))

        # add center triangle
        triangles0 = np.tile(np.array([[0, 1, 2]]), (len(idx_bevel), 1))

        triangles_bevel = np.array(
            [
                2 * idx_bevel + idx_offset,
                2 * idx_bevel + (1 - idx_offset),
                n_centers + np.arange(len(idx_bevel)),
            ]
        ).T

        # add all new centers, offsets, and triangles
        centers = np.concatenate([centers, centers_bevel])
        offsets = np.concatenate([offsets, offsets_bevel])
        triangles = np.concatenate([triangles, triangles_bevel])

    # extracting vectors (~4x faster than np.moveaxis)
    a, b, c = tuple((centers + offsets)[triangles][:, i] for i in range(3))
    # flip negative oriented triangles
    flip_idx = _sign_cross(b - a, c - a) < 0
    triangles[flip_idx] = np.flip(triangles[flip_idx], axis=-1)

    return centers, offsets, triangles
