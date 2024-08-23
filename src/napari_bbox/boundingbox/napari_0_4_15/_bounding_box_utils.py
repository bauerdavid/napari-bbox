# A copy of napari.layers.shapes._shapes_utils
import itertools

import numpy as np

from vispy.geometry import PolygonData
from vispy.visuals.tube import _frenet_frames

from napari.layers.utils.layer_utils import segment_normal
from napari.utils.translations import trans


def orientation(p, q, r):
    """Determines oritentation of ordered triplet (p, q, r)

    Parameters
    ----------
    p : (2,) array
        Array of first point of triplet
    q : (2,) array
        Array of second point of triplet
    r : (2,) array
        Array of third point of triplet

    Returns
    -------
    val : int
        One of (-1, 0, 1). 0 if p, q, r are collinear, 1 if clockwise, and -1
        if counterclockwise.
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    val = np.sign(val)

    return val


def is_collinear(points):
    """Determines if a list of 2D points are collinear.

    Parameters
    ----------
    points : (N, 2) array
        Points to be tested for collinearity

    Returns
    -------
    val : bool
        True is all points are collinear, False otherwise.
    """
    if len(points) < 3:
        return True

    # The collinearity test takes three points, the first two are the first
    # two in the list, and then the third is iterated through in the loop
    return all(orientation(points[0], points[1], p) == 0 for p in points[2:])


def create_box(data):
    """Creates the axis aligned interaction box of a list of points

    Parameters
    ----------
    data : np.ndarray
        Nx2 array of points whose interaction box is to be found

    Returns
    -------
    box : np.ndarray
        9x2 array of vertices of the interaction box. The first 8 points are
        the corners and midpoints of the box in clockwise order starting in the
        upper-left corner. The last point is the center of the box
    """
    min_val = [data[:, 0].min(axis=0), data[:, 1].min(axis=0)]
    max_val = [data[:, 0].max(axis=0), data[:, 1].max(axis=0)]
    tl = np.array([min_val[0], min_val[1]])
    tr = np.array([max_val[0], min_val[1]])
    br = np.array([max_val[0], max_val[1]])
    bl = np.array([min_val[0], max_val[1]])
    box = np.array(
        [
            tl,
            (tl + tr) / 2,
            tr,
            (tr + br) / 2,
            br,
            (br + bl) / 2,
            bl,
            (bl + tl) / 2,
            (tl + tr + br + bl) / 4,
        ]
    )
    return box


def rectangle_to_box(data):
    """Converts the four corners of a rectangle into a interaction box like
    representation. If the rectangle is not axis aligned the resulting box
    representation will not be axis aligned either

    Parameters
    ----------
    data : np.ndarray
        4xD array of corner points to be converted to a box like representation

    Returns
    -------
    box : np.ndarray
        9xD array of vertices of the interaction box. The first 8 points are
        the corners and midpoints of the box in clockwise order starting in the
        upper-left corner. The last point is the center of the box
    """
    if not data.shape[0] == 4:
        raise ValueError(
            trans._(
                "Data shape does not match expected `[4, D]` shape specifying corners for the rectangle",
                deferred=True,
            )
        )
    box = np.array(
        [
            data[0],
            (data[0] + data[1]) / 2,
            data[1],
            (data[1] + data[2]) / 2,
            data[2],
            (data[2] + data[3]) / 2,
            data[3],
            (data[3] + data[0]) / 2,
            data.mean(axis=0),
        ]
    )
    return box


def find_corners(data):
    """Finds the four corners of the interaction box defined by an array of
    points

    Parameters
    ----------
    data : np.ndarray
        Nx2 array of points whose interaction box is to be found

    Returns
    -------
    corners : np.ndarray
        4x2 array of corners of the bounding box
    """
    min_val = data.min(axis=0)
    max_val = data.max(axis=0)
    tl = np.array([min_val[0], min_val[1]])
    tr = np.array([max_val[0], min_val[1]])
    br = np.array([max_val[0], max_val[1]])
    bl = np.array([min_val[0], max_val[1]])
    corners = np.array([tl, tr, br, bl])
    return corners


def find_bbox_corners(data):
    min_val = data.min(axis=0, keepdims=True)
    max_val = data.max(axis=0, keepdims=True)
    border_vals = np.concatenate([min_val, max_val])
    dims = border_vals.shape[-1]
    corners = np.where(list(map(tuple, itertools.product([False, True], repeat=dims))), max_val, min_val)
    return corners


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
    vertices, triangles = PolygonData(vertices=data).triangulate()
    triangles = triangles.astype(np.uint32)

    return vertices, triangles


def generate_tube_meshes(path, closed=False, tube_points=10):
    """Generates list of mesh vertices and triangles from a path

    Adapted from vispy.visuals.TubeVisual
    https://github.com/vispy/vispy/blob/master/vispy/visuals/tube.py

    Parameters
    ----------
    path : (N, 3) array
        Vertices specifying the path.
    closed : bool
        Bool which determines if the path is closed or not.
    tube_points : int
        The number of points in the circle-approximating polygon of the
        tube's cross section.

    Returns
    -------
    centers : (M, 3) array
        Vertices of all triangles for the lines
    offsets : (M, D) array
        offsets of all triangles for the lines
    triangles : (P, 3) array
        Vertex indices that form the mesh triangles
    """
    points = np.array(path).astype(float)

    if closed and not np.all(points[0] == points[-1]):
        points = np.concatenate([points, [points[0]]], axis=0)

    tangents, normals, binormals = _frenet_frames(points, closed)

    segments = len(points) - 1

    # get the positions of each vertex
    grid = np.zeros((len(points), tube_points, 3))
    grid_off = np.zeros((len(points), tube_points, 3))
    for i in range(len(points)):
        pos = points[i]
        normal = normals[i]
        binormal = binormals[i]

        # Add a vertex for each point on the circle
        v = np.arange(tube_points, dtype=float) / tube_points * 2 * np.pi
        cx = -1.0 * np.cos(v)
        cy = np.sin(v)
        grid[i] = pos
        grid_off[i] = cx[:, np.newaxis] * normal + cy[:, np.newaxis] * binormal

    # construct the mesh
    indices = []
    for i in range(segments):
        for j in range(tube_points):
            ip = (i + 1) % segments if closed else i + 1
            jp = (j + 1) % tube_points

            index_a = i * tube_points + j
            index_b = ip * tube_points + j
            index_c = ip * tube_points + jp
            index_d = i * tube_points + jp

            indices.append([index_a, index_b, index_d])
            indices.append([index_b, index_c, index_d])
    triangles = np.array(indices, dtype=np.uint32)

    centers = grid.reshape(grid.shape[0] * grid.shape[1], 3)
    offsets = grid_off.reshape(grid_off.shape[0] * grid_off.shape[1], 3)

    return centers, offsets, triangles


def triangulate_edge(path, closed=False):
    """Determines the triangulation of a path. The resulting `offsets` can
    multiplied by a `width` scalar and be added to the resulting `centers`
    to generate the vertices of the triangles for the triangulation, i.e.
    `vertices = centers + width*offsets`. Using the `centers` and `offsets`
    representation thus allows for the computed triangulation to be
    independent of the line width.

    Parameters
    ----------
    path : np.ndarray
        Nx2 or Nx3 array of central coordinates of path to be triangulated
    closed : bool
        Bool which determines if the path is closed or not.

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

    path = np.asanyarray(path)

    # Remove any equal adjacent points
    if len(path) > 2:
        idx = np.concatenate([[True], ~np.all(path[1:] == path[:-1], axis=-1)])
        clean_path = path[idx].copy()

        if clean_path.shape[0] == 1:
            clean_path = np.concatenate((clean_path, clean_path), axis=0)
    else:
        clean_path = path

    if clean_path.shape[-1] == 2:
        centers, offsets, triangles = generate_2D_edge_meshes(
            clean_path, closed=closed
        )
    else:
        centers, offsets, triangles = generate_tube_meshes(
            clean_path, closed=closed
        )

    return centers, offsets, triangles


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
    clean_path = np.array(path).astype(float)

    if closed:
        if np.all(clean_path[0] == clean_path[-1]) and len(clean_path) > 2:
            clean_path = clean_path[:-1]
        full_path = np.concatenate(
            ([clean_path[-1]], clean_path, [clean_path[0]]), axis=0
        )
        normals = [
            segment_normal(full_path[i], full_path[i + 1])
            for i in range(len(clean_path))
        ]
        normals = np.array(normals)
        full_path = np.concatenate((clean_path, [clean_path[0]]), axis=0)
        full_normals = np.concatenate((normals, [normals[0]]), axis=0)
    else:
        full_path = np.concatenate((clean_path, [clean_path[-2]]), axis=0)
        normals = [
            segment_normal(full_path[i], full_path[i + 1])
            for i in range(len(clean_path))
        ]
        normals[-1] = -normals[-1]
        normals = np.array(normals)
        full_path = clean_path
        full_normals = np.concatenate(([normals[0]], normals), axis=0)

    miters = np.array(
        [full_normals[i : i + 2].mean(axis=0) for i in range(len(full_path))]
    )
    miters = np.array(
        [
            miters[i] / np.dot(miters[i], full_normals[i])
            if np.dot(miters[i], full_normals[i]) != 0
            else full_normals[i]
            for i in range(len(full_path))
        ]
    )
    miter_lengths = np.linalg.norm(miters, axis=1)
    miters = 0.5 * miters
    vertex_offsets = []
    central_path = []
    triangles = []
    m = 0

    for i in range(len(full_path)):
        if i == 0:
            if (bevel or miter_lengths[i] > limit) and closed:
                offset = np.array([miters[i, 1], -miters[i, 0]])
                offset = 0.5 * offset / np.linalg.norm(offset)
                flip = np.sign(np.dot(offset, full_normals[i]))
                vertex_offsets.append(offset)
                vertex_offsets.append(
                    -flip * miters[i] / miter_lengths[i] * limit
                )
                vertex_offsets.append(-offset)
                central_path.append(full_path[i])
                central_path.append(full_path[i])
                central_path.append(full_path[i])
                triangles.append([0, 1, 2])
                m = m + 1
            else:
                vertex_offsets.append(-miters[i])
                vertex_offsets.append(miters[i])
                central_path.append(full_path[i])
                central_path.append(full_path[i])
        elif i == len(full_path) - 1:
            if closed:
                a = vertex_offsets[m + 1]
                b = vertex_offsets[1]
                ray = full_path[i] - full_path[i - 1]
                if np.cross(a, ray) * np.cross(b, ray) > 0:
                    triangles.append([m, m + 1, 1])
                    triangles.append([m, 0, 1])
                else:
                    triangles.append([m, m + 1, 1])
                    triangles.append([m + 1, 0, 1])
            else:
                vertex_offsets.append(-miters[i])
                vertex_offsets.append(miters[i])
                central_path.append(full_path[i])
                central_path.append(full_path[i])
                a = vertex_offsets[m + 1]
                b = vertex_offsets[m + 3]
                ray = full_path[i] - full_path[i - 1]
                if np.cross(a, ray) * np.cross(b, ray) > 0:
                    triangles.append([m, m + 1, m + 3])
                    triangles.append([m, m + 2, m + 3])
                else:
                    triangles.append([m, m + 1, m + 3])
                    triangles.append([m + 1, m + 2, m + 3])
        elif bevel or miter_lengths[i] > limit:
            offset = np.array([miters[i, 1], -miters[i, 0]])
            offset = 0.5 * offset / np.linalg.norm(offset)
            flip = np.sign(np.dot(offset, full_normals[i]))
            vertex_offsets.append(offset)
            vertex_offsets.append(-flip * miters[i] / miter_lengths[i] * limit)
            vertex_offsets.append(-offset)
            central_path.append(full_path[i])
            central_path.append(full_path[i])
            central_path.append(full_path[i])
            a = vertex_offsets[m + 1]
            b = vertex_offsets[m + 3]
            ray = full_path[i] - full_path[i - 1]
            if np.cross(a, ray) * np.cross(b, ray) > 0:
                triangles.append([m, m + 1, m + 3])
                triangles.append([m, m + 2, m + 3])
            else:
                triangles.append([m, m + 1, m + 3])
                triangles.append([m + 1, m + 2, m + 3])
            triangles.append([m + 2, m + 3, m + 4])
            m = m + 3
        else:
            vertex_offsets.append(-miters[i])
            vertex_offsets.append(miters[i])
            central_path.append(full_path[i])
            central_path.append(full_path[i])
            a = vertex_offsets[m + 1]
            b = vertex_offsets[m + 3]
            ray = full_path[i] - full_path[i - 1]
            if np.cross(a, ray) * np.cross(b, ray) > 0:
                triangles.append([m, m + 1, m + 3])
                triangles.append([m, m + 2, m + 3])
            else:
                triangles.append([m, m + 1, m + 3])
                triangles.append([m + 1, m + 2, m + 3])
            m = m + 2
    centers = np.array(central_path)
    offsets = np.array(vertex_offsets)
    triangles = np.array(triangles)

    return centers, offsets, triangles


def path_to_mask(mask_shape, vertices):
    """Converts a path to a boolean mask with `True` for points lying along
    each edge.

    Parameters
    ----------
    mask_shape : array (2,)
        Shape of mask to be generated.
    vertices : array (N, 2)
        Vertices of the path.

    Returns
    -------
    mask : np.ndarray
        Boolean array with `True` for points along the path
    """
    mask = np.zeros(mask_shape, dtype=bool)
    vertices = np.round(
        np.clip(vertices, 0, np.subtract(mask_shape, 1))
    ).astype(int)
    for i in range(len(vertices) - 1):
        start = vertices[i]
        stop = vertices[i + 1]
        step = np.ceil(np.max(abs(stop - start))).astype(int)
        x_vals = np.linspace(start[0], stop[0], step)
        y_vals = np.linspace(start[1], stop[1], step)
        for x, y in zip(x_vals, y_vals):
            mask[int(x), int(y)] = 1
    return mask


def poly_to_mask(mask_shape, vertices):
    """Converts a polygon to a boolean mask with `True` for points
    lying inside the shape. Uses the bounding box of the vertices to reduce
    computation time.

    Parameters
    ----------
    mask_shape : np.ndarray | tuple
        1x2 array of shape of mask to be generated.
    vertices : np.ndarray
        Nx2 array of the vertices of the polygon.

    Returns
    -------
    mask : np.ndarray
        Boolean array with `True` for points inside the polygon
    """
    mask = np.zeros(mask_shape, dtype=bool)
    bottom = vertices.min(axis=0).astype('int')
    bottom = np.clip(bottom, 0, np.subtract(mask_shape, 1))
    top = np.ceil(vertices.max(axis=0)).astype('int')
    # top = np.append([top], [mask_shape], axis=0).min(axis=0)
    top = np.clip(top, 0, np.subtract(mask_shape, 1))
    if np.all(top > bottom):
        bb_mask = grid_points_in_poly(top - bottom, vertices - bottom)
        mask[bottom[0] : top[0], bottom[1] : top[1]] = bb_mask
    return mask


def grid_points_in_poly(shape, vertices):
    """Converts a polygon to a boolean mask with `True` for points
    lying inside the shape. Loops through all indices in the grid

    Parameters
    ----------
    shape : np.ndarray | tuple
        1x2 array of shape of mask to be generated.
    vertices : np.ndarray
        Nx2 array of the vertices of the polygon.

    Returns
    -------
    mask : np.ndarray
        Boolean array with `True` for points inside the polygon
    """
    points = np.array(
        [(x, y) for x in range(shape[0]) for y in range(shape[1])], dtype=int
    )
    inside = points_in_poly(points, vertices)
    mask = inside.reshape(shape)
    return mask


def points_in_poly(points, vertices):
    """Tests points for being inside a polygon using the ray casting algorithm

    Parameters
    ----------
    points : np.ndarray
        Mx2 array of points to be tested
    vertices : np.ndarray
        Nx2 array of the vertices of the polygon.

    Returns
    -------
    inside : np.ndarray
        Length M boolean array with `True` for points inside the polygon
    """
    n_verts = len(vertices)
    inside = np.zeros(len(points), dtype=bool)
    j = n_verts - 1
    for i in range(n_verts):
        # Determine if a horizontal ray emanating from the point crosses the
        # line defined by vertices i-1 and vertices i.
        cond_1 = np.logical_and(
            vertices[i, 1] <= points[:, 1], points[:, 1] < vertices[j, 1]
        )
        cond_2 = np.logical_and(
            vertices[j, 1] <= points[:, 1], points[:, 1] < vertices[i, 1]
        )
        cond_3 = np.logical_or(cond_1, cond_2)
        d = vertices[j] - vertices[i]
        if d[1] == 0:
            # If y vertices are aligned avoid division by zero
            cond_4 = 0 < d[0] * (points[:, 1] - vertices[i, 1])
        else:
            cond_4 = points[:, 0] < (
                d[0] * (points[:, 1] - vertices[i, 1]) / d[1] + vertices[i, 0]
            )
        cond_5 = np.logical_and(cond_3, cond_4)
        inside[cond_5] = 1 - inside[cond_5]
        j = i

    # If the number of crossings is even then the point is outside the polygon,
    # if the number of crossings is odd then the point is inside the polygon

    return inside


def get_bounding_box_ndim(data):
    """Checks whether data is a list of bounding boxes or one shape.

    Parameters
    ----------
    data : (N, ) list of array
        List of bounding box data, where each element is an (N, D) array of the
        N vertices of a bounding box in D dimensions.

    Returns
    -------
    ndim : int
        Dimensionality of the bounding box/es in data
    """
    # list of bounding boxes
    if np.array(data).ndim == 3:
        ndim = np.array(data).shape[2]
    # just one shape
    else:
        ndim = np.array(data).shape[1]
    # list of different shapes
    return ndim


def number_of_bounding_boxes(data):
    """Determine number of bounding boxes in the data.

    Parameters
    ----------
    data : list or np.ndarray
        Can either be no bounding boxes, if empty, a
        single bounding box or a list of bounding boxes.

    Returns
    -------
    n_bounding_boxes : int
        Number of new bounding box
    """
    if len(data) == 0:
        # If no new shapes
        n_bounding_boxes = 0
    elif np.array(data).ndim == 2:
        # If a single array for a shape
        n_bounding_boxes = 1
    else:
        n_bounding_boxes = len(data)

    return n_bounding_boxes


def validate_num_vertices(
    data
):
    """Raises error if a bounding box in data has invalid number of vertices.

    Checks whether all bounding boxes in data have a valid number of vertices.
    Bounding boxes should have 2**D vertices.

    One of valid_vertices or min_vertices must be passed to the
    function.

    Parameters
    ----------
    data : Array | Tuple(Array,str) | List[Array | Tuple(Array, str)] | Tuple(List[Array], str)
        List of bounding box data, where each element is an (N, D) array of the
        N vertices of a bounding box in D dimensions. Can be an 3-dimensional array
        if each bounding box has the same number of vertices.
    min_vertices : int or None
        Minimum number of vertices for the shape type, by default None
    valid_vertices : Tuple(int) or None
        Valid number of vertices for the shape type in data, by default None

    Raises
    ------
    ValueError
        Raised if a shape is found with invalid number of vertices
    """
    n_bounding_boxes = number_of_bounding_boxes(data)
    ndim = get_bounding_box_ndim(data)
    valid_vertices = [2**ndim]
    # single array of vertices
    if n_bounding_boxes == 1 and np.array(data).ndim == 2:
        # wrap in extra dimension so we can iterate through shape not vertices
        data = [data]
    for bounding_box in data:
        if len(bounding_box) not in valid_vertices:
            raise ValueError(
                trans._(
                    "{bounding_box} has invalid number of vertices: {shape_length}.",
                    deferred=True,
                    bounding_box=bounding_box,
                    shape_length=len(bounding_box),
                )
            )