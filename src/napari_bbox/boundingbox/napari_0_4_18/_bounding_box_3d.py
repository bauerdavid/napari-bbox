"""Geometry helpers for editing axis-aligned bounding boxes in 3D views.

All functions work on the displayed (3) dimensions of the layer data. A box is
described by its lower and upper corners ``lo`` and ``hi``. Face handles are
indexed as ``2 * axis + side`` where ``side`` is 0 for the face at ``lo`` and 1
for the face at ``hi``.
"""
import numpy as np

from ..napari_0_4_15._bounding_box_utils import find_bbox_corners

N_HANDLES = 6

# Handles whose axis is closer than this to the view direction are hidden,
# as dragging them would barely move the mouse along their axis
HANDLE_HIDE_COS = np.cos(np.deg2rad(15))


def box_bounds(data, dims):
    """Lower and upper corners of a bounding box in the given dimensions."""
    data = np.asarray(data)[:, dims]
    return data.min(axis=0), data.max(axis=0)


def set_box_bounds(data, dims, lo, hi):
    """Corner vertices of ``data`` with the bounds in ``dims`` replaced."""
    data = np.asarray(data)
    lo_nd = data.min(axis=0)
    hi_nd = data.max(axis=0)
    lo_nd[dims] = lo
    hi_nd[dims] = hi
    return find_bbox_corners(np.stack([lo_nd, hi_nd]))


def ray_box_intersection(origin, direction, lo, hi):
    """Entry and exit ray parameters of a ray with an axis-aligned box.

    Returns ``None`` if the infinite line along the ray misses the box.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        t0 = (lo - origin) / direction
        t1 = (hi - origin) / direction
    parallel = direction == 0
    inside = (origin >= lo) & (origin <= hi)
    if np.any(parallel & ~inside):
        return None
    t_near = np.where(parallel, -np.inf, np.minimum(t0, t1))
    t_far = np.where(parallel, np.inf, np.maximum(t0, t1))
    t_in, t_out = t_near.max(), t_far.min()
    if t_in > t_out:
        return None
    return t_in, t_out


def pick_box(origin, direction, los, his):
    """Index of the box under a ray.

    The box hit first along the ray wins, except that a box nested along the
    ray inside the winner takes precedence. Boxes are drawn as wireframes, so
    an inner box is visible through its enclosing box and must stay pickable.
    """
    hits = []
    for i, (lo, hi) in enumerate(zip(los, his)):
        hit = ray_box_intersection(origin, direction, lo, hi)
        if hit is not None:
            hits.append((hit[0], hit[1], i))
    if not hits:
        return None
    hits.sort(key=lambda h: (h[0], -h[2]))
    t_in, t_out, best = hits[0]
    for h_in, h_out, i in hits[1:]:
        if h_in >= t_in and h_out <= t_out:
            t_in, t_out, best = h_in, h_out, i
    return best


def face_handles(lo, hi):
    """(6, 3) array with the center of each face of a box."""
    center = (lo + hi) / 2
    handles = np.tile(center, (N_HANDLES, 1))
    for k in range(N_HANDLES):
        axis, side = divmod(k, 2)
        handles[k, axis] = hi[axis] if side else lo[axis]
    return handles


def visible_handles(view_direction):
    """Indices of handles that can be dragged along their axis."""
    if view_direction is None:
        return np.arange(N_HANDLES)
    axis_alignment = np.abs(np.asarray(view_direction))
    return np.array(
        [k for k in range(N_HANDLES) if axis_alignment[k // 2] < HANDLE_HIDE_COS],
        dtype=int,
    )


def ray_point_offsets(origin, direction, points):
    """Ray parameter of the closest approach to each point, and the offset."""
    direction = direction / np.linalg.norm(direction)
    t = (points - origin) @ direction
    closest = origin + t[:, None] * direction
    return t, points - closest


def resize_bounds(lo, hi, handle, distance, min_size):
    """Move the face of ``handle`` by ``distance`` without crossing the other."""
    lo, hi = lo.copy(), hi.copy()
    axis, side = divmod(handle, 2)
    if side:
        hi[axis] = max(hi[axis] + distance, lo[axis] + min_size)
    else:
        lo[axis] = min(lo[axis] + distance, hi[axis] - min_size)
    return lo, hi


def axis_parameter_under_ray(point, axis, ray_origin, ray_direction):
    """Parameter of the point on a line closest to a ray.

    The line is ``point + s * axis``. Following the closest point while the
    mouse moves keeps a dragged face handle under the cursor. Returns ``None``
    if the line is (nearly) parallel to the ray.
    """
    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    a = axis @ axis
    b = axis @ ray_direction
    denominator = a - b * b
    if denominator <= 1e-9 * a:
        return None
    offset = point - ray_origin
    return (b * (ray_direction @ offset) - axis @ offset) / denominator
