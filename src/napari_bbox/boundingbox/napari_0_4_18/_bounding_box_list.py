# A copy of napari.layers.shapes._shape_list
from ..napari_0_4_15._bounding_box_list import BoundingBoxList
from collections.abc import Iterable
from typing import Sequence, Union

import numpy as np

from napari.utils.geometry import (
    inside_triangles,
    intersect_line_with_triangles,
    line_in_triangles_3d,
)
from napari.utils.translations import trans
from .bounding_box import BoundingBox


class BoundingBoxList(BoundingBoxList):
    """List of bounding boxes class.

    Parameters
    ----------
    data : list
        List of BoundingBox objects
    ndisplay : int
        Number of displayed dimensions.

    Attributes
    ----------
    bounding_boxes : (N, ) list
        Bounding box objects.
    data : (N, ) list of (M, D) array
        Data arrays for each bounding box.
    ndisplay : int
        Number of displayed dimensions.
    slice_keys : (N, 2, P) array
        Array of slice keys for each bounding box. Each slice key has the min and max
        values of the P non-displayed dimensions, useful for slicing
        multidimensional bounding boxes. If the both min and max values of bounding box are
        equal then the bounding box is entirely contained within the slice specified
        by those values.
    edge_color : (N x 4) np.ndarray
        Array of RGBA edge colors for each bounding box.
    face_color : (N x 4) np.ndarray
        Array of RGBA face colors for each bounding box.
    edge_widths : (N, ) list of float
        Edge width for each bounding box.
    z_indices : (N, ) list of int
        z-index for each bounding box.

    Notes
    -----
    _vertices : np.ndarray
        Mx2 array of all displayed vertices from all bounding boxes
    _index : np.ndarray
        Length M array with the index (0, ..., N-1) of each bounding box that each
        vertex corresponds to
    _z_index : np.ndarray
        Length N array with z_index of each bounding box
    _z_order : np.ndarray
        Length N array with z_order of each bounding box. This must be a permutation
        of (0, ..., N-1).
    _mesh : Mesh
        Mesh object containing all the mesh information that will ultimately
        be rendered.
    """


    def _set_color(self, colors, attribute):
        """Set the face_color or edge_color property

        Parameters
        ----------
        colors : (N, 4) np.ndarray
            The value for setting edge or face_color. There must
            be one color for each bounding box
        attribute : str in {'edge', 'face'}
            The name of the attribute to set the color of.
            Should be 'edge' for edge_color or 'face' for face_color.
        """
        n_bounding_boxes = len(self.data)
        if not np.all(colors.shape == (n_bounding_boxes, 4)):
            raise ValueError(
                trans._(
                    '{attribute}_color must have shape ({n_bounding_boxes}, 4)',
                    deferred=True,
                    attribute=attribute,
                    n_bounding_boxes=n_bounding_boxes,
                )
            )

        update_method = getattr(self, f'update_{attribute}_colors')
        indices = np.arange(len(colors))
        update_method(indices, colors, update=False)
        self._update_displayed()

    def _update_displayed(self):
        """Update the displayed data based on the slice key."""
        # The list slice key is repeated to check against both the min and
        # max values stored in the shapes slice key.
        slice_key = np.array(self.slice_key)
        if len(slice_key) != self.slice_keys.shape[-1]:
            return
        # Slice key must exactly match mins and maxs of shape as then the
        # shape is entirely contained within the current slice.
        if len(self.bounding_boxes) > 0:
            self._displayed = np.all(
                np.logical_and(self.slice_keys[:, 0, :] <= slice_key, slice_key <= self.slice_keys[:, 1, :]), axis=1)
        else:
            self._displayed = []
        disp_indices = np.where(self._displayed)[0]

        z_order = self._mesh.triangles_z_order
        disp_tri = np.isin(
            self._mesh.triangles_index[z_order, 0], disp_indices
        )
        self._mesh.displayed_triangles = self._mesh.triangles[z_order][
            disp_tri
        ]
        self._mesh.displayed_triangles_index = self._mesh.triangles_index[
            z_order
        ][disp_tri]
        self._mesh.displayed_triangles_colors = self._mesh.triangles_colors[
            z_order
        ][disp_tri]
        disp_vert = np.isin(self._index, disp_indices)
        self.displayed_vertices = self._vertices[disp_vert]
        self.displayed_index = self._index[disp_vert]

    def add(
        self,
        bounding_box: Union[BoundingBox, Sequence[BoundingBox]],
        face_color=None,
        edge_color=None,
        bounding_box_index=None,
        z_refresh=True,
    ):
        """Adds a single BoundingBox object

        Parameters
        ----------
        bounding_box : BoundingBox
            The bounding box to add
        face_color : color (or iterable of colors of same length as shape)
        edge_color : color (or iterable of colors of same length as shape)
        bounding_box_index : None | int
            If int then edits the bounding box date at current index. To be used in
            conjunction with `remove` when renumber is `False`. If None, then
            appends a new bounding box to end of bounding boxes list
        z_refresh : bool
            If set to true, the mesh elements are reindexed with the new z order.
            When bounding_box_index is provided, z_refresh will be overwritten to false,
            as the z indices will not change.
            When adding a batch of bounding boxes, set to false  and then call
            BoundingBoxList._update_z_order() once at the end.
        """
        # single shape mode
        if issubclass(type(bounding_box), BoundingBox):
            self._add_single_bounding_box(
                bounding_box=bounding_box,
                face_color=face_color,
                edge_color=edge_color,
                bounding_box_index=bounding_box_index,
                z_refresh=z_refresh,
            )
        # multiple shape mode
        elif isinstance(bounding_box, Iterable):
            if bounding_box_index is not None:
                raise ValueError(
                    trans._(
                        'bounding_box_index must be None when adding multiple bounding boxes',
                        deferred=True,
                    )
                )
            self._add_multiple_bounding_boxes(
                bounding_boxes=bounding_box,
                face_colors=face_color,
                edge_colors=edge_color,
                z_refresh=z_refresh,
            )
        else:
            raise TypeError(
                trans._(
                    'Cannot add single nor multiple bounding box',
                    deferred=True,
                )
            )

    def _add_single_bounding_box(
        self,
        bounding_box,
        face_color=None,
        edge_color=None,
        bounding_box_index=None,
        z_refresh=True,
    ):
        """Adds a single BoundingBox object

        Parameters
        ----------
        bounding_box : subclass BoundingBox
            Must be a BoundingBox
        bounding_box_index : None | int
            If int then edits the bounding box date at current index. To be used in
            conjunction with `remove` when renumber is `False`. If None, then
            appends a new bounding box to end of bounding boxes list
        z_refresh : bool
            If set to true, the mesh elements are reindexed with the new z order.
            When bounding_box_index is provided, z_refresh will be overwritten to false,
            as the z indices will not change.
            When adding a batch of bounding boxes, set to false  and then call
            BoundingBoxList._update_z_order() once at the end.
        """
        if not issubclass(type(bounding_box), BoundingBox):
            raise TypeError(
                trans._(
                    'bounding_box must be subclass of BoundingBox',
                    deferred=True,
                )
            )

        if bounding_box_index is None:
            bounding_box_index = len(self.bounding_boxes)
            self.bounding_boxes.append(bounding_box)
            self._z_index = np.append(self._z_index, bounding_box.z_index)

            if face_color is None:
                face_color = np.array([1, 1, 1, 1])
            self._face_color = np.vstack([self._face_color, face_color])
            if edge_color is None:
                edge_color = np.array([0, 0, 0, 1])
            self._edge_color = np.vstack([self._edge_color, edge_color])
        else:
            z_refresh = False
            self.bounding_boxes[bounding_box_index] = bounding_box
            self._z_index[bounding_box_index] = bounding_box.z_index

            if face_color is None:
                face_color = self._face_color[bounding_box_index]
            else:
                self._face_color[bounding_box_index, :] = face_color
            if edge_color is None:
                edge_color = self._edge_color[bounding_box_index]
            else:
                self._edge_color[bounding_box_index, :] = edge_color

        self._vertices = np.append(
            self._vertices, bounding_box.data_displayed, axis=0
        )
        index = np.repeat(bounding_box_index, len(bounding_box.data_displayed)) # TODO check
        self._index = np.append(self._index, index, axis=0)

        # Add faces to mesh
        m = len(self._mesh.vertices)
        vertices = bounding_box._face_vertices
        self._mesh.vertices = np.append(self._mesh.vertices, vertices, axis=0)
        vertices = bounding_box._face_vertices
        self._mesh.vertices_centers = np.append(
            self._mesh.vertices_centers, vertices, axis=0
        )
        vertices = np.zeros(bounding_box._face_vertices.shape)
        self._mesh.vertices_offsets = np.append(
            self._mesh.vertices_offsets, vertices, axis=0
        )
        index = np.repeat([[bounding_box_index, 0]], len(vertices), axis=0)
        self._mesh.vertices_index = np.append(
            self._mesh.vertices_index, index, axis=0
        )

        triangles = bounding_box._face_triangles + m
        self._mesh.triangles = np.append(
            self._mesh.triangles, triangles, axis=0
        )
        index = np.repeat([[bounding_box_index, 0]], len(triangles), axis=0)
        self._mesh.triangles_index = np.append(
            self._mesh.triangles_index, index, axis=0
        )
        color_array = np.repeat([face_color], len(triangles), axis=0)
        self._mesh.triangles_colors = np.append(
            self._mesh.triangles_colors, color_array, axis=0
        )

        # Add edges to mesh
        m = len(self._mesh.vertices)
        vertices = (
                bounding_box._edge_vertices + bounding_box.edge_width * bounding_box._edge_offsets
        )
        self._mesh.vertices = np.append(self._mesh.vertices, vertices, axis=0)
        vertices = bounding_box._edge_vertices
        self._mesh.vertices_centers = np.append(
            self._mesh.vertices_centers, vertices, axis=0
        )
        vertices = bounding_box._edge_offsets
        self._mesh.vertices_offsets = np.append(
            self._mesh.vertices_offsets, vertices, axis=0
        )
        index = np.repeat([[bounding_box_index, 1]], len(vertices), axis=0)
        self._mesh.vertices_index = np.append(
            self._mesh.vertices_index, index, axis=0
        )

        triangles = bounding_box._edge_triangles + m
        self._mesh.triangles = np.append(
            self._mesh.triangles, triangles, axis=0
        )
        index = np.repeat([[bounding_box_index, 1]], len(triangles), axis=0)
        self._mesh.triangles_index = np.append(
            self._mesh.triangles_index, index, axis=0
        )
        color_array = np.repeat([edge_color], len(triangles), axis=0)
        self._mesh.triangles_colors = np.append(
            self._mesh.triangles_colors, color_array, axis=0
        )

        if z_refresh:
            # Set z_order
            self._update_z_order()

    def _add_multiple_bounding_boxes(
        self,
        bounding_boxes,
        face_colors=None,
        edge_colors=None,
        z_refresh=True,
    ):
        """Add multiple bounding boxes at once (faster than adding them one by one)

        Parameters
        ----------
        bounding_boxes : iterable of BoundingBox
        face_colors : iterable of face_color
        edge_colors : iterable of edge_color
        z_refresh : bool
            If set to true, the mesh elements are reindexed with the new z order.
            When bounding_box_index is provided, z_refresh will be overwritten to false,
            as the z indices will not change.
            When adding a batch of bounding boxes, set to false  and then call
            BoundingBoxList._update_z_order() once at the end.

        TODO: Currently shares a lot of code with `add()`, with the
        difference being that `add()` supports inserting bounding boxes at a specific
        `bounding_box_index`, whereas `add_multiple` will append them as a full batch
        """

        def _make_index(length, bounding_box_index, cval=0):
            """Same but faster than `np.repeat([[bounding_box_index, cval]], length, axis=0)`"""
            index = np.empty((length, 2), np.int32)
            index.fill(cval)
            index[:, 0] = bounding_box_index
            return index

        all_z_index = []
        all_vertices = []
        all_index = []
        all_mesh_vertices = []
        all_mesh_vertices_centers = []
        all_mesh_vertices_offsets = []
        all_mesh_vertices_index = []
        all_mesh_triangles = []
        all_mesh_triangles_index = []
        all_mesh_triangles_colors = []

        m_mesh_vertices_count = len(self._mesh.vertices)

        if face_colors is None:
            face_colors = np.tile(np.array([1, 1, 1, 1]), (len(bounding_boxes), 1))
        else:
            face_colors = np.asarray(face_colors)

        if edge_colors is None:
            edge_colors = np.tile(np.array([0, 0, 0, 1]), (len(bounding_boxes), 1))
        else:
            edge_colors = np.asarray(edge_colors)

        if not len(face_colors) == len(edge_colors) == len(bounding_boxes):
            raise ValueError(
                trans._(
                    'bounding_boxes, face_colors, and edge_colors must be the same length',
                    deferred=True,
                )
            )

        if not all(issubclass(type(bounding_box), BoundingBox) for bounding_box in bounding_boxes):
            raise ValueError(
                trans._(
                    'all bounding boxes must be subclass of BoundingBox',
                    deferred=True,
                )
            )

        for bounding_box, face_color, edge_color in zip(
            bounding_boxes, face_colors, edge_colors
        ):
            bounding_box_index = len(self.bounding_boxes)
            self.bounding_boxes.append(bounding_box)
            all_z_index.append(bounding_box.z_index)
            all_vertices.append(bounding_box.data_displayed)
            all_index.append([bounding_box_index] * len(bounding_box.data_displayed))

            # Add faces to mesh
            m_tmp = m_mesh_vertices_count
            all_mesh_vertices.append(bounding_box._face_vertices)
            m_mesh_vertices_count += len(bounding_box._face_vertices)
            all_mesh_vertices_centers.append(bounding_box._face_vertices)
            vertices = np.zeros(bounding_box._face_vertices.shape)
            all_mesh_vertices_offsets.append(vertices)
            all_mesh_vertices_index.append(
                _make_index(len(vertices), bounding_box_index, cval=0)
            )

            triangles = bounding_box._face_triangles + m_tmp
            all_mesh_triangles.append(triangles)
            all_mesh_triangles_index.append(
                _make_index(len(triangles), bounding_box_index, cval=0)
            )

            color_array = np.repeat([face_color], len(triangles), axis=0)
            all_mesh_triangles_colors.append(color_array)

            # Add edges to mesh
            m_tmp = m_mesh_vertices_count

            vertices = (
                bounding_box._edge_vertices + bounding_box.edge_width * bounding_box._edge_offsets
            )
            all_mesh_vertices.append(vertices)
            m_mesh_vertices_count += len(vertices)

            all_mesh_vertices_centers.append(bounding_box._edge_vertices)

            all_mesh_vertices_offsets.append(bounding_box._edge_offsets)

            all_mesh_vertices_index.append(
                _make_index(len(bounding_box._edge_offsets), bounding_box_index, cval=1)
            )

            triangles = bounding_box._edge_triangles + m_tmp
            all_mesh_triangles.append(triangles)
            all_mesh_triangles_index.append(
                _make_index(len(triangles), bounding_box_index, cval=1)
            )

            color_array = np.repeat([edge_color], len(triangles), axis=0)
            all_mesh_triangles_colors.append(color_array)

        # assemble properties
        self._z_index = np.append(self._z_index, np.array(all_z_index), axis=0)
        self._face_color = np.vstack((self._face_color, face_colors))
        self._edge_color = np.vstack((self._edge_color, edge_colors))
        self._vertices = np.vstack((self._vertices, np.vstack(all_vertices)))
        self._index = np.append(self._index, np.concatenate(all_index), axis=0)

        self._mesh.vertices = np.vstack(
            (self._mesh.vertices, np.vstack(all_mesh_vertices))
        )
        self._mesh.vertices_centers = np.vstack(
            (self._mesh.vertices_centers, np.vstack(all_mesh_vertices_centers))
        )
        self._mesh.vertices_offsets = np.vstack(
            (self._mesh.vertices_offsets, np.vstack(all_mesh_vertices_offsets))
        )
        self._mesh.vertices_index = np.vstack(
            (self._mesh.vertices_index, np.vstack(all_mesh_vertices_index))
        )

        self._mesh.triangles = np.vstack(
            (self._mesh.triangles, np.vstack(all_mesh_triangles))
        )
        self._mesh.triangles_index = np.vstack(
            (self._mesh.triangles_index, np.vstack(all_mesh_triangles_index))
        )
        self._mesh.triangles_colors = np.vstack(
            (self._mesh.triangles_colors, np.vstack(all_mesh_triangles_colors))
        )

        if z_refresh:
            # Set z_order
            self._update_z_order()

    def update_edge_colors(self, indices, edge_colors, update=True):
        """same as update_edge_color() but for multiple indices/edgecolors at once"""
        self._edge_color[indices] = edge_colors
        all_indices = np.bitwise_and(
            np.isin(self._mesh.triangles_index[:, 0], indices),
            self._mesh.triangles_index[:, 1] == 1,
        )
        self._mesh.triangles_colors[all_indices] = self._edge_color[
            self._mesh.triangles_index[all_indices, 0]
        ]
        if update:
            self._update_displayed()

    def update_face_colors(self, indices, face_colors, update=True):
        """same as update_face_color() but for multiple indices/facecolors at once"""
        self._face_color[indices] = face_colors
        all_indices = np.bitwise_and(
            np.isin(self._mesh.triangles_index[:, 0], indices),
            self._mesh.triangles_index[:, 1] == 0,
        )
        self._mesh.triangles_colors[all_indices] = self._face_color[
            self._mesh.triangles_index[all_indices, 0]
        ]
        if update:
            self._update_displayed()

    def inside(self, coord):
        """Determines if any bounding box at given coord by looking inside triangle
        meshes. Looks only at displayed bounding boxes

        Parameters
        ----------
        coord : sequence of float
            Image coordinates to check if any bounding boxes are at.

        Returns
        -------
        bounding_box : int | None
            Index of bounding box if any that is at the coordinates. Returns `None`
            if no bounding box is found.
        """
        if len(self._mesh.vertices) == 0:
            return None
        triangles = self._mesh.vertices[self._mesh.displayed_triangles]
        indices = inside_triangles(triangles - coord)
        bounding_boxes = self._mesh.displayed_triangles_index[indices, 0]

        if len(bounding_boxes) == 0:
            return None

        z_list = self._z_order.tolist()
        order_indices = np.array([z_list.index(m) for m in bounding_boxes])
        ordered_bounding_boxes = bounding_boxes[np.argsort(order_indices)]
        return ordered_bounding_boxes[0]

    def _inside_3d(self, ray_position: np.ndarray, ray_direction: np.ndarray):
        """Determines if any bounding box is intersected by a ray by looking inside triangle
        meshes. Looks only at displayed bounding boxes.

        Parameters
        ----------
        ray_position : np.ndarray
            (3,) array containing the location that was clicked. This
            should be in the same coordinate system as the vertices.
        ray_direction : np.ndarray
            (3,) array describing the direction camera is pointing in
            the scene. This should be in the same coordinate system as
            the vertices.

        Returns
        -------
        bounding_box : int | None
            Index of bounding_box if any that is at the coordinates. Returns `None`
            if no bounding_box is found.
        intersection_point : Optional[np.ndarray]
            The point where the ray intersects the mesh face. If there was
            no intersection, returns None.
        """
        if len(self._mesh.vertices) == 0:
            return None, None
        triangles = self._mesh.vertices[self._mesh.displayed_triangles]
        inside = line_in_triangles_3d(
            line_point=ray_position,
            line_direction=ray_direction,
            triangles=triangles,
        )
        intersected_bounding_boxes = self._mesh.displayed_triangles_index[inside, 0]
        if len(intersected_bounding_boxes) == 0:
            return None, None

        intersection_points = self._triangle_intersection(
            triangle_indices=inside,
            ray_position=ray_position,
            ray_direction=ray_direction,
        )
        start_to_intersection = intersection_points - ray_position
        distances = np.linalg.norm(start_to_intersection, axis=1)
        closest_bounding_box_index = np.argmin(distances)
        bounding_box = intersected_bounding_boxes[closest_bounding_box_index]
        intersection = intersection_points[closest_bounding_box_index]
        return bounding_box, intersection

    def _triangle_intersection(
        self,
        triangle_indices: np.ndarray,
        ray_position: np.ndarray,
        ray_direction: np.ndarray,
    ):
        """Find the intersection of a ray with specified triangles.

        Parameters
        ----------
        triangle_indices : np.ndarray
            (n,) array of bounding box indices to find the intersection with the ray. The indices should
            correspond with self._mesh.displayed_triangles.
        ray_position : np.ndarray
            (3,) array with the coordinate of the starting point of the ray in layer coordinates.
            Only provide the 3 displayed dimensions.
        ray_direction : np.ndarray
            (3,) array of the normal direction of the ray in layer coordinates.
            Only provide the 3 displayed dimensions.

        Returns
        -------
        intersection_points : np.ndarray
            (n x 3) array of the intersection of the ray with each of the specified bounding boxes in layer coordinates.
            Only the 3 displayed dimensions are provided.
        """
        if len(self._mesh.vertices) == 0:
            return np.empty((0, 3))
        triangles = self._mesh.vertices[self._mesh.displayed_triangles]
        intersected_triangles = triangles[triangle_indices]
        intersection_points = intersect_line_with_triangles(
            line_point=ray_position,
            line_direction=ray_direction,
            triangles=intersected_triangles,
        )
        return intersection_points
