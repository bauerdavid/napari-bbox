# A copy of napari.layers.shapes._shapes_models.shape
from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Literal

import numpy as np
import numpy.typing as npt

from napari.layers.shapes._accelerated_triangulate_dispatch import (
    remove_path_duplicates,
)

# [TODO] check the function in the utils and re implement 
from napari.layers.shapes._shapes_utils import (
    _save_failed_triangulation,
    find_planar_axis,
    is_collinear,
    path_to_mask,
    poly_to_mask,
    triangulate_edge,
    triangulate_face,
    triangulate_face_and_edges,
    triangulate_face_triangle,
    triangulate_face_vispy,
)
from ..napari_0_4_18._bounding_box_utils import (
    find_bbox_corners,
    rectangle_to_box,
    find_corners
)

from napari.layers.shapes.shape_types import (
    BoxArray,
    CoordinateArray,
    TriangleArray,
)
from napari.utils.misc import argsort
from napari.utils.translations import trans
from napari.utils.triangulation_backend import TriangulationBackend

try:
    import bermuda
except ImportError:
    bermuda = None

try:
    from PartSegCore_compiled_backend import (
        triangulate as partsegcore_triangulate,
    )
except ImportError:
    partsegcore_triangulate = None


TRIANGULATION_BACKEND = TriangulationBackend.pure_python

# [NOTE] added information to the documnentation
class BoundingBox(ABC):
    """Base class for a single bounding box

    Parameters
    ----------
    data :
        Vertices specifying the bounding box.
        Either:
            - a (2, D) array specifying the tlf and brb corners of a bounding box
            - a (4, D) array specifying the four corners of a bounding box in 2D
            - a (8, D) array specifying the eight corners of a bounding box in 3D
        These need not be axis aligned.
    edge_width : float
        thickness of lines and edges.
    z_index : int
        Specifier of z order priority. Bounding boxes with higher z order are displayed
        ontop of others.
    dims_order : (D,) list
        Order that the dimensions are to be rendered in.
    ndisplay : int
        Number of displayed dimensions.

    Attributes
    ----------
    data :
        Vertices specifying the bounding box.
        Either:
            - a (2, D) array specifying the tlf and brb corners of a bounding box
            - a (4, D) array specifying the four corners of a bounding box in 2D
            - a (8, D) array specifying the eight corners of a bounding box in 3D
        These need not be axis aligned.
    data_displayed : (N, 2) array or (N, 3) array # [NOTE] key difference with Shape in napari
        Vertices of the bounding box that are currently displayed. 
    edge_width : float
        thickness of lines and edges.
    z_index : int
        Specifier of z order priority. Bounding boxes with higher z order are displayed
        ontop of others.
    dims_order : (D,) list
        Order that the dimensions are rendered in.
    ndisplay : int
        Number of dimensions to be displayed.
    displayed : tuple
        List of dimensions that are displayed.
    not_displayed : tuple
        List of dimensions that are not displayed.
    slice_key : (2, M) array
        Min and max values of the M non-displayed dimensions, useful for
        slicing multidimensional bounding boxes.

    Notes
    -----
    _box : np.ndarray
        9x2 array of vertices of the interaction box. The first 8 points are
        the corners and midpoints of the box in clockwise order starting in the
        upper-left corner. The last point is the center of the box
    _face_vertices : np.ndarray
        Qx2 array of vertices of all triangles for the bounding box face
    _face_triangles : np.ndarray
        Px3 array of vertex indices that form the triangles for the bounding box face
    _edge_vertices : np.ndarray
        Rx2 array of centers of vertices of triangles for the bounding box edge.
        These values should be added to the scaled `_edge_offsets` to get the
        actual vertex positions. The scaling corresponds to the width of the
        edge
    _edge_offsets : np.ndarray
        Sx2 array of offsets of vertices of triangles for the bounding box edge. For
        These values should be scaled and added to the `_edge_vertices` to get
        the actual vertex positions. The scaling corresponds to the width of
        the edge
    _edge_triangles : np.ndarray
        Tx3 array of vertex indices that form the triangles for the bounding box edge
    _filled : bool
        Flag if array is filled or not.
    _use_face_vertices : bool
        Flag to use face vertices for mask generation.
    """

    slice_key: np.ndarray[tuple[Literal[2], int], np.dtype[np.int64]]

    def __init__(
        self,
        data, # [NOTE] important to implement back 
        *,
        edge_width=1,
        z_index=0,
        dims_order=None,
        ndisplay=2,
    ) -> None:
        self._dims_order = dims_order or list(range(2))
        self._ndisplay = ndisplay
        self.slice_key: npt.NDArray

        self._face_vertices: CoordinateArray = np.empty(
            (0, self.ndisplay), dtype=np.float32
        )
        self._face_triangles: TriangleArray = np.empty((0, 3), dtype=np.uint32)  # type: ignore[assignment]
        self._edge_vertices: CoordinateArray = np.empty(
            (0, self.ndisplay), dtype=np.float32
        )
        self._edge_offsets: CoordinateArray = np.empty(
            (0, self.ndisplay), dtype=np.float32
        )
        self._edge_triangles: TriangleArray = np.empty((0, 3), dtype=np.uint32)  # type: ignore[assignment]
        self._box: BoxArray = np.empty((9, 2), dtype=np.float32)  # type: ignore[assignment]

        self._closed = True
        self._filled = True
        self._bounding_box = np.empty((0, self.ndisplay)) # [NOTE] important to implement before it can be written if data
        self.data = data # [NOTE] important to implement back
        self._use_face_vertices = False
        self.edge_width = edge_width
        self.z_index = z_index
        self.name = ''

        self._data: npt.NDArray

    def __new__(cls, *args, **kwargs):
        if (
            TRIANGULATION_BACKEND
            in {
                TriangulationBackend.bermuda,
                TriangulationBackend.fastest_available,
            }
            and bermuda is not None
        ):
            cls._set_meshes = cls._set_meshes_compiled_bermuda
            cls._triangulate_edge = cls._triangulate_edge_bermuda
        elif (
            TRIANGULATION_BACKEND
            in {
                TriangulationBackend.partsegcore,
                TriangulationBackend.fastest_available,
            }
            and partsegcore_triangulate is not None
        ):
            cls._set_meshes = cls._set_meshes_compiled_partseg
            cls._triangulate_edge = cls._triangulate_edge_partseg
        elif (
            TRIANGULATION_BACKEND
            in {
                TriangulationBackend.triangle,
                TriangulationBackend.fastest_available,
            }
            and 'triangle' in sys.modules
        ):
            cls._set_meshes = cls._set_meshes_triangle
        else:
            cls._set_meshes = cls._set_meshes_py
        return super().__new__(cls)

    @property
    # [NOTE] implemented direcly in class so removing the @abstractmethod decorator
    def data(self):
        # user writes own docstring
        return self._data

    @data.setter 
    def data(self, data):
        data = np.array(data).astype(np.float32)

        if len(self.dims_order) != data.shape[1]:
            self._dims_order = list(range(data.shape[1]))

        if len(data) == 2:
            data = find_bbox_corners(data)
            print(f"data: {data}")

        # [NOTE] added more context on error 
        if len(data) != 2**len(self.dims_order):
            raise ValueError(
                trans._(
                    'Data shape does not match a bounding box in {dim}D. Expected {corners} corner vertices, {number} provided.',
                    deferred=True,
                    number=len(data),
                    dim=len(self.dims_order),
                    corners=2**len(self.dims_order)
                )
            )

        self._data = data

        # [NOTE] works for all type of data array
        # since a bounding box is already expected to be axis-aligned,
        # this just returns the (2, D) array
        # specifying the tlf and brb corners for that bounding box
        self._bounding_box = np.array(
            [
                np.min(data, axis=0),
                np.max(data, axis=0),
            ]
        )

        self._update_displayed_data()

    # [NOTE] implemented direcly in class so removing the @abstractmethod decorator
    def _update_displayed_data(self):
        """Update the data that is to be displayed."""
        # Add four boundary lines and then two triangles for each
        self._clean_cache()
        data_displayed = self.data_displayed

        if self.ndisplay == 2:
            self._set_meshes(data_displayed, face=False)
            self._face_vertices = data_displayed
            self._face_triangles = np.array([[0, 1, 2], [0, 2, 3]])
            # The data displayed are in this case the four corners
            self._box = rectangle_to_box(data_displayed)  # type: ignore[arg-type]

        if len(self.data_displayed) == 4: # quad in 3D
            # [NOTE] /!\ changed the ordered of vertices to work - regardless of using new or old add_bounding_box in mouse bindings
            ordered_vertices = data_displayed[[0, 1, 2, 3, 0]]
            self._set_meshes(ordered_vertices, closed=False, face=False)
            self._face_vertices = data_displayed
            self._face_triangles = np.array([
                    [0, 1, 2],
                    [2, 3, 0],
                ])
        else: # box in 3D
            ordered_vertices = data_displayed[
                    [0, 1, 3, 2, 0, 4, 6, 2, 3, 7, 6, 4, 5, 7, 3, 1, 5]
                ]
            self._set_meshes(ordered_vertices, closed=False, face=False)
            # [NOTE] ignoring faces for 3D box
            # self._face_triangles = np.array([
                #     [0, 1, 2],
                #     [2, 1, 3],
                #     [0, 4, 1],
                #     [1, 4, 5],
                #     [0, 2, 6],
                #     [0, 6, 4],
                #     [7, 4, 6],
                #     [7, 5, 4],
                #     [7, 6, 2],
                #     [7, 2, 3],
                #     [7, 3, 1],
                #     [7, 1, 5]
                # ])
            # self._face_triangles = np.append(self._face_triangles, np.fliplr(self._face_triangles), 0)
            # self._face_vertices = self.data_displayed
            self._face_triangles = np.array([[0, 0, 0]])
        self._update_slice_key()

    # [NOTE] keeping old implementation but _bounding_box now used for this in napari
    # [TODO] check if can rely on internal napari _bounding_box instead
    def _update_slice_key(self):
        data_not_displayed = self.data[:, self.dims_not_displayed]
        self.slice_key = np.array(
            [
                np.min(data_not_displayed, axis=0),
                np.max(data_not_displayed, axis=0),
            ]
        )    

    @property
    def ndisplay(self):
        """int: Number of displayed dimensions."""
        return self._ndisplay

    @ndisplay.setter
    def ndisplay(self, ndisplay):
        if self.ndisplay == ndisplay:
            return
        self._ndisplay = ndisplay
        self._update_displayed_data()

    @property
    def dims_order(self):
        """(D,) list: Order that the dimensions are rendered in."""
        return self._dims_order

    @dims_order.setter
    def dims_order(self, dims_order):
        if self.dims_order == dims_order:
            return
        self._dims_order = dims_order
        self._update_displayed_data()

    @cached_property
    def dims_displayed(self):
        """tuple: Dimensions that are displayed."""
        return self.dims_order[-self.ndisplay :]

    # [NOTE] new addition, internal napari bounding_box, not to be confused
    # [TODO] could try without extra edges
    @property
    def bounding_box(self) -> np.ndarray:
        """(2, N) array, bounding box of the object."""
        # We add +-0.5 to handle edge width
        return self._bounding_box[:, self.dims_displayed] + [
            [-0.5 * self.edge_width],
            [0.5 * self.edge_width],
        ]

    @property
    def dims_not_displayed(self):
        """tuple: Dimensions that are not displayed."""
        return self.dims_order[: -self.ndisplay]

    # [NOTE] data_displayed not cached property in order to use the setter
    # [TODO] check that works
    @property
    def data_displayed(self) -> CoordinateArray:
        """(N, 2) or (N, 3) array: Vertices of the bounding box that are currently displayed."""
        displayed = np.unique(self.data[:, self.dims_displayed], axis=0)
        if len(self.dims_displayed) == 2: # 2D box
            return find_corners(displayed) # accounts for data being (2, D) with tlf and brb corners only
        return displayed
    
    @data_displayed.setter
    def data_displayed(self, new_value):
        data = self.data[:, self.dims_displayed]
        data_sorted = data.copy()
        idx_sort = np.arange(len(data))
        # [NOTE] going through the dims displayed and sorting vertices
        for d in range(data.shape[1]):
            sub_idx_sort = data_sorted[:, -1 - d].argsort(kind='mergesort')
            data_sorted = data_sorted[sub_idx_sort]
            idx_sort = idx_sort[sub_idx_sort]
        # [NOTE] when collapsing from higher dimension to lower → duplicate vertices → need to group them
        vals, idx_start, count = np.unique(data_sorted, return_counts=True, return_index=True, axis=0)
        # [NOTE] get the unique vertices and move the corresponding collapsed group together
        # [NOTE] key to the bbox plugin to maintain 2D → 3D consistency 
        uniqe_idx = np.split(idx_sort, idx_start[1:])
        for val, idx_list in zip(new_value, uniqe_idx):
            self._data[idx_list[:, np.newaxis], self.dims_displayed] = val

    @property
    def edge_width(self):
        """float: thickness of lines and edges."""
        return self._edge_width

    @edge_width.setter
    def edge_width(self, edge_width):
        self._edge_width = edge_width

    @property
    def z_index(self):
        """int: z order priority of shape. Shapes with higher z order displayed
        ontop of others.
        """
        return self._z_index

    @z_index.setter
    def z_index(self, z_index):
        self._z_index = z_index

    @property
    def vertices_count(self) -> int:
        """int: Number of vertices in the shape."""
        return self._edge_vertices.shape[0] + self._face_vertices.shape[0]

    @property
    def triangles_count(self) -> int:
        """int: Number of triangles in the shape."""
        return self._face_triangles.shape[0] + self._edge_triangles.shape[0]

    @property
    def face_triangles_count(self) -> int:
        """int: Number of triangles in the face of the shape."""
        return self._face_triangles.shape[0]

    @property
    def face_vertices_count(self) -> int:
        """int: Number of vertices in the face of the shape."""
        return self._face_vertices.shape[0]

    @property
    def edge_triangles_count(self) -> int:
        """int: Number of triangles in the edge of the shape."""
        return self._edge_triangles.shape[0]

    @property
    def edge_vertices_count(self) -> int:
        """int: Number of vertices in the edge of the shape."""
        return self._edge_vertices.shape[0]

    def _set_empty_edge(self) -> None:
        self._edge_vertices = np.empty((0, self.ndisplay), dtype=np.float32)
        self._edge_offsets = np.empty((0, self.ndisplay), dtype=np.float32)
        self._edge_triangles = np.empty((0, 3), dtype=np.uint32)  # type: ignore[assignment]

    def _set_empty_face(self) -> None:
        self._face_vertices = np.empty((0, self.ndisplay), dtype=np.float32)
        self._face_triangles = np.empty((0, 3), dtype=np.uint32)  # type: ignore[assignment]

    # [NOTE] left but used for unsupported bermuda backend
    def _set_meshes_compiled_3d(
        self,
        data: CoordinateArray,
        closed: bool = True,
        face: bool = True,
        edge: bool = True,
    ):
        if face:
            face_triangles, face_vertices = (
                bermuda.triangulate_polygons_face_3d([data])
            )
            self._face_vertices = face_vertices
            self._face_triangles = face_triangles
        else:
            self._set_empty_face()

        if edge:
            centers, offsets, edge_triangles = triangulate_edge(
                data, closed=closed
            )
            self._edge_vertices = centers
            self._edge_offsets = offsets
            self._edge_triangles = edge_triangles
        else:
            self._set_empty_edge()

    def _set_meshes(  # noqa: B027
        self,
        data: CoordinateArray,
        closed: bool = True,
        face: bool = True,
        edge: bool = True,
    ) -> None: ...

    # [NOTE] left but unimplemented backend
    def _set_meshes_compiled_bermuda(
        self,
        data: CoordinateArray,
        closed: bool = True,
        face: bool = True,
        edge: bool = True,
    ) -> None:
        """Sets the face and edge meshes from a set of points.

        Uses bermuda compiled backend for triangulation.

        Parameters
        ----------
        data : np.ndarray
            Nx2 or Nx3 array specifying the shape to be triangulated
        closed : bool
            Bool which determines if the edge is closed or not
        face : bool
            Bool which determines if the face need to be traingulated
        edge : bool
            Bool which determines if the edge need to be traingulated
        """
        if data.shape[1] == 3:
            self._set_meshes_compiled_3d(
                data, closed=closed, face=face, edge=edge
            )
            return

        # if we are computing both edge and face triangles, we can do so
        # with a single call to the compiled backend
        if edge and face:
            try:
                (triangles, vertices), (centers, offsets, edge_triangles) = (
                    bermuda.triangulate_polygons_with_edge([data])
                )
            except BaseException as e:  # pragma: no cover
                path, text_path = _save_failed_triangulation(
                    data, backend='bermuda'
                )
                raise RuntimeError(
                    f'Triangulation failed. Data saved to {path} and {text_path}'
                ) from e

            self._edge_vertices = centers
            self._edge_offsets = offsets
            self._edge_triangles = edge_triangles
            self._face_vertices = vertices
            self._face_triangles = triangles
            return

        # otherwise, we make individual calls to specialized functions
        if edge:
            self._edge_vertices, self._edge_offsets, self._edge_triangles = (
                bermuda.triangulate_path_edge(data, closed=closed)
            )
        else:
            self._set_empty_edge()
        if face:
            self._face_triangles, self._face_vertices = (
                bermuda.triangulate_polygons_face([data])
            )
        else:
            self._set_empty_face()

    # [NOTE] left but unimplemented backend
    def _set_meshes_compiled_partseg(
        self,
        data: CoordinateArray,
        closed: bool = True,
        face: bool = True,
        edge: bool = True,
    ) -> None:
        """Sets the face and edge meshes from a set of points.

        Uses PartSegCore compiled backend for triangulation.

        Parameters
        ----------
        data : np.ndarray
            Nx2 or Nx3 array specifying the shape to be triangulated
        closed : bool
            Bool which determines if the edge is closed or not
        face : bool
            Bool which determines if the face need to be traingulated
        edge : bool
            Bool which determines if the edge need to be traingulated
        """
        if data.shape[1] == 3:
            self._set_meshes_py(data, closed=closed, face=face, edge=edge)
            return

        # if we are computing both edge and face triangles, we can do so
        # with a single call to the compiled backend
        if edge and face:
            try:
                (triangles, vertices), (centers, offsets, edge_triangles) = (
                    partsegcore_triangulate.triangulate_polygon_with_edge_numpy_li(
                        [data], split_edges=True
                    )
                )
            except BaseException as e:  # pragma: no cover
                path, text_path = _save_failed_triangulation(
                    data, backend='partsegcore'
                )
                raise RuntimeError(
                    f'Triangulation failed. Data saved to {path} and {text_path}'
                ) from e

            self._edge_vertices = centers
            self._edge_offsets = offsets
            self._edge_triangles = edge_triangles
            self._face_vertices = vertices
            self._face_triangles = triangles
            return

        # otherwise, we make individual calls to specialized functions
        if edge:
            self._edge_vertices, self._edge_offsets, self._edge_triangles = (
                partsegcore_triangulate.triangulate_path_edge_numpy(
                    data, closed=closed
                )
            )
        else:
            self._set_empty_edge()
        if face:
            self._face_triangles, self._face_vertices = (
                partsegcore_triangulate.triangulate_polygon_numpy_li([data])
            )
        else:
            self._set_empty_face()

    # [NOTE] left but unimplemented backend
    def _set_meshes_triangle(
        self,
        data: CoordinateArray,
        closed: bool = True,
        face: bool = True,
        edge: bool = True,
    ) -> None:
        """Sets the face and edge meshes from a set of points.

        Uses the triangle package to triangulate the polygon face

        Parameters
        ----------
        data : np.ndarray
            Nx2 or Nx3 array specifying the shape to be triangulated
        closed : bool
            Bool which determines if the edge is closed or not
        face : bool
            Bool which determines if the face need to be traingulated
        edge : bool
            Bool which determines if the edge need to be traingulated
        """
        data = remove_path_duplicates(data, closed=closed)
        if edge and face:
            (f_vertices, f_triangles), (centers, offsets, triangles) = (
                triangulate_face_and_edges(data, triangulate_face_triangle)
            )
            self._edge_vertices = centers
            self._edge_offsets = offsets
            self._edge_triangles = triangles
            self._face_vertices = f_vertices
            self._face_triangles = f_triangles
            return

        if edge:
            centers, offsets, triangles = triangulate_edge(data, closed=closed)
            self._edge_vertices = centers
            self._edge_offsets = offsets
            self._edge_triangles = triangles
        else:
            self._set_empty_edge()
        ndim = data.shape[1]
        # this method is called right before display, on sliced data, so
        # ndim can only be 2 or 3. If 3D, shapes must be confined to a plane
        # along *some* axis. We find that axis and the plane coordinate, then
        # proceed as if 2D. If 2D, the data is passed through unchanged. And
        # if there is no planar axis, we cannot triangulate and we return an
        # empty data array
        data2d, axis, value = find_planar_axis(data)

        # set empty data as fallback
        self._set_empty_face()
        if face and not is_collinear(data2d):
            vertices, triangles = triangulate_face(
                data2d, triangulate_face_triangle
            )
            if ndim == 3 and axis is not None and value is not None:
                # axis and value can be None if data 3D but not limited to an
                # axis-aligned plane. However in that situation data2d will be
                # empty, is_collinear is True, and we will never get here. But
                # we check anyway for mypy's sake
                vertices = np.insert(vertices, axis, value, axis=1)
            if len(triangles) > 0:
                self._face_vertices = vertices
                self._face_triangles = triangles

    # [NOTE] python backend to implement with bbox plugin
    def _set_meshes_py(
        self,
        data: CoordinateArray,
        closed: bool = True,
        face: bool = True,
        edge: bool = True,
    ) -> None:
        """Sets the face and edge meshes from a set of points.

        Parameters
        ----------
        data : np.ndarray
            Nx2 or Nx3 array specifying the shape to be triangulated
        closed : bool
            Bool which determines if the edge is closed or not
        face : bool
            Bool which determines if the face need to be traingulated
        edge : bool
            Bool which determines if the edge need to be traingulated
        """
        # [NOTE] left but of no use for bounding boxes as implemented in plugin for now
        if edge and face:
            (f_vertices, f_triangles), (centers, offsets, triangles) = (
                triangulate_face_and_edges(data, triangulate_face_vispy)
            )
            self._edge_vertices = centers
            self._edge_offsets = offsets
            self._edge_triangles = triangles
            self._face_vertices = f_vertices
            self._face_triangles = f_triangles
            return

        if edge:
            centers, offsets, triangles = triangulate_edge(data, closed=closed)
            self._edge_vertices = centers
            self._edge_offsets = offsets
            self._edge_triangles = triangles
        else: # [NOTE] new cleaning logic
            self._set_empty_edge()

        # set empty data as fallback
        self._set_empty_face()

        # removing consecutive duplicate points
        # [NOTE] no need for hack in remove_path_duplicates function
        idx = np.concatenate(
            [[True], ~np.all(data[1:] == data[:-1], axis=-1)]
            )
        clean_data = data[idx].copy() # [NOTE] removes consecutive duplicate points.

        if face and not is_collinear(clean_data): # [NOTE] if all points in the same line → no face
            # [NOTE] check 2D
            if clean_data.shape[1] == 2:
                vertices, triangles = triangulate_face(
                    clean_data, triangulate_face_vispy
                )
            else: # [NOTE] 3D
                # [NOTE] in a single plane
                data2d, axis, value = find_planar_axis(clean_data)
                if axis is not None and value is not None: # [NOTE] found single plane
                    vertices, triangles = triangulate_face(
                        data2d, triangulate_face_vispy
                        )
                    vertices = np.insert(vertices, axis, value, axis=1)
                    if len(triangles) > 0:
                        self._face_vertices = vertices
                        self._face_triangles = triangles
                    return
        # [NOTE] for 3D bounding boxes can ignore faces

    # [NOTE] leaving the option for the legacy implementation for now
    # [TODO] check if can be removed
    def legacy_set_meshes(self, data, closed=True, face=True, edge=True):
        """Sets the face and edge meshes from a set of points.

        Parameters
        ----------
        data : np.ndarray
            Nx2 or Nx3 array specifying the bounding box to be triangulated
        closed : bool
            Bool which determines if the edge is closed or not
        face : bool
            Bool which determines if the face need to be traingulated
        edge : bool
            Bool which determines if the edge need to be traingulated
        """
        if edge:
            centers, offsets, triangles = triangulate_edge(data, closed=closed)
            self._edge_vertices = centers
            self._edge_offsets = offsets
            self._edge_triangles = triangles
        else:
            self._edge_vertices = np.empty((0, self.ndisplay))
            self._edge_offsets = np.empty((0, self.ndisplay))
            self._edge_triangles = np.empty((0, 3), dtype=np.uint32)

        if face: 
            idx = np.concatenate(
                [[True], ~np.all(data[1:] == data[:-1], axis=-1)]
            )
            clean_data = data[idx].copy() # [NOTE] removes consecutive duplicate points.

            if not is_collinear(clean_data[:, -2:]): #[NOTE] if all points in the same line → no face
                if clean_data.shape[1] == 2: # [NOTE] check 2D
                    vertices, triangles = triangulate_face(clean_data)
                elif len(np.unique(clean_data[:, 0])) == 1: # [NOTE] planar check if all points on the same plane for first dim (e.g. Z)
                    val = np.unique(clean_data[:, 0])
                    vertices, triangles = triangulate_face(clean_data[:, -2:]) # [NOTE] triangulate in 2D easier if all in same plane
                    exp = np.expand_dims(np.repeat(val, len(vertices)), axis=1)
                    vertices = np.concatenate([exp, vertices], axis=1)
                else: #[NOTE] if in 3D no face drawn
                    triangles = []
                    vertices = []
                if len(triangles) > 0:
                    self._face_vertices = vertices
                    self._face_triangles = triangles
                else:
                    self._face_vertices = np.empty((0, self.ndisplay))
                    self._face_triangles = np.empty((0, 3), dtype=np.uint32)
            else:
                self._face_vertices = np.empty((0, self.ndisplay))
                self._face_triangles = np.empty((0, 3), dtype=np.uint32)
        else:
            self._face_vertices = np.empty((0, self.ndisplay))
            self._face_triangles = np.empty((0, 3), dtype=np.uint32)

    def _triangulate_edge(
        self, data: CoordinateArray, closed: bool
    ) -> tuple[CoordinateArray, CoordinateArray, TriangleArray]:
        """Triangulate the edge of the shape.

        Parameters
        ----------
        data : np.ndarray
            Nx2 or Nx3 array specifying the shape to be triangulated
        closed : bool
            Bool which determines if the edge is closed or not

        Returns
        -------
        tuple
            Tuple of (centers, offsets, triangles) where centers is a 2D array
            of the centers of the triangles, offsets is a 2D array of the
            offsets of the triangles, and triangles is a 2D array of the
            triangles.
        """
        return triangulate_edge(data, closed=closed)

    # [NOTE] left but unimplemented backend
    def _triangulate_edge_partseg(
        self, data: CoordinateArray, closed: bool
    ) -> tuple[CoordinateArray, CoordinateArray, TriangleArray]:
        return partsegcore_triangulate.triangulate_path_edge_numpy(
            data, closed=closed
        )

    # [NOTE] left but unimplemented backend
    def _triangulate_edge_bermuda(
        self, data: CoordinateArray, closed: bool
    ) -> tuple[CoordinateArray, CoordinateArray, TriangleArray]:
        return bermuda.triangulate_path_edge(data, closed=closed)

    # [NOTE] left but unimplemented backend
    def _all_triangles(self):
        """Return all triangles for the shape

        Returns
        -------
        np.ndarray
            Nx3 array of vertex indices that form the triangles for the shape
        """
        return np.vstack(
            [
                self._face_vertices[self._face_triangles],
                (self._edge_vertices + self.edge_width * self._edge_offsets)[
                    self._edge_triangles
                ],
            ]
        )

    def transform(self, transform: npt.NDArray) -> None:
        """Performs a linear transform on the shape

        Parameters
        ----------
        transform : np.ndarray
            2x2 array specifying linear transform.
        """
        self._box = self._box @ transform.T
        self._data[:, self.dims_displayed] = (
            self._data[:, self.dims_displayed] @ transform.T
        )
        self._face_vertices = self._face_vertices @ transform.T
        self.__dict__.pop('data_displayed', None)  # clear cache
        points = self.data_displayed
        points = remove_path_duplicates(points, closed=self._closed)
        centers, offsets, triangles = self._triangulate_edge(
            points, closed=self._closed
        )
        self._edge_vertices = centers
        self._edge_offsets = offsets
        self._edge_triangles = triangles
        self._bounding_box = np.array( # [NOTE] addition internal _bounding_box
            [
                np.min(self._data, axis=0),
                np.max(self._data, axis=0),
            ]
        )
        self._update_slice_key() # [TODO] check if can rely on internal _bounding_box instead
        self._clean_cache()

    def shift(self, shift: npt.NDArray) -> None:
        """Performs a 2D shift on the shape

        Parameters
        ----------
        shift : np.ndarray
            length 2 array specifying shift of shapes.
        """
        shift = np.array(shift)

        self._face_vertices = self._face_vertices + shift
        self._edge_vertices = self._edge_vertices + shift
        self._box = self._box + shift
        self.data_displayed = self.data_displayed + shift # [NOTE] adapated for plugin → do not want to change self._data based on data_displayed
        self._bounding_box[:, self.dims_displayed] = (
            self._bounding_box[:, self.dims_displayed] + shift
        )
        self._clean_cache()

    def scale(self, scale, center=None):
        """Performs a scaling on the shape

        Parameters
        ----------
        scale : float, list
            scalar or list specifying rescaling of shape.
        center : list
            length 2 list specifying coordinate of center of scaling.
        """
        if isinstance(scale, list | np.ndarray):
            transform = np.array([[scale[0], 0], [0, scale[1]]])
        else:
            transform = np.array([[scale, 0], [0, scale]])
        if center is None:
            self.transform(transform)
        else:
            center = np.array(center)
            self.shift(-center)
            self.transform(transform)
            self.shift(center)

    # [NOTE] not implemented in bounding_box plugin → bbox should be axis-aligned 
    #def rotate(self, angle, center=None):
    #    """Performs a rotation on the shape
    #
    #    Parameters
    #    ----------
    #    angle : float
    #        angle specifying rotation of shape in degrees. CCW is positive.
    #    center : list
    #        length 2 list specifying coordinate of fixed point of the rotation.
    #    """
    #    theta = np.radians(angle)
    #    transform = np.array(
    #        [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    #    )
    #    if center is None:
    #        self.transform(transform)
    #    else:
    #        center = np.array(center)
    #        self.shift(-center)
    #        self.transform(transform)
    #        self.shift(center)

    # [NOTE] not implemented in bounding_box plugin
    #def flip(self, axis, center=None):
    #    """Performs a flip on the shape, either horizontal or vertical.
    #
    #    Parameters
    #    ----------
    #    axis : int
    #        integer specifying axis of flip. `0` flips horizontal, `1` flips
    #        vertical.
    #    center : list
    #        length 2 list specifying coordinate of center of flip axes.
    #    """
    #    if axis == 0:
    #        transform = np.array([[1, 0], [0, -1]])
    #    elif axis == 1:
    #        transform = np.array([[-1, 0], [0, 1]])
    #    else:
    #        raise ValueError(
    #            trans._(
    #                'Axis not recognized, must be one of "{{0, 1}}"',
    #                deferred=True,
    #            )
    #        )
    #    if center is None:
    #        self.transform(transform)
    #    else:
    #        self.shift(-center)
    #        self.transform(transform)
    #        self.shift(-center)

    def to_mask(self, mask_shape=None, zoom_factor=1, offset=(0, 0)):
        """Convert the shape vertices to a boolean mask.

        Set points to `True` if they are lying inside the shape if the shape is
        filled, or if they are lying along the boundary of the shape if the
        shape is not filled. Negative points or points outside the mask_shape
        after the zoom and offset are clipped.

        Parameters
        ----------
        mask_shape : (D,) array
            Shape of mask to be generated. If non specified, takes the max of
            the displayed vertices.
        zoom_factor : float
            Premultiplier applied to coordinates before generating mask. Used
            for generating as downsampled mask.
        offset : 2-tuple
            Offset subtracted from coordinates before multiplying by the
            zoom_factor. Used for putting negative coordinates into the mask.

        Returns
        -------
        mask : np.ndarray
            Boolean array with `True` for points inside the shape
        """
        if mask_shape is None:
            mask_shape = np.round(self.data_displayed.max(axis=0)).astype(
                'int'
            )

        if len(mask_shape) == 2:
            embedded = False
            shape_plane = mask_shape
        elif len(mask_shape) == self.data.shape[1]:
            embedded = True
            shape_plane = [mask_shape[d] for d in self.dims_displayed]
        else:
            raise ValueError(
                trans._(
                    'mask shape length must either be 2 or the same as the dimensionality of the shape, expected {expected} got {received}.',
                    deferred=True,
                    expected=self.data.shape[1],
                    received=len(mask_shape),
                )
            )

        if self._use_face_vertices:
            data = self._face_vertices
        else:
            data = self.data_displayed

        data = data[:, -len(shape_plane) :]

        if self._filled:
            mask_p = poly_to_mask(shape_plane, (data - offset) * zoom_factor)
        else:
            mask_p = path_to_mask(shape_plane, (data - offset) * zoom_factor)

        # If the mask is to be embedded in a larger array, compute array
        # and embed as a slice.
        if embedded:
            mask = np.zeros(mask_shape, dtype=bool)
            slice_key: list[int | slice] = [0] * len(mask_shape)
            for i in range(len(mask_shape)):
                if i in self.dims_displayed:
                    slice_key[i] = slice(None)
                elif self.slice_key is not None:
                    slice_key[i] = slice(
                        self.slice_key[0, i], self.slice_key[1, i] + 1
                    )
                else:
                    raise RuntimeError(
                        'Internal error: self.slice_key is None'
                    )
            displayed_order = argsort(self.dims_displayed)
            mask[tuple(slice_key)] = mask_p.transpose(displayed_order)
        else:
            mask = mask_p

        return mask

    # [NOTE] no cache for data_displayed
    def _clean_cache(self) -> None:
        if 'dims_displayed' in self.__dict__:
            del self.__dict__['dims_displayed']
        #if 'data_displayed' in self.__dict__:
        #    del self.__dict__['data_displayed']