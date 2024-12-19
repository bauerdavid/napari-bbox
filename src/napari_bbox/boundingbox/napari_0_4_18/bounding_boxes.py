# A copy of napari.layers.shapes.shapes
from copy import copy, deepcopy
from typing import Dict, List, Tuple, Union

import numpy as np

from napari.layers.base import Layer, no_op
from napari.layers.base._base_constants import ActionType
from napari.layers.base._base_mouse_bindings import (
    highlight_box_handles,
    transform_with_box,
)

from .bounding_box import BoundingBox
from ._bounding_box_list import BoundingBoxList
from ._bounding_box_constants import (
    Box,
    ColorMode,
    Mode,
    SizeMode
)
from ._bounding_box_mouse_bindings import (
    add_bounding_box,
    highlight,
    select,
)
from ._bounding_box_utils import (
    number_of_bounding_boxes,
)
from ..napari_0_4_15.bounding_boxes import BoundingBoxLayer
from ..._helper_functions import layer_slice_indices, layer_dims_order, layer_ndisplay
from ..._utils import NAPARI_VERSION
from napari.layers.utils.color_manager_utils import (
    map_property,
)
from napari.layers.utils.color_transformations import (
    normalize_and_broadcast_colors,
    transform_color_with_defaults,
)
from napari.layers.utils.interactivity_utils import (
    nd_line_segment_to_displayed_data_ray,
)
from napari.layers.utils.layer_utils import _FeatureTable, _unique_element
from napari.layers.utils.text_manager import TextManager
from napari.utils.colormaps.colormap_utils import ColorType
from napari.utils.colormaps.standardize_color import (
    transform_color,
)
from napari.utils.events import Event
from napari.utils.events.custom_types import Array
from napari.utils.misc import ensure_iterable
from napari.utils.translations import trans


DEFAULT_COLOR_CYCLE = np.array([[1, 0, 1, 1], [0, 1, 0, 1]])


class BoundingBoxLayer(BoundingBoxLayer):
    """Bounding box layer.

    Parameters
    ----------
    data : list or array
        List of bounding box data, where each element is an (N, D) array of the
        N vertices of a Bounding box in D dimensions. Can be an 3-dimensional
        array.
    ndim : int
        Number of dimensions for bounding boxes. When data is not None, ndim must be D.
        An empty bounding box layer can be instantiated with arbitrary ndim.
    features : dict[str, array-like] or Dataframe-like
        Features table where each row corresponds to a bounding box and each column
        is a feature.
    feature_defaults : dict[str, Any] or Dataframe-like
        The default value of each feature in a table with one row.
    properties : dict {str: array (N,)}, DataFrame
        Properties for each bounding box. Each property should be an array of length N,
        where N is the number of bounding boxes.
    property_choices : dict {str: array (N,)}
        possible values for each property.
    text : str, dict
        Text to be displayed with the bounding boxes. If text is set to a key in properties,
        the value of that property will be displayed. Multiple properties can be
        composed using f-string-like syntax (e.g., '{property_1}, {float_property:.2f}).
        A dictionary can be provided with keyword arguments to set the text values
        and display properties. See TextManager.__init__() for the valid keyword arguments.
        For example usage, see /napari/examples/add_shapes_with_text.py.
    edge_width : float or list
        Thickness of lines and edges. If a list is supplied it must be the
        same length as the length of `data` and each element will be
        applied to each bounding box otherwise the same value will be used for all
        bounding boxes.
    edge_color : str, array-like
        If string can be any color name recognized by vispy or hex value if
        starting with `#`. If array-like must be 1-dimensional array with 3
        or 4 elements. If a list is supplied it must be the same length as
        the length of `data` and each element will be applied to each bounding box
        otherwise the same value will be used for all bounding boxes.
    edge_color_cycle : np.ndarray, list
        Cycle of colors (provided as string name, RGB, or RGBA) to map to edge_color if a
        categorical attribute is used color the vectors.
    edge_colormap : str, napari.utils.Colormap
        Colormap to set edge_color if a continuous attribute is used to set face_color.
    edge_contrast_limits : None, (float, float)
        clims for mapping the property to a color map. These are the min and max value
        of the specified property that are mapped to 0 and 1, respectively.
        The default value is None. If set the none, the clims will be set to
        (property.min(), property.max())
    face_color : str, array-like
        If string can be any color name recognized by vispy or hex value if
        starting with `#`. If array-like must be 1-dimensional array with 3
        or 4 elements. If a list is supplied it must be the same length as
        the length of `data` and each element will be applied to each bounding box
        otherwise the same value will be used for all bounding boxes.
    face_color_cycle : np.ndarray, list
        Cycle of colors (provided as string name, RGB, or RGBA) to map to face_color if a
        categorical attribute is used color the vectors.
    face_colormap : str, napari.utils.Colormap
        Colormap to set face_color if a continuous attribute is used to set face_color.
    face_contrast_limits : None, (float, float)
        clims for mapping the property to a color map. These are the min and max value
        of the specified property that are mapped to 0 and 1, respectively.
        The default value is None. If set the none, the clims will be set to
        (property.min(), property.max())
    z_index : int or list
        Specifier of z order priority. Bounding boxes with higher z order are
        displayed ontop of others. If a list is supplied it must be the
        same length as the length of `data` and each element will be
        applied to each bounding box otherwise the same value will be used for all
        bounding boxes.
    name : str
        Name of the layer.
    metadata : dict
        Layer metadata.
    scale : tuple of float
        Scale factors for the layer.
    translate : tuple of float
        Translation values for the layer.
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    shear : 1-D array or n-D array
        Either a vector of upper triangular values, or an nD shear matrix with
        ones along the main diagonal.
    affine : n-D array or napari.utils.transforms.Affine
        (N+1, N+1) affine transformation matrix in homogeneous coordinates.
        The first (N, N) entries correspond to a linear transform and
        the final column is a length N translation vector and a 1 or a napari
        `Affine` transform object. Applied as an extra transform on top of the
        provided scale, rotate, and shear values.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}.
    visible : bool
        Whether the layer visual is currently being displayed.
    cache : bool
        Whether slices of out-of-core datasets should be cached upon retrieval.
        Currently, this only applies to dask arrays.

    Attributes
    ----------
    data : (N, ) list of array
        List of bounding box data, where each element is an (N, D) array of the
        N vertices of a bounding box in D dimensions.
    features : Dataframe-like
        Features table where each row corresponds to a bounding box and each column
        is a feature.
    feature_defaults : DataFrame-like
        Stores the default value of each feature in a table with one row.
    properties : dict {str: array (N,)}, DataFrame
        Properties for each bounding box. Each property should be an array of length N,
        where N is the number of bounding boxes.
    text : str, dict
        Text to be displayed with the bounding boxes. If text is set to a key in properties,
        the value of that property will be displayed. Multiple properties can be
        composed using f-string-like syntax (e.g., '{property_1}, {float_property:.2f}).
        For example usage, see /napari/examples/add_shapes_with_text.py.
    edge_color : str, array-like
        Color of the bounding box border. Numeric color values should be RGB(A).
    face_color : str, array-like
        Color of the bounding box face. Numeric color values should be RGB(A).
    edge_width : (N, ) list of float
        Edge width for each bounding box.
    z_index : (N, ) list of int
        z-index for each bounding box.
    current_edge_width : float
        Thickness of lines and edges of the next bounding box to be added or the
        currently selected bounding box.
    current_edge_color : str
        Color of the edge of the next bounding box to be added or the currently
        selected bounding box.
    current_face_color : str
        Color of the face of the next bounding box to be added or the currently
        selected bounding box.
    selected_data : set
        List of currently selected bounding boxes.
    nbounding_boxes : int
        Total number of bounding boxes.
    mode : Mode
        Interactive mode. The normal, default mode is PAN_ZOOM, which
        allows for normal interactivity with the canvas.

        The SELECT mode allows for entire bounding boxes to be selected, moved and
        resized.

        The ADD_BOUNDING_BOX mode allows for bounding boxes to be added.

    Notes
    -----
    _data_dict : Dict of BoundingBoxList
        Dictionary containing all the bounding box data indexed by slice tuple
    _data_view : BoundingBoxList
        Object containing the currently viewed bounding box data.
    _selected_data_history : set
        Set of currently selected captured on press of <space>.
    _selected_data_stored : set
        Set of selected previously displayed. Used to prevent rerendering the
        same highlighted bounding boxes when no data has changed.
    _selected_box : None | np.ndarray
        `None` if no bounding boxes are selected, otherwise a 10x2 array of vertices of
        the interaction box. The first 8 points are the corners and midpoints
        of the box. The 9th point is the center of the box, and the last point
        is the location of the rotation handle that can be used to rotate the
        box.
    _drag_start : None | np.ndarray
        If a drag has been started and is in progress then a length 2 array of
        the initial coordinates of the drag. `None` otherwise.
    _drag_box : None | np.ndarray
        If a drag box is being created to select bounding boxes then this is a 2x2
        array of the two extreme corners of the drag. `None` otherwise.
    _drag_box_stored : None | np.ndarray
        If a drag box is being created to select bounding boxes then this is a 2x2
        array of the two extreme corners of the drag that have previously been
        rendered. `None` otherwise. Used to prevent rerendering the same
        drag box when no data has changed.
    _is_moving : bool
        Bool indicating if any bounding boxes are currently being moved.
    _is_selecting : bool
        Bool indicating if a drag box is currently being created in order to
        select bounding boxes.
    _is_creating : bool
        Bool indicating if any bounding boxes are currently being created.
    _fixed_aspect : bool
        Bool indicating if aspect ratio of bounding boxes should be preserved on
        resizing.
    _aspect_ratio : float
        Value of aspect ratio to be preserved if `_fixed_aspect` is `True`.
    _fixed_vertex : None | np.ndarray
        If a scaling or rotation is in progress then a length 2 array of the
        coordinates that are remaining fixed during the move. `None` otherwise.
    _fixed_index : int
        If a scaling or rotation is in progress then the index of the vertex of
        the bounding box that is remaining fixed during the move. `None`
        otherwise.
    _update_properties : bool
        Bool indicating if properties are to allowed to update the selected
        bounding boxes when they are changed. Blocking this prevents circular loops
        when bounding boxes are selected and the properties are changed based on that
        selection
    _allow_thumbnail_update : bool
        Flag set to true to allow the thumbnail to be updated. Blocking the thumbnail
        can be advantageous where responsiveness is critical.
    _clipboard : dict
        Dict of bounding box objects that are to be used during a copy and paste.
    _colors : list
        List of supported vispy color names.
    _vertex_size : float
        Size of the vertices of the bounding boxes in Canvas
        coordinates.
    _rotation_handle_length : float
        Length of the rotation handle of the bounding box in Canvas
        coordinates.
    _input_ndim : int
        Dimensions of bounding box data.
    _thumbnail_update_thresh : int
        If there are more than this number of bounding boxes, the thumbnail
        won't update during interactive events
    """

    _modeclass = Mode

    _drag_modes = {
        Mode.PAN_ZOOM: no_op,
        Mode.TRANSFORM: transform_with_box,
        Mode.SELECT: select,
        Mode.ADD_BOUNDING_BOX: add_bounding_box
    }

    _move_modes = {
        Mode.PAN_ZOOM: no_op,
        Mode.TRANSFORM: highlight_box_handles,
        Mode.SELECT: highlight,
        Mode.ADD_BOUNDING_BOX: no_op
    }

    _double_click_modes = {
        Mode.PAN_ZOOM: no_op,
        Mode.TRANSFORM: no_op,
        Mode.SELECT: no_op,
        Mode.ADD_BOUNDING_BOX: no_op,
    }

    _cursor_modes = {
        Mode.PAN_ZOOM: 'standard',
        Mode.TRANSFORM: 'standard',
        Mode.SELECT: 'pointing',
        Mode.ADD_BOUNDING_BOX: 'cross'
    }

    _interactive_modes = {
        Mode.PAN_ZOOM,
    }

    def __init__(
        self,
        data=None,
        *,
        ndim=None,
        features=None,
        feature_defaults=None,
        properties=None,
        property_choices=None,
        text=None,
        edge_width=1,
        edge_color='green',
        edge_color_cycle=None,
        edge_colormap='viridis',
        edge_contrast_limits=None,
        face_color='transparent',
        face_color_cycle=None,
        face_colormap='viridis',
        face_contrast_limits=None,
        z_index=0,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        rotate=None,
        shear=None,
        affine=None,
        opacity=0.7,
        blending='translucent',
        visible=True,
        cache=True,
        experimental_clipping_planes=None,
    ) -> None:
        if data is None or len(data) == 0:
            if ndim is None:
                ndim = 2
            data = np.empty((0, 0, ndim))
        else:
            data = np.asarray(data)
            data_ndim = data.shape[-1]
            if ndim is not None and ndim != data_ndim:
                raise ValueError(
                    trans._(
                        "Bounding box dimensions must be equal to ndim",
                        deferred=True,
                    )
                )
            ndim = data_ndim

        Layer.__init__(
            self,
            data,
            ndim=ndim,
            name=name,
            metadata=metadata,
            scale=scale,
            translate=translate,
            rotate=rotate,
            shear=shear,
            affine=affine,
            opacity=opacity,
            blending=blending,
            visible=visible,
            cache=cache,
            experimental_clipping_planes=experimental_clipping_planes,
        )

        self.events.add(
            edge_width=Event,
            edge_color=Event,
            face_color=Event,
            properties=Event,
            current_edge_color=Event,
            current_face_color=Event,
            current_properties=Event,
            highlight=Event,
            features=Event,
            feature_defaults=Event,
            size_mode=Event,
            size_multiplier=Event,
            size_constant=Event
        )

        # Flag set to false to block thumbnail refresh
        self._allow_thumbnail_update = True

        self._display_order_stored = []
        self._ndisplay_stored = self._slice_input.ndisplay

        self._feature_table = _FeatureTable.from_layer(
            features=features,
            feature_defaults=feature_defaults,
            properties=properties,
            property_choices=property_choices,
            num_data=number_of_bounding_boxes(data),
        )

        # The following bounding box properties are for the new bounding boxes that will
        # be drawn. Each bounding box has a corresponding property with the
        # value for itself
        if np.isscalar(edge_width):
            self._current_edge_width = edge_width
        else:
            self._current_edge_width = 1

        self._data_view = BoundingBoxList(ndisplay=self._slice_input.ndisplay)
        self._data_view.slice_key = np.array(layer_slice_indices(self))[
            self._slice_input.not_displayed
        ]

        self._value = (None, None)
        self._value_stored = (None, None)
        self._moving_value = (None, None)
        self._selected_data = set()
        self._selected_data_stored = set()
        self._selected_data_history = set()
        self._selected_box = None
        self._last_cursor_position = None
        self._last_dim_point = None

        self._drag_start = None
        self._fixed_vertex = None
        self._fixed_aspect = False
        self._aspect_ratio = 1
        self._is_moving = False

        # _moving_coordinates are needed for fixing aspect ratio during
        # a resize, it stores the last pointer coordinate value that happened
        # during a mouse move to that pressing/releasing shift
        # can trigger a redraw of the bounding box with a fixed aspect ratio.
        self._moving_coordinates = None

        self._fixed_index = 0
        self._is_selecting = False
        self._drag_box = None
        self._drag_box_stored = None
        self._is_creating = False
        self._clipboard = {}

        self._status = self.mode
        self._size_mode = None
        self.size_mode = SizeMode.AVERAGE
        self._size_multiplier = 1.
        self._size_constant = 20.
        self._init_bounding_boxes(
            data,
            edge_width=edge_width,
            edge_color=edge_color,
            edge_color_cycle=edge_color_cycle,
            edge_colormap=edge_colormap,
            edge_contrast_limits=edge_contrast_limits,
            face_color=face_color,
            face_color_cycle=face_color_cycle,
            face_colormap=face_colormap,
            face_contrast_limits=face_contrast_limits,
            z_index=z_index,
        )

        # set the current_* properties
        if len(data) > 0:
            self._current_edge_color = self.edge_color[-1]
            self._current_face_color = self.face_color[-1]
        elif len(data) == 0 and len(self.properties) > 0:
            self._initialize_current_color_for_empty_layer(edge_color, 'edge')
            self._initialize_current_color_for_empty_layer(face_color, 'face')
        elif len(data) == 0 and len(self.properties) == 0:
            self._current_edge_color = transform_color_with_defaults(
                num_entries=1,
                colors=edge_color,
                elem_name="edge_color",
                default="black",
            )
            self._current_face_color = transform_color_with_defaults(
                num_entries=1,
                colors=face_color,
                elem_name="face_color",
                default="black",
            )

        self._text = TextManager._from_layer(
            text=text,
            features=self.features,
        )

        # Trigger generation of view slice and thumbnail
        self._mouse_down = False
        self.refresh()

    def _initialize_current_color_for_empty_layer(
        self, color: ColorType, attribute: str
    ):
        """Initialize current_{edge,face}_color when starting with empty layer.

        Parameters
        ----------
        color : (N, 4) array or str
            The value for setting edge or face_color
        attribute : str in {'edge', 'face'}
            The name of the attribute to set the color of.
            Should be 'edge' for edge_color or 'face' for face_color.
        """
        color_mode = getattr(self, f'_{attribute}_color_mode')
        if color_mode == ColorMode.DIRECT:
            curr_color = transform_color_with_defaults(
                num_entries=1,
                colors=color,
                elem_name=f'{attribute}_color',
                default="white",
            )

        elif color_mode == ColorMode.CYCLE:
            color_cycle = getattr(self, f'_{attribute}_color_cycle')
            curr_color = transform_color(next(color_cycle))

            # add the new color cycle mapping
            color_property = getattr(self, f'_{attribute}_color_property')
            prop_value = self.feature_defaults[color_property][0]
            color_cycle_map = getattr(self, f'{attribute}_color_cycle_map')
            color_cycle_map[prop_value] = np.squeeze(curr_color)
            setattr(self, f'{attribute}_color_cycle_map', color_cycle_map)

        elif color_mode == ColorMode.COLORMAP:
            color_property = getattr(self, f'_{attribute}_color_property')
            prop_value = self.feature_defaults[color_property][0]
            colormap = getattr(self, f'{attribute}_colormap')
            contrast_limits = getattr(self, f'_{attribute}_contrast_limits')
            curr_color, _ = map_property(
                prop=prop_value,
                colormap=colormap,
                contrast_limits=contrast_limits,
            )
        setattr(self, f'_current_{attribute}_color', curr_color)

    @property
    def data(self):
        """list: Each element is an (N, D) array of the vertices of a bounding box."""
        return self._data_view.data

    @data.setter
    def data(self, data):
        self._finish_drawing()
        
        data = np.asarray(data)
        n_new_bounding_boxes = number_of_bounding_boxes(data)

        edge_widths = self._data_view.edge_widths
        edge_color = self._data_view.edge_color
        face_color = self._data_view.face_color
        z_indices = self._data_view.z_indices

        # fewer bounding boxes, trim attributes
        if self.nbounding_boxes > n_new_bounding_boxes:
            edge_widths = edge_widths[:n_new_bounding_boxes]
            z_indices = z_indices[:n_new_bounding_boxes]
            edge_color = edge_color[:n_new_bounding_boxes]
            face_color = face_color[:n_new_bounding_boxes]
        # more bounding boxes, add attributes
        elif self.nbounding_boxes < n_new_bounding_boxes:
            n_bounding_boxes_difference = n_new_bounding_boxes - self.nbounding_boxes
            edge_widths = edge_widths + [1] * n_bounding_boxes_difference
            z_indices = z_indices + [0] * n_bounding_boxes_difference
            edge_color = np.concatenate(
                (
                    edge_color,
                    self._get_new_bounding_box_color(n_bounding_boxes_difference, 'edge'),
                )
            )
            face_color = np.concatenate(
                (
                    face_color,
                    self._get_new_bounding_box_color(n_bounding_boxes_difference, 'face'),
                )
            )

        self._data_view = BoundingBoxList(ndisplay=self._slice_input.ndisplay)
        self._data_view.slice_key = np.array(layer_slice_indices(self))[
            self._slice_input.not_displayed
        ]
        self.add(
            data,
            edge_width=edge_widths,
            edge_color=edge_color,
            face_color=face_color,
            z_index=z_indices,
        )

        self._update_dims()
        self.events.data(value=self.data)
        self._reset_editable()

    def _on_selection(self, selected: bool):
        # this method is slated for removal.  don't add anything new.
        if not selected:
            self._finish_drawing()

    @property
    def current_properties(self) -> Dict[str, np.ndarray]:
        """dict{str: np.ndarray(1,)}: properties for the next added shape."""
        return self._feature_table.currents()

    @current_properties.setter
    def current_properties(self, current_properties):
        update_indices = None
        if (
                self._update_properties
                and len(self.selected_data) > 0
                and self._mode in [Mode.SELECT, Mode.PAN_ZOOM]
        ):
            update_indices = list(self.selected_data)
        self._feature_table.set_currents(
            current_properties, update_indices=update_indices
        )
        if update_indices is not None:
            self.refresh_colors()
            self.events.properties()
            self.events.features()
        self.events.current_properties()
        self.events.feature_defaults()

    @property
    def selected_data(self):
        """set: set of currently selected bounding boxes."""
        return self._selected_data

    @selected_data.setter
    def selected_data(self, selected_data):
        self._selected_data = set(selected_data)
        self._selected_box = self.interaction_box(self._selected_data)

        # Update properties based on selected bounding boxes
        if len(selected_data) > 0:
            selected_data_indices = list(selected_data)
            selected_face_colors = self._data_view._face_color[
                selected_data_indices
            ]
            if (
                unique_face_color := _unique_element(selected_face_colors)
            ) is not None:
                with self.block_update_properties():
                    self.current_face_color = unique_face_color

            selected_edge_colors = self._data_view._edge_color[
                selected_data_indices
            ]
            if (
                unique_edge_color := _unique_element(selected_edge_colors)
            ) is not None:
                with self.block_update_properties():
                    self.current_edge_color = unique_edge_color

            unique_edge_width = _unique_element(
                np.array(
                    [
                        self._data_view.bounding_boxes[i].edge_width
                        for i in selected_data
                    ]
                )
            )
            if unique_edge_width is not None:
                with self.block_update_properties():
                    self.current_edge_width = unique_edge_width

            unique_properties = {}
            for k, v in self.properties.items():
                unique_properties[k] = _unique_element(
                    v[selected_data_indices]
                )

            if all(p is not None for p in unique_properties.values()):
                with self.block_update_properties():
                    self.current_properties = unique_properties

    def _get_state(self):
        """Get dictionary of layer state.

        Returns
        -------
        state : dict
            Dictionary of layer state.
        """
        state = self._get_base_state()
        face_color = self.face_color
        edge_color = self.edge_color
        if not face_color.size:
            face_color = self._current_face_color
        if not edge_color.size:
            edge_color = self._current_edge_color
        state.update(
            {
                'ndim': self.ndim,
                'properties': self.properties,
                'property_choices': self.property_choices,
                'text': self.text.dict(),
                'opacity': self.opacity,
                'z_index': self.z_index,
                'edge_width': self.edge_width,
                'face_color': face_color,
                'face_color_cycle': self.face_color_cycle,
                'face_colormap': self.face_colormap.name,
                'face_contrast_limits': self.face_contrast_limits,
                'edge_color': edge_color,
                'edge_color_cycle': self.edge_color_cycle,
                'edge_colormap': self.edge_colormap.name,
                'edge_contrast_limits': self.edge_contrast_limits,
                'data': self.data,
                'features': self.features,
                'feature_defaults': self.feature_defaults,
            }
        )
        return state

    @Layer.mode.getter
    def mode(self):
        """MODE: Interactive mode. The normal, default mode is PAN_ZOOM, which
        allows for normal interactivity with the canvas.

        The SELECT mode allows for entire bounding boxes to be selected, moved and
        resized.

        The ADD_BOUNDING_BOX mode allows for bounding boxes to be added.
        """
        return str(self._mode)

    @mode.setter
    def mode(self, mode: Union[str, Mode]):
        mode = self._mode_setter_helper(mode)
        if mode == self._mode:
            return

        self._mode = mode
        self.events.mode(mode=mode)

        draw_modes = {
            Mode.SELECT,
        }

        # don't update thumbnail on mode changes
        with self.block_thumbnail_update():
            if not (mode in draw_modes and self._mode in draw_modes):
                # BoundingBoxLayer._finish_drawing() calls BoundingBoxLayer.refresh()
                self._finish_drawing()
            else:
                self.refresh()

    def add(
        self,
        data,
        *,
        edge_width=None,
        edge_color=None,
        face_color=None,
        z_index=None,
        gui=False
    ):
        """Add bounding boxes to the current layer.

        Parameters
        ----------
        data : Array | List[Array]
            List of bounding box data, where each element is an (N, D) array of the
            N vertices of a bounding box in D dimensions. Can be an 3-dimensional array.
        edge_width : float | list
            thickness of lines and edges. If a list is supplied it must be the
            same length as the length of `data` and each element will be
            applied to each bounding box otherwise the same value will be used for all
            bounding boxes.
        edge_color : str | tuple | list
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements. If a list is supplied it must be the same length as
            the length of `data` and each element will be applied to each bounding box
            otherwise the same value will be used for all bounding boxes.
        face_color : str | tuple | list
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements. If a list is supplied it must be the same length as
            the length of `data` and each element will be applied to each bounding box
            otherwise the same value will be used for all bounding boxes.
        z_index : int | list
            Specifier of z order priority. Bounding boxes with higher z order are
            displayed ontop of others. If a list is supplied it must be the
            same length as the length of `data` and each element will be
            applied to each bounding box otherwise the same value will be used for all
            bounding boxes.
        """
        data = np.asarray(data)
        if edge_width is None:
            edge_width = self.current_edge_width

        n_new_bounding_boxes = number_of_bounding_boxes(data)

        if edge_color is None:
            edge_color = self._get_new_bounding_box_color(
                n_new_bounding_boxes, attribute='edge'
            )
        if face_color is None:
            face_color = self._get_new_bounding_box_color(
                n_new_bounding_boxes, attribute='face'
            )
        if self._data_view is not None:
            z_index = z_index or max(self._data_view._z_index, default=-1) + 1
        else:
            z_index = z_index or 0

        if n_new_bounding_boxes > 0:
            total_bounding_boxes = n_new_bounding_boxes + self.nbounding_boxes
            self._feature_table.resize(total_bounding_boxes)
            self.text.apply(self.features)
            if NAPARI_VERSION >= "0.4.19":
                self.events.data(
                    value=self.data,
                    action=ActionType.ADDING,
                    data_indices=(-1,),
                    vertex_indices=((),),
                )
            self._add_bounding_boxes(
                data,
                edge_width=edge_width,
                edge_color=edge_color,
                face_color=face_color,
                z_index=z_index,
            )
            if NAPARI_VERSION >= "0.4.19":
                if not gui:
                    self.events.data(
                        value=self.data,
                        action=ActionType.ADDED,
                        data_indices=(-1,),
                        vertex_indices=((),),
                    )
            else:
                self.events.data(
                    value=self.data,
                    action=ActionType.ADD.value,
                    data_indices=(-1,),
                    vertex_indices=((),),
                )

    def _add_bounding_boxes(
        self,
        data,
        *,
        edge_width=None,
        edge_color=None,
        face_color=None,
        z_index=None
    ):
        """Add bounding boxes to the data view.

        Parameters
        ----------
        data : Array | List[Array]
            List of bounding box data, where each element is an (N, D) array of the
            N vertices of a bounding box in D dimensions. Can be an 3-dimensional array.
        edge_width : float | list
            thickness of lines and edges. If a list is supplied it must be the
            same length as the length of `data` and each element will be
            applied to each bounding box otherwise the same value will be used for all
            bounding boxes.
        edge_color : str | tuple | list
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements. If a list is supplied it must be the same length as
            the length of `data` and each element will be applied to each bounding box
            otherwise the same value will be used for all bounding boxes.
        face_color : str | tuple | list
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements. If a list is supplied it must be the same length as
            the length of `data` and each element will be applied to each bounding box
            otherwise the same value will be used for all bounding boxes.
        z_index : int | list
            Specifier of z order priority. Bounding boxes with higher z order are
            displayed ontop of others. If a list is supplied it must be the
            same length as the length of `data` and each element will be
            applied to each bounding box otherwise the same value will be used for all
            bounding boxes.
        """
        if edge_width is None:
            edge_width = self.current_edge_width
        if edge_color is None:
            edge_color = self._current_edge_color
        if face_color is None:
            face_color = self._current_face_color
        if self._data_view is not None:
            z_index = z_index or max(self._data_view._z_index, default=-1) + 1
        else:
            z_index = z_index or 0

        if len(data) > 0:
            if np.array(data[0]).ndim == 1:
                # If a single array for a bounding box has been passed turn into list
                data = [data]

            # transform the colors
            transformed_ec = transform_color_with_defaults(
                num_entries=len(data),
                colors=edge_color,
                elem_name="edge_color",
                default="white",
            )
            transformed_edge_color = normalize_and_broadcast_colors(
                len(data), transformed_ec
            )
            transformed_fc = transform_color_with_defaults(
                num_entries=len(data),
                colors=face_color,
                elem_name="face_color",
                default="white",
            )
            transformed_face_color = normalize_and_broadcast_colors(
                len(data), transformed_fc
            )

            # Turn input arguments into iterables
            bounding_box_inputs = zip(
                data,
                ensure_iterable(edge_width),
                transformed_edge_color,
                transformed_face_color,
                ensure_iterable(z_index),
            )

            self._add_bounding_boxes_to_view(bounding_box_inputs, self._data_view)

        self._display_order_stored = copy(layer_dims_order(self))
        self._ndisplay_stored = copy(layer_ndisplay(self))
        self._update_dims()

    def _add_bounding_boxes_to_view(self, bounding_box_inputs, data_view):
        """Build new bounding boxes and add them to the _data_view"""

        bounding_box_inputs = tuple(bounding_box_inputs)

        # build all shapes
        bb_inp = tuple(
            (
                BoundingBox(
                    d,
                    edge_width=ew,
                    z_index=z,
                    dims_order=self._slice_input.order,
                    ndisplay=self._slice_input.ndisplay,
                ),
                ec,
                fc,
            )
            for d, ew, ec, fc, z in bounding_box_inputs
        )

        bounding_boxes, edge_colors, face_colors = tuple(zip(*bb_inp))

        # Add all shapes at once (faster than adding them one by one)
        data_view.add(
            bounding_box=bounding_boxes,
            edge_color=edge_colors,
            face_color=face_colors,
            z_refresh=False,
        )

        data_view._update_z_order()

    @property
    def text(self) -> TextManager:
        """TextManager: The TextManager object containing the text properties"""
        return self._text

    @text.setter
    def text(self, text):
        self._text._update_from_layer(
            text=text,
            features=self.features,
        )

    def _update_thumbnail(self, event=None):
        """Update thumbnail with current bounding boxes and colors."""
        # Set the thumbnail to black, opacity 1
        colormapped = np.zeros(self._thumbnail_shape)
        colormapped[..., 3] = 1
        # if the shapes layer is empty, don't update, just leave it black
        if len(self.data) == 0:
            self.thumbnail = colormapped
        # don't update the thumbnail if dragging a shape
        elif self._is_moving is False and self._allow_thumbnail_update is True:
            # calculate min vals for the vertices and pad with 0.5
            # the offset is needed to ensure that the top left corner of the shapes
            # corresponds to the top left corner of the thumbnail
            de = self._extent_data
            offset = (
                np.array([de[0, d] for d in self._slice_input.displayed]) + 0.5
            )
            # calculate range of values for the vertices and pad with 1
            # padding ensures the entire bounding box can be represented in the thumbnail
            # without getting clipped
            bounding_box = np.ceil(
                [de[1, d] - de[0, d] + 1 for d in self._slice_input.displayed]
            ).astype(int)
            zoom_factor = np.divide(
                self._thumbnail_shape[:2], bounding_box[-2:]
            ).min()

            colormapped = self._data_view.to_colors(
                colors_shape=self._thumbnail_shape[:2],
                zoom_factor=zoom_factor,
                offset=offset[-2:],
                max_bounding_boxes=self._max_bounding_boxes_thumbnail,
            )

            self.thumbnail = colormapped

    def remove_selected(self):
        """Remove any selected bounding boxes."""
        index = list(self.selected_data)
        to_remove = sorted(index, reverse=True)
        for ind in to_remove:
            self._data_view.remove(ind)

        if len(index) > 0:
            if NAPARI_VERSION >= "0.4.19":
                self.events.data(
                    value=self.data,
                    action=ActionType.REMOVING,
                    data_indices=tuple(
                        index,
                    ),
                    vertex_indices=((),),
                )
            self._feature_table.remove(index)
            self.text.remove(index)
            self._data_view._edge_color = np.delete(
                self._data_view._edge_color, index, axis=0
            )
            self._data_view._face_color = np.delete(
                self._data_view._face_color, index, axis=0
            )
        self.selected_data = set()
        self._finish_drawing()
        if NAPARI_VERSION >= "0.4.19":
            self.events.data(
                value=self.data,
                action=ActionType.REMOVED,
                data_indices=tuple(
                    index,
                ),
                vertex_indices=((),),
            )
        else:
            self.events.data(
                value=self.data,
                action=ActionType.REMOVE.value,
                data_indices=tuple(
                    index,
                ),
                vertex_indices=((),),
            )

    def _scale_box(self, scale, center=(0, 0)):
        """Perform a scaling on the selected box.

        Parameters
        ----------
        scale : float, list
            scalar or list specifying rescaling of bounding box.
        center : list
            coordinates of center of rotation.
        """
        if not isinstance(scale, (list, np.ndarray)):
            scale = [scale, scale]
        box = self._selected_box - center
        box = np.array(box * scale)
        # TODO check if that's correct
        if not np.all(box[Box.TOP_CENTER] == box[Box.HANDLE]):
            r = self._rotation_handle_length * self.scale_factor
            handle_vec = box[Box.HANDLE] - box[Box.TOP_CENTER]
            cur_len = np.linalg.norm(handle_vec)
            box[Box.HANDLE] = box[Box.TOP_CENTER] + r * handle_vec / cur_len
        self._selected_box = box + center

    def _get_value_3d(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        dims_displayed: List[int],
    ) -> Tuple[Union[float, int], None]:
        """Get the layer data value along a ray

        Parameters
        ----------
        start_point : np.ndarray
            The start position of the ray used to interrogate the data.
        end_point : np.ndarray
            The end position of the ray used to interrogate the data.
        dims_displayed : List[int]
            The indices of the dimensions currently displayed in the Viewer.

        Returns
        -------
        value
            The data value along the supplied ray.
        vertex : None
            Index of vertex if any that is at the coordinates. Always returns `None`.
        """
        value, _ = self._get_index_and_intersection(
            start_point=start_point,
            end_point=end_point,
            dims_displayed=dims_displayed,
        )

        return value, None

    def _get_index_and_intersection(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        dims_displayed: List[int],
    ) -> Tuple[Union[None, float, int], Union[None, np.ndarray]]:
        """Get the shape index and intersection point of the first shape
        (i.e., closest to start_point) along the specified 3D line segment.

        Note: this method is meant to be used for 3D intersection and returns
        (None, None) when used in 2D (i.e., len(dims_displayed) is 2).

        Parameters
        ----------
        start_point : np.ndarray
            The start position of the ray used to interrogate the data in
            layer coordinates.
        end_point : np.ndarray
            The end position of the ray used to interrogate the data in
            layer coordinates.
        dims_displayed : List[int]
            The indices of the dimensions currently displayed in the Viewer.

        Returns
        -------
        value Union[None, float, int]
            The data value along the supplied ray.
        intersection_point : Union[None, np.ndarray]
            (n,) array containing the point where the ray intersects the first shape
            (i.e., the shape most in the foreground). The coordinate is in layer
            coordinates.
        """
        if len(dims_displayed) != 3:
            # return None if in 2D mode
            return None, None
        if (start_point is None) or (end_point is None):
            # return None if the ray doesn't intersect the data bounding box
            return None, None

        # Get the normal vector of the click plane
        start_position, ray_direction = nd_line_segment_to_displayed_data_ray(
            start_point=start_point,
            end_point=end_point,
            dims_displayed=dims_displayed,
        )
        value, intersection = self._data_view._inside_3d(
            start_position, ray_direction
        )

        # add the full nD coords to intersection
        intersection_point = start_point.copy()
        intersection_point[dims_displayed] = intersection

        return value, intersection_point

    def get_index_and_intersection(
        self,
        position: np.ndarray,
        view_direction: np.ndarray,
        dims_displayed: List[int],
    ) -> Tuple[Union[float, int], None]:
        """Get the shape index and intersection point of the first shape
        (i.e., closest to start_point) "under" a mouse click.

        See examples/add_points_on_nD_shapes.py for example usage.

        Parameters
        ----------
        position : tuple
            Position in either data or world coordinates.
        view_direction : Optional[np.ndarray]
            A unit vector giving the direction of the ray in nD world coordinates.
            The default value is None.
        dims_displayed : Optional[List[int]]
            A list of the dimensions currently being displayed in the viewer.
            The default value is None.

        Returns
        -------
        value
            The data value along the supplied ray.
        intersection_point : np.ndarray
            (n,) array containing the point where the ray intersects the first shape
            (i.e., the shape most in the foreground). The coordinate is in layer
            coordinates.
        """
        start_point, end_point = self.get_ray_intersections(
            position, view_direction, dims_displayed
        )
        if (start_point is not None) and (end_point is not None):
            shape_index, intersection_point = self._get_index_and_intersection(
                start_point=start_point,
                end_point=end_point,
                dims_displayed=dims_displayed,
            )
        else:
            shape_index = (None,)
            intersection_point = None
        return shape_index, intersection_point

    def _copy_data(self):
        """Copy selected shapes to clipboard."""
        if len(self.selected_data) > 0:
            index = list(self.selected_data)
            self._clipboard = {
                'data': [
                    deepcopy(self._data_view.bounding_boxes[i])
                    for i in self._selected_data
                ],
                'edge_color': deepcopy(self._data_view._edge_color[index]),
                'face_color': deepcopy(self._data_view._face_color[index]),
                'features': deepcopy(self.features.iloc[index]),
                'indices': layer_slice_indices(self),
                'text': self.text._copy(index),
            }
        else:
            self._clipboard = {}

    def _paste_data(self):
        """Paste any shapes from clipboard and then selects them."""
        cur_bboxes = self.nbounding_boxes
        if len(self._clipboard.keys()) > 0:
            # Calculate offset based on dimension shifts
            offset = [
                layer_slice_indices(self)[i] - self._clipboard['indices'][i]
                for i in layer_dims_not_displayed(self)
            ]

            self._feature_table.append(self._clipboard['features'])
            self.text._paste(**self._clipboard['text'])

            # Add new shape data
            for i, bb in enumerate(self._clipboard['data']):
                bbox = deepcopy(bb)
                data = copy(bbox.data)
                not_disp = layer_dims_not_displayed(self)
                data[:, not_disp] = data[:, not_disp] + np.array(offset)
                bbox.data = data
                face_color = self._clipboard['face_color'][i]
                edge_color = self._clipboard['edge_color'][i]
                self._data_view.add(
                    bbox, face_color=face_color, edge_color=edge_color
                )

            self.selected_data = set(
                range(cur_bboxes, cur_bboxes + len(self._clipboard['data']))
            )

            self.move_to_front()

    def _store_last_dim_point(self, last_dim_point):
        self._last_dim_point = last_dim_point
