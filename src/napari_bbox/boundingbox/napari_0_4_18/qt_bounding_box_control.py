# A copy of napari._qt.layer_controls.qt_shapes_controls
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
from napari._qt.qt_resources import get_current_stylesheet
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QButtonGroup, QCheckBox, QHBoxLayout, QLabel, QComboBox
from superqt import QDoubleSlider

from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
from napari._qt.utils import (
    qt_signals_blocked,
    set_widgets_enabled_with_opacity,
)
from napari._qt.widgets._slider_compat import QSlider
from napari._qt.widgets.qt_color_swatch import QColorSwatchEdit
from napari._qt.widgets.qt_mode_buttons import (
    QtModePushButton,
    QtModeRadioButton,
)
from ...resources import cube_style_path
from ._bounding_box_constants import Mode
from napari.utils.action_manager import action_manager
from napari.utils.events import disconnect_events
from napari.utils.interactions import Shortcut
from napari.utils.translations import trans
from superqt.sliders import QLabeledDoubleSlider

if TYPE_CHECKING:
    import napari.layers


class QtBoundingBoxControls(QtLayerControls):
    # TODO: review comments
    """Qt view and controls for the napari BoundingBoxLayer layer.

    Parameters
    ----------
    layer : napari.layers.BoundingBoxLayer
        An instance of a napari BoundingBoxLayer layer.

    Attributes
    ----------
    button_group : qtpy.QtWidgets.QButtonGroup
        Button group for bounding boxes layer modes
        (SELECT, PAN_ZOOM, ADD_BOUNDING_BOX).
    delete_button : qtpy.QtWidgets.QtModePushButton
        Button to delete selected bounding boxes
    edgeColorEdit : QColorSwatchEdit
        Widget allowing user to set edge color of points.
    faceColorEdit : QColorSwatchEdit
        Widget allowing user to set face color of points.
    layer : napari.layers.BoundingBoxLayer
        An instance of a BoundingBoxLayer layer.
    panzoom_button : qtpy.QtWidgets.QtModeRadioButton
        Button to pan/zoom bounding box layer.
    bounding_box_button : qtpy.QtWidgets.QtModeRadioButton
        Button to add rectangles to bounding box layer.
    select_button : qtpy.QtWidgets.QtModeRadioButton
        Button to select bounding boxes.
    textDispCheckBox : qtpy.QtWidgets.QCheckBox
        Checkbox to control if text should be displayed
    widthSlider : qtpy.QtWidgets.QSlider
        Slider controlling line edge width of bounding boxes.

    Raises
    ------
    ValueError
        Raise error if bounding box mode is not recognized.
    """

    layer: 'napari.layers.Shapes'

    def __init__(self, layer) -> None:
        super().__init__(layer)

        self.layer.events.mode.connect(self._on_mode_change)
        self.layer.events.size_mode.connect(self._on_size_mode_change)
        self.layer.events.size_multiplier.connect(self._on_size_multiplier_change)
        self.layer.events.size_constant.connect(self._on_size_constant_change)
        self.layer.events.edge_width.connect(self._on_edge_width_change)
        self.layer.events.current_edge_color.connect(
            self._on_current_edge_color_change
        )
        self.layer.events.current_face_color.connect(
            self._on_current_face_color_change
        )
        self.layer.events.editable.connect(self._on_editable_or_visible_change)
        self.layer.events.visible.connect(self._on_editable_or_visible_change)
        self.layer.text.events.visible.connect(self._on_text_visibility_change)

        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(40)
        sld.setSingleStep(1)
        value = self.layer.current_edge_width
        if isinstance(value, Iterable):
            if isinstance(value, list):
                value = np.asarray(value)
            value = value.mean()
        sld.setValue(int(value))
        sld.valueChanged.connect(self.changeWidth)
        self.widthSlider = sld

        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sld.setMinimum(2)
        sld.setMaximum(200)
        sld.setSingleStep(1)
        value = self.layer.text.size
        if isinstance(value, Iterable):
            if isinstance(value, list):
                value = np.asarray(value)
            value = value.mean()
        sld.setValue(int(value))
        sld.valueChanged.connect(self.changeTextSize)
        self.textSlider = sld

        def _radio_button(
            parent,
            btn_name,
            mode,
            action_name,
            extra_tooltip_text='',
            **kwargs,
        ):
            """
            Convenience local function to create a RadioButton and bind it to
            an action at the same time.

            Parameters
            ----------
            parent : Any
                Parent of the generated QtModeRadioButton
            btn_name : str
                name fo the button
            mode : Enum
                Value Associated to current button
            action_name : str
                Action triggered when button pressed
            extra_tooltip_text : str
                Text you want added after the automatic tooltip set by the
                action manager
            **kwargs:
                Passed to QtModeRadioButton

            Returns
            -------
            button: QtModeRadioButton
                button bound (or that will be bound to) to action `action_name`

            Notes
            -----
            When shortcuts are modifed/added/removed via the action manager, the
            tooltip will be updated to reflect the new shortcut.
            """
            action_name = f'napari:{action_name}'
            btn = QtModeRadioButton(parent, btn_name, mode, **kwargs)
            '''action_manager.bind_button(
                action_name,
                btn,
                extra_tooltip_text='',
            )'''
            return btn

        self.select_button = _radio_button(
            layer, 'select', Mode.SELECT, "activate_select_mode"
        )



        self.panzoom_button = _radio_button(
            layer,
            'pan',
            Mode.PAN_ZOOM,
            "activate_bb_pan_zoom_mode",
            extra_tooltip_text=trans._('(or hold Space)'),
            checked=True,
        )

        self.bounding_box_button = _radio_button(
            layer,
            'bounding box',
            Mode.ADD_BOUNDING_BOX,
            "activate_add_bb_mode",
        )
        self.bounding_box_button.setStyleSheet(get_current_stylesheet([cube_style_path]))

        self.delete_button = QtModePushButton(
            layer,
            'delete_shape',
            slot=self.layer.remove_selected,
            tooltip=trans._(
                "Delete selected bounding boxes ({shortcut})",
                shortcut=Shortcut('Backspace').platform,
            ),
        )

        self._EDIT_BUTTONS = (
            self.select_button,
            self.bounding_box_button,
            self.delete_button,
        )

        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.select_button)
        self.button_group.addButton(self.panzoom_button)
        self.button_group.addButton(self.bounding_box_button)
        self._on_editable_or_visible_change()

        button_row = QHBoxLayout()
        button_row.addWidget(self.delete_button)
        button_row.addWidget(self.select_button)
        button_row.addWidget(self.panzoom_button)
        button_row.addWidget(self.bounding_box_button)
        button_row.setContentsMargins(0, 0, 0, 5)
        button_row.setSpacing(4)

        bb_size_mode_combobox = QComboBox()
        bb_size_mode_combobox.addItem("average")
        bb_size_mode_combobox.addItem("constant")
        bb_size_mode_combobox.activated[str].connect(self.changeSizeMode)
        self.bb_size_mode_combobox = bb_size_mode_combobox

        bb_size_mult_slider = QLabeledDoubleSlider(Qt.Horizontal, parent=self)
        bb_size_mult_slider.setFocusPolicy(Qt.NoFocus)
        bb_size_mult_slider.setMinimum(0.1)
        bb_size_mult_slider.setMaximum(10)
        bb_size_mult_slider.setSingleStep(0.1)
        bb_size_mult_slider.valueChanged.connect(self.changeSizeMultiplier)
        self.bb_size_mult_slider = bb_size_mult_slider
        self.bb_size_mult_label = QLabel(trans._('size multiplier:'), parent=self)
        self._on_size_multiplier_change()

        bb_size_const_slider = QLabeledDoubleSlider(Qt.Horizontal, parent=self)
        bb_size_const_slider.setFocusPolicy(Qt.NoFocus)
        bb_size_const_slider.setMinimum(1)
        bb_size_const_slider.setMaximum(100)
        bb_size_const_slider.setSingleStep(1)
        bb_size_const_slider.valueChanged.connect(self.changeSizeConst)
        self.bb_size_const_slider = bb_size_const_slider
        self.bb_size_const_label = QLabel(trans._('size constant: '), parent=self)
        self._on_size_constant_change()
        self._on_size_mode_change()


        self.faceColorEdit = QColorSwatchEdit(
            initial_color=self.layer.current_face_color,
            tooltip=trans._('click to set current face color'),
        )
        self._on_current_face_color_change()
        self.edgeColorEdit = QColorSwatchEdit(
            initial_color=self.layer.current_edge_color,
            tooltip=trans._('click to set current edge color'),
        )
        self._on_current_edge_color_change()
        self.textColorEdit = QColorSwatchEdit(
            initial_color=self.layer.text.color.constant,
            tooltip=trans._('click to set current text color'),
        )
        self._on_current_text_color_change()

        self.faceColorEdit.color_changed.connect(self.changeFaceColor)
        self.edgeColorEdit.color_changed.connect(self.changeEdgeColor)

        text_disp_cb = QCheckBox()
        text_disp_cb.setToolTip(trans._('toggle text visibility'))
        text_disp_cb.setChecked(self.layer.text.visible)
        text_disp_cb.stateChanged.connect(self.change_text_visibility)
        self.textDispCheckBox = text_disp_cb

        self.layout().addRow(button_row)
        self.layout().addRow(self.opacityLabel, self.opacitySlider)
        self.layout().addRow(trans._('edge width:'), self.widthSlider)
        self.layout().addRow(trans._('blending:'), self.blendComboBox)
        self.layout().addRow(trans._('size mode:'), self.bb_size_mode_combobox)
        self.layout().addRow(self.bb_size_mult_label, self.bb_size_mult_slider)
        self.layout().addRow(self.bb_size_const_label, self.bb_size_const_slider)
        self.layout().addRow(trans._('face color:'), self.faceColorEdit)
        self.layout().addRow(trans._('edge color:'), self.edgeColorEdit)
        self.layout().addRow(trans._('display text:'), self.textDispCheckBox)
        self.layout().addRow(trans._('text color:'), self.textColorEdit)
        self.layout().addRow(trans._('text size:'), self.textSlider)

    def _on_mode_change(self, event):
        """Update ticks in checkbox widgets when bounding boxes layer mode changed.

        Available modes for bounding boxes layer are:
        * SELECT
        * PAN_ZOOM
        * ADD_BOUNDING_BOX

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.

        Raises
        ------
        ValueError
            Raise error if event.mode is not ADD_BOUNDING_BOX, PAN_ZOOM, or SELECT.
        """
        mode_buttons = {
            Mode.SELECT: self.select_button,
            Mode.PAN_ZOOM: self.panzoom_button,
            Mode.ADD_BOUNDING_BOX: self.bounding_box_button,
        }

        if event.mode in mode_buttons:
            mode_buttons[event.mode].setChecked(True)
        elif event.mode != Mode.TRANSFORM:
            raise ValueError(
                trans._("Mode '{mode}'not recognized", mode=event.mode)
            )

    def changeFaceColor(self, color: np.ndarray):
        """Change face color of bounding boxes.

        Parameters
        ----------
        color : np.ndarray
            Face color for bounding boxes, color name or hex string.
            Eg: 'white', 'red', 'blue', '#00ff00', etc.
        """
        with self.layer.events.current_face_color.blocker():
            self.layer.current_face_color = color

    def changeEdgeColor(self, color: np.ndarray):
        """Change edge color of bounding boxes.

        Parameters
        ----------
        color : np.ndarray
            Edge color for bounding boxes, color name or hex string.
            Eg: 'white', 'red', 'blue', '#00ff00', etc.
        """
        with self.layer.events.current_edge_color.blocker():
            self.layer.current_edge_color = color

    def changeTextColor(self, color: np.ndarray):
        """Change edge color of bounding boxes.

        Parameters
        ----------
        color : np.ndarray
            Edge color for bounding boxes, color name or hex string.
            Eg: 'white', 'red', 'blue', '#00ff00', etc.
        """
        with self.layer.text.events.color.blocker():
            self.layer.text.color = color
            self.layer.refresh()

    def changeWidth(self, value):
        """Change edge line width of bounding boxes on the layer model.

        Parameters
        ----------
        value : float
            Line width of bounding boxes.
        """
        self.layer.current_edge_width = float(value) / 2

    def changeTextSize(self, value):
        """Change edge line width of bounding boxes on the layer model.

        Parameters
        ----------
        value : float
            Line width of bounding boxes.
        """
        self.layer.text.size = float(value) / 2

    def changeSizeMode(self, value=None):
        self.layer.size_mode = value

    def changeSizeMultiplier(self, value):
        self.layer.size_multiplier = value

    def changeSizeConst(self, value):
        self.layer.size_constant = value

    def change_text_visibility(self, state):
        """Toggle the visibility of the text.

        Parameters
        ----------
        state : int
            Integer value of Qt.CheckState that indicates the check state of textDispCheckBox
        """
        self.layer.text.visible = Qt.CheckState(state) == Qt.CheckState.Checked

    def _on_text_visibility_change(self):
        """Receive layer model text visibiltiy change change event and update checkbox."""
        with self.layer.text.events.visible.blocker():
            self.textDispCheckBox.setChecked(self.layer.text.visible)

    def _on_edge_width_change(self):
        """Receive layer model edge line width change event and update slider."""
        with self.layer.events.edge_width.blocker():
            value = self.layer.current_edge_width
            value = np.clip(int(2 * value), 0, 40)
            self.widthSlider.setValue(value)

    def _on_current_edge_color_change(self):
        """Receive layer model edge color change event and update color swatch."""
        with qt_signals_blocked(self.edgeColorEdit):
            self.edgeColorEdit.setColor(self.layer.current_edge_color)

    def _on_current_face_color_change(self):
        """Receive layer model face color change event and update color swatch."""
        with qt_signals_blocked(self.faceColorEdit):
            self.faceColorEdit.setColor(self.layer.current_face_color)

    def _on_current_text_color_change(self, event=None):
        """Receive layer model face color change event and update color swatch.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method, by default None.
        """
        with qt_signals_blocked(self.textColorEdit):
            self.textColorEdit.setColor(self.layer.text.color.constant)

    def _on_ndisplay_changed(self):
        self.layer.editable = self.ndisplay == 2

    def _on_editable_or_visible_change(self):
        """Receive layer model editable/visible change event & enable/disable buttons."""
        set_widgets_enabled_with_opacity(
            self,
            self._EDIT_BUTTONS,
            self.layer.editable and self.layer.visible,
        )

    def _on_size_mode_change(self, event=None):
        size_mode = self.layer.size_mode
        self.bb_size_mode_combobox.setCurrentText(size_mode)
        if size_mode == "average":
            self.bb_size_const_label.setVisible(False)
            self.bb_size_const_slider.setVisible(False)
            self.bb_size_mult_label.setVisible(True)
            self.bb_size_mult_slider.setVisible(True)
        elif size_mode == "constant":
            self.bb_size_const_label.setVisible(True)
            self.bb_size_const_slider.setVisible(True)
            self.bb_size_mult_label.setVisible(False)
            self.bb_size_mult_slider.setVisible(False)

    def _on_size_multiplier_change(self, event=None):
        with self.layer.events.size_multiplier.blocker():
            self.bb_size_mult_slider.setValue(self.layer.size_multiplier)

    def _on_size_constant_change(self, event=None):
        with self.layer.events.size_multiplier.blocker():
            self.bb_size_const_slider.setValue(self.layer.size_constant)

    def close(self):
        """Disconnect events when widget is closing."""
        disconnect_events(self.layer.text.events, self)
        super().close()

from napari._qt.layer_controls.qt_layer_controls_container import layer_to_controls
def register_layer_control(layer_type):
    layer_to_controls[layer_type] = QtBoundingBoxControls
