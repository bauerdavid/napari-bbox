# A copy of napari.qt_layer_controls_base.py and napari._qt.layer_controls.qt_shapes_controls
from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
from napari._qt.layer_controls.widgets import (
    QtFaceColorControl,
    QtTextVisibilityControl,
    QtOpacityBlendingControls,
    QtWidgetControlsBase,
)

from napari._qt.layer_controls.widgets._shapes import (
    QtEdgeColorControl,
    QtEdgeWidthSliderControl,
)

# [NOTE] removed for bounding box plugin
#from napari.layers.shapes._shapes_constants import Mode
from napari.layers.base.base import Layer

from napari._qt.utils import set_widgets_enabled_with_opacity
from napari._qt.widgets.qt_mode_buttons import QtModePushButton, QtModeRadioButton
from napari.utils.action_manager import action_manager
from napari.utils.events import disconnect_events
from napari.utils.interactions import Shortcut
from napari.utils.translations import trans

# [NOTE] addition for bounding box plugin
from ._bounding_box_constants import Mode
from .bounding_boxes import BoundingBoxLayer

from qtpy.QtCore import Qt
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import (
    QWidget,
    QButtonGroup,
    QFormLayout,
    QFrame,
    QGridLayout,
    QMessageBox,
    QComboBox,
    QLabel
)

# [NOTE] addition for bounding box plugin
from superqt.sliders import QLabeledDoubleSlider, QLabeledSlider
from napari._qt.qt_resources import get_current_stylesheet
from ...resources import cube_style_path

from collections.abc import Iterable
import numpy as np
from napari._qt.utils import qt_signals_blocked
from napari.utils.events.event_utils import connect_setattr

# [TODO] check [CLANKER] optimisation
class QtBBEdgeWidthSliderControl(QtEdgeWidthSliderControl):
    """
    Class that wraps the connection of events/signals between the current edge
    width layer attribute and Qt widgets.

    Parameters
    ----------
    parent : qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : BoundingBoxLayer
        An instance of a BoundingBoxLayer.
    scale_factor : float
        A scale factor for dividing BoundingBoxLayer edge width.

    Attributes
    ----------
    edge_width_slider : superqt.QLabeledDoubleSlider
        Slider controlling line edge width of layer.
    edge_width_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the current edge width widget.
    scale_factor : float
        A scale factor for dividing BoundingBoxLayer edge width.
    """
    def __init__(
        self,
        parent: QWidget,
        layer: BoundingBoxLayer,
        scale_factor: float = 0.5,
    ) -> None:
        self.scale_factor = scale_factor

        # initialize parent
        super().__init__(parent, layer)

        # disconnect original connection if needed
        try:
            self.edge_width_slider.valueChanged.disconnect()
        except TypeError:
            pass

        # set slider from scaled layer value
        value = self._layer.current_edge_width

        if isinstance(value, Iterable):
            if isinstance(value, list):
                value = np.asarray(value)
            value = value.mean()

        scaled_value = value / self.scale_factor
        self.edge_width_slider.setValue(int(scaled_value))

        # reconnect with scaling
        self.edge_width_slider.valueChanged.connect(
            self._on_slider_changed
        )

    def _on_slider_changed(self, value: int) -> None:
        """Convert slider value -> actual edge width."""
        self._layer.current_edge_width = float(
            value * self.scale_factor
        )

    def _on_edge_width_change(self) -> None:
        """Convert actual edge width -> slider value."""
        with qt_signals_blocked(self.edge_width_slider):
            value = self._layer.current_edge_width

            scaled_value = value / self.scale_factor
            scaled_value = np.clip(int(scaled_value), 0, 40)

            self.edge_width_slider.setValue(scaled_value)

class LayerFormLayout(QFormLayout):
    """Reusable form layout for subwidgets in each QtLayerControls class"""

    def __init__(self, QWidget=None) -> None:
        super().__init__(QWidget)
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(4)
        self.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)


class QtSuperBoundingBoxControls(QFrame):
    """Superclass for BoundingBoxControl classes.

    This class is never directly instantiated anywhere.

    Parameters
    ----------
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    MODE : Enum
        Available modes in the associated layer.
    PAN_ZOOM_ACTION_NAME : str
        String id for the pan-zoom action to bind to the pan_zoom button.
    TRANSFORM_ACTION_NAME : str
        String id for the transform action to bind to the transform button.
    _opacity_blending_controls: napari._qt.layer_controls.widgets.QtOpacityBlendingControls
        Wrapper widget with a dropdown widget to select the layer blending mode and
        a slider for the layer opacity.
    button_grid : qtpy.QtWidgets.QGridLayout
        GridLayout for the layer mode buttons
    button_group : qtpy.QtWidgets.QButtonGroup
        Button group for image based layer modes (PAN_ZOOM TRANSFORM).
    layer : napari.layers.Layer
        An instance of a napari layer.
    panzoom_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to activate move camera mode for layer.
    transform_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to transform layer.
    """

    MODE = Mode
    PAN_ZOOM_ACTION_NAME = ''
    TRANSFORM_ACTION_NAME = ''

    def __init__(self, layer: BoundingBoxLayer) -> None:
        super().__init__()

        self._ndisplay: int = 2
        self._EDIT_BUTTONS: tuple = ()
        self._MODE_BUTTONS: dict = {}

        self.layer = layer
        self.layer.events.mode.connect(self._on_mode_change)
        self.layer.events.editable.connect(self._on_editable_or_visible_change)
        self.layer.events.visible.connect(self._on_editable_or_visible_change)

        self.setObjectName('layer')
        self.setMouseTracking(True)

        self.setLayout(LayerFormLayout(self))

        # Buttons
        self.button_group = QButtonGroup(self)
        # TODO:
        self.panzoom_button = self._radio_button(
            layer,
            'pan',
            self.MODE.PAN_ZOOM,
            False,
            self.PAN_ZOOM_ACTION_NAME,
            extra_tooltip_text=trans._(
                '\n(or hold Space)\n(hold Shift to pan in 3D)'
                '\n(hold Alt to zoom via ROI selection)',
            ),
            checked=True,
        )
        self.transform_button = self._radio_button(
            layer,
            'transform',
            self.MODE.TRANSFORM,
            True,
            self.TRANSFORM_ACTION_NAME,
            extra_tooltip_text=trans._(
                '\nAlt + Left mouse click over this button to reset'
            ),
        )
        self.transform_button.installEventFilter(self)
        self._on_editable_or_visible_change()

        self.button_grid = QGridLayout()
        self.button_grid.addWidget(self.panzoom_button, 0, 5) # [NOTE] changed to work with bounding box plugin
        self.button_grid.addWidget(self.transform_button, 0, 6) # [NOTE] changed to work with bounding box plugin
        self.button_grid.setContentsMargins(5, 0, 0, 5)
        self.button_grid.setColumnStretch(0, 1)
        self.button_grid.setSpacing(4)
        self.layout().addRow(self.button_grid)

        # Setup widgets controls
        self._opacity_blending_controls = QtOpacityBlendingControls(
            self, layer
        )
        self._add_widget_controls(self._opacity_blending_controls)

    def _add_widget_controls(
        self,
        wrapper: QtWidgetControlsBase,
    ) -> None:
        """
        Add widget controls.

        Parameters
        ----------
        wrapper : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWidgetControlsBase
            An instance of a `QtWidgetControlsBase` subclass that setups
            widgets for a layer attribute.
        """
        controls = wrapper.get_widget_controls()

        for label_text, control_widget in controls:
            self.layout().addRow(label_text, control_widget)

    def changeProjectionMode(self, text):
        with self.layer.events.blocker(self._on_projection_mode_change):
            self.layer.projection_mode = text

    def _radio_button(
        self,
        layer,
        btn_name,
        mode,
        edit_button,
        action_name,
        extra_tooltip_text='',
        **kwargs,
    ):
        """
        Convenience local function to create a RadioButton and bind it to
        an action at the same time.

        Parameters
        ----------
        layer : napari.layers.Layer
            The layer instance that this button controls.n
        btn_name : str
            name fo the button
        mode : Enum
            Value Associated to current button
        edit_button: bool
            True if the button corresponds to edition operations. False otherwise.
        action_name : str
            Action triggered when button pressed
        extra_tooltip_text : str
            Text you want added after the automatic tooltip set by the
            action manager
        **kwargs:
            Passed to napari._qt.widgets.qt_mode_button.QtModeRadioButton

        Returns
        -------
        button: napari._qt.widgets.qt_mode_button.QtModeRadioButton
            button bound (or that will be bound to) to action `action_name`

        Notes
        -----
        When shortcuts are modifed/added/removed via the action manager, the
        tooltip will be updated to reflect the new shortcut.
        """
        action_name = f'napari:{action_name}'
        btn = QtModeRadioButton(layer, btn_name, mode, **kwargs)
        # [NOTE] editted out in bounding box plugin
        action_manager.bind_button(
            action_name,
            btn,
            extra_tooltip_text=extra_tooltip_text,
        )
        self._MODE_BUTTONS[mode] = btn
        self.button_group.addButton(btn)
        if edit_button:
            self._EDIT_BUTTONS += (btn,)
        return btn

    def _on_mode_change(self, event):
        """
        Update ticks in checkbox widgets when image based layer mode changed.

        Available modes for base layer are:
        * PAN_ZOOM
        * TRANSFORM

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.

        Raises
        ------
        ValueError
            Raise error if event.mode is not PAN_ZOOM or TRANSFORM.
        """
        if event.mode in self._MODE_BUTTONS:
            self._MODE_BUTTONS[event.mode].setChecked(True)
        else:
            raise ValueError(
                trans._("Mode '{mode}' not recognized", mode=event.mode)
            )

    def _on_editable_or_visible_change(self):
        """Receive layer model editable/visible change event & enable/disable buttons."""
        set_widgets_enabled_with_opacity(
            self,
            self._EDIT_BUTTONS,
            self.layer.editable and self.layer.visible,
        )
        self._set_transform_tool_state()

    @property
    def ndisplay(self) -> int:
        """The number of dimensions displayed in the canvas."""
        return self._ndisplay

    @ndisplay.setter
    def ndisplay(self, ndisplay: int) -> None:
        self._ndisplay = ndisplay
        self._on_ndisplay_changed()

    def _on_ndisplay_changed(self) -> None:
        """Respond to a change to the number of dimensions displayed in the viewer.

        This is needed because some layer controls may have options that are specific
        to 2D or 3D visualization only like the transform mode button.
        """
        self._set_transform_tool_state()

    def _set_transform_tool_state(self):
        """
        Enable/disable transform button taking into account:
            * Layer visibility.
            * Layer editability.
            * Number of dimensions being displayed.
        """
        set_widgets_enabled_with_opacity(
            self,
            [self.transform_button],
            self.layer.editable and self.layer.visible and self.ndisplay == 2,
        )

    def _disconnect_child_widget_controls(self, child) -> None:
        disconnect_method = getattr(child, 'disconnect_widget_controls', None)
        if disconnect_method is not None:
            disconnect_method()

    def eventFilter(self, qobject, event):
        """
        Event filter implementation to handle the Alt + Left mouse click interaction to
        reset the layer transform.

        For more info about Qt Event Filters you can check:
            https://doc.qt.io/qt-6/eventsandfilters.html#event-filters
        """
        if (
            qobject == self.transform_button
            and event.type() == QMouseEvent.MouseButtonRelease
            and event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() == Qt.AltModifier
        ):
            result = QMessageBox.warning(
                self,
                trans._('Reset transform'),
                trans._('Are you sure you want to reset transforms?'),
                QMessageBox.Yes | QMessageBox.No,
            )
            if result == QMessageBox.Yes:
                self.layer._reset_affine()
                return True
        return super().eventFilter(qobject, event)

    def deleteLater(self):
        disconnect_events(self.layer.events, self)
        for child in self.children():
            self._disconnect_child_widget_controls(child)
        super().deleteLater()

    def close(self):
        """Disconnect events when widget is closing."""
        disconnect_events(self.layer.events, self)
        for child in self.children():
            close_method = getattr(child, 'close', None)
            self._disconnect_child_widget_controls(child)
            if close_method is not None:
                close_method()
        return super().close()

class QtBoundingBoxControls(QtSuperBoundingBoxControls):
    """Qt view and controls for the BoudingBoxLayer.

    Parameters
    ----------
    layer : BoudingBoxLayer
        An instance of a BoudingBoxLayer.

    Attributes
    ----------
    _edge_color_control : napari._qt.layer_controls.widgets._shapes.QtEdgeColorControl
        Widget that wraps a ColorSwatchEdit controlling current edge color of the layer.
    _edge_width_slider_control : QtBBEdgeWidthSliderControl
        Widget that wraps a slider controlling line edge width of layer.
    _face_color_control : napari._qt.layer_controls.widgets.QtFaceColorControl
        Widget that wraps a ColorSwatchEdit controlling current face color of the layer.
    _text_visibility_control : napari._qt.layer_controls.widgets.QtTextVisibilityControl
        WIdget that wraps a checkbox controlling if text on the layer is visible or not.
    delete_button : qtpy.QtWidgets.QtModePushButton
        Button to delete selected bounding boxes.
    rectangle_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to add bounding boxes to bounding box layer.
    select_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to select bounding boxes.

    Raises
    ------
    ValueError
        Raise error if bounding box mode is not recognized.
    """

    layer: 'BoundingBoxLayer'
    MODE = Mode
    PAN_ZOOM_ACTION_NAME = 'activate_bb_pan_zoom_mode'
    TRANSFORM_ACTION_NAME = 'activate_transform_mode'

    def __init__(self, layer) -> None:
        super().__init__(layer)

        # Setup buttons
        self.select_button = self._radio_button(
            layer, 'select', Mode.SELECT, True, 'activate_bb_select_mode'
        )
        
        self.bounding_box_button = self._radio_button(
            layer,
            'bounding box',
            Mode.ADD_BOUNDING_BOX,
            True,
            'activate_add_bb_mode',
        )
        self.bounding_box_button.setStyleSheet(get_current_stylesheet([cube_style_path]))
        
        self.delete_button = QtModePushButton(
            layer,
            'delete_shape',
            slot=self.layer.remove_selected,
            tooltip=trans._(
                'Delete selected bounding boxes ({shortcut})',
                shortcut=Shortcut('Backspace').platform,
            ),
        )
        # [NOTE] select_button and bounding_box_button should be added to _EDIT_BUTTONS automatically with _radio_button()
        self._EDIT_BUTTONS += (
            self.delete_button,
        )
        self._on_editable_or_visible_change()

        self.button_grid.addWidget(self.delete_button, 0, 0)
        self.button_grid.addWidget(self.select_button, 0, 1)
        self.button_grid.addWidget(self.bounding_box_button, 0, 2)
        self.button_grid.setContentsMargins(5, 0, 0, 5)
        self.button_grid.setColumnStretch(0, 1)
        self.button_grid.setSpacing(4)

        # Setup widgets controls
        self._edge_width_slider_control = QtBBEdgeWidthSliderControl(self, layer)
        self._add_widget_controls(self._edge_width_slider_control)
        self._edge_color_control = QtEdgeColorControl(
            self,
            layer,
            tooltip=trans._(
                'Click to set the edge color of currently selected bounding boxes and any added afterwards'
            ),
        )
        self._add_widget_controls(self._edge_color_control)
        self._face_color_control = QtFaceColorControl(
            self,
            layer,
            tooltip=trans._(
                'Click to set the face color of currently selected bounding boxes and any added afterwards.'
            ),
        )
        self._add_widget_controls(self._face_color_control)
        self._text_visibility_control = QtTextVisibilityControl(self, layer)
        self._add_widget_controls(self._text_visibility_control)

        # [NOTE] bounding box plugin specific
        self.layer.events.size_mode.connect(self._on_size_mode_change)
        self.layer.events.size_multiplier.connect(self._on_size_multiplier_change)
        self.layer.events.size_constant.connect(self._on_size_constant_change)

        bb_size_mode_combobox = QComboBox()
        bb_size_mode_combobox.addItem("average")
        bb_size_mode_combobox.addItem("constant")
        bb_size_mode_combobox.activated.connect(self.changeSizeMode)
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

        self.layout().addRow(trans._('size mode:'), self.bb_size_mode_combobox)
        self.layout().addRow(self.bb_size_mult_label, self.bb_size_mult_slider)
        self.layout().addRow(self.bb_size_const_label, self.bb_size_const_slider)

    def _on_mode_change(self, event):
        """Update ticks in checkbox widgets when bounding box layer mode changed.

        Available modes for bounding box layer are:
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
            Raise error if event.mode is not one of the available modes.
        """
        super()._on_mode_change(event)

    def _on_ndisplay_changed(self):
        self.layer.editable = self.ndisplay == 2
        super()._on_ndisplay_changed()

    # [NOTE] bounding box plugin specific
    def changeSizeMode(self, value=None):
        mode = self.bb_size_mode_combobox.itemText(value)
        self.layer.size_mode = mode

    def changeSizeMultiplier(self, value):
        self.layer.size_multiplier = value

    def changeSizeConst(self, value):
        self.layer.size_constant = value
    
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

# [NOTE] have to register the BoundingBoxControls
from napari._qt.layer_controls.qt_layer_controls_container import layer_to_controls
def register_layer_control(layer_type):
    layer_to_controls[layer_type] = QtBoundingBoxControls