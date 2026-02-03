# A copy of napari._qt.layer_controls.qt_shapes_controls (adapted for napari>=0.6)
from ..napari_0_4_15.qt_bounding_box_control import QtBoundingBoxControls
from collections.abc import Iterable

import numpy as np
from napari._qt.qt_resources import get_current_stylesheet
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QButtonGroup, QCheckBox, QHBoxLayout, QLabel, QComboBox

from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
from napari._qt.utils import (
    set_widgets_enabled_with_opacity,
    qt_signals_blocked,
)
from napari._qt.widgets.qt_color_swatch import QColorSwatchEdit
from napari._qt.widgets.qt_mode_buttons import (
    QtModePushButton,
    QtModeRadioButton,
)
from ...resources import cube_style_path
from ._bounding_box_constants import Mode
from napari.utils.interactions import Shortcut
from napari.utils.translations import trans
from superqt.sliders import QLabeledDoubleSlider, QLabeledSlider


class QtBoundingBoxControls(QtBoundingBoxControls):
    """Qt view and controls for the napari BoundingBoxLayer layer."""

    def __init__(self, layer) -> None:
        QtLayerControls.__init__(self, layer)

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

        sld = QLabeledSlider(Qt.Orientation.Horizontal)
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

        sld = QLabeledSlider(Qt.Orientation.Horizontal)
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

        # Opacity slider (napari>=0.6 no longer provides opacitySlider on base)
        opacity_sld = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        opacity_sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        opacity_sld.setMinimum(0.0)
        opacity_sld.setMaximum(1.0)
        opacity_sld.setSingleStep(0.01)
        opacity_sld.setValue(float(self.layer.opacity))
        opacity_sld.valueChanged.connect(self._on_opacity_change)
        self.opacitySlider = opacity_sld
        self.layer.events.opacity.connect(self._on_layer_opacity_change)

        # Blending combo (napari>=0.6 no longer provides blendComboBox on base)
        blend_combo = QComboBox()
        blending_modes = getattr(self.layer, "_blending_modes", None)
        if blending_modes is None:
            blending_modes = getattr(self.layer, "blending_modes", None)
        if blending_modes is None:
            blending_modes = ("opaque", "translucent", "additive")
        for mode in blending_modes:
            blend_combo.addItem(mode)
        if self.layer.blending in blending_modes:
            blend_combo.setCurrentText(self.layer.blending)
        blend_combo.activated.connect(self._on_blend_combo_change)
        self.blendComboBox = blend_combo
        self.layer.events.blending.connect(self._on_layer_blending_change)

        def _radio_button(
            parent,
            btn_name,
            mode,
            action_name,
            extra_tooltip_text='',
            **kwargs,
        ):
            action_name = f'napari:{action_name}'
            btn = QtModeRadioButton(parent, btn_name, mode, **kwargs)
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
        self.layout().addRow(trans._('opacity:'), self.opacitySlider)
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

    def _on_opacity_change(self, value):
        self.layer.opacity = float(value)

    def _on_layer_opacity_change(self, event=None):
        if hasattr(self, "opacitySlider"):
            with qt_signals_blocked(self.opacitySlider):
                self.opacitySlider.setValue(float(self.layer.opacity))

    def _on_blend_combo_change(self, index):
        self.layer.blending = self.blendComboBox.itemText(index)

    def _on_layer_blending_change(self, event=None):
        if hasattr(self, "blendComboBox"):
            with qt_signals_blocked(self.blendComboBox):
                self.blendComboBox.setCurrentText(self.layer.blending)

    def _on_mode_change(self, event):
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

    def change_text_visibility(self, state):
        self.layer.text.visible = Qt.CheckState(state) == Qt.CheckState.Checked

    def _on_ndisplay_changed(self):
        self.layer.editable = self.ndisplay == 2

    def _on_editable_or_visible_change(self, event=None):
        set_widgets_enabled_with_opacity(
            self,
            self._EDIT_BUTTONS,
            self.layer.editable and self.layer.visible,
        )


from napari._qt.layer_controls.qt_layer_controls_container import layer_to_controls
def register_layer_control(layer_type):
    layer_to_controls[layer_type] = QtBoundingBoxControls
