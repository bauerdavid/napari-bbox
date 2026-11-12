import napari
import numpy as np
import pytest
from qtpy.QtWidgets import QLabel

from napari_bbox import BoundingBoxLayer


@pytest.fixture
def controls():
    viewer = napari.Viewer(show=False)
    layer = BoundingBoxLayer(
        [np.array([[10, 10], [30, 30]], dtype=float)], name="boxes"
    )
    viewer.layers.append(layer)
    yield layer, viewer.window._qt_viewer.controls.widgets[layer]
    viewer.close()


def _label_count(widget, text):
    return sum(
        1
        for label in widget.findChildren(QLabel)
        if label.text().strip().lower() == text
    )


def _opacity_and_blending_widgets(widget):
    """The slider and combo box of the one opacity and blending control."""
    base = getattr(widget, "_opacity_blending_controls", None)
    if base is not None:
        return base.opacity_slider, base.blend_combobox
    return widget.opacitySlider, widget.blendComboBox


def test_opacity_and_blending_are_shown_once(controls):
    _, widget = controls
    assert _label_count(widget, "opacity:") == 1
    assert _label_count(widget, "blending:") == 1


def test_opacity_slider_and_layer_stay_in_sync(controls):
    layer, widget = controls
    slider, _ = _opacity_and_blending_widgets(widget)

    def slider_opacity():
        # older napari versions use an integer slider from 0 to 100
        return slider.value() / slider.maximum()

    layer.opacity = 0.3
    assert slider_opacity() == pytest.approx(0.3, abs=0.01)

    value = 0.6 * slider.maximum()
    slider.setValue(round(value) if isinstance(slider.value(), int) else value)
    assert layer.opacity == pytest.approx(0.6, abs=0.01)


def test_blending_combo_follows_the_layer(controls):
    layer, widget = controls
    _, combo = _opacity_and_blending_widgets(widget)

    layer.blending = "additive"
    assert combo.currentText().lower() == "additive"
