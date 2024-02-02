"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from .boundingbox import BoundingBoxLayer
from qtpy.QtWidgets import QPushButton, QHBoxLayout, QVBoxLayout, QSpinBox, QLabel, QFrame, QFileDialog, QWidget
from ._writer import write_single_bbox
from ._reader import read_bbox
from napari import Viewer
from napari.utils.notifications import show_info


class BoundingBoxCreator(QWidget):
    def __init__(self, viewer: Viewer):
        super().__init__()
        self.viewer = viewer
        self._active_layer = None
        viewer.layers.selection.events.active.connect(self.update_active_layer)
        layout = QVBoxLayout()

        ndims_spinbox = QSpinBox()
        ndims_spinbox.setMinimum(2)
        ndims_spinbox.setMaximum(10)
        ndims_spinbox.setValue(2)
        ndims_layout = QHBoxLayout()
        ndims_layout.addWidget(QLabel("ndims:"))
        ndims_layout.addWidget(ndims_spinbox)
        layout.addLayout(ndims_layout)

        create_button = QPushButton("Create")
        create_button.clicked.connect(lambda: viewer.add_bounding_boxes(ndim=ndims_spinbox.value()))
        layout.addWidget(create_button)
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        layout.addWidget(separator)
        open_button = QPushButton("Open")
        open_button.clicked.connect(self.open_layer)
        layout.addWidget(open_button)
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        layout.addWidget(separator)
        # layout.addWidget(self.bounding_boxes.combobox)
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_layer)
        self.update_active_layer()
        layout.addWidget(self.save_button)
        self.setLayout(layout)

    def save_layer(self):
        layer = self._active_layer
        if layer is None:
            raise ValueError("No bounding box layer selected")
        path = QFileDialog.getSaveFileName(filter="*.csv")[0]
        if not path:
            return
        write_single_bbox(path, layer.data, {})
        show_info("layer '%s' saved successfully" % layer.name)

    def open_layer(self):
        path = QFileDialog.getOpenFileName(filter="*.csv")[0]
        if not path:
            return
        self.viewer.add_layer(BoundingBoxLayer(read_bbox(path)[0][0]))

    def update_active_layer(self, e=None):
        self._active_layer = e.source.active if e else None
        self.save_button.setEnabled(is_bbox_layer := type(self._active_layer) is BoundingBoxLayer)
        self.save_button.setToolTip("Select a bounding box layer from the list to save" if not is_bbox_layer else "")
