import time
import warnings
from enum import Enum
from functools import partial

from qtpy.QtWidgets import QSpacerItem, QSizePolicy
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QSpinBox, QCheckBox
from qtpy.QtWidgets import QTableWidget, QTableWidgetItem, QWidget, QGridLayout, QPushButton, QFileDialog
from qtpy.QtCore import Qt
from magicgui.widgets import Table
from napari._qt.qthreading import thread_worker
from qtpy.QtCore import QTimer

from magicgui import magic_factory
from ._functions import topographic_image, topographic_points, topographic_surface
from napari.types import ImageData, LayerDataTuple
from typing import List


import pyqtgraph as pg
import numpy as np
import napari
from napari_tools_menu import register_dock_widget

@register_dock_widget(menu="Measurement > Plot profile")
class PlotProfile(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer
        napari_viewer.layers.selection.events.changed.connect(self._on_selection)

        self._data = None
        self._former_line = None

        graph_container = QWidget()

        # histogram view
        self._graphics_widget = pg.GraphicsLayoutWidget()
        self._graphics_widget.setBackground(None)

        #graph_container.setMaximumHeight(100)
        graph_container.setLayout(QHBoxLayout())
        graph_container.layout().addWidget(self._graphics_widget)

        # individual layers: legend
        self._labels = QWidget()
        self._labels.setLayout(QVBoxLayout())
        self._labels.layout().setSpacing(0)

        # setup layout
        self.setLayout(QVBoxLayout())

        self.layout().addWidget(graph_container)
        self.layout().addWidget(self._labels)

        num_points_container = QWidget()
        num_points_container.setLayout(QHBoxLayout())

        lbl = QLabel("Number of points")
        num_points_container.layout().addWidget(lbl)
        self._sp_num_points = QSpinBox()
        self._sp_num_points.setMinimum(2)
        self._sp_num_points.setMaximum(10000000)
        self._sp_num_points.setValue(100)
        num_points_container.layout().addWidget(self._sp_num_points)
        num_points_container.layout().setSpacing(0)
        self.layout().addWidget(num_points_container)


        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self._on_selection)
        self.layout().addWidget(btn_refresh)

        self._cb_live_update = QCheckBox("Live update")
        self._cb_live_update.setChecked(True)
        self.layout().addWidget(self._cb_live_update)

        btn_list_values = QPushButton("List values")
        btn_list_values.clicked.connect(self._list_values)
        self.layout().addWidget(btn_list_values)

        verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.layout().addItem(verticalSpacer)
        # self.layout().setSpacing(0)

        # we're using a thread worker because layer.mouse_drag_callbacks doesn't work
        # as expected in napari 0.4.10.
        # Todo: check later if the thread_worker can be replaced with a mouse/move/drag event
        # https://napari.org/guides/stable/threading.html
        #@thread_worker
        #def loop_run():
        #    while True:  # endless loop
        #        time.sleep(0.5)
        #        yield True

        #worker = loop_run()

        #def update_layer(whatever):
        #    if self.cb_live_update.isChecked():
        #        self.redraw()
        #    if not self.isVisible():
        #        worker.quit()

        # Start the loop
        #worker.yielded.connect(update_layer)
        #worker.start()

        self._timer = QTimer()
        self._timer.setInterval(500)

        @self._timer.timeout.connect
        def update_layer(*_):
            if self._cb_live_update.isChecked():
                self.redraw()
            if not self.isVisible():
                self._timer.stop()

        self._timer.start()

        self.redraw()

    @property
    def data(self):
        warnings.warn("PlotProfile().data is deprecated. Use PlotProfile().to_table() instead.", DeprecationWarning)
        return self._data

    def _on_selection(self, event):
        # redraw when layer selection has changed
        self.redraw(force_redraw=True)


    def to_table(self):
        table = {}
        for my_profile in self._data:
            positions = np.asarray(my_profile['positions'])
            for i, x in enumerate(positions[0]):
                table[my_profile['name'] + '_pos' + str(i)] = positions[:, i]

            table[my_profile['name'] + '_intensity'] = my_profile['intensities']
            table[my_profile['name'] + '_distance'] = my_profile['distances']
        return table

    def _list_values(self):
        table = self.to_table()

        # turn table into a widget
        first_selected_layer = self.selected_image_layers()[0]
        first_selected_layer.properties = table
        from napari_skimage_regionprops import add_table
        add_table(first_selected_layer, self._viewer)

    def _get_current_line(self):
        """Return the current line as an (n_vertices, viewer_ndim) array in world coordinates.

        Displayed dimensions come from the line; all other dimensions (e.g. time,
        channel, z in 2D view) come from the viewer's current slider position, so
        the profile follows the slice being looked at.
        """
        for layer in self._viewer.layers.selection:
            if isinstance(layer, napari.layers.Shapes):
                selection = list(layer.selected_data)
                if len(selection) > 0:
                    return line_to_world(layer.data[selection[0]], layer, self._viewer.dims)
                if len(layer.data) > 0:
                    return line_to_world(layer.data[-1], layer, self._viewer.dims)
        return None

    def redraw(self, force_redraw : bool = False):

        line = self._get_current_line()

        if line is None:
            #self._reset_plot()
            return


        if not force_redraw:
            if self._former_line is not None and self._former_line.shape == line.shape \
                    and np.array_equal(line, self._former_line):
                return

        self._reset_plot()

        self._former_line = line.copy()

        # clean up
        layout = self._labels.layout()
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().setParent(None)

        # visualize plots
        num_bins = self._sp_num_points.value()
        colors = []
        self._data = []
        for i, layer in enumerate(self.selected_image_layers()):
            # plot profile
            my_profile = profile(layer, line, num_points=num_bins)
            my_profile['name'] = layer.name
            self._data.append(my_profile)

            colormap = layer.colormap.colors
            color = np.asarray(colormap[-1, 0:3]) * 255
            colors.append(color)

            intensities = my_profile['intensities']
            if len(intensities) > 0:
                self.p2.plot(my_profile['distances'], intensities, pen=color, name=layer.name)

                text = '[%0.2f .. %0.2f], %0.2f +- %0.2f' % (np.min(intensities),np.max(intensities),np.mean(intensities),np.std(intensities))

                row = LayerLabelWidget(layer, text, colors[i], self)
                layout.addWidget(row)


    def _reset_plot(self):
        if not hasattr(self, "p2"):
            self.p2 = self._graphics_widget.addPlot()
            axis = self.p2.getAxis('bottom')
            axis.setLabel("Distance")
            axis = self.p2.getAxis('left')
            axis.setLabel("Intensity")
        else:
            self.p2.clear()

    def selected_image_layers(self):
        return [layer for layer in self._viewer.layers if (isinstance(layer, napari.layers.Image) and layer.visible)]

class LayerLabelWidget(QWidget):
    def __init__(self, layer, text, color, gui):
        super().__init__()

        self.setLayout(QHBoxLayout())

        lbl = QLabel(layer.name + text)
        lbl.setStyleSheet('color: #%02x%02x%02x' % tuple(color.astype(int)))
        self.layout().addWidget(lbl)

def line_to_world(line, shapes_layer, dims):
    """Convert a shape's vertices to world coordinates spanning all viewer dimensions.

    Coordinates along the displayed dimensions are taken from the shape; the
    remaining (sliced) dimensions are taken from the viewer's current position.
    This makes a 2D line drawn on e.g. a 5D (t, c, z, y, x) dataset sample the
    currently shown time point / channel / plane.
    """
    current = np.asarray(dims.point, dtype=float)
    displayed = list(dims.displayed)
    world_line = np.tile(current, (len(line), 1))
    for i, vertex in enumerate(np.asarray(line, dtype=float)):
        vertex_world = np.asarray(shapes_layer.data_to_world(vertex), dtype=float)
        full = current.copy()
        n = min(len(vertex_world), len(full))
        full[len(full) - n:] = vertex_world[len(vertex_world) - n:]
        world_line[i, displayed] = full[displayed]
    return world_line


def _layer_level0_data(layer):
    """Full-resolution array of an image layer (first pyramid level for multiscale data)."""
    return layer.data[0] if layer.multiscale else layer.data


def _read_points(data, positions):
    """Read values at integer positions (n_points, ndim) without loading the whole array."""
    if len(positions) == 0:
        return np.asarray([])
    index = tuple(positions.T)
    if isinstance(data, np.ndarray):
        return data[index]
    if hasattr(data, 'vindex'):
        # dask and zarr arrays: point-wise indexing, only touches the chunks hit by the line
        return np.asarray(data.vindex[index])
    # other array-likes: read the bounding box around the samples only
    lower = positions.min(axis=0)
    upper = positions.max(axis=0) + 1
    block = np.asarray(data[tuple(slice(lo, hi) for lo, hi in zip(lower, upper))])
    return block[tuple((positions - lower).T)]


def profile(layer, line, num_points : int = 256):
    """Sample intensities of an image layer along a polyline.

    Parameters
    ----------
    layer : napari.layers.Image
        Image layer; may be n-dimensional, multiscale (e.g. OME-Zarr pyramids) and
        backed by lazy arrays (dask, zarr). Only the pixels along the line are read.
    line : array of shape (n_vertices, n_dims)
        Polyline in world coordinates. If it has more dimensions than the layer, the
        trailing ones are used; if fewer, leading dimensions are taken as 0.
    num_points : int
        Number of sample points along the line.

    Returns
    -------
    dict with 'positions' (integer pixel indices in the layer's full-resolution data),
    'distances' (along the line, in world units) and 'intensities'.
    """
    line = np.asarray(line, dtype=float)
    if line.ndim != 2 or len(line) == 0:
        return {'positions': [], 'distances': [], 'intensities': []}

    # cumulative world-space distance at every vertex
    segment_lengths = np.linalg.norm(np.diff(line, axis=0), axis=1)
    vertex_distances = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = vertex_distances[-1]

    # world positions of evenly spaced samples along the polyline
    sample_distances = np.linspace(0, total, num_points) if total > 0 else np.zeros(1)
    if total > 0:
        samples_world = np.stack([np.interp(sample_distances, vertex_distances, line[:, d])
                                  for d in range(line.shape[1])], axis=1)
    else:
        samples_world = line[:1]

    # world -> layer data coordinates (handles scale, translate, rotate, affine, and
    # world/layer dimensionality mismatch); nearest pixel
    data_positions = np.array([layer.world_to_data(p) for p in samples_world], dtype=float)
    data_positions = np.round(data_positions).astype(int)

    data = _layer_level0_data(layer)
    spatial_shape = np.asarray(data.shape[:layer.ndim])  # excludes RGB(A) channel axis
    inside = np.all((data_positions >= 0) & (data_positions < spatial_shape), axis=1)
    positions = data_positions[inside]
    distances = sample_distances[inside]

    intensities = _read_points(data, positions)

    return {
        'positions': list(positions),
        'distances': list(distances),
        'intensities': list(intensities)
    }

def min_max(data):
    return data.min(), data.max()

class TopographicalVisualization(Enum):
    Image = partial(topographic_image)
    Points = partial(topographic_points)
    Surface = partial(topographic_surface)

@register_dock_widget(menu="Visualization > Topographical view (npp)")
@magic_factory(step_size={"visible": True})
def topographical_view(image: ImageData, visualize_as:TopographicalVisualization = TopographicalVisualization.Image,
                       step_size: int = 1) -> List[LayerDataTuple]:
    """Return a 3D topographical view from a 2D image.

    This function warps pixels intensities to heights and returns a 3D visualization as specified.

    Parameters
    ----------
    image : 2D-array
        Grayscale 2D input image; will be converted to int internally
    visualize_as: TopographicalVisualization
        Type of visualization: Image(s), Points or Surfaces
    step_size : uint
        Grid-size for the visualization

    Returns
    -------
    napari layers : list of LayerDataTuple
        napari layers displaying pixel intensities as heights.
    """
    image = np.asarray(image)
    if not isinstance(image.dtype, int):
        image = image.astype(int)

    return visualize_as.value(image, step_size)


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [PlotProfile, topographical_view]
