"""Profiles on n-dimensional, multiscale, lazily loaded data such as OME-Zarr (t, c, z, y, x)."""
import numpy as np
import pytest
import napari
from napari.components import Dims

from napari_plot_profile._dock_widget import profile, line_to_world

T, C, Z, Y, X = 3, 2, 4, 64, 80
SCALE = (1.0, 2.0, 0.5, 0.5)
TRANSLATE = (0.0, 3.0, 10.0, 20.0)


@pytest.fixture
def volume():
    return np.random.default_rng(0).integers(0, 1000, size=(T, C, Z, Y, X)).astype(np.uint16)


def _channel_layer(volume, c, lazy):
    channel = volume[:, c]
    if lazy:
        da = pytest.importorskip("dask.array")
        pyramid = [da.from_array(channel, chunks=(1, 1, 32, 32)),
                   da.from_array(channel[..., ::2, ::2], chunks=(1, 1, 32, 32))]
    else:
        pyramid = [channel, channel[..., ::2, ::2]]
    return napari.layers.Image(pyramid, multiscale=True, scale=SCALE, translate=TRANSLATE)


def _world(t, z, y, x):
    """World coordinates of a full-resolution pixel."""
    return np.array([t, z, y, x]) * SCALE + TRANSLATE


@pytest.mark.parametrize("lazy", [False, True])
def test_profile_multiscale_4d(volume, lazy):
    layer = _channel_layer(volume, 1, lazy)
    line = np.stack([_world(2, 1, 5, 6), _world(2, 1, 40, 70)])

    result = profile(layer, line, num_points=50)

    positions = np.asarray(result['positions'])
    assert positions.shape == (50, 4)
    assert np.all(positions[:, :2] == [2, 1])
    assert tuple(positions[0]) == (2, 1, 5, 6)
    assert tuple(positions[-1]) == (2, 1, 40, 70)
    expected = volume[2, 1, 1][positions[:, 2], positions[:, 3]]
    np.testing.assert_array_equal(result['intensities'], expected)
    # distances are in world (physical) units
    assert result['distances'][-1] == pytest.approx(np.linalg.norm(line[1] - line[0]))


def test_profile_clips_points_outside_image(volume):
    layer = _channel_layer(volume, 0, lazy=False)
    line = np.stack([_world(0, 0, 10, -20), _world(0, 0, 10, 40)])

    result = profile(layer, line, num_points=61)

    positions = np.asarray(result['positions'])
    assert positions[:, 3].min() == 0
    assert len(positions) == 41


@pytest.mark.parametrize("shape_ndim", [2, 4])
def test_line_to_world_uses_current_slice(shape_ndim):
    dims = Dims(ndim=4, ndisplay=2)
    dims.range = [(0, 3, 1), (3, 11, 2), (10, 42, 0.5), (20, 60, 0.5)]
    dims.point = (2, 5, 10, 20)

    yx = np.array([[12.5, 23.0], [30.0, 55.0]])
    vertices = yx if shape_ndim == 2 else np.hstack([np.zeros((2, 2)), yx])  # drawn at t=0, z=0
    shapes = napari.layers.Shapes([vertices], shape_type='path')

    world = line_to_world(vertices, shapes, dims)

    np.testing.assert_array_equal(world[:, :2], [[2, 5], [2, 5]])  # follows the sliders
    np.testing.assert_array_equal(world[:, 2:], yx)
