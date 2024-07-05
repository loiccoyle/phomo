import numpy as np
import pytest
from PIL import Image

from phomo import Master
from phomo.grid import Grid


@pytest.fixture
def master_array():
    return np.hstack(
        [
            np.ones((64, 72, 3), dtype="uint8") * 255,
            np.zeros((64, 56, 3), dtype="uint8"),
        ]
    )


@pytest.fixture
def master(master_array):
    return Master.from_image(Image.fromarray(master_array))


@pytest.fixture
def grid(master):
    return Grid(master, mosaic_shape=(64, 128), tile_shape=(16, 16))


def test_slices(grid: Grid, master_array):
    assert len(grid.slices) == (master_array.shape[0] // 16) * (
        master_array.shape[1] // 16
    )


def test_arrays(grid: Grid, master_array):
    assert len(grid.arrays) == (master_array.shape[0] // 16) * (
        master_array.shape[1] // 16
    )


def test_subdivide(grid: Grid):
    prev_len = len(grid.slices)
    grid.subdivide(0.1)
    new_len = len(grid.slices)
    # 4 tiles get divided into 4 which adds 4*3 tiles
    assert new_len == prev_len + 4 * 3


def test_origin(grid: Grid, master: Master):
    assert grid.origin == (0, 0)
    grid = Grid(master, mosaic_shape=(64, 128), tile_shape=(12, 12))
    assert grid.origin == (2, 4)


def test_remove_origin(master: Master):
    grid = Grid(master, mosaic_shape=(64, 128), tile_shape=(12, 12))
    # has an starting offset
    assert grid.slices[0][0].start == 2
    assert grid.slices[0][0].stop == 14
    assert grid.slices[0][1].start == 4
    assert grid.slices[0][1].stop == 16
    new_slices = grid.remove_origin(grid.slices[0])
    # no starting offset
    assert new_slices[0].start == 0
    assert new_slices[0].stop == 12
    assert new_slices[1].start == 0
    assert new_slices[1].stop == 12


def test_plot(grid: Grid):
    assert isinstance(grid.plot(), Image.Image)
