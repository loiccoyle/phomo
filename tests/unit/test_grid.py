from unittest import TestCase

import numpy as np
from PIL import Image

from phomo import Master
from phomo.grid import Grid


class TestMaster(TestCase):
    def setUp(self):
        # create a test image object
        self.master_array = np.hstack(
            [
                np.ones((64, 72, 3), dtype="uint8") * 255,
                np.zeros((64, 56, 3), dtype="uint8"),
            ]
        )
        self.master_img = Image.fromarray(self.master_array)
        # create test master
        self.master = Master.from_image(self.master_img)
        self.grid = Grid(self.master, mosaic_shape=(64, 128), tile_shape=(16, 16))

    def test_slices(self):
        assert len(self.grid.slices) == (self.master_array.shape[0] // 16) * (
            self.master_array.shape[1] // 16
        )

    def test_subdivide(self):
        prev_len = len(self.grid.slices)
        self.grid.subdivide(0.1)
        new_len = len(self.grid.slices)
        # 4 tiles get divided into 4 which adds 4*3 tiles
        assert new_len == prev_len + 4 * 3

    def test_origin(self):
        assert self.grid.origin == (0, 0)
        grid = Grid(self.master, mosaic_shape=(64, 128), tile_shape=(12, 12))
        assert grid.origin == (2, 4)

    def test_remove_origin(self):
        grid = Grid(self.master, mosaic_shape=(64, 128), tile_shape=(12, 12))
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

    def test_plot(self):
        assert isinstance(self.grid.plot(), Image.Image)
