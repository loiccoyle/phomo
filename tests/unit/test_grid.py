from unittest import TestCase

from PIL import Image
import numpy as np

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

    def test_plot(self):
        assert isinstance(self.grid.plot(), Image.Image)
