from pathlib import Path
from unittest import TestCase
from shutil import rmtree

from PIL import Image
import numpy as np

from mosaic import Pool, utils
from mosaic import Master, Pool, Mosaic


class TestMosaic(TestCase):
    def setUp(self) -> None:
        self.test_dir = Path("test_mosaic")
        if not self.test_dir.is_dir():
            self.test_dir.mkdir()

        self.master_shape = (550, 512)
        self.master_path = self.test_dir / "master.png"
        # create a test image object
        self.master_array = np.ones((*self.master_shape, 3), dtype="uint8") * 255
        self.master_img = Image.fromarray(self.master_array)
        # create a test image file
        self.master_img.save(self.master_path)
        # create test master
        self.master = Master.from_file(self.master_path)

        # rainbow tile directory
        self.tile_dir = self.test_dir / "rainbow"
        if not self.tile_dir.is_dir():
            self.tile_dir.mkdir()
        utils.rainbow_of_squares(
            self.tile_dir, size=(50, 50), range_params=(0, 255, 60)
        )
        self.tile_paths = list(self.tile_dir.glob("*"))
        # create test pool
        self.pool = Pool.from_dir(self.tile_dir)

        self.mosaic = Mosaic(self.master, self.pool)

    def test_tile_shape(self):
        assert self.mosaic.tile_shape == self.pool.arrays[0].shape[:-1]

    def test_size(self):
        assert self.mosaic.size == (500, 550)

    def test_n_leftover(self):
        assert self.mosaic.n_leftover == 15

    def test_build(self):
        mosaic_img = self.mosaic.build()
        assert mosaic_img.size == self.mosaic.size

    def tearDown(self):
        if self.test_dir.is_dir():
            rmtree(self.test_dir)
