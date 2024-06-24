from pathlib import Path
from shutil import rmtree
from unittest import TestCase

import numpy as np
from PIL import Image

from phomo import Master, Mosaic, Pool, utils


class TestMosaic(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("test_mosaic")
        if not cls.test_dir.is_dir():
            cls.test_dir.mkdir()

        cls.master_shape = (550, 512)
        cls.master_path = cls.test_dir / "master.png"
        # create a test image object
        cls.master_array = np.ones((*cls.master_shape, 3), dtype="uint8") * 255
        cls.master_img = Image.fromarray(cls.master_array)
        # create a test image file
        cls.master_img.save(cls.master_path)
        # create test master
        cls.master = Master.from_file(cls.master_path)

        # rainbow tile directory
        cls.tile_dir = cls.test_dir / "rainbow"
        if not cls.tile_dir.is_dir():
            cls.tile_dir.mkdir()
        utils.rainbow_of_squares(cls.tile_dir, size=(50, 50), range_params=(0, 255, 60))
        cls.tile_paths = list(cls.tile_dir.glob("*"))
        # create test pool
        cls.pool = Pool.from_dir(cls.tile_dir)

        cls.mosaic = Mosaic(cls.master, cls.pool)

    def test_tile_shape(self):
        assert self.mosaic.tile_shape == self.pool.arrays[0].shape[:-1]

    def test_size(self):
        assert self.mosaic.size == (500, 550)

    def test_n_leftover(self):
        assert self.mosaic.n_leftover == 15

    def test_build(self):
        mosaic_img = self.mosaic.build(self.mosaic.d_matrix(workers=1))
        assert mosaic_img.size == self.mosaic.size

        mosaic_img = self.mosaic.build(self.mosaic.d_matrix(workers=2))
        assert mosaic_img.size == self.mosaic.size

        with self.assertRaises(ValueError):
            mosaic_img = self.mosaic.build(self.mosaic.d_matrix(workers=0))

    @classmethod
    def tearDownClass(cls):
        if cls.test_dir.is_dir():
            rmtree(cls.test_dir)
