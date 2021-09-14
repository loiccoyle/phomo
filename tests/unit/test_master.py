from pathlib import Path
from shutil import rmtree
from unittest import TestCase

import numpy as np
from PIL import Image

from phomo import Master


class TestMaster(TestCase):
    @classmethod
    def setUp(cls):
        cls.test_dir = Path("test_master")
        if not cls.test_dir.is_dir():
            cls.test_dir.mkdir()
        cls.master_path = cls.test_dir / "master.png"
        # create a test image object
        cls.master_array = np.ones((64, 64, 3), dtype="uint8") * 255
        cls.master_img = Image.fromarray(cls.master_array)
        # create a test image file
        cls.master_img.save(cls.master_path)
        # create test master
        cls.master = Master.from_file(cls.master_path)

    def test_constructors(self):
        master_img = Master.from_image(self.master_img)
        master_file = Master.from_file(self.master_path)
        assert (master_img.array == master_file.array).all()
        # make sure it works for single channel modes
        master = Master.from_image(self.master_img.convert("L"))
        assert master.array.shape[-1] == 3

    def test_img(self):
        assert isinstance(self.master.img, Image.Image)

    def test_pixels(self):
        assert self.master.pixels.shape[-1] == 3
        assert (
            self.master.pixels.shape[0]
            == self.master_array.shape[0] * self.master_array.shape[1]
        )

    # Palette methods
    def test_palette(self):
        self.master.palette()

    def test_cdfs(self):
        self.master.cdfs()

    def test_plot(self):
        self.master.plot()

    @classmethod
    def tearDown(cls):
        if cls.test_dir.is_dir():
            rmtree(cls.test_dir)
