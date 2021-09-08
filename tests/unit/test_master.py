from pathlib import Path
from shutil import rmtree
from unittest import TestCase

import numpy as np
from PIL import Image

from phomo import Master


class TestMaster(TestCase):
    def setUp(self):
        self.test_dir = Path("test_master")
        if not self.test_dir.is_dir():
            self.test_dir.mkdir()
        self.master_path = self.test_dir / "master.png"
        # create a test image object
        self.master_array = np.ones((64, 64, 3), dtype="uint8") * 255
        self.master_img = Image.fromarray(self.master_array)
        # create a test image file
        self.master_img.save(self.master_path)
        # create test master
        self.master = Master.from_file(self.master_path)

    def test_from_image(self):
        Master.from_image(self.master_img)
        # make sure it works for single channel modes
        master = Master.from_image(self.master_img.convert("L"))
        assert master.array.shape[-1] == 3

    def test_from_file(self):
        Master.from_file(self.master_path)

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

    def tearDown(self):
        if self.test_dir.is_dir():
            rmtree(self.test_dir)
