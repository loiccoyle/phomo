from pathlib import Path
from shutil import rmtree
from unittest import TestCase

import numpy as np
from PIL import Image

from phomo import utils


class TestUtils(TestCase):
    def setUp(self):
        # create test directory
        self.test_dir = Path("test_utils")
        if not self.test_dir.is_dir():
            self.test_dir.mkdir()
        # rainbow tile directory
        self.rainbow_dir = self.test_dir / "rainbow"
        if not self.rainbow_dir.is_dir():
            self.rainbow_dir.mkdir()
        # create a test image object
        self.test_array = np.ones((64, 64, 3), dtype="uint8") * 255
        self.test_img = Image.fromarray(self.test_array)
        # create a test image file
        self.test_img_file = self.test_dir / "test_img.png"
        self.test_img.save(self.test_img_file)

    def test_rainbow_of_squares(self):
        # create the squares
        range_params = (0, 256, 15)
        shape = (20, 10)
        utils.rainbow_of_squares(
            self.rainbow_dir, size=shape, range_params=range_params
        )
        tiles = list(self.rainbow_dir.glob("*"))
        # check the number of tiles created
        assert len(tiles) == len(list(range(*range_params))) ** 3
        # check the size of the tiles
        img = Image.open(tiles[0])
        assert img.size == shape

    def test_crop_to_ratio(self):
        # create a test white image
        img_cropped = utils.crop_to_ratio(self.test_img, ratio=2)
        # check the aspect ration of the img
        assert img_cropped.size[0] / img_cropped.size[1] == 2
        assert img_cropped.size == (64, 32)

    def test_open_img_file(self):
        # just open the file
        img = utils.open_img_file(self.test_img_file)
        assert isinstance(img, Image.Image)
        # crop to ratio
        img = utils.open_img_file(self.test_img_file, crop_ratio=2)
        assert img.size[0] / img.size[1] == 2
        assert img.size == (64, 32)
        # change image size
        img = utils.open_img_file(self.test_img_file, crop_ratio=2, size=(32, 64))
        assert img.size == (32, 64)
        # convert to mode
        img = utils.open_img_file(
            self.test_img_file, crop_ratio=2, size=(32, 64), convert="L"
        )
        assert img.size == (32, 64)
        assert img.mode == "L"

    def test_resize_array(self):
        resized = utils.resize_array(self.test_array, (32, 64))
        assert resized.shape == (64, 32, 3)

    def test_to_ucs(self):
        utils.to_ucs(self.test_array)

    def test_to_rgb(self):
        ucs = utils.to_ucs(self.test_array)
        rgb = utils.to_rgb(ucs)
        # test transitivity
        assert np.allclose(self.test_array, rgb, atol=1)

    def tearDown(self):
        if self.test_dir.is_dir():
            rmtree(self.test_dir)
