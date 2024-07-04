from pathlib import Path
from shutil import rmtree
from unittest import TestCase

import numpy as np
from PIL import Image

from phomo import utils


class TestUtils(TestCase):
    @classmethod
    def setUpClass(cls):
        # create test directory
        cls.test_dir = Path("test_utils")
        if not cls.test_dir.is_dir():
            cls.test_dir.mkdir()
        # rainbow tile directory
        cls.rainbow_dir = cls.test_dir / "rainbow"
        if not cls.rainbow_dir.is_dir():
            cls.rainbow_dir.mkdir()
        # create a test image object
        cls.test_array = np.ones((64, 64, 3), dtype="uint8") * 255
        cls.test_img = Image.fromarray(cls.test_array)
        # create a test image file
        cls.test_img_file = cls.test_dir / "test_img.png"
        cls.test_img.save(cls.test_img_file)

    def test_rainbow_of_squares(self):
        # create the squares
        channel_range = range(0, 256, 15)
        shape = (20, 10)
        utils.rainbow_of_squares(
            self.rainbow_dir,
            size=shape,
            r_range=channel_range,
            g_range=channel_range,
            b_range=channel_range,
        )
        tiles = list(self.rainbow_dir.glob("*"))
        # check the number of tiles created
        assert len(tiles) == len(list(channel_range)) ** 3
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
            self.test_img_file, crop_ratio=2, size=(32, 64), mode="L"
        )
        assert img.size == (32, 64)
        assert img.mode == "L"

    def test_resize_array(self):
        resized = utils.resize_array(self.test_array, (32, 64))
        assert resized.shape == (64, 32, 3)

    @classmethod
    def tearDownClass(cls):
        if cls.test_dir.is_dir():
            rmtree(cls.test_dir)
