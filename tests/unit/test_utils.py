import numpy as np
import pytest
from PIL import Image

from phomo import utils


@pytest.fixture
def img_array():
    return np.ones((64, 64, 3), dtype="uint8") * 255


@pytest.fixture
def img(img_array):
    return Image.fromarray(img_array)


@pytest.fixture
def img_file(tmp_path, img):
    img_file = tmp_path / "test_img.png"
    img.save(img_file)
    return img_file


def test_rainbow_of_squares(tmp_path):
    # create the squares
    channel_range = range(0, 256, 15)
    shape = (20, 10)
    utils.rainbow_of_squares(
        tmp_path,
        size=shape,
        r_range=channel_range,
        g_range=channel_range,
        b_range=channel_range,
    )
    tiles = list(tmp_path.glob("*"))
    # check the number of tiles created
    assert len(tiles) == len(list(channel_range)) ** 3
    # check the size of the tiles
    img = Image.open(tiles[0])
    assert img.size == shape


def test_crop_to_ratio(img):
    # create a test white image
    img_cropped = utils.crop_to_ratio(img, ratio=2)
    # check the aspect ration of the img
    assert img_cropped.size[0] / img_cropped.size[1] == 2
    assert img_cropped.size == (64, 32)


def test_resize_array(img_array):
    resized = utils.resize_array(img_array, (32, 64))
    assert resized.shape == (64, 32, 3)


def test_open_img_file(img_file):
    # just open the file
    img = utils.open_img_file(img_file)
    assert isinstance(img, Image.Image)
    # crop to ratio
    img = utils.open_img_file(img_file, crop_ratio=2)
    assert img.size[0] / img.size[1] == 2
    assert img.size == (64, 32)
    # change image size
    img = utils.open_img_file(img_file, crop_ratio=2, size=(32, 64))
    assert img.size == (32, 64)
    # convert to mode
    img = utils.open_img_file(img_file, crop_ratio=2, size=(32, 64), mode="L")
    assert img.size == (32, 64)
    assert img.mode == "L"
