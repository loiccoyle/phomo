import numpy as np
import pytest
from PIL import Image

from phomo import Master


@pytest.fixture
def master_array():
    return np.ones((64, 64, 3), dtype="uint8") * 255


@pytest.fixture
def master_img(master_array):
    return Image.fromarray(master_array)


@pytest.fixture
def master_path(tmp_path, master_img):
    path = tmp_path / "master.png"
    master_img.save(path)
    return path


@pytest.fixture
def master(master_path):
    return Master.from_file(master_path)


def test_constructors(master_img: Image.Image, master_path):
    master_from_img = Master.from_image(master_img)
    master_from_file = Master.from_file(master_path)
    assert (master_from_img.array == master_from_file.array).all()
    # make sure it works for single channel modes
    master = Master.from_image(master_img.convert("L"))
    assert master.array.shape[-1] == 3


def test_img(master: Master):
    assert isinstance(master.img, Image.Image)


def test_pixels(master: Master, master_array):
    assert master.pixels.shape[-1] == 3
    assert master.pixels.shape[0] == master_array.shape[0] * master_array.shape[1]


def test_plot(master: Master):
    master.plot()
