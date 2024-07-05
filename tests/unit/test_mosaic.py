import numpy as np
import pytest
from PIL import Image

from phomo import Master, Mosaic, Pool


@pytest.fixture
def mosaic(pool_big_tiles):
    master_shape = (550, 512)
    # create a test image object
    master_array = np.ones((*master_shape, 3), dtype="uint8") * 255

    # create test master
    master = Master.from_image(Image.fromarray(master_array))
    return Mosaic(master, pool_big_tiles)


def test_tile_shape(mosaic: Mosaic, pool_big_tiles: Pool):
    assert mosaic.tile_shape == pool_big_tiles.array[0].shape[:-1]


def test_size(mosaic: Mosaic):
    assert mosaic.size == (500, 550)


def test_n_leftover(mosaic: Mosaic):
    assert mosaic.n_leftover == 15


def test_build(mosaic: Mosaic):
    mosaic_img = mosaic.build(mosaic.d_matrix(workers=1))
    assert mosaic_img.size == mosaic.size

    mosaic_img = mosaic.build(mosaic.d_matrix(workers=2))
    assert mosaic_img.size == mosaic.size

    with pytest.raises(ValueError):
        mosaic_img = mosaic.build(mosaic.d_matrix(workers=0))
