import pytest
from PIL import Image

from phomo import Pool
from phomo.pool import PoolTiles


@pytest.fixture
def tile_paths(tile_dir):
    return list(tile_dir.glob("*"))


def test_from_dir(tile_dir, tile_paths):
    pool = Pool.from_dir(tile_dir)
    assert len(pool) == len(tile_paths)


def test_form_files(tile_paths):
    pool = Pool.from_files(tile_paths)
    assert len(pool) == len(tile_paths)


def test_tiles(pool: Pool):
    tiles = pool.tiles
    assert isinstance(tiles, PoolTiles)
    assert isinstance(tiles[0], Image.Image)


def test_pixels(pool: Pool, tile_paths):
    pixels = pool.pixels
    assert pixels.ndim == 2
    assert pixels.shape[-1] == 3
    assert (
        pixels.shape[0]
        == len(tile_paths) * pool.tiles[0].size[0] * pool.tiles[0].size[0]
    )


def test_len(pool: Pool, tile_paths):
    assert len(tile_paths) == len(pool)


def test_plot(pool: Pool):
    pool.plot()
