from pathlib import Path
from shutil import rmtree
from unittest import TestCase

from PIL import Image

from phomo import Pool, utils
from phomo.pool import PoolTiles


class TestPool(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("test_pool")
        if not cls.test_dir.is_dir():
            cls.test_dir.mkdir()
        # rainbow tile directory
        cls.tile_dir = cls.test_dir / "rainbow"
        if not cls.tile_dir.is_dir():
            cls.tile_dir.mkdir()
        channel_range = range(0, 255, 60)
        utils.rainbow_of_squares(
            cls.tile_dir,
            size=(10, 10),
            r_range=channel_range,
            g_range=channel_range,
            b_range=channel_range,
        )
        cls.tile_paths = list(cls.tile_dir.glob("*"))

        # create test pool
        cls.pool = Pool.from_dir(cls.tile_dir)

    def test_from_dir(self):
        pool = Pool.from_dir(self.tile_dir)
        assert len(pool) == len(self.tile_paths)

    def test_form_files(self):
        pool = Pool.from_files(self.tile_paths)
        assert len(pool) == len(self.tile_paths)

    def test_tiles(self):
        tiles = self.pool.tiles
        assert isinstance(tiles, PoolTiles)
        assert isinstance(tiles[0], Image.Image)

    def test_pixels(self):
        pixels = self.pool.pixels
        assert pixels.ndim == 2
        assert pixels.shape[-1] == 3
        assert pixels.shape[0] == len(self.tile_paths) * 10 * 10

    def test_len(self):
        assert len(self.tile_paths) == len(self.pool)

    # Palette methods
    def test_palette(self):
        self.pool.palette()

    def test_plot(self):
        self.pool.plot()

    @classmethod
    def tearDownClass(cls):
        if cls.test_dir.is_dir():
            rmtree(cls.test_dir)
