from pathlib import Path
from shutil import rmtree
from unittest import TestCase

import numpy as np
from PIL import Image

from phomo import Pool, utils
from phomo.pool import PoolTiles


class TestPool(TestCase):
    def setUp(self):
        self.test_dir = Path("test_pool")
        if not self.test_dir.is_dir():
            self.test_dir.mkdir()
        # rainbow tile directory
        self.tile_dir = self.test_dir / "rainbow"
        if not self.tile_dir.is_dir():
            self.tile_dir.mkdir()
        utils.rainbow_of_squares(
            self.tile_dir, size=(10, 10), range_params=(0, 255, 60)
        )
        self.tile_paths = list(self.tile_dir.glob("*"))

        # create test pool
        self.pool = Pool.from_dir(self.tile_dir)

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

    def test_cdfs(self):
        self.pool.cdfs()

    def test_plot(self):
        self.pool.plot()

    def tearDown(self):
        if self.test_dir.is_dir():
            rmtree(self.test_dir)
