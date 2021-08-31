import logging
import os
import random
from functools import partial
from multiprocessing import Pool as MpPool
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from .grid import Grid
from .master import Master
from .pool import Pool
from .utils import resize_array


# TODO: this does not work
class MosaicUnstructured:
    def __init__(self, master: Master, pool: Pool, workers: Optional[int] = None):
        self.master = master
        self.pool = pool
        if workers is None:
            workers = os.cpu_count()

        self.workers = workers
        self._log = logging.getLogger(__name__)

    def build(self, shuffle: bool = True, n_points: int = 64):
        """Construct the mosaic.

        Args:
            shuffle: shuffle the order of the tiles before building the mosaic.
            n_points: number of coordinate to try for each tile.
        """
        tiles = self.pool.arrays
        if shuffle:
            self._log.info("Shuffling pool.")
            tiles = random.sample(tiles, len(tiles))

        mosaic_array = np.zeros_like(self.master.array)
        with MpPool(self.workers) as pool:
            for tile_array in tqdm(tiles, desc="Building"):
                possible_ij = list(
                    zip(
                        np.random.choice(
                            np.arange(
                                0,
                                self.master.array.shape[0] - tile_array.shape[0],
                                dtype=np.uint,
                            ),
                            n_points,
                            replace=False,
                        ),
                        np.random.choice(
                            np.arange(
                                0,
                                self.master.array.shape[1] - tile_array.shape[1],
                                dtype=np.uint,
                            ),
                            n_points,
                            replace=False,
                        ),
                    )
                )
                out = pool.map(
                    partial(self._loss, tile_array=tile_array),
                    possible_ij,
                    chunksize=len(possible_ij) // self.workers
                    if self.workers
                    else None,
                )
                min_ij = possible_ij[np.argmin(out)]
                mosaic_array[
                    min_ij[0] : int(min_ij[0] + tile_array.shape[0]),
                    min_ij[1] : int(min_ij[1] + tile_array.shape[1]),
                ] = tile_array
        return Image.fromarray(mosaic_array.astype(np.uint8))

    def _loss(self, ij: Tuple[int, int], tile_array: np.ndarray) -> float:
        i, j = int(ij[0]), int(ij[1])
        return np.linalg.norm(
            (
                self.master.array[
                    i : i + tile_array.shape[0],
                    j : j + tile_array.shape[1],
                ]
                - tile_array
            ).reshape((-1, tile_array.shape[-1])),
            ord=1,
        )


class Mosaic:
    def __init__(
        self,
        master: Master,
        pool: Pool,
        n_appearances: int = 1,
    ) -> None:
        """Construct a regular grid mosaic.

        Note:
            The Pool's tiles should  all be the same size.

        Args:
            master: Master image to reconstruct.
            pool: Tile image pool with which to reconstruct the Master image.
            n_appearances: Number of times a tile can appear in the mosaic.

        Examples:
            Creating a Mosaic instance.

            >>> Mosaic(master, pool, mosaic_size=(1280, 1280), tile_size=(64. 64))
        """
        self._log = logging.getLogger(__name__)
        self.master = master
        if len(set([array.size for array in pool.arrays])) != 1:
            raise ValueError("Pool tiles sizes are not identical.")
        self.pool = pool
        self.tile_size = (self.pool.arrays[0].shape[0], self.pool.arrays[0].shape[1])
        self.n_appearances = n_appearances
        self.grid = Grid(self.master, self.shape, self.tile_size)

    @property
    def shape(self) -> Tuple[int, int]:
        """The size of the mosaic image.

        It can be different from the master image size as an integer number of
        tiles should fit within it.
        """
        return (
            self.master.array.shape[0] - self.master.array.shape[0] % self.tile_size[0],
            self.master.array.shape[1] - self.master.array.shape[1] % self.tile_size[1],
        )

    @property
    def n_leftover(self) -> int:
        return len(self.pool) * self.n_appearances - len(self.grid.slices)

    def _d_matrix(self, ord: Optional[int] = None) -> np.ndarray:
        """Compute the distance matrix between all the master's tiles and the
        pool tiles.

        Returns:
            Distance matrix, shape: (number of master arrays, number of tiles in the pool).
        """
        # Compute the distance matrix.
        d_matrix = np.zeros((len(self.grid.slices), len(self.pool.arrays)))
        self._log.debug("d_matrix shape: %s", d_matrix.shape)

        for i, slices in tqdm(
            enumerate(self.grid.slices),
            total=len(self.grid.slices),
            desc="Building distance matrix",
        ):
            array = self.master.array[slices[0], slices[1]]
            if array[:-1].shape != self.tile_size:
                array = resize_array(array, (self.tile_size[1], self.tile_size[0]))
            d_matrix[i, :] = [
                np.linalg.norm(
                    (np.int16(tile) - np.int16(array)).reshape(-1, 3),
                    ord=ord,
                )
                for tile in self.pool.arrays
            ]

        return d_matrix

    def build(self, ord: Optional[int] = None) -> Image.Image:
        """Construct the mosaic image.

        Args:
            ord: Order of the norm used to compute the distance. See np.linalg.norm.

        Returns:
            The PIL.Image instance of the mosaic.
        """
        mosaic = np.zeros((*self.shape, 3))

        # Compute the distance matrix.
        d_matrix = self._d_matrix(ord=ord)

        # Keep track of tiles and sub arrays.
        placed_master_arrays = set()
        placed_tiles = set()
        n_appearances = [0] * len(self.pool)

        pbar = tqdm(total=d_matrix.shape[0], desc="Building mosaic")
        # from: https://stackoverflow.com/questions/29046162/numpy-array-loss-of-dimension-when-masking
        sorted_master_slices_i, sorted_tiles = np.unravel_index(
            np.argsort(d_matrix, axis=None), d_matrix.shape
        )
        for slices_i, tile in zip(sorted_master_slices_i, sorted_tiles):
            if slices_i in placed_master_arrays or tile in placed_tiles:
                continue
            slices = self.grid.slices[slices_i]
            tile_array = self.pool.arrays[tile]
            # if the grid has been subdivided then the tile should be shrunk to
            # the size of the subdivision
            array_size = (
                slices[1].stop - slices[1].start,
                slices[0].stop - slices[0].start,
            )
            if tile_array.shape[:-1] != array_size[::-1]:
                tile_array = resize_array(tile_array, array_size)

            # TODO: will need to substract the remainders here when centering
            # the mosaic
            mosaic[slices[0], slices[1]] = tile_array
            placed_master_arrays.add(slices_i)
            n_appearances[tile] += 1
            if n_appearances[tile] == self.n_appearances:
                placed_tiles.add(tile)
            pbar.update(1)
        pbar.close()
        return Image.fromarray(np.uint8(mosaic))

    def __repr__(self) -> str:
        # indent these guys
        master = repr(self.master).replace("\n", "\n    ")
        pool = repr(self.pool).replace("\n", "\n    ")
        grid = repr(self.grid).replace("\n", "\n    ")
        return f"""{self.__class__.__module__}.{self.__class__.__name__} at {hex(id(self))}:
    n_appearances: {self.n_appearances}
    mosaic shape: {self.shape}
    tile size: {self.tile_size}
    leftover tiles: {self.n_leftover}
    {grid}
    {master}
    {pool}"""
