import logging
from functools import partial
from multiprocessing.pool import Pool as MpPool
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from scipy.optimize import linear_sum_assignment

from .grid import Grid
from .master import Master
from .metrics import METRICS, MetricCallable
from .pool import Pool
from .utils import resize_array


class Mosaic:
    def __init__(
        self,
        master: Master,
        pool: Pool,
        n_appearances: int = 1,
    ) -> None:
        """Construct a regular grid mosaic.

        Note:
            The Pool's tiles should all be the same size.

        Args:
            master: Master image to reconstruct.
            pool: Tile image pool with which to reconstruct the Master image.
            n_appearances: Number of times a tile can appear in the mosaic.

        Examples:
            Building a mosaic.

            >>> Mosaic(master, pool, n_appearances=1).build()
        """
        self._log = logging.getLogger(__name__)
        self.master = master
        if len(set([array.size for array in pool.arrays])) != 1:
            raise ValueError("Pool tiles sizes are not identical.")
        self.pool = pool
        self.tile_shape = (self.pool.arrays[0].shape[0], self.pool.arrays[0].shape[1])
        self.n_appearances = n_appearances
        self.grid = Grid(self.master, (self.size[1], self.size[0]), self.tile_shape)

    @property
    def size(self) -> Tuple[int, int]:
        """The size of the mosaic image.

        It can be different from the master image size as an integer number of
        tiles should fit within it.

        Returns:
            The width and height of the mosaic image.
        """
        return (
            self.master.array.shape[1]
            - self.master.array.shape[1] % self.tile_shape[1],
            self.master.array.shape[0]
            - self.master.array.shape[0] % self.tile_shape[0],
        )

    @property
    def n_leftover(self) -> int:
        """The number of tiles which will be unused when building the mosaic."""
        return len(self.pool) * self.n_appearances - len(self.grid.slices)

    def _d_matrix_worker(
        self, array: np.ndarray, metric_func: MetricCallable, **kwargs
    ) -> np.ndarray:
        """Parallel worker. Computes one row of the distance matrix."""
        # if the tile grid was subdivided the master array can be smaller
        # than the tiles, need to resize to match the shapes
        if array.shape[:-1] != self.tile_shape:
            # this isn't exact because we are upscalling the master array
            # we should be shrinking all the tile arrays but that is slower
            array = resize_array(array, (self.tile_shape[1], self.tile_shape[0]))
        return metric_func(array, self.pool.arrays, **kwargs)

    def compute_d_matrix(
        self,
        workers: int = 1,
        metric: Union[str, MetricCallable] = "norm",
        **kwargs,
    ) -> np.ndarray:
        """Compute the distance matrix between all the master's tiles and the
        pool tiles.

        Args:
            workers: The number of worker to use.
            metric: The distance metric used for the distance matrix. Either
                provide a string, for implemented metrics see ``phomo.metrics.METRICS``.
                Or a callable, which should take two ``np.ndarray``s and return a float.
            **kwargs: Passed to `metric`.

        Returns:
            Distance matrix, shape: (number of master arrays, number of tiles in the pool).
        """
        if isinstance(metric, str):
            if metric not in METRICS.keys():
                raise KeyError(
                    "'%s' not in available metrics: %s",
                    metric,
                    repr(list(METRICS.keys())),
                )
            self._log.info("Using metric '%s'", metric)
            metric_func = METRICS[metric]
        else:
            self._log.info("Using user provided distance metric function.")
            metric_func = metric

        # Compute the distance matrix.
        worker = partial(self._d_matrix_worker, metric_func=metric_func, **kwargs)
        if workers != 1:
            self._log.info("Computing distance matrix with %i workers.", workers)
            with MpPool(processes=workers) as pool:
                d_matrix = np.array(
                    list(
                        tqdm(
                            pool.imap(
                                worker,
                                self.grid.arrays,
                                chunksize=len(self.grid) // workers,
                            ),
                            total=len(self.grid.slices),
                            desc="Building distance matrix",
                        )
                    )
                )
        else:
            # get rid of pool overhead if serial computation is desired.
            self._log.info("Computing distance matrix in serial.")
            d_matrix = np.array(
                [
                    worker(array)
                    for array in tqdm(self.grid.arrays, desc="Building distance matrix")
                ]
            )
        self._log.debug("d_matrix shape: %s", d_matrix.shape)
        return d_matrix

    def build_greedy(
        self,
        workers: int = 1,
        metric: Union[str, MetricCallable] = "norm",
        d_matrix: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Image.Image:
        """Construct the mosaic image using a greedy tile assignement algorithm.

        Args:
            workers: The number of workers to use when computing the
                distance matrix.
            metric: The distance metric used for the distance matrix. Either
                provide a string, for implemented metrics see ``phomo.metrics.METRICS``.
                Or a callable, which should take two ``np.ndarray``s and return a float.
            d_matrix: Use a pre-computed distance matrix.
            **kwargs: Passed to the `metric` function.

        Returns:
            The PIL.Image instance of the mosaic.
        """
        mosaic = np.zeros((self.size[1], self.size[0], 3))

        # Compute the distance matrix.
        if d_matrix is None:
            d_matrix = self.compute_d_matrix(workers=workers, metric=metric, **kwargs)

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

            # shift slices back so that the centering of the mosaic within the
            # master image is removed
            slices = self.grid.remove_origin(slices)
            mosaic[slices[0], slices[1]] = tile_array
            placed_master_arrays.add(slices_i)
            n_appearances[tile] += 1
            if n_appearances[tile] == self.n_appearances:
                placed_tiles.add(tile)
            pbar.update(1)
        pbar.close()
        return Image.fromarray(np.uint8(mosaic))

    def build(
        self,
        workers: int = 1,
        metric: Union[str, MetricCallable] = "norm",
        d_matrix: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Image.Image:
        """Construct the mosaic image by solving the linear sum assignment problem.
        See: https://en.wikipedia.org/wiki/Assignment_problem

        Args:
            workers: The number of workers to use when computing the
                distance matrix.
            metric: The distance metric used for the distance matrix. Either
                provide a string, for implemented metrics see ``phomo.metrics.METRICS``.
                Or a callable, which should take two ``np.ndarray``s and return a float.
            d_matrix: Use a pre-computed distance matrix.
            **kwargs: Passed to the `metric` function.

        Returns:
            The PIL.Image instance of the mosaic.
        """
        mosaic = np.zeros((self.size[1], self.size[0], 3))

        # Compute the distance matrix.
        if d_matrix is None:
            d_matrix = self.compute_d_matrix(workers=workers, metric=metric, **kwargs)

        # expand the dmatrix to allow for repeated tiles
        if self.n_appearances > 0:
            d_matrix = np.tile(d_matrix, self.n_appearances)
        print("dmatrix", d_matrix.shape)

        self._log.info("Computing optimal tile assignment.")
        row_ind, col_ind = linear_sum_assignment(d_matrix)
        pbar = tqdm(total=d_matrix.shape[0], desc="Building mosaic")
        for row, col in zip(row_ind, col_ind):
            slices = self.grid.slices[row]
            tile_array = self.pool.arrays[col % len(self.pool.arrays)]
            # if the grid has been subdivided then the tile should be shrunk to
            # the size of the subdivision
            array_size = (
                slices[1].stop - slices[1].start,
                slices[0].stop - slices[0].start,
            )
            if tile_array.shape[:-1] != array_size[::-1]:
                tile_array = resize_array(tile_array, array_size)

            # shift slices back so that the centering of the mosaic within the
            # master image is removed
            slices = self.grid.remove_origin(slices)
            mosaic[slices[0], slices[1]] = tile_array
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
    mosaic size: {self.size}
    tile shape: {self.tile_shape}
    leftover tiles: {self.n_leftover}
    {grid}
    {master}
    {pool}"""
