import logging
import math
from functools import partial
from multiprocessing.pool import Pool as MpPool
from os import PathLike
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

from .grid import Grid
from .master import Master
from .metrics import METRICS, MetricCallable
from .pool import Pool
from .utils import resize_array

LOGGER = logging.getLogger(__name__)


class Mosaic:
    @classmethod
    def from_file_and_dir(
        cls,
        master_file: PathLike,
        tile_dir: PathLike,
        *args,
        master_crop_ratio: Optional[float] = None,
        master_size: Optional[Tuple[int, int]] = None,
        master_mode: Optional[str] = None,
        tile_crop_ratio: Optional[float] = None,
        tile_size: Optional[Tuple[int, int]] = None,
        tile_mode: Optional[str] = None,
        **kwargs,
    ) -> "Mosaic":
        """Construct a `Mosaic` from a master image file and a directory containing the file images.

        Args:
            master_file: The master image file.
            tile_dir: the directory containing the tile images.

        Returns:
            A `Mosaic` to construct the `master_file` using the tile images in the `tile_dir`.
        """
        master = Master.from_file(
            master_file,
            crop_ratio=master_crop_ratio,
            img_size=master_size,
            mode=master_mode,
        )
        pool = Pool.from_dir(
            tile_dir, tile_size=tile_size, crop_ratio=tile_crop_ratio, mode=tile_mode
        )
        return cls(master, pool, *args, **kwargs)

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
            master: `Master` image to reconstruct.
            pool: Tile image pool with which to reconstruct the `Master` image.
            n_appearances: Number of times a tile can appear in the mosaic.

        Examples:
            Building a mosaic.

            >>> pool = Pool.from_dir("tiles")
            >>> master = Master.from_file("master.png")
            >>> mosaic = Mosaic(master, pool, n_appearances=1)
            >>> mosaic.build(mosaic.d_matrix())
        """
        self.master = master
        if len(set([array.size for array in pool.array])) != 1:
            raise ValueError("Pool tiles sizes are not identical.")
        self.pool = pool
        self.tile_shape = (self.pool.array[0].shape[0], self.pool.array[0].shape[1])
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
        return metric_func(array, self.pool.array, **kwargs)

    def d_matrix(
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
            LOGGER.info("Using metric '%s'", metric)
            metric_func = METRICS[metric]
        else:
            LOGGER.info("Using user provided distance metric function.")
            metric_func = metric

        # Compute the distance matrix.
        worker = partial(self._d_matrix_worker, metric_func=metric_func, **kwargs)
        if workers != 1:
            LOGGER.info("Computing distance matrix with %i workers.", workers)
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
            LOGGER.info("Computing distance matrix in serial.")
            d_matrix = np.array(
                [
                    worker(array)
                    for array in tqdm(self.grid.arrays, desc="Building distance matrix")
                ]
            )
        LOGGER.debug("d_matrix shape: %s", d_matrix.shape)
        return d_matrix

    def d_matrix_cuda(self, metric: str = "norm") -> np.ndarray:
        """Compute the distance matrix using CUDA for GPU acceleration.

        Args:
            metric: The distance metric used for the distance matrix. Either "norm" or "greyscale".

        Returns:
            Distance matrix, shape: (number of master arrays, number of tiles in the pool).
        """

        try:
            from numba import cuda
        except ImportError:
            raise ImportError(
                "Numba is required for CUDA support, run \"pip install 'phomo[cuda]'\" to install it."
            )

        if metric not in ["norm", "greyscale"]:
            raise ValueError(
                f"Invalid metric '{metric}'. When using gpu `metric' must be 'norm' or 'greyscale'."
            )

        LOGGER.info("Computing distance matrix with CUDA.")

        # when the grid has been subdivided the master arrays will be smaller, so we grow them to match
        # the tile size
        grid_arrays = [
            array
            if array.shape == self.tile_shape
            else resize_array(array, self.tile_shape)
            for array in self.grid.arrays
        ]
        pool_arrays = self.pool.array
        if metric == "greyscale":
            grid_arrays = [array.sum(axis=-1, keepdims=True) for array in grid_arrays]
            pool_arrays = [array.sum(axis=-1, keepdims=True) for array in pool_arrays]

        # Transfer the master and pool arrays to the GPU.
        master_arrays_device = cuda.to_device(grid_arrays)
        pool_arrays_device = cuda.to_device(pool_arrays)

        # Allocate memory for the distance matrix on the GPU.
        d_matrix_device = cuda.device_array((len(grid_arrays), len(pool_arrays)))

        # Define the CUDA kernel for computing the distance matrix.
        @cuda.jit
        def compute_d_matrix_kernel(master_arrays, pool_arrays, d_matrix):
            i, j = cuda.grid(2)  # type: ignore
            if i < master_arrays.shape[0] and j < pool_arrays.shape[0]:
                distance = 0.0
                for x in range(master_arrays.shape[1]):
                    for y in range(master_arrays.shape[2]):
                        for c in range(master_arrays.shape[3]):
                            diff = master_arrays[i, x, y, c] - pool_arrays[j, x, y, c]
                            distance += diff * diff
                d_matrix[i, j] = math.sqrt(distance)

        # Define the number of threads per block and blocks per grid.
        threads_per_block = (16, 16)
        blocks_per_grid_x = math.ceil(len(grid_arrays) / threads_per_block[0])
        blocks_per_grid_y = math.ceil(len(pool_arrays) / threads_per_block[1])
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        # Launch the kernel.
        compute_d_matrix_kernel[blocks_per_grid, threads_per_block](  # type: ignore
            master_arrays_device, pool_arrays_device, d_matrix_device
        )

        LOGGER.debug("d_matrix shape: %s", d_matrix_device.shape)
        # Copy the result back to the host.
        return d_matrix_device.copy_to_host()

    def build_greedy(self, d_matrix: np.ndarray) -> Image.Image:
        """Construct the mosaic image using a greedy tile assignement algorithm.

        This leads to less accurate mosaics, but is significantly faster than the
        optimal assignement algorithm, especialy when the distance matrix is large.

        Args:
            d_matrix: The computed distance matrix.

        Returns:
            The `PIL.Image` instance of the mosaic.
        """
        mosaic = np.zeros((self.size[1], self.size[0], 3))

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
            tile_array = self.pool.array[tile]
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

    def build(self, d_matrix: np.ndarray) -> Image.Image:
        """Construct the mosaic image by solving the linear sum assignment problem.
        See: https://en.wikipedia.org/wiki/Assignment_problem

        Args:
            d_matrix: The computed distance matrix.

        Returns:
            The `PIL.Image` instance of the mosaic.

        Examples:
            Building a mosaic.

            >>> mosaic.build(mosaic.d_matrix())

            On a GPU.

            >>> mosaic.build(mosaic.d_matrix_cuda())
        """
        mosaic = np.zeros((self.size[1], self.size[0], 3))

        # expand the dmatrix to allow for repeated tiles
        if self.n_appearances > 0:
            d_matrix = np.tile(d_matrix, self.n_appearances)

        LOGGER.info("Computing optimal tile assignment.")
        row_ind, col_ind = linear_sum_assignment(d_matrix)
        pbar = tqdm(total=d_matrix.shape[0], desc="Building mosaic")
        for row, col in zip(row_ind, col_ind):
            slices = self.grid.slices[row]
            tile_array = self.pool.array[col % len(self.pool.array)]
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
