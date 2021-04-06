import logging
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

from PIL import Image
from tqdm.auto import tqdm

from .utils import crop_square
from .utils import open_exif


class Mosaic(object):
    def __init__(
        self,
        master_img: Image,
        tiles: List[Path],
    ):
        """Create a mosaic of the 'master_img' using the 'tiles'.

        Args:
            master_img: PIL.Image instance of the master image to reconstruct
                using the tiles.
            tiles: List of file paths of the tiles with which to reconstruct
                the master image.
        """
        self.master_img = master_img
        self.tiles = tiles
        self._log = logging.getLogger(__name__)
        self._log.info("Mosaic grid %s", self.grid)
        self._log.info("Master size: %s", self.master_img.size)
        self._log.info("Number of Tiles: %s", len(self.tiles))
        self._log.info("Tile size: %s", self.tile_size)
        self._log.info("Mosaic size: %s", self.mosaic_size)

        self.master_coords, self.master_arrays = self.get_master_arrays()
        self.tile_arrays = self.get_tile_arrays()

    @property
    def tiles(self):
        return self._tiles

    @tiles.setter
    def tiles(self, value):
        self._tiles = value
        self._grid = [np.sqrt(len(self.tiles)) * i for i in [self.width_to_height, 1]]
        self._tile_size = np.ceil(np.divide(self.master_img.size, self.grid)).astype(
            int
        )
        self._mosaic_size = self.master_img.size - self.master_img.size % self.tile_size

    @property
    def master_img(self):
        return self._master_img

    @master_img.setter
    def master_img(self, value):
        self._master_img = value
        self._width_to_height = self.master_img.size[0] / self.master_img.size[1]

    @property
    def grid(self):
        return self._grid

    @property
    def tile_size(self):
        return self._tile_size

    @property
    def mosaic_size(self):
        return self._mosaic_size

    @property
    def width_to_height(self):
        return self._width_to_height

    def get_master_arrays(self) -> Tuple[List[Tuple[int, int]], List[np.ndarray]]:
        """Divide the master image into tile sized arrays.

        Returns:
            A list containing the the pixel coordinates of the top left corner
            of each tile.
            A list containing the np.ndarrays of the master image.
        """
        master_arrays = []
        master_coords = []
        master_ar = np.array(self.master_img).astype("int16")
        for x in range(0, self.mosaic_size[0], self.tile_size[0]):
            for y in range(0, self.mosaic_size[1], self.tile_size[1]):
                master_coords.append((x, y))
                master_arrays.append(
                    master_ar[y : y + self.tile_size[0], x : x + self.tile_size[1]],
                )
        return master_coords, master_arrays

    def _load_square(self, img_file: Path) -> Union[np.ndarray, None]:
        """Load an image and crop it to square."""
        try:
            with open_exif(img_file) as tile:
                image = crop_square(tile)
                image = image.resize(self.tile_size, Image.ANTIALIAS)
                if image.mode != self.master_img.mode:
                    image = image.convert(mode=self.master_img.mode)
            return np.array(image).astype("int16")
        except (IndexError, OSError, ValueError):
            self._log.error("Error skipping %s", img_file, exc_info=True)

    def get_tile_arrays(self) -> List[np.ndarray]:
        """Goes through the tile files, load the arrays and crop to square.

        Returns:
            A list of arrays of the tile images.
        """
        tile_arrays = [
            self._load_square(image)
            for image in tqdm(self.tiles, desc="Loading and cropping tiles")
        ]
        return [array for array in tile_arrays if array is not None]

    def build(self) -> Image:
        """Builds the Photo Mosaic.

        Returns:
            The PIL.Image instance containing the mosaic.
        """
        # Init mosaic array.
        n_channels = len(self.master_img.getbands())
        if n_channels == 1:
            mosaic = np.zeros(self.mosaic_size)
        else:
            mosaic = np.zeros((*self.mosaic_size, n_channels))

        # Compute the distance matrix.
        d_matrix = np.zeros((len(self.master_arrays), len(self.tile_arrays)))
        for i, tile in tqdm(
            enumerate(self.tile_arrays),
            total=len(self.tile_arrays),
            desc="Building distance matrix",
        ):
            d_matrix[:, i] = [
                np.linalg.norm(tile - master_tile) for master_tile in self.master_arrays
            ]

        d_matrix = np.ma.array(d_matrix)
        pbar = tqdm(total=d_matrix.shape[0], desc="Building mosaic")
        while d_matrix[~d_matrix.mask].size != 0:
            # while the distance matrix isn't completely masked
            min_ind = np.where(d_matrix == np.min(d_matrix[~d_matrix.mask]))
            for row, col in zip(min_ind[0], min_ind[1]):
                if d_matrix.mask.shape != () and d_matrix.mask[row, col]:
                    continue
                self._log.debug("%s, row:%s, col:%s", np.min(d_matrix), row, col)
                x, y = self.master_coords[row]
                array = self.tile_arrays[col]
                mosaic[y : y + self.tile_size[1], x : x + self.tile_size[0]] = array
                d_matrix[:, col] = np.ma.masked
                d_matrix[row, :] = np.ma.masked
                pbar.update(1)
        pbar.close()
        return Image.fromarray(np.uint8(mosaic))
