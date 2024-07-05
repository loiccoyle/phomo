import logging
from os import PathLike
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike
from PIL import Image
from tqdm.auto import tqdm

from .palette import Palette
from .utils import open_img_file

LOGGER = logging.getLogger(__name__)


class Pool(Palette):
    @classmethod
    def from_dir(
        cls,
        tile_dir: PathLike,
        crop_ratio: Optional[float] = None,
        tile_size: Optional[Tuple[int, int]] = None,
        mode: Optional[str] = None,
    ) -> "Pool":
        """Create a `Pool` instance from the images in a directory.

        Args:
            tile_dir: path to directory containing the images.
            crop_ratio: width to height ratio to crop the tile images to. 1 results in a
                square image.
            tile_size: resize the image to the provided size, width followed by height.
            mode: convert the images to the provided mode.
                See [PIL Modes](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes).
        """
        if not isinstance(tile_dir, Path):
            tile_dir = Path(tile_dir)
        if not tile_dir.is_dir():
            raise ValueError(f"'{tile_dir}' is not a directory.")
        array = cls._load_files(
            list(tile_dir.glob("*")),
            crop_ratio=crop_ratio,
            size=tile_size,
            mode=mode,
        )
        return cls(array)

    @classmethod
    def from_files(
        cls,
        files: Sequence[PathLike],
        crop_ratio: Optional[float] = None,
        tile_size: Optional[Tuple[int, int]] = None,
        mode: Optional[str] = None,
    ) -> "Pool":
        """Create a `Pool` instance from a list of images.

        Args:
            files: list of paths to the tile images.
            crop_ratio: width to height ratio to crop the master image to. 1 results in a square image.
            tile_size: resize the image to the provided size, width followed by height.
            mode: mode the image to the provided mode.
                See [PIL Modes](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes).
        """
        array = cls._load_files(files, crop_ratio=crop_ratio, size=tile_size, mode=mode)
        return cls(array)

    def __init__(
        self,
        array: ArrayLike,
    ) -> None:
        """A `Pool` of tile images, to use in contructing the photo mosaic.

        Args:
            array: `Pool` image data array. Should be (n_tiles, height, width, 3)
        """
        super().__init__(array)

    @property
    def tiles(self) -> "PoolTiles":
        """Access the Pool's tile images.

        Examples:
            Show the first image in the pool.

            >>> pool.tiles[0].show()
        """
        return PoolTiles(self.array)

    @property
    def pixels(self) -> np.ndarray:
        """Array containing the 3-channel pixel values of all the images in the Pool."""
        return np.vstack([array.reshape(-1, array.shape[-1]) for array in self.array])

    @staticmethod
    def _load_files(files: Sequence[PathLike], **kwargs) -> List[np.ndarray]:
        arrays = []
        for tile in tqdm(files, desc="Loading tiles"):
            img = open_img_file(tile, **kwargs)
            array = np.asarray(img)
            # make sure the arrays have 3 channels even in black and white
            if array.ndim == 2:
                array = np.stack([array] * 3, -1)
            arrays.append(array)
        return arrays

    def __len__(self) -> int:
        return len(self.array)

    def __repr__(self) -> str:
        return f"""{self.__class__.__module__}.{self.__class__.__name__} at {hex(id(self))}:
    len: {self.__len__()}"""


class PoolTiles:
    """Helper interface to access of `PIL.Image` instances of the tiles."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def __getitem__(self, index) -> Image.Image:
        return Image.fromarray(self._array[index].round(0).astype("uint8"), mode="RGB")
