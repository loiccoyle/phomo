import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from .palette import Palette
from .utils import open_img_file, to_rgb, to_ucs


class Pool(Palette):
    @classmethod
    def from_dir(
        cls,
        tile_dir: Path,
        *args,
        crop_ratio: Optional[float] = None,
        tile_size: Optional[Tuple[int, int]] = None,
        convert: Optional[str] = None,
        **kwargs,
    ) -> "Pool":
        """Create a Pool instance from the images in a directory.

        Args:
            tile_dir: path to directory containing the images.
            crop_ratio: width to height ratio to crop the master image to. 1 results in a square image.
            image_size: resize the image to the provided size, width followed by height.
            convert: convert the image to the provided mode. See PIL Modes.
        """
        if not tile_dir.is_dir():
            raise ValueError(f"'{tile_dir}' is not a directory.")
        arrays = cls._load_files(
            list(tile_dir.glob("*")),
            crop_ratio=crop_ratio,
            img_size=tile_size,
            convert=convert,
        )
        return cls(arrays, *args, **kwargs)

    @classmethod
    def from_files(
        cls,
        files: List[Path],
        *args,
        crop_ratio: Optional[float] = None,
        tile_size: Optional[Tuple[int, int]] = None,
        convert: Optional[str] = None,
        **kwargs,
    ) -> "Pool":
        """Create a Pool instance from a list of images.

        Args:
            files: list of paths to the tile images.
            crop_ratio: width to height ratio to crop the master image to. 1 results in a square image.
            image_size: resize the image to the provided size, width followed by height.
            convert: convert the image to the provided mode. See PIL Modes.
        """
        arrays = cls._load_files(
            files, crop_ratio=crop_ratio, img_size=tile_size, convert=convert
        )
        return cls(arrays, *args, **kwargs)

    def __init__(
        self,
        arrays: List[np.ndarray],
    ) -> None:
        """A Pool of images tiles.

        Args:
            arrays: list of arrays containing the image pixel values. Should containing
                3 colour channels.
        """
        self.arrays = arrays
        self._log = logging.getLogger(__name__)
        self._log.info("Number of tiles: %s", len(self.arrays))
        self._space = "rgb"

    @property
    def tiles(self) -> "PoolTiles":
        """Access the Pool's tile images.

        Examples:
            Show the first image in the pool.

            >>> pool.tiles[0].show()
        """
        if self._space == "rgb":
            arrays = self.arrays
        else:
            arrays = [to_rgb(array) for array in self.arrays]
        return PoolTiles(arrays)

    @property
    def pixels(self) -> np.ndarray:
        """Array containing the 3-channel pixel values of all the images in the Pool."""
        # if self._colors is None:
        #     self._log.debug("Computing colors.")
        #     self._colors = self._flatten_arrays(self.arrays)
        return np.vstack([array.reshape(-1, array.shape[-1]) for array in self.arrays])

    @property
    def space(self) -> str:
        """Colour space of the tiles in the pool.

        Either "rgb" or "ucs".
        """
        return self._space

    @staticmethod
    def _load_files(files: List[Path], **kwargs) -> List[np.ndarray]:
        arrays = []
        for tile in tqdm(files, desc="Loading arrays"):
            img = open_img_file(tile, **kwargs)
            array = np.asarray(img)
            # make sure the arrays have 3 channels even in black and white
            if array.ndim == 2:
                array = np.stack([array] * 3, -1)
            arrays.append(array)
        return arrays

    def to_ucs(self) -> "Pool":
        if self._space == "ucs":
            raise ValueError("Color space is already UCS.")
        out = Pool(
            [
                to_ucs(array)
                for array in tqdm(self.arrays, desc="Converting tiles to UCS")
            ]
        )
        out._space = "ucs"
        return out

    def to_rgb(self) -> "Pool":
        if self._space == "rgb":
            raise ValueError("Color space is already RGB.")
        out = Pool(
            [
                to_rgb(array)
                for array in tqdm(self.arrays, desc="Converting tiles to RGB")
            ]
        )
        out._space = "rgb"
        return out

    def __len__(self) -> int:
        return len(self.arrays)

    def __repr__(self) -> str:
        return f"""{self.__class__.__module__}.{self.__class__.__name__} at {hex(id(self))}:
    space: {self.space}
    len: {self.__len__()}"""


class PoolTiles:
    """Helper interface to access of PIL.Image instances of the tiles."""

    def __init__(self, arrays: List[np.ndarray]) -> None:
        self._arrays = arrays

    def __getitem__(self, index) -> Union[List[Image.Image], Image.Image]:
        selected = self._arrays[index]
        if isinstance(selected, list):
            return [Image.fromarray(selected.round(0).astype("uint8"), mode="RGB")]
        else:
            return Image.fromarray(selected.round(0).astype("uint8"), mode="RGB")
