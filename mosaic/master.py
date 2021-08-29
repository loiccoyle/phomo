import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from .palette import Palette
from .pool import Pool
from .utils import open_img_file, to_rgb, to_ucs


class Master(Palette):
    @classmethod
    def from_file(
        cls,
        master_image_file: Path,
        *args,
        crop_ratio: Optional[float] = None,
        img_size: Optional[Tuple[int, int]] = None,
        convert: Optional[str] = None,
        **kwargs,
    ) -> "Master":
        """Create a master image from file.

        Args:
            master_image_file: path to image file.
            crop_ratio: width to height ratio to crop the master image to. 1 results in a square image.
            image_size: resize the image to the provided size, width followed by height.
            convert: convert the image to the provided mode. See PIL Modes.

        Returns:
            Master instance.

        Examples:
            For black and white square 1280x1280 image.

            >>> Master.from_file("master.png", crop_ratio=1, img_size=(1280, 1280), convert="L")
        """
        img = open_img_file(
            master_image_file, crop_ratio=crop_ratio, img_size=img_size, convert=convert
        )
        return cls.from_image(img, *args, **kwargs)

    @classmethod
    def from_image(cls, master_image: Image.Image, *args, **kwargs) -> "Master":
        """Create a master image from PIL.Image.Image

        Args:
            master_image: PIL.Image instance.

        Returns:
            Master instance.
        """
        array = np.asarray(master_image)
        # make sure the arrays have 3 channels even in black and white
        if array.ndim == 2:
            array = np.stack([array] * 3, -1)
        return cls(array, *args, **kwargs)

    def __init__(self, array: np.ndarray):
        """The master image.

        Args:
            array: numpy array of the image, should contain 3 channels.

        Returns:
            Master instance.
        """
        self.array = array
        self._log = logging.getLogger(__name__)
        self._log.info("Shape: %s", self.array.shape)
        self._space = "rgb"

    @property
    def img(self):
        """PIL.Image of the Master image."""
        if self._space == "ucs":
            img_array = to_rgb(self.array)
        else:
            img_array = self.array
        return Image.fromarray(img_array.round(0).astype("uint8"), mode="RGB")

    @property
    def pixels(self) -> np.ndarray:
        """Array containing the 3-channel pixel values of the Master image."""
        return self.array.reshape(-1, self.array.shape[-1])

    @property
    def space(self) -> str:
        """Colour space of the tiles in the pool.

        Either "rgb" or "ucs".
        """
        return self._space

    def to_ucs(self) -> "Master":
        """Convert master to uniform colour space.

        Returns:
            Master instance converted to UCS colour space.
        """
        if self._space == "ucs":
            raise ValueError("Color space is already UCS.")
        out = Master(to_ucs(self.array))
        out._space = "ucs"
        return out

    def to_rgb(self) -> "Master":
        """Convert master to RGB colours space.

        Returns:
            Master instance converted to RGB colour space.
        """
        if self._space == "rgb":
            raise ValueError("Color space is already RGB.")
        out = Master(to_rgb(self.array))
        out._space = "rgb"
        return out

    def match(self, pool: Pool) -> "Master":
        """Match the colour distribution of the master with the pool.

        Args:
            pool: Pool instance to match the colour distribution.

        Returns:
            Master instance matched with its colour distribution matched to the pool.
        """
        if self._space != pool._space:
            raise ValueError("Master and Pool have different color spaces.")
        match_function = self._match_function(pool)
        array = match_function(self.array)
        # if self._space == "ucs":
        #     array = to_RGB(array)
        # elif self._space == "rgb":
        #     array = array.round(0).astype('uint8')
        out = Master(array)
        out._space = self._space
        return out

    def __repr__(self) -> str:
        return f"""{self.__class__.__module__}.{self.__class__.__name__} at {hex(id(self))}:
    space: {self.space}
    size: {self.array.shape}"""
