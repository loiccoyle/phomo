import logging
from os import PathLike
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from .palette import Palette
from .utils import open_img_file

LOGGER = logging.getLogger(__name__)


class Master(Palette):
    @classmethod
    def from_file(
        cls,
        master_image_file: PathLike,
        crop_ratio: Optional[float] = None,
        img_size: Optional[Tuple[int, int]] = None,
        mode: Optional[str] = None,
    ) -> "Master":
        """Create a master image from file.

        Args:
            master_image_file: path to image file.
            crop_ratio: width to height ratio to crop the master image to. 1 results in a square image.
            img_size: resize the image to the provided size, width followed by height.
            mode: convert the image to the provided mode. See PIL Modes.

        Returns:
            Master image instance.

        Examples:
            For black and white square 1280x1280 image.

            >>> Master.from_file("master.png", crop_ratio=1, img_size=(1280, 1280), convert="L")
        """
        img = open_img_file(
            master_image_file, crop_ratio=crop_ratio, size=img_size, mode=mode
        )
        return cls.from_image(img)

    @classmethod
    def from_image(cls, master_image: Image.Image) -> "Master":
        """Create a master image from PIL.Image.Image

        Args:
            master_image: `PIL.Image` instance.

        Returns:
            Master image instance.
        """
        array = np.asarray(master_image)
        # make sure the arrays have 3 channels even in black and white
        if array.ndim == 2:
            array = np.stack([array] * 3, -1)
        return cls(array)

    def __init__(self, array: np.ndarray) -> None:
        """The master image.

        Args:
            array: numpy array of the image, should contain 3 channels.

        Returns:
            Master image instance.
        """
        super().__init__(array)

    @property
    def img(self):
        """`PIL.Image` of the master image."""
        return Image.fromarray(self.array.round(0).astype("uint8"), mode="RGB")

    @property
    def pixels(self) -> np.ndarray:
        """Array containing the 3-channel pixel values of the master image."""
        return self.array.reshape(-1, self.array.shape[-1])

    def __repr__(self) -> str:
        return f"""{self.__class__.__module__}.{self.__class__.__name__} at {hex(id(self))}:
    shape: {self.array.shape}"""
