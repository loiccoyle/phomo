import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from .palette import Palette
from .utils import open_img_file


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
            master_image_file, crop_ratio=crop_ratio, size=img_size, convert=convert
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

    def __init__(self, array: np.ndarray) -> None:
        """The master image.

        Args:
            array: numpy array of the image, should contain 3 channels.

        Returns:
            Master instance.
        """
        self.array = array
        self._log = logging.getLogger(__name__)
        self._log.info("master shape: %s", self.array.shape)

    @property
    def img(self):
        """PIL.Image of the Master image."""
        return Image.fromarray(self.array.round(0).astype("uint8"), mode="RGB")

    @property
    def pixels(self) -> np.ndarray:
        """Array containing the 3-channel pixel values of the Master image."""
        return self.array.reshape(-1, self.array.shape[-1])

    def __repr__(self) -> str:
        return f"""{self.__class__.__module__}.{self.__class__.__name__} at {hex(id(self))}:
    shape: {self.array.shape}"""
