from os import PathLike
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from PIL.ImageOps import exif_transpose
from tqdm.auto import tqdm


def rainbow_of_squares(
    target_dir: Path,
    size: Tuple[int, int] = (10, 10),
    r_range: range = range(0, 256, 15),
    g_range: range = range(0, 256, 15),
    b_range: range = range(0, 256, 15),
) -> None:
    """Generate a bunch of solid-color tiles for experimentation and testing.

    Args:
        target_dir: direcotry in which to place the rainbow tiles.
        size: size of the images, width followed by height.
        r_range_params: Passed to ``range()`` to stride through the red channel.
        g_range_params: Passed to ``range()`` to stride through the green channel.
        b_range_params: Passed to ``range()`` to stride through the blue channel.
    """
    target_dir.mkdir(exist_ok=True)
    with tqdm(
        total=len(r_range) + len(g_range) + len(b_range), desc="Generating color tiles"
    ) as pbar:
        canvas = np.ones((*size[::-1], 3))
        for r in r_range:
            for g in g_range:
                for b in b_range:
                    img = (canvas * [r, g, b]).astype(np.uint8)
                    filename = "{:03d}-{:03d}-{:03d}.png".format(r, g, b)
                    img = Image.fromarray(img, mode="RGB")
                    img.save(target_dir / filename)
                    pbar.update()


def crop_to_ratio(image: Image.Image, ratio: float = 1) -> Image.Image:
    """Reshapes an image to the specified ratio by cropping along the larger
    dimension.

    Args:
        image: PIL.Image to crop.
        ratio: width to height ratio to which to crop the image. Use 1 to obtain a
            square image.

    Returns:
        Cropped PIL.Image.
    """

    width, height = image.size

    def crop_height(image, rx):
        return image.crop(
            (
                0,
                (rx / 2),
                width,
                height - (rx / 2),
            )
        )

    def crop_width(image, rx):
        return image.crop(
            (
                (rx / 2),
                0,
                width - (rx / 2),
                height,
            )
        )

    # Find the delta change.
    rxheight = width / ratio - height
    rxwidth = height * ratio - width

    # Can only crop pixels, not add them.
    if rxheight < 0 and rxwidth < 0:
        # If both sides can be cropped to get what we want:
        # Select the largest (because both are negative)
        if rxheight > rxwidth:
            return crop_height(image, rxheight * -1)
        else:
            return crop_width(image, rxwidth * -1)

    elif rxheight < 0:
        # Trim height to fit aspect ratio
        return crop_height(image, rxheight * -1)

    elif rxwidth < 0:
        # Trim width to fit aspect ratio
        return crop_width(image, rxwidth * -1)

    else:
        # Can't do anything in this case
        return image


def open_img_file(
    img_file: PathLike,
    crop_ratio: Optional[float] = None,
    size: Optional[Tuple[int, int]] = None,
    mode: Optional[str] = None,
) -> Image.Image:
    """Open an image file with some extra bells and whistles.

    Args:
        img_file: path to the image.
        crop_ratio: width to height to which to crop the image.
        size: resize image.
        mode: convert the image to the provided mode. See PIL image modes.

    Returns:
        Image instance.
    """
    with Image.open(img_file) as img:
        img_t = exif_transpose(img)
        img = img_t if img_t is not None else img
        if crop_ratio is not None:
            img = crop_to_ratio(img, crop_ratio)
        if size is not None:
            img = img.resize(size)
        if mode is not None:
            img = img.convert(mode)
        else:
            img = img.convert("RGB")
    return img


def resize_array(
    array: np.ndarray, size: Tuple[int, int], *args, **kwargs
) -> np.ndarray:
    """Resize an array representing and image.

    Args:
        array: array containing the images data.
        size: desired size, width followed by height.
        *args, **kwargs: passed to `PIL.Image.resize`.

    Returns:
        Array containing the resized image data.
    """
    return np.asarray(Image.fromarray(array).resize(size, *args, **kwargs))
