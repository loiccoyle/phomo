from pathlib import Path
from typing import Optional, Tuple

import colour
import numpy as np
from PIL import Image
from PIL.ImageOps import exif_transpose
from tqdm.auto import tqdm


def rainbow_of_squares(target_dir: Path, shape=(10, 10), range_params=(0, 256, 15)):
    """
    Generate 5832 small solid-color tiles for experimentation and testing.
    Parameters
    ----------
    target_dir : string
    shape : tuple, optional
        default is (10, 10)
    range_params : tuple, optional
        Passed to ``range()`` to stride through each color channel.
        Default is ``(0, 256, 15)``.
    """
    target_dir.mkdir(exist_ok=True)
    with tqdm(total=3 * len(range(*range_params))) as pbar:
        canvas = np.ones(shape + (3,))
        for r in range(*range_params):
            for g in range(*range_params):
                for b in range(*range_params):
                    img = (canvas * [r, g, b]).astype(np.uint8)
                    filename = "{:03d}-{:03d}-{:03d}.png".format(r, g, b)
                    img = Image.fromarray(img, mode="RGB")
                    img.save(target_dir / filename)
                    pbar.update()


def crop_to_ratio(image: Image.Image, ratio: float = 1) -> Image.Image:
    """Reshapes an image to the specified ratio by cropping along the larger
    dimension that doesn't meet the specified aspect ratio.

    Args:
        image: PIL.Image to crop.
        ratio: width to height ratio to which to crop the image. Use 1 to obtain a square image.

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
    img_file: Path,
    crop_ratio: Optional[float] = None,
    img_size: Optional[Tuple[int, int]] = None,
    convert: Optional[str] = None,
) -> Image.Image:
    """Open an image file with some extra bells and whistles.

    Args:
        img_file: path to the image.
        crop_ratio: width to height to which to crop the image.
        img_size: resize image.
        convert: convert the image to the provided mode. See PIL image modes.

    Returns:
        Image instance.
    """
    with Image.open(img_file) as img:
        img = exif_transpose(img)
        if crop_ratio is not None:
            img = crop_to_ratio(img, crop_ratio)
        if img_size is not None:
            img = img.resize(img_size)
        if convert is not None:
            img = img.convert(convert)
        else:
            img = img.convert("RGB")
    return img


def to_ucs(array_sRGB: np.ndarray) -> np.ndarray:
    """Convert image pixel array to UCS.

    Args:
        array_sRGB: 3D array containing the pixel colours. It should be a 3 channel
            array with colour bound from 0 to 255.

    Returns:
        3D array containing the values in UCS.
    """
    array_xyz = colour.sRGB_to_XYZ(array_sRGB / 255)
    return colour.XYZ_to_UCS(array_xyz)


def to_rgb(array_ucs: np.ndarray) -> np.ndarray:
    """Convert image pixel array to RGB.

    Args:
        array_ucs: 3D array containing the pixel colours. It should be a 3 channel
            array.

    Returns:
        3D array containing the values in RGB.
    """
    array_xyz = colour.UCS_to_XYZ(array_ucs)
    return np.clip(colour.XYZ_to_sRGB(array_xyz), 0, 1) * 255


# TODO: implement latest CAM16 models:
# https://github.com/colour-science/colour#54129cam16-lcd-cam16-scd-and-cam16-ucs-colourspaces---li-et-al-2017
# def to_UCS(array_sRGB: np.ndarray) -> np.ndarray:
#     """sRGB -> XYZ -> CAM16 -> CAM16UCS"""
#     array_xyz = colour.sRGB_to_XYZ(array_sRGB / 255)
#     XYZ_w = [95.05, 100.00, 108.88]
#     L_A = 318.31
#     Y_b = 20.0
#     surround = colour.VIEWING_CONDITIONS_CIECAM02["Average"]
#     specification = colour.XYZ_to_CAM16(array_xyz, XYZ_w, L_A, Y_b, surround)
#     JMh = np.stack([specification.J, specification.M, specification.h], axis=-1)
#     array_ucs = colour.JMh_CAM16_to_CAM16UCS(JMh)
#     return array_ucs


# def to_RGB(array_UCS: np.ndarray) -> np.ndarray:
#     XYZ_w = [95.05, 100.00, 108.88]
#     L_A = 318.31
#     Y_b = 20.0
#     surround = colour.VIEWING_CONDITIONS_CIECAM02["Average"]
#     array_cam16 = colour.CAM16UCS_to_JMh_CAM16(array_UCS)
#     array_xyz = colour.CAM16_to_XYZ(
#         colour.CAM_Specification_CAM16(
#             J=array_cam16[:, :, 0], C=array_cam16[:, :, 1], h=array_cam16[:, :, 2]
#         ),
#         XYZ_w,
#         L_A,
#         Y_b,
#         surround,
#     )
#     array_rgb = np.clip(colour.XYZ_to_sRGB(array_xyz), 0, 1) * 255
#     return array_rgb
