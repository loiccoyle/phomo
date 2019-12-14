import numpy as np
from PIL import Image, ExifTags


def crop_square(image_to_crop):
    '''Crop image_to_crop to square by croping the largest dimension
    '''
    pleb_size = image_to_crop.size
    min_dim_i = np.argmin(pleb_size)
    min_dim = pleb_size[min_dim_i]

    if min_dim_i == 0:
        crop_box = (0, (pleb_size[1] - min_dim) / 2,
                    pleb_size[0], (pleb_size[1] + min_dim) / 2)
    if min_dim_i == 1:
        crop_box = ((pleb_size[0] - min_dim) / 2, 0,
                    (pleb_size[0] + min_dim) / 2, pleb_size[1])

    crop_image = image_to_crop.crop(crop_box)
    return crop_image


def open_exif(image_file):
    '''Opens image_file and takes into account the orientation tag to conserve the
    correct orientation
    '''
    img = Image.open(image_file)
    try:
        exif = dict((ExifTags.TAGS[k], v)
                    for k, v in img._getexif().items() if k in ExifTags.TAGS)
        if exif['Orientation'] == 3:
            img = img.rotate(180, expand=True)
        elif exif['Orientation'] == 6:
            img = img.rotate(270, expand=True)
        elif exif['Orientation'] == 8:
            img = img.rotate(90, expand=True)
        return img
    except KeyError:
        return img
