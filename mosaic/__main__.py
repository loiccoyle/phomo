import argparse
import logging
from pathlib import Path

import numpy as np

from . import logger
from .mosaic import MosaicGrid


# TODO: This is all outdated
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("master", help="Master image path.", type=str)
    parser.add_argument(
        "tile_dir", help="Directory containing the tile images.", type=str
    )
    parser.add_argument("output", help="Output path.")
    parser.add_argument(
        "-f",
        "--usage_factor",
        default=0.9,
        help="Ratio of tile images to use.",
        type=float,
    )
    parser.add_argument(
        "-u",
        "--upscale",
        help="Master image upscale coefficient.",
        default=1,
        type=float,
    )
    parser.add_argument(
        "-s", "--show", help="Show mosaic after building.", action="store_true"
    )
    parser.add_argument("-v", "--verbose", help="Verbosity.", action="count", default=0)
    parser.add_argument(
        "-b", "--black_and_white", help="Black and white.", action="store_true"
    )
    args = parser.parse_args()

    tile_dir = Path(args.tile_dir)
    mode = "RGB"

    if tile_dir.is_dir():
        tiles = list(tile_dir.glob("*"))
    else:
        raise ValueError(f"'{args.tile_dir}' is not a directory.")

    if args.verbose > 0:
        verbose_map = {1: logging.INFO, 2: logging.DEBUG}
        level = verbose_map[args.verbose]
        # from https://docs.python.org/3/howto/logging.html#configuring-logging
        logger.setLevel(level)
        # create console handler and set level to debug
        handler = logging.StreamHandler()
        handler.setLevel(level)
        # create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        # add formatter to handler
        handler.setFormatter(formatter)
        # add ch to logger
        logger.addHandler(handler)

    master_im = open_exif(args.master)
    scaled = [int(i * args.upscale) for i in master_im.size]
    master_im = master_im.resize(scaled)

    if args.black_and_white:
        mode = "L"
        print("Converting to black and white")
        master_im = master_im.convert(mode=mode)

    tiles = np.random.choice(tiles, int(len(tiles) * args.usage_factor), replace=False)

    mosaic = Mosaic(master_im, tiles)
    mosaic_im = mosaic.build()
    if args.show:
        mosaic_im.show()
    mosaic_im.save(args.output)


if __name__ == "__main__":
    main()
