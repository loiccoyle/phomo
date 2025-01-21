import argparse
import logging
import sys
from pathlib import Path
from typing import List

from . import logger
from .metrics import METRICS
from .mosaic import Master, Mosaic, Pool


def parse_args(args: List[str]) -> argparse.Namespace:
    """Parse the command line arguments.

    Args:
        args: list of command line arguments.

    Returns:
        argparse Namespace of the parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "master",
        help="Master image path.",
        type=str,
    )
    parser.add_argument(
        "tile_dir",
        help="Directory containing the tile images.",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Mosiac output path.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-c",
        "--master-crop-ratio",
        help="Crop the master image to width/height ratio.",
        default=None,
        type=float,
    )
    parser.add_argument(
        "-s",
        "--master-size",
        help="Resize master image to width, height.",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "-C",
        "--tile-crop-ratio",
        help="Crop the tile images to width/height ratio.",
        default=None,
        type=float,
    )
    parser.add_argument(
        "-S",
        "--tile-size",
        help="Resize tile images to width, height.",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "-n",
        "--n-appearances",
        help="The number of times a tile can appear in the mosaic.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-b",
        "--black-and-white",
        help="Convert master and tile images to black and white.",
        action="store_true",
    )
    parser.add_argument(
        "-g",
        "--show-grid",
        help="Show the tile grid, don't build the mosiac.",
        action="store_true",
    )
    parser.add_argument(
        "-d",
        "--subdivisions",
        help="Grid subdivision thresholds.",
        nargs="+",
        default=[],
        type=float,
    )
    parser.add_argument(
        "-G",
        "--gpu",
        help="Use GPU for distance matrix computation. Requires installing with `pip install 'phomo[cuda]'`.",
        action="store_true",
    )
    parser.add_argument(
        "-m",
        "--metric",
        help="Distance metric.",
        default="norm",
        type=str,
        choices=METRICS.keys(),
    )
    parser.add_argument(
        "-j",
        "--workers",
        help="Number of workers use to run when computing the distance matrix.",
        default=1,
        type=int,
    )
    parser.add_argument(
        "-e",
        "--equalize",
        help="Equalize the colour distributions to cover the full colour space.",
        action="store_true",
    )
    parser.add_argument(
        "--match-master-to-tiles",
        help="Match the master image's colour distribution with the tile image colours.",
        action="store_true",
    )
    parser.add_argument(
        "--match-tiles-to-master",
        help="Match the tile images' colour distribution with the master image colours.",
        action="store_true",
    )
    parser.add_argument(
        "--greedy",
        help="Use a greedy tile assignment algorithm. Should improve performance at the expense of accuracy.",
        action="store_true",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Verbosity.",
        action="count",
        default=0,
    )
    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])

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

    mode = None
    if args.black_and_white:
        mode = "L"

    tile_size = args.tile_size
    if args.tile_size is not None:
        if len(args.tile_size) != 2:
            raise ValueError("Provided tile size is not of length 2.")
        tile_size = tuple(tile_size)

    master_size = args.master_size
    if args.master_size is not None:
        if len(args.master_size) != 2:
            raise ValueError("Provided master size is not of length 2.")
        master_size = tuple(master_size)

    master = Master.from_file(
        Path(args.master),
        crop_ratio=args.master_crop_ratio,
        img_size=master_size,  # type: ignore
        mode=mode,
    )

    pool = Pool.from_dir(
        Path(args.tile_dir),
        crop_ratio=args.tile_crop_ratio,
        tile_size=tile_size,  # type: ignore
        mode=mode,
    )
    if args.equalize:
        master = master.equalize()
        pool = pool.equalize()
    elif args.match_tiles_to_master:
        pool = pool.match(master)
    elif args.match_master_to_tiles:
        master = master.match(pool)

    mosaic = Mosaic(master, pool, n_appearances=args.n_appearances)
    for threshold in args.subdivisions:
        mosaic.grid.subdivide(threshold)

    logger.debug("mosaic:\n%s", repr(mosaic))
    logger.info("Number of unused tiles: %i", mosaic.n_leftover)

    if args.show_grid:
        grid_im = mosaic.grid.plot()
        grid_im.show()
    else:
        d_matrix = (
            mosaic.d_matrix(metric=args.metric, workers=args.workers)
            if not args.gpu
            else mosaic.d_matrix_cuda(metric=args.metric)
        )
        mosaic_im = (
            mosaic.build_greedy(d_matrix) if args.greedy else mosaic.build(d_matrix)
        )
        if args.output is None:
            mosaic_im.show()
        else:
            mosaic_im.save(args.output)
