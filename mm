#/usr/bin/env python3

import argparse
from mosaic_maker import Mosaic
from mosaic_maker.utils import open_exif

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("tile_dir", help="path of directory containing the tiles of the mosaic", type=str)
    parser.add_argument("master", help= "path of master image", type=str)
    parser.add_argument("output", help="path of output")
    parser.add_argument("-u", "--upscale", help="upscale coefficient master", default=1, type=float)
    parser.add_argument("-s", "--show", help="show mosaic after building", action="store_true")
    parser.add_argument("-v", "--verbose", help="verbosity", action="store_true")
    parser.add_argument("-b", "--black_and_white", help="black and white", action="store_true")
    args = parser.parse_args()

    master = args.master
    tile_dir = args.tile_dir
    upscale = args.upscale
    verbose = args.verbose
    output = args.output
    mode = "RGB"

    master_im = open_exif(master)
    scaled = [int(i*upscale) for i in master_im.size]
    master_im = master_im.resize(scaled)

    if args.black_and_white:
        mode = "L"
        print("Converting to black and white")
        master_im = master_im.convert(mode=mode)

    mosaic = Mosaic(master_im, tile_dir, verbose=verbose, mode=mode)
    mosaic.tile_load()
    mosaic_im = mosaic.build_mosaic()
    if args.show:
        mosaic_im.show()
    mosaic_im.save(output)

