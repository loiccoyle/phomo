<h1 align="center">Phomo</h1>
<h3 align="center"><img src="https://i.imgur.com/rMze8u5.png" width="1000"></h3>
<h5 align="center">Python package and CLI utility to create photo mosaics.</h5>

<p align="center">
  <a href="https://github.com/loiccoyle/phomo/actions?query=workflow%3Atests"><img src="https://github.com/loiccoyle/phomo/workflows/tests/badge.svg"></a>
  <a href="./LICENSE.md"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>

## Instalation

Requires python 3

In a terminal:

```sh
git clone https://github.com/loiccoyle/phomo
cd phomo
pip install .
```

## Usage

Once it is installed, you can use the `phomo` command.

```
usage: phomo [-h] [-o OUTPUT] [-c MASTER_CROP_RATIO] [-s MASTER_SIZE [MASTER_SIZE ...]]
              [-C TILE_CROP_RATIO] [-S TILE_SIZE [TILE_SIZE ...]] [-n N_APPEARANCES] [-v] [-b] [-g]
              [-d SUBDIVISIONS [SUBDIVISIONS ...]]
              master tile_dir

positional arguments:
  master                Master image path.
  tile_dir              Directory containing the tile images.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Mosiac output path.
  -c MASTER_CROP_RATIO, --master-crop-ratio MASTER_CROP_RATIO
                        Crop the master image to width/height ratio.
  -s MASTER_SIZE [MASTER_SIZE ...], --master-size MASTER_SIZE [MASTER_SIZE ...]
                        Resize master image to width, height.
  -C TILE_CROP_RATIO, --tile-crop-ratio TILE_CROP_RATIO
                        Crop the tile images to width/height ratio.
  -S TILE_SIZE [TILE_SIZE ...], --tile-size TILE_SIZE [TILE_SIZE ...]
                        Resize tile images to width, height.
  -n N_APPEARANCES, --n-appearances N_APPEARANCES
                        The number of times a tile can appear in the mosaic.
  -v, --verbose         Verbosity.
  -b, --black_and_white
                        Black and white.
  -g, --show-grid       Show the tile grid, don't build the mosiac.
  -d SUBDIVISIONS [SUBDIVISIONS ...], --subdivisions SUBDIVISIONS [SUBDIVISIONS ...]
                        Subdivision thresholds.
```
