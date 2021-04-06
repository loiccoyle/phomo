# Mosaic

## Instalation

Requires python 3

In a terminal:

```sh
git clone https://github.com/loiccoyle/mosaic
cd mosaic
pip install .
```

## Usage
Once it is installed, you can use the `mosaic` command.

```
usage: mosaic [-h] [-f USAGE_FACTOR] [-u UPSCALE] [-s] [-v] [-b] tile_dir master output

positional arguments:
  tile_dir              Directory containing the tile images.
  master                Master image path.
  output                Output path.

optional arguments:
  -h, --help            show this help message and exit
  -f USAGE_FACTOR, --usage_factor USAGE_FACTOR
                        Ratio of tile images to use.
  -u UPSCALE, --upscale UPSCALE
                        Master image upscale coefficient.
  -s, --show            Show mosaic after building.
  -v, --verbose         Verbosity.
  -b, --black_and_white
                        Black and white.
```
