<p align="center"><img src="https://i.imgur.com/rMze8u5.png" width="1000"></p>
<p align="center"><b>Python package and CLI utility to create photo mosaics.</b></p>

<p align="center">
  <a href="https://github.com/loiccoyle/phomo/actions"><img src="https://github.com/loiccoyle/phomo/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://docs.loiccoyle.com/phomo"><img src="https://img.shields.io/github/deployments/loiccoyle/phomo/github-pages?label=docs"></a>
  <a href="https://pypi.org/project/phomo/"><img src="https://img.shields.io/pypi/v/phomo"></a>
  <a href="./LICENSE.md"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <img src="https://img.shields.io/badge/platform-linux%20%7C%20macOS%20%7C%20windows-informational">
</p>

> Prefer rust? Or a `npm` package? Check out [`loiccoyle/phomo-rs`](https://github.com/loiccoyle/phomo-rs)!

`phomo` lets you create [photographic mosaics](https://en.wikipedia.org/wiki/Photographic_mosaic).
It arranges the tile images to best recreate a master image. To achieve this, `phomo` computes a distance matrix between all the tiles and the master image regions, looking not just at the average colour but the norm of the colour distributions differences.
Once this distance matrix is computed, each tile is assigned to the optimal master image region by solving the linear sum assignment problem.

## 📦 Installation

In a terminal:

```sh
pip install phomo

# or for GPU acceleration:

pip install 'phomo[cuda]'
```

As always, it is usually a good idea to use a [virtual environment](https://docs.python.org/3/library/venv.html).

If you're just interested in command line usage, consider using [pipx](https://pypa.github.io/pipx/).

> [!NOTE]
> For GPU acceleration you'll need a CUDA compatible GPU and the CUDA toolkit installed. See [numba docs](https://numba.readthedocs.io/en/stable/cuda/overview.html#requirements) for details.

## 📋 Usage

### Python package

Check out the [docs](https://loiccoyle.com/phomo) and the [`examples`](./examples).

### CLI

Once it is installed, you can use the `phomo` command.

It would go something like:

```sh
phomo master.png tile_directory/ -S 20 20 -o mosaic.png
```

If in doubt see the help:

<!-- help start -->

```console
$ phomo -h
usage: phomo [-h] [-o OUTPUT] [-c MASTER_CROP_RATIO]
             [-s MASTER_SIZE [MASTER_SIZE ...]] [-C TILE_CROP_RATIO]
             [-S TILE_SIZE [TILE_SIZE ...]] [-n N_APPEARANCES] [-b] [-g]
             [-d SUBDIVISIONS [SUBDIVISIONS ...]] [-G]
             [-m {greyscale,norm,luv_approx}] [-j WORKERS] [-e]
             [--match-master-to-tiles] [--match-tiles-to-master] [--greedy]
             [-v]
             master tile_dir

positional arguments:
  master                Master image path.
  tile_dir              Directory containing the tile images.

options:
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
  -b, --black-and-white
                        Convert master and tile images to black and white.
  -g, --show-grid       Show the tile grid, don't build the mosiac.
  -d SUBDIVISIONS [SUBDIVISIONS ...], --subdivisions SUBDIVISIONS [SUBDIVISIONS ...]
                        Grid subdivision thresholds.
  -G, --gpu             Use GPU for distance matrix computation. Requires
                        installing with `pip install 'phomo[cuda]'`.
  -m {greyscale,norm,luv_approx}, --metric {greyscale,norm,luv_approx}
                        Distance metric.
  -j WORKERS, --workers WORKERS
                        Number of workers use to run when computing the
                        distance matrix.
  -e, --equalize        Equalize the colour distributions to cover the full
                        colour space.
  --match-master-to-tiles
                        Match the master image's colour distribution with the
                        tile image colours.
  --match-tiles-to-master
                        Match the tile images' colour distribution with the
                        master image colours.
  --greedy              Use a greedy tile assignment algorithm. Should improve
                        performance at the expense of accuracy.
  -v, --verbose         Verbosity.
```

<!-- help end -->

## 🤩 Credit

- [photomosaic](https://pypi.org/project/photomosaic/) for the grid subdivision feature.
