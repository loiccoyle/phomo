<h3 align="center"><img src="https://i.imgur.com/rMze8u5.png" width="1000"></h3>
<h5 align="center">Python package and CLI utility to create photo mosaics.</h5>

<p align="center">
  <a href="https://github.com/loiccoyle/phomo/actions?query=workflow%3Atests"><img src="https://github.com/loiccoyle/phomo/workflows/tests/badge.svg"></a>
  <a href="https://pypi.org/project/phomo/"><img src="https://img.shields.io/pypi/v/phomo"></a>
  <a href="./LICENSE.md"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <img src="https://img.shields.io/badge/platform-linux%20%7C%20macOS%20%7C%20windows-informational">
</p>

`phomo` lets you create [photographic mosaics](https://en.wikipedia.org/wiki/Photographic_mosaic).
It arranges the tile images to best recreate a master image. To achieve this, `phomo` computes a distance matrix between all the tiles and the master image regions, looking not just at the average colour but the norm of the colour distributions differences.
Once this distance matrix is computed, each tile is assigned to the region of the master with the smallest distance between the colour distributions.

## 📦 Installation

Requires python 3

In a terminal:

```sh
$ pip install phomo
```

As always, it is usually a good idea to use a [virtual environment](https://docs.python.org/3/library/venv.html).

If you're just interested in command line usage, consider using [pipx](https://pypa.github.io/pipx/).

## 📋 Usage

### Python package

See the [`examples`](./examples) folder for usage as a python package.

### CLI

Once it is installed, you can use the `phomo` command.

It would go something like:

```sh
$ phomo master.png tile_directory -S 20 20 -o mosaic.png
```

If in doubt see the help:

<!-- help start -->

<!-- help end -->

## 🤩 Credit

- [photomosaic](https://pypi.org/project/photomosaic/) for the grid subdivision feature.

## ✔️ TODO

- [x] look into parallelizing/multithreading
- [ ] look into non greedy tile assignments
- [ ] palette matching
- [ ] documentation
- [ ] shell completion
- [ ] hex grid
