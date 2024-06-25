<p align="center"><img src="https://i.imgur.com/rMze8u5.png" width="1000"></p>
<p align="center"><b>Python package and CLI utility to create photo mosaics.</b></p>

<p align="center">
  <a href="https://github.com/loiccoyle/phomo/actions"><img src="https://github.com/loiccoyle/phomo/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/phomo/"><img src="https://img.shields.io/pypi/v/phomo"></a>
  <a href="./LICENSE.md"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <img src="https://img.shields.io/badge/platform-linux%20%7C%20macOS%20%7C%20windows-informational">
</p>

# Overview

`phomo` lets you quickly and easily create [photographic mosaics](https://en.wikipedia.org/wiki/Photographic_mosaic).

# Features

- **Simple:** CLI interface to create photo mosaics in seconds.
- **Configurable:** Python package to create custom mosaics with ease.
- **Fast:** GPU acceleration for large mosaics.

# Installation

`phomo` requires python 3.9 or later. It can be installed with or without GPU acceleration.

/// admonition | Gpu acceleration
For GPU acceleration you'll need a CUDA compatible GPU and the CUDA toolkit installed. See [numba docs](https://numba.readthedocs.io/en/stable/cuda/overview.html#requirements) for details.
///

## CLI

If you only need the command line interface, you can use [`pipx`](https://pypa.github.io/pipx/). It installs the `phomo` package in an isolated environment and makes it easy to uninstall.

```bash
pipx install 'phomo'
# or for gpu acceleration
pipx install 'phomo[cuda]'
```

## Python package

To install the Python package, use `pip`. It will install the CLI as well.
It is recommended to use a
[virtual environment](https://docs.python.org/3/tutorial/venv.html) to not mess with your system packages.

```bash
pip install 'phomo'
# or for gpu acceleration
pip install 'phomo[cuda]'
```
