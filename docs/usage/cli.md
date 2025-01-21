# Command Line Interface

## Basic Usage

To use the `phomo` CLI, open your terminal and type:

```bash
phomo <master-image-path> <tile-directory-path> [options]
```

### Positional Arguements

- `<master-image-path>`: Path to the master image we want to reconstruct as a photo mosaic.
- `<tile-directory-path>`: Directory containing the tile images. The images in this directory will be used to reconstruct the master image. The more images, the better the mosaic.

### Options

- `-h, --help`: Show the help message and exit.
- `-o OUTPUT, --output OUTPUT`: Specify the mosaic output path.
- `-c MASTER_CROP_RATIO, --master-crop-ratio MASTER_CROP_RATIO`: Crop the master image to width/height ratio.
- `-s MASTER_SIZE [MASTER_SIZE ...], --master-size MASTER_SIZE [MASTER_SIZE ...]`: Resize master image to width, height.
- `-C TILE_CROP_RATIO, --tile-crop-ratio TILE_CROP_RATIO`: Crop the tile images to width/height ratio.
- `-S TILE_SIZE [TILE_SIZE ...], --tile-size TILE_SIZE [TILE_SIZE ...]`: Resize tile images to width, height.
- `-n N_APPEARANCES, --n-appearances N_APPEARANCES`: The number of times a tile can appear in the mosaic.
- `-b, --black-and-white`: Convert master and tile images to black and white.
- `-g, --show-grid`: Show the tile grid, don't build the mosaic.
- `-d SUBDIVISIONS [SUBDIVISIONS ...], --subdivisions SUBDIVISIONS [SUBDIVISIONS ...]`: Grid subdivision thresholds.
- `-G, --gpu`: Use GPU for distance matrix computation. Requires installing with `pip install 'phomo[cuda]'`.
- `-m {greyscale,norm,luv_approx}, --metric {greyscale,norm,luv_approx}`: Distance metric.
- `-j WORKERS, --workers WORKERS`: Number of workers use to run when computing the distance matrix.
- `-e, --equalize`: Equalize the colour distributions to cover the full colour space.
- `--match-master-to-tiles`: Match the master image's colour distribution with the tile image colours.
- `--match-tiles-to-master`: Match the tile images' colour distribution with the master image colours.
- `--greedy`: Use a greedy tile assignment algorithm. Should improve performance at the expense of accuracy.
- `-v, --verbose`: Verbosity.

## Examples

### With 20x20 mosaic tiles and each tile appearing at most twice

```bash
phomo master.jpg tiles/ -o mosaic.jpg -S 20 20 -n 2
```

### Resize master image to 1080x1080 and use 10x10 tiles with the greyscale metric

```bash
phomo master.jpg tiles/ -o mosaic.jpg -s 1920 1080 -S 10 10 -m greyscale
```

### Subdivide tile regions with high contrast and run on the GPU

```bash
phomo master.jpg tiles/ -o mosaic.jpg -S 40 40 -G -d 0.1 0.1
```
