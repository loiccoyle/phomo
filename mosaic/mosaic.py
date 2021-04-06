from tqdm import tqdm
from PIL import Image
from pathlib import Path
from multiprocessing import Pool
import numpy as np

from .utils import crop_square
from .utils import open_exif


class Mosaic(object):
    def __init__(
        self, master_img, tile_dir, verbose=False, mode="RGB", usage_factor=0.9
    ):
        self.master_img = master_img
        self.usage_factor = usage_factor
        self.tile_dir = Path(tile_dir)
        self.verbose = verbose
        self.mode = mode
        if self.verbose:
            print(f"Mosaic grid {self.grid}")
            print(f"Master size: {self.master_img.size}")
            print(f"Number of Tiles: {len(self.tiles)}")
            print(f"Tile size: {self.tile_size}")
            print(f"Mosaic size: {self.mosaic_size}")
        self.master_arrays = self.get_master_arrays()

    @property
    def tile_dir(self):
        return self._tile_dir

    @tile_dir.setter
    def tile_dir(self, value):
        self._tile_dir = value
        self._tiles = self.find_tiles()
        self._grid = [
            np.sqrt(len(self.tiles) * self.usage_factor) * i
            for i in [self.width_to_height, 1]
        ]
        self._tile_size = np.ceil(np.divide(self.master_img.size, self.grid)).astype(
            int
        )
        self._mosaic_size = self.master_img.size - self.master_img.size % self.tile_size

    @property
    def master_img(self):
        return self._master_img

    @master_img.setter
    def master_img(self, value):
        self._master_img = value
        self._width_to_height = self.master_img.size[0] / self.master_img.size[1]

    @property
    def tiles(self):
        return self._tiles

    @property
    def grid(self):
        return self._grid

    @property
    def tile_size(self):
        return self._tile_size

    @property
    def mosaic_size(self):
        return self._mosaic_size

    @property
    def width_to_height(self):
        return self._width_to_height

    def get_master_arrays(self):
        master_arrays = []
        master_ar = np.array(self.master_img).astype("int16")
        for x in range(0, self.mosaic_size[0], self.tile_size[0]):
            for y in range(0, self.mosaic_size[1], self.tile_size[1]):
                if self.mode == "L":
                    master_arrays.append(
                        [
                            x,
                            y,
                            master_ar[
                                y : y + self.tile_size[0], x : x + self.tile_size[1]
                            ],
                        ]
                    )
                elif self.mode == "RGB":
                    master_arrays.append(
                        [
                            x,
                            y,
                            master_ar[
                                y : y + self.tile_size[0], x : x + self.tile_size[1], :
                            ],
                        ]
                    )
        return master_arrays

    def find_tiles(self):
        files = list(self.tile_dir.glob("*"))
        files = [f for f in files if f.name not in [".DS_Store"]]
        return files

    def load_square(self, im):

        try:
            tile = open_exif(im)
            image = crop_square(tile)
            image = image.resize(self.tile_size, Image.ANTIALIAS)
            if image.mode != self.mode:
                image = image.convert(mode=self.mode)
            # self.tile_arrays.append([im,np.array(image)])
            # tile.close()
        except (IndexError, OSError, ValueError) as e:
            print("Error {} skipping {}".format(e, im))
            # tile.close()
            return
        tile.close()
        return [im, np.array(image).astype("int16")]

    def tile_load(self):
        """Goes through the pleb_dir files and gets the avg colour stores in
        pleb_avg.
        """
        self.tile_arrays = []
        with Pool() as p:
            out = p.imap_unordered(self.load_square, self.tiles)
            self.tile_arrays = list(
                tqdm(out, total=len(self.tiles), desc="Cropping and loading plebs")
            )

        # for im in tqdm(self.tiles,desc="Cropping and loading plebs"):
        #     self.load_square(im)

    def calc_distance(self, tile, col):
        tile_ar = tile[1]
        self.d_matrix[:, col] = [
            *map(
                lambda x: np.sqrt(
                    np.sum(np.square(np.abs(tile_ar - x[2]).mean(axis=(0, 1))))
                ),
                self.master_arrays,
            )
        ]
        return

    def build_mosaic(self):
        """
        Builds a proper mosaic, much more time consuming
        """
        w, h = self.mosaic_size[0], self.mosaic_size[1]
        print(w, h)
        if self.mode == "L":
            self.mosaic = np.zeros((h, w))
        elif self.mode == "RGB":
            self.mosaic = np.zeros((h, w, 3))
        self.d_matrix = np.zeros((len(self.master_arrays), len(self.tile_arrays)))
        print(self.d_matrix.shape)

        for i, tile in tqdm(
            enumerate(self.tile_arrays),
            total=len(self.tile_arrays),
            desc="Building distance matrix  ",
        ):
            self.calc_distance(tile, i)
            # print(tile)
        # print(self.d_matrix)

        d_temp = np.ma.array(self.d_matrix)
        pbar = tqdm(total=d_temp.shape[0], desc="Building mosaic           ")
        while d_temp[~d_temp.mask].size != 0:
            min_ind = np.where(d_temp == np.min(d_temp[~d_temp.mask]))
            for ind in zip(min_ind[0], min_ind[1]):
                row = ind[0]
                col = ind[1]
                if d_temp.mask.shape != () and d_temp.mask[row, col]:
                    continue
                if self.verbose:
                    print(f"{np.min(d_temp)}, row:{row}, col:{col}")
                x = self.master_arrays[row][0]
                y = self.master_arrays[row][1]
                array = self.tile_arrays[col][1]
                if self.mode == "L":
                    self.mosaic[
                        y : y + self.tile_size[1], x : x + self.tile_size[0]
                    ] = array
                elif self.mode == "RGB":
                    self.mosaic[
                        y : y + self.tile_size[1], x : x + self.tile_size[0], :
                    ] = array
                d_temp[:, col] = np.ma.masked
                d_temp[row, :] = np.ma.masked
                pbar.update(1)
        im = Image.fromarray(np.uint8(self.mosaic))
        pbar.close()
        self.mosaic = im
        return self.mosaic
