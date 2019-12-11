#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
from tqdm import tqdm
from PIL import Image, ExifTags
from multiprocessing import Pool
import numpy as np


def crop_square(image_to_crop):
    '''
    Crop image_to_crop to square by croping the largest dimension
    '''
    pleb_size = image_to_crop.size
    min_dim_i = np.argmin(pleb_size)
    min_dim = pleb_size[min_dim_i]

    if min_dim_i == 0:
        crop_box = (0, (pleb_size[1]-min_dim)/2, pleb_size[0], (pleb_size[1]+min_dim)/2)
    if min_dim_i == 1:
        crop_box = ((pleb_size[0]-min_dim)/2, 0, (pleb_size[0]+min_dim)/2 ,pleb_size[1])

    crop_image = image_to_crop.crop(crop_box)
    return crop_image

def open_exif(image_file):
    '''
    Opens image_file and takes into account the orientation tag to conserve the
    correct orientation
    '''
    img = Image.open(image_file)
    try :
        exif=dict((ExifTags.TAGS[k], v) for k, v in img._getexif().items() if k in ExifTags.TAGS)
        if   exif['Orientation'] == 3:
            img = img.rotate(180, expand=True)
        elif exif['Orientation'] == 6:
            img = img.rotate(270, expand=True)
        elif exif['Orientation'] == 8:
            img = img.rotate(90, expand=True)
        return img
    except AttributeError:
        return img

class Mosaic(object):
    def __init__(self, master_img, tile_dir, verbose=False, mode="RGB"):
        self.master_img = master_img
        self.tile_dir = tile_dir
        self.verbose = verbose
        self.mode = mode
        if self.verbose:
            print("Mosaic grid {}".format(self.grid))
            print('Master size: {}'.format(self.master_img.size))
            print('Number of Tiles: {}'.format(len(self.tiles)))
            print('Tile size: {}'.format(self.tile_size))
            print('Mosaic size: {}'.format(self.mosaic_size))
        self.master_arrays =self.get_master_arrays()

    @property
    def tile_dir(self):
        return self._tile_dir

    @tile_dir.setter
    def tile_dir(self, value):
        self._tile_dir = value
        self._tiles = self.find_tiles()
        self._grid = [np.sqrt(len(self.tiles)*0.9) * i for i in [self.width_to_height, 1]]
        self._tile_size = np.ceil(np.divide(self.master_img.size,self.grid)).astype(int)
        self._mosaic_size = self.master_img.size - self.master_img.size%self.tile_size

    @property
    def master_img(self):
        return self._master_img

    @master_img.setter
    def master_img(self, value):
        self._master_img = value
        self._width_to_height = self.master_img.size[0]/self.master_img.size[1]

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
        master_ar = np.array(self.master_img)
        for x in range(0,self.mosaic_size[0],self.tile_size[0]):
            for y in range(0,self.mosaic_size[1],self.tile_size[1]):
                if self.mode == "L":
                    master_arrays.append([x,y,master_ar[y:y+self.tile_size[0],x:x+self.tile_size[1]]])
                elif self.mode == "RGB":
                    master_arrays.append([x,y,master_ar[y:y+self.tile_size[0],x:x+self.tile_size[1],:]])
        return master_arrays

    def find_tiles(self):
        files = glob.glob(self.tile_dir+"/**/*")
        return files

    def load_square(self, im):

        try:
            tile = Image.open(im)
            image = crop_square(tile)
            image = image.resize(self.tile_size, Image.ANTIALIAS)
            if image.mode != self.mode:
                image = image.convert(mode=self.mode)
            # self.tile_arrays.append([im,np.array(image)])
            tile.close()
        except (IndexError,OSError,ValueError) as e:
            print('Error {} skipping {}'.format(e,im))
            tile.close()
            return
        return [im,np.array(image)]

    def tile_load(self):
        '''
        goes through the pleb_dir files and gets the avg colour stores in pleb_avg
        '''
        self.tile_arrays = []
        with Pool() as p:
            out = p.imap_unordered(self.load_square, self.tiles)
            self.tile_arrays = list(tqdm(out, total=len(self.tiles), desc='Cropping and loading plebs'))

        # for im in tqdm(self.tiles,desc="Cropping and loading plebs"):
        #     self.load_square(im)

    def calc_distance(self, tile, col):
        tile_ar = tile[1]
        self.d_matrix[:,col] = [*map(lambda x: np.sqrt(np.sum(np.square(np.abs(tile_ar - x[2]).mean(axis=(0,1))))), self.master_arrays)]
        return

    def build_mosaic(self):
        '''
        Builds a proper mosaic, much more time consuming
        '''
        w, h = self.mosaic_size[0], self.mosaic_size[1]
        if self.mode == "L":
            self.mosaic = np.zeros((h, w))
        elif self.mode == "RGB":
            self.mosaic = np.zeros((h, w, 3))
        self.d_matrix = np.zeros((len(self.master_arrays),len(self.tile_arrays)))

        for i, tile in tqdm(enumerate(self.tile_arrays), total=len(self.tile_arrays), desc='Building distance matrix  '):
            self.calc_distance(tile, i)

        d_temp = np.ma.array(self.d_matrix)
        pbar = tqdm(total=d_temp.shape[0], desc="Building mosaic           ")
        while d_temp[~d_temp.mask].size != 0 :
            min_ind = np.where(d_temp==np.min(d_temp[~d_temp.mask]))
            for ind in zip(min_ind[0],min_ind[1]):
                row = ind[0]
                col = ind[1]
                if d_temp.mask.shape != () and d_temp.mask[row,col]:
                    continue
                if self.verbose:
                    print('{}, row:{}, col:{}'.format(np.min(d_temp),row,col))
                x = self.master_arrays[row][0]
                y = self.master_arrays[row][1]
                array = self.tile_arrays[col][1]
                if self.mode == "L":
                    self.mosaic[y:y+self.tile_size[1], x:x+self.tile_size[0]] = array
                elif self.mode == "RGB":
                    self.mosaic[y:y+self.tile_size[1], x:x+self.tile_size[0],:] = array
                d_temp[:,col] = np.ma.masked
                d_temp[row,:] = np.ma.masked
                pbar.update(1)
        im = Image.fromarray(np.uint8(self.mosaic))
        pbar.close()
        self.mosaic = im
        return self.mosaic

#%%
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
