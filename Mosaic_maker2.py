#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
from tqdm import tqdm
from PIL import Image, ExifTags
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

def distance(c1, c2):
    return np.sqrt((c1[0]-c2[0])**2+(c1[1]-c2[1])**2+(c1[2]-c2[2])**2)

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

def pleb_load(pleb_dir,size):
    '''
    goes through the pleb_dir files and gets the avg colour stores in pleb_avg
    '''
    pleb_ar = []
    for im in tqdm(pleb_dir,desc="Cropping and loading plebs"):
        try:
            pleb = Image.open(im)
            image = crop_square(pleb)
            image = image.resize(size, Image.ANTIALIAS)
            if image.mode != 'RGB':
                image = image.convert(mode='RGB')
            pleb_ar.append([im,np.array(image)])
            pleb.close()
        except (IndexError,OSError,ValueError) as e:
            print('Error {} skipping {}'.format(e,im))
            pleb.close()
            continue
    return pleb_ar


def build_mosaic(mosaic_size, master, pleb_list, path=None, verbose=False):
    '''
    Builds a proper mosaic, much more time consuming
    '''
    w, h = mosaic_size[0], mosaic_size[1]
    mosaic = np.zeros((h,w,3))
    d_matrix = np.zeros((len(master),len(pleb_list)))
    pbar = tqdm(total=len(pleb_list), desc='Building distance matrix  ')
    for i,pleb in enumerate(pleb_list):
        pleb_ar = pleb[1]
        d_matrix[:,i] = [*map(lambda x: np.sqrt(np.sum(np.square(np.abs(pleb_ar - x[2]).mean(axis=(0,1))))), master)]
        pbar.update(1)
    pbar.close()

    d_temp = np.ma.array(d_matrix)
    pbar = tqdm(total=d_temp.shape[0], desc="Building mosaic           ")
    while d_temp[~d_temp.mask].size != 0 :
        min_ind = np.where(d_temp==np.min(d_temp[~d_temp.mask]))
        for ind in zip(min_ind[0],min_ind[1]):
            row = ind[0]
            col = ind[1]
            if d_temp.mask.shape != () and d_temp.mask[row,col]:
                continue
            if verbose:
                print('{}, row:{}, col:{}'.format(np.min(d_temp),row,col))
            x = master[row][0]
            y = master[row][1]
            array = pleb_list[col][1]
            mosaic[y:y+TILE_SIZE[1], x:x+TILE_SIZE[0],:] = array
            d_temp[:,col] = np.ma.masked
            d_temp[row,:] = np.ma.masked
            pbar.update(1)
    im = Image.fromarray(np.uint8(mosaic))
    if path != None:
        print('Saving mosaic at: {}'.format(path))
        im.save(path)
    pbar.close()
    return im

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("tile_dir", help="path of directory containing the tiles of the mosaic", type=str)
    parser.add_argument("master", help= "path of master image", type=str)
    parser.add_argument("output", help="path of output")
    parser.add_argument("-u", "--upscale", help="upscale coefficient master")
    parser.add_argument("-r", "--replacement", help="use tile replacement", action="store_true")
    parser.add_argument("-s", "--show", help="show mosaic after building", action="store_true")
    parser.add_argument("-v", "--verbose", help="verbosity", action="store_true")
    args = parser.parse_args()

    MASTER = args.master
    glob_string = args.tile_dir
    REPLACEMENT = args.replacement
    TILE_DIR = glob.glob(glob_string)
    output = args.output

    NUM_PLEB = len(TILE_DIR)

    master = open_exif(MASTER)
    if args.upscale != None:
        MOSAIC_REQUESTED_SIZE = [i*args.upscale for i in master.size]
    else:
        MOSAIC_REQUESTED_SIZE = [i for i in master.size]

    master = master.resize(MOSAIC_REQUESTED_SIZE,Image.BICUBIC)

    width_to_height = master.size[0]/master.size[1]

    NUM_GRID = [np.sqrt(NUM_PLEB*0.9) * i for i in [width_to_height, 1]]

    TILE_SIZE = np.ceil(np.divide(master.size,NUM_GRID))
    TILE_SIZE = TILE_SIZE.astype(int)
    MOSAIC_SIZE = master.size - master.size%TILE_SIZE

    if args.verbose:
        print("Mosaic grid {}".format(NUM_GRID))
        print('Master size: {}'.format(master.size))
        print('Number of Tiles: {}'.format(NUM_PLEB))
        print('Tile size: {}'.format(TILE_SIZE))
        print('Mosaic size: {}'.format(MOSAIC_SIZE))

    #%% Loads the cropped plebs in memory of the plebs and stores it
    tile_arrays = pleb_load(TILE_DIR,TILE_SIZE)

    #%% Splits the master into the sub regions
    master_arrays = []
    master_ar = np.array(master)
    for x in range(0,MOSAIC_SIZE[0],TILE_SIZE[0]):
        for y in range(0,MOSAIC_SIZE[1],TILE_SIZE[1]):
            master_arrays.append([x,y,master_ar[y:y+TILE_SIZE[0],x:x+TILE_SIZE[1],:]])

    mosaic = build_mosaic(MOSAIC_SIZE, master_arrays, tile_arrays)
    if args.show:
        mosaic.show()
    mosaic.save(output)
