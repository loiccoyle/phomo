# MosaicMaker

## Instalation

Requires python 3

In a terminal:

```
git clone https://github.com/loiccoyle/MosaicMaker
cd MosaicMaker
python setup.py install 
```

## Usage
Once it is installed, you should be able to run the `mm` executable.

Like so:

```
./mm path/to/folder/containing/images path/to/master/img path/to/output/image
```

Additional arguments:
```
-b: black and white
-u float: upscale output image factor (by default the output image will be the same resolution as the master image)
-v: verbose
-f: usage factor, ratio of how many of the tile images should be used
```
