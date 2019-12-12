PKGNAME=mosaic_maker

default: build

all: install

install:
	pip install .

install-dev:
	pip install -e .

clean:
	python setup.py clean --all

