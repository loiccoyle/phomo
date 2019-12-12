#!/usr/bin/env python
# Filename: setup.py
"""
The mosaic_maker setup script.
"""
from setuptools import setup

with open('requirements.txt') as fobj:
    REQUIREMENTS = [l.strip() for l in fobj.readlines()]

# try:
#     with open("README.md") as fh:
#         LONG_DESCRIPTION = fh.read()
# except UnicodeDecodeError:
#     LONG_DESCRIPTION = ""

setup(
    name='mosaic_maker',
    url='',
    description='',
    long_description='',
    author='Loic Coyle',
    author_email='loic.thomas.coyle@cern.ch',
    packages=['mosaic_maker'],
    include_package_data=True,
    platforms='any',
    install_requires=REQUIREMENTS,
    python_requires='>=3.6',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
    ],
)

__author__ = 'Loic Coyle'
