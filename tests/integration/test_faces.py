import os
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from phomo import Master, Mosaic, Pool

DATA_DIR = Path(__file__).parents[1] / "data"
FACES_TAR = DATA_DIR / "faces.tar.gz"
EXPECTED_MOSAIC = DATA_DIR / "mosaic.png"


def is_within_directory(directory, target):
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)

    prefix = os.path.commonprefix([abs_directory, abs_target])

    return prefix == abs_directory


def safe_extract(tar, path=Path("."), members=None, *, numeric_owner=False):
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not is_within_directory(path, member_path):
            raise Exception("Attempted Path Traversal in Tar File")

    tar.extractall(path, members, numeric_owner=numeric_owner)


@pytest.fixture
def faces_dir(tmp_path):
    with tarfile.open(FACES_TAR) as tar:
        safe_extract(tar, tmp_path)
    return tmp_path


@pytest.fixture
def master_file(faces_dir):
    return list((faces_dir).glob("*"))[1000]


def test__mosaic_faces(faces_dir, master_file):
    pool = Pool.from_dir(faces_dir, crop_ratio=1, tile_size=(20, 20))
    master = Master.from_file(master_file, crop_ratio=1, img_size=(200, 200))
    mosaic = Mosaic(master, pool)
    mosaic_img = mosaic.build(mosaic.d_matrix(workers=2))
    assert np.allclose(np.array(mosaic_img), np.array(Image.open(EXPECTED_MOSAIC)))


def test__mosaic_faces_cli(faces_dir, master_file, tmp_path):
    outfile = tmp_path / "mosaic.png"
    subprocess.run(
        f"phomo {str(master_file)} {str(faces_dir)} -c 1 -s 200 200 -C 1 -S 20 20 -o {str(outfile)}",
        check=True,
        shell=True,
    )
    assert (outfile).is_file()
    assert np.allclose(
        np.array(Image.open(outfile)), np.array(Image.open(EXPECTED_MOSAIC))
    )
