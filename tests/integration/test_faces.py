import os
import subprocess
import tarfile
from pathlib import Path
from random import sample
from shutil import rmtree
from unittest import TestCase

from phomo import Master, Mosaic, Pool

FACES_TAR = Path(__file__).parents[1] / "data" / "faces.tar.gz"


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


class TestFaces(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("test_faces")
        if not cls.test_dir.is_dir():
            cls.test_dir.mkdir()
        cls.data_dir = cls.test_dir / "faces"

        with tarfile.open(FACES_TAR) as tar:
            safe_extract(tar, cls.data_dir)

        cls.master_file = sample(list((cls.data_dir).glob("*")), 1)[0]

    def test_mosaic(self):
        pool = Pool.from_dir(self.data_dir, crop_ratio=1, tile_size=(20, 20))
        master = Master.from_file(self.master_file, crop_ratio=1, img_size=(200, 200))
        mosaic = Mosaic(master, pool)
        mosaic_img = mosaic.build(workers=2)
        assert mosaic_img.size == mosaic.size

    def test_cli(self):
        subprocess.run(
            f"phomo {str(self.master_file)} {str(self.data_dir)} -c 1 -s 200 200 -C 1 -S 20 20 -b -o {str(self.test_dir / 'mosaic.jpg')}",
            check=True,
            shell=True,
        )
        assert (self.test_dir / "mosaic.jpg").is_file()

    @classmethod
    def tearDownClass(cls):
        if cls.test_dir.is_dir():
            rmtree(cls.test_dir)
