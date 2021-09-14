import subprocess
import tarfile
from pathlib import Path
from random import sample
from shutil import rmtree
from unittest import TestCase

from phomo import Master, Mosaic, Pool

FACES_TAR = Path(__file__).parents[1] / "data" / "faces.tar.gz"


class TestFaces(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("test_faces")
        if not cls.test_dir.is_dir():
            cls.test_dir.mkdir()
        cls.data_dir = cls.test_dir / "faces"

        with tarfile.open(FACES_TAR) as tar:
            tar.extractall(cls.data_dir)
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
