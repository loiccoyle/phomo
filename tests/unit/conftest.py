from pathlib import Path
import shutil
import pytest

from phomo import Pool, utils


TEST_PATH = Path("test_tiles/")


@pytest.fixture(scope="session")
def tile_dir():
    tile_dir = TEST_PATH / "rainbow"
    if not tile_dir.is_dir():
        tile_dir.mkdir(parents=True)
    channel_range = range(0, 255, 60)
    utils.rainbow_of_squares(
        tile_dir,
        size=(50, 50),
        r_range=channel_range,
        g_range=channel_range,
        b_range=channel_range,
    )
    yield tile_dir
    shutil.rmtree(TEST_PATH)


@pytest.fixture(scope="session")
def pool(tile_dir: Path):
    # create test pool
    return Pool.from_dir(tile_dir, tile_size=(10, 10))


@pytest.fixture(scope="session")
def pool_big_tiles(tile_dir: Path):
    # create test pool
    return Pool.from_dir(tile_dir, tile_size=(50, 50))
