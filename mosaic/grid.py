import logging
from typing import Iterator, List, Tuple

import numpy as np
from PIL import Image

from .master import Master
from .plot import plot_grid


class Grid:
    def __init__(
        self,
        master: Master,
        mosaic_shape: Tuple[int, int],
        tile_size: Tuple[int, int],
    ) -> None:
        """Mosaic tile grid.

        Args:
            master: Master image.
            mosaic_shape: mosaic image shape.
            tile_size: size of the tiles.
        """
        self.master = master
        self.mosaic_shape = mosaic_shape
        self.tile_size = tile_size
        self.thresholds = []
        self._log = logging.getLogger(__name__)
        self._slices = None
        self._shape = None

    @property
    def slices(self) -> List[Tuple[slice, slice]]:
        """Mosaic grid slices."""
        if self._slices is None:
            self._log.debug("Computing slices.")
            self._slices = list(self._compute_slices())
        return self._slices

    def _compute_slices(self) -> Iterator[Tuple[slice, slice]]:
        for x in range(0, self.mosaic_shape[1], self.tile_size[1]):
            for y in range(0, self.mosaic_shape[0], self.tile_size[0]):
                yield (slice(y, y + self.tile_size[0]), slice(x, x + self.tile_size[1]))

    @staticmethod
    def _subdivide(
        slices: Tuple[slice, slice]
    ) -> Tuple[
        Tuple[slice, slice],
        Tuple[slice, slice],
        Tuple[slice, slice],
        Tuple[slice, slice],
    ]:
        """Create four tiles from the four quadrants of the input tile.

        Examples:
            Subdivide a grid element.

            >>> mosaic.grid._subdivide((slice(0, 30, None), slice(0, 45, None)))
            ((slice(0, 15, None), slice(0, 22, None)),
             (slice(0, 15, None), slice(22, 45, None)),
             (slice(15, 30, None), slice(0, 22, None)),
             (slice(15, 30, None), slice(22, 45, None)))
        """
        height = slices[0].stop - slices[0].start
        width = slices[1].stop - slices[1].start
        tile_dims = [(s.stop - s.start) // 2 for s in slices]
        return (
            (
                slice(slices[0].start, slices[0].start + tile_dims[0]),
                slice(slices[1].start, slices[1].start + tile_dims[1]),
            ),
            (
                slice(slices[0].start, slices[0].start + tile_dims[0]),
                slice(
                    slices[1].start + tile_dims[1],
                    slices[1].start + 2 * tile_dims[1] + width % 2,
                ),
            ),
            (
                slice(
                    slices[0].start + tile_dims[0],
                    slices[0].start + 2 * tile_dims[0] + height % 2,
                ),
                slice(slices[1].start, slices[1].start + tile_dims[1]),
            ),
            (
                slice(
                    slices[0].start + tile_dims[0],
                    slices[0].start + 2 * tile_dims[0] + height % 2,
                ),
                slice(
                    slices[1].start + tile_dims[1],
                    slices[1].start + 2 * tile_dims[1] + width % 2,
                ),
            ),
        )

    def _yield_subdivide(self, threshold: float) -> Iterator[Tuple[slice, slice]]:
        for i, slices in enumerate(self.slices):
            pixels = self.master.array[slices[0], slices[1]].reshape(-1, 3)
            contrast = np.mean(np.std(pixels, axis=0))
            self._log.debug("Contrast slice %s: %s", i, contrast)
            if contrast > threshold:
                self._log.debug("Dividing slice %s", i)
                yield from self._subdivide(slices)
            else:
                yield slices

    def subdivide(self, threshold: float) -> None:
        """Subdivide grid based on contrast.

        Note:
            Modifies in place.

        Args:
            threshold: contrast threshold at which to divide the slice into 4
                smaller slices.
        """
        self._slices = list(self._yield_subdivide(threshold))
        self.thresholds.append(threshold)

    def __repr__(self) -> str:
        return f"""{self.__class__.__module__}.{self.__class__.__name__} at {hex(id(self))}:
    len slices: {len(self.slices)}
    thresholds: {self.thresholds}"""

    def plot(self, colour: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
        """Plot the grid layout."""
        return plot_grid(self.master.array, self.slices, colour)
