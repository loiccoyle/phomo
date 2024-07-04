import logging
from typing import Iterator, List, Tuple

import numpy as np
from PIL import Image

from .master import Master

LOGGER = logging.getLogger(__name__)


class Grid:
    def __init__(
        self,
        master: Master,
        mosaic_shape: Tuple[int, int],
        tile_shape: Tuple[int, int],
    ) -> None:
        """Mosaic tile grid.

        Args:
            master: Master image.
            mosaic_shape: mosaic image shape.
            tile_shape: size of the tiles.
        """
        self.master = master
        self.mosaic_shape = mosaic_shape
        self.tile_shape = tile_shape
        self.thresholds = []
        self.origin = (
            self.master.array.shape[0] % self.tile_shape[0] // 2,
            self.master.array.shape[1] % self.tile_shape[1] // 2,
        )
        self._slices = None
        self._arrays = None

    @property
    def arrays(self) -> List[np.ndarray]:
        """List of arrays containing the pixel values of all the slices."""
        if self._arrays is None:
            LOGGER.debug("Computing arrays.")
            self._arrays = [
                self.master.array[slices[0], slices[1]] for slices in self.slices
            ]
        return self._arrays

    @property
    def slices(self) -> List[Tuple[slice, slice]]:
        """Mosaic grid slices."""
        if self._slices is None:
            LOGGER.debug("Computing slices.")
            self._slices = list(self._compute_slices())
        return self._slices

    def _compute_slices(self) -> Iterator[Tuple[slice, slice]]:
        for x in range(
            self.origin[1], self.mosaic_shape[1] - self.origin[1], self.tile_shape[1]
        ):
            for y in range(
                self.origin[0],
                self.mosaic_shape[0] - self.origin[0],
                self.tile_shape[0],
            ):
                yield (
                    slice(y, y + self.tile_shape[0]),
                    slice(x, x + self.tile_shape[1]),
                )

    def remove_origin(self, slices: Tuple[slice, slice]) -> Tuple[slice, slice]:
        """Remove the origin of the mosaic from a slice tuple."""
        return (
            slice(slices[0].start - self.origin[0], slices[0].stop - self.origin[0]),
            slice(slices[1].start - self.origin[1], slices[1].stop - self.origin[1]),
        )

    @staticmethod
    def _subdivide(
        slices: Tuple[slice, slice],
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
        n_divisions = 0
        for i, slices in enumerate(self.slices):
            pixels = self.master.array[slices[0], slices[1]].reshape(-1, 3)
            contrast = np.mean(np.std(pixels / 255, axis=0))
            LOGGER.debug("Contrast slice %i: %f", i, contrast)
            if contrast > threshold:
                LOGGER.debug("Dividing slice %i", i)
                yield from self._subdivide(slices)
                n_divisions += 1
            else:
                yield slices
        LOGGER.info("Subdivided %i tiles.", n_divisions)

    def subdivide(self, threshold: float) -> None:
        """Subdivide grid based on contrast.

        Note:
            Modifies the grid in place.

        Args:
            threshold: contrast threshold at which to divide the slice into 4
                smaller slices. Between 0 and 1.
        """
        LOGGER.info("Subdividing with threshold %f", threshold)
        self._slices = list(self._yield_subdivide(threshold))
        # clear the cached arrays
        self._arrays = None
        self.thresholds.append(threshold)

    def plot(self, colour: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
        """Draw the tile edges on a copy of master image image.

        Args:
            colour: Colour of tile edges.

        Returns:
            Master image with the tile grid overlayed ontop.
        """
        annotated_image = self.master.array.copy()
        for y, x in self.slices:
            annotated_image[y, x.start] = colour
            annotated_image[y, x.stop - 1] = colour
            annotated_image[y.start, x] = colour
            annotated_image[y.stop - 1, x] = colour
        return Image.fromarray(annotated_image)

    def __len__(self) -> int:
        return len(self.slices)

    def __repr__(self) -> str:
        return f"""{self.__class__.__module__}.{self.__class__.__name__} at {hex(id(self))}:
    origin: {self.origin}
    len slices: {len(self.slices)}
    thresholds: {self.thresholds}"""
