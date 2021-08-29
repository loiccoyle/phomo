import logging
from typing import List, Tuple, Iterator

from .mosaic import Master


class Grid:
    def __init__(
        self, master: Master, mosaic_shape: Tuple[int, int], tile_size: Tuple[int, ...]
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

    @property
    def shape(self) -> Tuple[int, int]:
        """The shape of mosaic grid.

        Returns:
            The number of tiles along the vertical axis and the horizontal axis.
        """
        if self._shape is None:
            self._log.debug("Computing shape.")
            self._shape = (
                self.master.array.shape[0] // self.tile_size[0],
                self.master.array.shape[1] // self.tile_size[1],
            )
        return self._shape

    @staticmethod
    def _subdivide(
        master_slice: Tuple[slice, slice]
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
        height = master_slice[0].stop - master_slice[0].start
        width = master_slice[1].stop - master_slice[1].start
        tile_dims = [(s.stop - s.start) // 2 for s in master_slice]
        return (
            (
                slice(master_slice[0].start, master_slice[0].start + tile_dims[0]),
                slice(master_slice[1].start, master_slice[1].start + tile_dims[1]),
            ),
            (
                slice(master_slice[0].start, master_slice[0].start + tile_dims[0]),
                slice(
                    master_slice[1].start + tile_dims[1],
                    master_slice[1].start + 2 * tile_dims[1] + width % 2,
                ),
            ),
            (
                slice(
                    master_slice[0].start + tile_dims[0],
                    master_slice[0].start + 2 * tile_dims[0] + height % 2,
                ),
                slice(master_slice[1].start, master_slice[1].start + tile_dims[1]),
            ),
            (
                slice(
                    master_slice[0].start + tile_dims[0],
                    master_slice[0].start + 2 * tile_dims[0] + height % 2,
                ),
                slice(
                    master_slice[1].start + tile_dims[1],
                    master_slice[1].start + 2 * tile_dims[1] + width % 2,
                ),
            ),
        )

    def __repr__(self) -> str:
        return f"""{self.__class__.__module__}.{self.__class__.__name__} at {hex(id(self))}:
    shape: {self.shape}
    len slices: {len(self.slices)}"""
