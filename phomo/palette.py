from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np



class Palette:
    """Colour palette methods."""

    pixels: np.ndarray

    def palette(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the colour distribution.

        Args:
            **kwargs: passed to `numpy.histogram`.

        Returns:
            Histogram edges and counts.
        """
        bins = kwargs.pop("bins", 256)
        values = []
        bin_edges = []
        for i in range(self.pixels.shape[1]):
            freqs, edges = np.histogram(self.pixels[:, i], bins=bins, **kwargs)
            bin_edges.append(edges)
            values.append(freqs)
        values = np.vstack(values).T
        bin_edges = np.vstack(bin_edges).T
        return bin_edges, values

    def cdfs(self):
        """Compute the cumulative distribution functions of the colours ditributions.

        Returns:
            Histogram edges and counts.
        """
        bins, frequencies = self.palette()
        return bins, self._cdfs(frequencies)

    @staticmethod
    def _cdfs(frequencies: np.ndarray) -> np.ndarray:
        cdfs = np.cumsum(frequencies, axis=0, dtype=float)
        cdfs = np.insert(cdfs, 0, 0, axis=0)
        cdfs /= cdfs[-1]
        return cdfs

    def plot(self, log: bool = False) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the colour distribution.

        Args:
            log: Plot y axis in log scale.

        Returns:
            Plot figure and axes.
        """

        bin_edges, values = self.palette()
        fig, axes = plt.subplots(3, figsize=(12, 6))
        for i, ax in enumerate(axes):
            ax.bar(
                bin_edges[:-1, i],
                values[:, i],
                width=np.diff(bin_edges[:, i]),
                align="edge",
            )
            if log:
                ax.set_yscale("log")
            ax.set_title(f"Channel {i+1}")
        fig.tight_layout()
        return fig, axes

    def _match_function(self, other: "Palette") -> Callable:
        self_bins, self_freqs = self.palette()
        # self_bins = self_bins[:-1]
        self_cdfs = self._cdfs(self_freqs)

        other_bins, other_freqs = other.palette()
        # other_bins = other_bins[:-1]
        other_cdfs = other._cdfs(other_freqs)

        def channel_map(arr, channel: int):
            """Rescale values in ``arr`` from this palette to another."""
            # Where in the old cdf did value(s) in arr fall?
            old_y = np.interp(arr, self_bins[:, channel], self_cdfs[:, channel])
            # Find the value at the corresponding position in the new cdf.
            new_x = np.interp(old_y, other_cdfs[:, channel], other_bins[:, channel])
            return new_x

        def map_function(image_array: np.ndarray) -> np.ndarray:
            # image_shape = image_array.shape
            # image_array = image_array.reshape(-1, image_array.shape[-1])

            new_image = np.empty_like(image_array)
            for channel in range(self.pixels.shape[-1]):
                new_image[:, :, channel] = channel_map(
                    image_array[:, :, channel], channel
                )
            return new_image  # .reshape(image_shape)

        return map_function
