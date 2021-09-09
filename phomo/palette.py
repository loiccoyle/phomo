from typing import Tuple

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
