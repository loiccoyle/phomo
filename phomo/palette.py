from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike


class Palette:
    """Colour palette methods."""

    array: np.ndarray

    def __init__(self, array: ArrayLike):
        self.array = np.array(array)
        self.plot = PalettePlotter(self)

    @property
    def pixels(self) -> np.ndarray:
        """Returns flattened pixels from the array."""
        return self.array.reshape(-1, self.array.shape[-1])

    def equalize(self):
        """Equalize the colour distribution using `cv2.equalizeHist`.

        Returns:
            A new `Palette` with equalized colour distribution.
        """
        # the array of the pool is (n_tiles, height, width, 3)
        # the array of the master is (height, width, 3)
        # so we flatten until the colour channels
        out_shape = self.array.shape
        array = self.array.reshape(-1, 3)
        matched_image = np.zeros_like(array)
        for i in range(3):  # Assuming 3 channels (RGB)
            matched_image[:, i] = cv2.equalizeHist(array[:, i]).squeeze()
        return self.__class__(matched_image.reshape(out_shape))

    def match(self, other: "Palette"):
        """Match the colour distribution of this object to the colour distribution of the
        `other` using the Reinhard colour transfer algorithm.

        See:
            https://api.semanticscholar.org/CorpusID:14088925

        Args:
            The other `Palette` to match this `Palette`'s colour distribution to.

        Returns:
            A new `Palette` with it's colour distribution matched the `other` `Palette`.
        """
        self_shape = self.array.shape
        self_array = self.array.reshape(-1, self.array.shape[1], 3)
        target_array = other.array.reshape(-1, other.array.shape[1], 3)

        source_lab = cv2.cvtColor(self_array, cv2.COLOR_RGB2LAB)
        target_lab = cv2.cvtColor(target_array, cv2.COLOR_RGB2LAB)

        # Compute the mean and standard deviation of each channel
        src_mean, src_std = cv2.meanStdDev(source_lab)
        tgt_mean, tgt_std = cv2.meanStdDev(target_lab)

        src_mean, src_std = src_mean.flatten(), src_std.flatten()
        tgt_mean, tgt_std = tgt_mean.flatten(), tgt_std.flatten()

        epsilon = 1e-5
        src_std = np.where(src_std < epsilon, epsilon, src_std)

        # Transfer color
        result_lab = source_lab.astype(float)
        for i in range(3):
            result_lab[:, :, i] -= src_mean[i]
            result_lab[:, :, i] = result_lab[:, :, i] * (tgt_std[i] / src_std[i])
            result_lab[:, :, i] += tgt_mean[i]

        # Clip values to valid range and convert back to uint8
        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        result_rgb = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)
        return self.__class__(result_rgb.reshape(self_shape))


class PalettePlotter:
    def __init__(self, palette: Palette):
        self._palette = palette

    def _colour_hist(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the 1D colour distributions.

        Args:
            **kwargs: passed to `numpy.histogram`.

        Returns:
            Histogram edges and counts.
        """
        bins = kwargs.pop("bins", range(256))
        values = []
        bin_edges = []
        for i in range(self._palette.pixels.shape[1]):
            freqs, edges = np.histogram(self._palette.pixels[:, i], bins=bins, **kwargs)
            bin_edges.append(edges)
            values.append(freqs)
        values = np.vstack(values).T
        bin_edges = np.vstack(bin_edges).T
        return bin_edges, values

    def _colour_hist_3d(
        self, bins: int = 256
    ) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        """Compute the 3D colour distribution."""
        hist, edges = np.histogramdd(
            self._palette.pixels,
            bins=bins,
            range=[(0, 255), (0, 255), (0, 255)],
        )
        return edges, hist

    def _colour_palette(self, depth: int = 3):
        pixels = self._palette.array.reshape(-1, 3)

        def split(pixels: np.ndarray, depth: int) -> list[np.ndarray]:
            if len(pixels) == 0 or depth == 0:
                return [pixels]

            ranges = np.ptp(pixels, axis=0)
            axis = np.argmax(ranges)
            median = np.median(pixels[:, axis])

            left = pixels[pixels[:, axis] <= median]
            right = pixels[pixels[:, axis] > median]

            return split(left, depth - 1) + split(right, depth - 1)

        quantized = split(pixels, depth)

        palette = [np.mean(region, axis=0) for region in quantized if len(region) > 0]
        palette = np.array(palette, dtype=np.uint8)

        return palette[::-1]

    def palette(self, depth: int = 3) -> Tuple[Figure, np.ndarray]:
        """Show the dominant colours of the palette using a median cut algorithm.

        See:
            https://en.wikipedia.org/wiki/Median_cut

        Args:
            depth: The number of splits to perform.

        Returns:
            `Figure` and `np.array` of `Axes`.
        """
        palette = self._colour_palette(depth=depth)

        square_size = 50
        palette_ar = np.zeros(
            (square_size, len(palette) * square_size, 3), dtype="uint8"
        )

        for i, color in enumerate(palette):
            palette_ar[:, i * square_size : (i + 1) * square_size, :] = color

        fig, ax = plt.subplots(
            1,
            figsize=(5, 5 * len(palette)),
            frameon=False,
        )
        ax.imshow(palette_ar, aspect="equal")
        ax.set_axis_off()
        ax.margins(0, 0)
        fig.tight_layout(pad=0)
        return fig, ax

    def distribution(self, log: bool = False) -> Tuple[Figure, np.ndarray]:
        """Plot the colour distribution of each channel.

        Args:
            log: Plot y axis in log scale.

        Returns:
            `Figure` and `np.array` of `Axes`.
        """

        bin_edges, values = self._colour_hist()
        fig, axs = plt.subplots(3, figsize=(12, 6))
        channels = ["Red", "Green", "Blue"]
        for i, (ax, channel) in enumerate(zip(axs, channels)):
            ax.bar(
                bin_edges[:-1, i],
                values[:, i],
                width=np.diff(bin_edges[:, i]),
                align="edge",
                color=channel,
            )
            if log:
                ax.set_yscale("log")
            ax.set_title(channel)
            ax.set_xlim(0, 255)
        fig.tight_layout()
        return fig, axs

    def distribution_2d(self) -> Tuple[Figure, np.ndarray]:
        """Plot 2D projections of the 3D colour distribution.

        Returns:
            `Figure` and `np.array` of `Axes`.
        """
        _, hist = self._colour_hist_3d()
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs = axs.ravel()

        titles = ["Red-Green", "Green-Blue", "Blue-Red"]
        for i, (ax, title) in enumerate(zip(axs, titles)):
            i = (i + 2) % 3
            proj = np.sum(hist, axis=i)
            if i != 1:
                proj = proj.T
            ax.imshow(
                proj,
                origin="lower",
                extent=[0, 255, 0, 255],
                aspect="auto",
                vmax=np.mean(proj) + 3 * np.std(proj),
            )
            ax.set_title(title)
            ax.set_xlabel(title.split("-")[0])
            ax.set_ylabel(title.split("-")[1])

        fig.tight_layout()
        return fig, axs

    def __call__(self):
        """Plot all the plots."""
        self.palette()
        self.distribution()
        self.distribution_2d()
