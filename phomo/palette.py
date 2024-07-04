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

    @property
    def pixels(self) -> np.ndarray:
        """Returns flattened pixels from the array."""
        return self.array.reshape(-1, self.array.shape[-1])

    def palette(self, bins: int = 256) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        """Compute the 3D colour distribution."""
        hist, edges = np.histogramdd(
            self.pixels,
            bins=bins,
            range=[(0, 255), (0, 255), (0, 255)],
        )
        return edges, hist

    def plot(self) -> Tuple[Figure, np.ndarray]:
        """Plot 2D projections of the 3D colour distribution."""
        _, hist = self.palette()
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
        """Match the colour distribution of the `Master` to the distribution of the
        `Pool` using the colour transfer algorithm explained in this paper:

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
