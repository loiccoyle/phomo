from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_palette(
    bin_edges: np.ndarray, values: np.ndarray, log: bool = False
) -> Tuple[plt.Figure, plt.Axes]:

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


def plot_grid(
    image: np.ndarray,
    slices: List[Tuple[slice, slice]],
    colour: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Draw the tile edges on a copy of image. Make a dot at each tile center.

    This is a utility for inspecting a tile layout, not a necessary step in
    the mosaic-building process.

    Args:
        image: image array.
        slices: list of pairs of slices.
        colour: value to "draw" onto ``image`` at tile boundaries

    Returns:
        Image with tile borders highlitghted in ``colour``.
    """
    annotated_image = image.copy()
    for y, x in slices:
        annotated_image[y, x.start] = colour  # tile edges
        annotated_image[y, x.stop - 1] = colour  # tile edges
        annotated_image[y.start, x] = colour  # tile edges
        annotated_image[y.stop - 1, x] = colour  # tile edges
    return Image.fromarray(annotated_image)
