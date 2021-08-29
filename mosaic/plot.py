from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


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
