from typing import Protocol

import numpy as np

METRICS = {}


class MetricCallable(Protocol):  # type: ignore
    def __call__(
        self, master_chunk: np.ndarray, tile_arrays: np.ndarray
    ) -> np.ndarray: ...


def register_metric(func):
    METRICS[func.__name__] = func
    return func


@register_metric
def greyscale(
    master_chunk: np.ndarray, tile_arrays: np.ndarray, **kwargs
) -> np.ndarray:
    """Compute the greyscale distance.

    This metric ignores colours and compares greyscale values. Should provide better
    photomosaics when using few tiles images.


    Args:
        master_chunk: array containing the RGB pixels with values between 0 and 255.
        tile_arrays: array tile pixel arrays, values between 0 and 255.
        **kwargs: passed to ``np.linalg.norm``.

    Returns:
        Colour distance approximation between the master chunk and all the tiles
            arrays.
    """
    delta = np.subtract(
        master_chunk.sum(axis=-1), tile_arrays.sum(axis=-1), dtype=float
    )
    return np.linalg.norm(delta.reshape(delta.shape[0], -1), axis=-1, **kwargs)


@register_metric
def norm(master_chunk: np.ndarray, tile_arrays: np.ndarray, **kwargs) -> np.ndarray:
    """Distance metric using ``np.linalg.norm``.

    Quick distance metric in RGB space.

    Args:
        master_chunk: array containing the RGB pixels with values between 0 and 255.
        tile_arrays: list of tile pixel arrays, values between 0 and 255.
        **kwargs: passed to ``np.linalg.norm``.

    Returns:
        Colour distance approximation between the master chunk and all the tiles
            arrays.
    """
    return np.linalg.norm(
        np.subtract(master_chunk, tile_arrays, dtype=float).reshape(
            tile_arrays.shape[0], -1, tile_arrays.shape[-1]
        ),
        axis=(1, 2),
        **kwargs,
    )


# def norm(master_chunk: np.ndarray, tile_arrays: np.ndarray, **kwargs) -> float:
#     """`np.linalg.norm` distance metric.

#     Args:
#         master_chunk: array containing the RGB pixels with values between 0
#             and 255.
#         tile_arrays: array containing the RGB pixels with values between 0
#             and 255.

#     Returns:
#         Colour distance approximation.
#     """
#     return np.linalg.norm(
#         np.linalg.norm(
#             np.subtract(master_chunk, tile_arrays, dtype=float),
#             axis=-1,
#
#             **kwargs,
#         )
#     )


@register_metric
def luv_approx(
    master_chunk: np.ndarray, tile_arrays: np.ndarray, **kwargs
) -> np.ndarray:
    """Distance metric using a L*U*V space approximation.

    This metric should provide more accurate colour matching.

    Reference:
        https://www.compuphase.com/cmetric.htm

    Args:
        master_chunk: array containing the RGB pixels with values between and 255.
        tile_arrays: array containing the RGB pixels with values between 0 and 255.
        **kwargs: passed to ``np.linalg.norm``.

    Returns:
        Colour distance approximation between the master chunk and all the tiles
            arrays.
    """
    r = (master_chunk[:, :, 0] + tile_arrays[:, :, :, 0]) // 2
    r = r.astype(float)
    d = np.subtract(master_chunk, tile_arrays, dtype=float)
    return np.linalg.norm(
        (
            ((512 + r) * d[:, :, :, 0] ** 2)
            + 1024 * d[:, :, :, 1] ** 2
            + ((767 - r) * d[:, :, :, 2] ** 2)
        ).reshape(d.shape[0], -1),
        axis=-1,
        **kwargs,
    )
