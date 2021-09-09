import functools
import sys

# prior to python 3.8 the protocol is in typing_extensions
if sys.version_info[0] == 3 and sys.version_info[1] < 8:
    from typing_extensions import Protocol
else:
    from typing import Protocol

import numpy as np

METRICS = {}


class MetricCallable(Protocol):  # type: ignore
    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        ...


def register_metric(func):
    METRICS[func.__name__] = func

    @functools.wraps(func)
    def metric(*args, **kwargs):
        return func(*args, **kwargs)

    return metric


@register_metric
def greyscale(img_a: np.ndarray, img_b: np.ndarray, *args, **kwargs) -> float:
    """Compute the greyscale distance.

    This metric ignores colours and compares greyscale values. Should provide better
    photomosaics when using few tiles images.

    Args:
        img_a: array containing the RGB pixels with values between 0 and 255.
        img_b: array containing the RGB pixels with values between 0 and 255.
        *args, **kwargs: passed to `np.linalg.norm`.

    Returns:
        Colour distance approximation.
    """
    return np.linalg.norm(
        np.subtract(img_a.mean(axis=-1), img_b.mean(axis=-1), dtype=float),
        *args,
        **kwargs,
    )


@register_metric
def norm(img_a: np.ndarray, img_b: np.ndarray, *args, **kwargs) -> float:
    """`np.linalg.norm` distance metric.

    Quick distance metric in RGB space.

    Args:
        img_a: array containing the RGB pixels with values between 0 and 255.
        img_b: array containing the RGB pixels with values between 0 and 255.
        *args, **kwargs: passed to `np.linalg.norm`.

    Returns:
        Colour distance approximation.
    """
    return np.linalg.norm(np.subtract(img_a, img_b, dtype=float), *args, **kwargs)


# def norm(img_a: np.ndarray, img_b: np.ndarray, *args, **kwargs) -> float:
#     """`np.linalg.norm` distance metric.

#     Args:
#         img_a: array containing the RGB pixels with values between 0
#             and 255.
#         img_b: array containing the RGB pixels with values between 0
#             and 255.

#     Returns:
#         Colour distance approximation.
#     """
#     return np.linalg.norm(
#         np.linalg.norm(
#             np.subtract(img_a, img_b, dtype=float),
#             axis=-1,
#             *args,
#             **kwargs,
#         )
#     )


@register_metric
def luv_approx(img_a: np.ndarray, img_b: np.ndarray, *args, **kwargs) -> float:
    """Distance metric using a L*U*V space approximation.

    This metric should provide more accurate colour matching.

    Reference:
        https://www.compuphase.com/cmetric.htm

    Args:
        img_a: array containing the RGB pixels with values between and 255.
        img_b: array containing the RGB pixels with values between and 255.
        *args, **kwargs: passed to `np.linalg.norm`.

    Returns:
        Colour distance approximation.
    """
    r = (img_a[:, :, 0] + img_b[:, :, 0]) // 2
    d = np.subtract(img_a, img_b, dtype=float)
    return np.linalg.norm(
        ((512 + r) * d[:, :, 0] ** 2)
        + 1024 * d[:, :, 1] ** 2
        + ((767 - r) * d[:, :, 2] ** 2),
        *args,
        **kwargs,
    )
