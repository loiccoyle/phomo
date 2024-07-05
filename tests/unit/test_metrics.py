import numpy as np

from phomo import metrics


def test_norm():
    a = np.ones((5, 4, 3))
    b = np.ones((1, 5, 4, 3))
    assert metrics.norm(a, b) == 0

    a = np.ones((5, 4, 3))
    b = np.zeros((1, 5, 4, 3))
    assert metrics.norm(a, b) > 0


def test_greyscale():
    a = np.ones((5, 4, 3))
    b = np.ones((1, 5, 4, 3))
    assert metrics.greyscale(a, b) == 0

    a = np.ones((5, 4, 3))
    b = np.zeros((1, 5, 4, 3))
    assert metrics.greyscale(a, b) > 0


def test_luv_approx():
    a = np.ones((5, 4, 3))
    b = np.ones((1, 5, 4, 3))
    assert metrics.luv_approx(a, b) == 0

    a = np.ones((5, 4, 3))
    b = np.zeros((1, 5, 4, 3))
    assert metrics.luv_approx(a, b) > 0


def test_metrics():
    # make sure this dictionary gets populated
    assert len(metrics.METRICS) > 0
