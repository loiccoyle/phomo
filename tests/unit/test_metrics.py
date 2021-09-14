from unittest import TestCase

import numpy as np

from phomo import metrics


class TestMetrics(TestCase):
    # TODO: these tests are not very good...
    def test_norm(self):
        a = np.ones((5, 4, 3))
        b = [np.ones((5, 4, 3))]
        assert metrics.norm(a, b) == 0

        a = np.ones((5, 4, 3))
        b = [np.zeros((5, 4, 3))]
        assert metrics.norm(a, b) > 0

    def test_greyscale(self):
        a = np.ones((5, 4, 3))
        b = [np.ones((5, 4, 3))]
        assert metrics.greyscale(a, b) == 0

        a = np.ones((5, 4, 3))
        b = [np.zeros((5, 4, 3))]
        assert metrics.greyscale(a, b) > 0

    def test_luv_approx(self):
        a = np.ones((5, 4, 3))
        b = [np.ones((5, 4, 3))]
        assert metrics.luv_approx(a, b) == 0

        a = np.ones((5, 4, 3))
        b = [np.zeros((5, 4, 3))]
        assert metrics.luv_approx(a, b) > 0

    def test_metrics(self):
        # make sure this dictionary gets populated
        assert len(metrics.METRICS) > 0
