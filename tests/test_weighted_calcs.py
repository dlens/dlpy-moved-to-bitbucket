from unittest import TestCase

from dlpy.weighted_calcs import *
from numpy import testing as npt


class Test(TestCase):
    def test_wtd_mean(self):
        # Several example calculations
        vals = [1, 2, 3]
        wgts = [1, 1, 1]
        mean = wtd_mean(vals, wgts)
        npt.assert_almost_equal(mean, 2.0)
        npt.assert_almost_equal(wtd_mean([1, 2], [1, 1]), 1.5)
        npt.assert_almost_equal(
            wtd_mean([1, 2, 3, 4, 5], [5, 4, 3, 1, 1]),
            2.2142857142857144
        )
        npt.assert_almost_equal(
            wtd_mean([1, 2, 3], [5, 4, 1]),
            1.6
        )

    def test_lin_interp(self):
        npt.assert_almost_equal(
            lin_interp(5, 0, 0, 10, 10),
            5
        )
        npt.assert_almost_equal(
            lin_interp(5, 0, 0, 20, 10),
            2.5
        )
        npt.assert_almost_equal(
            lin_interp(0, 10, 20, 10, 30),
            25
        )

    def test_approx_equal(self):
        self.assertTrue(approx_equal(1.1, 1.10000000005, 1e-10))
        self.assertTrue(approx_equal(1.1, 1.10000000000000005))
        self.assertFalse(approx_equal(1.1, 1.10000000005, 1e-11))
        self.assertTrue(approx_equal(1.1, 1.09999999999, 1e-11))

    def test_wtd_median(self):
        npt.assert_almost_equal(wtd_median([1, 2, 3], [1,1,1]), 2.0)
        npt.assert_almost_equal(wtd_median ([1, 2], [1, 1]), 1.5)
        npt.assert_almost_equal(
            wtd_median([1, 2, 3, 4, 5], [5, 4, 3, 1, 1]),
            2.0
        )
        npt.assert_almost_equal(
            wtd_median([1, 2, 3], [5, 4, 1]),
            1.5
        )

    def test_wtd_median_w_zeroes(self):
        # Trying to add 0 weight items 3-21 and see what happens
        # Turns out weightedstats is taking the average of the last 0 item and the first non-zero.
        # That is not correct by my definition and visualizations
        vals = [1, 2, 22]
        wgts = [4, 4, 8]
        for i in range(3, 22):
            vals.append(i)
            wgts.append(0)
        npt.assert_almost_equal(
            wtd_median(vals, wgts),
            12.0
        )
