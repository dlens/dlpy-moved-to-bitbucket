from unittest import TestCase

import numpy.testing as npt
import dlpy.maths as dlm
import numpy as np

class Test(TestCase):
    def test_linear_interp(self):
        npt.assert_almost_equal(dlm.linear_interp(0.5, 0, 0, 1, 1), 0.5)
        npt.assert_almost_equal(dlm.linear_interp(400, 100, 0, 500, 1), 0.75)
        npt.assert_almost_equal(dlm.linear_interp(0, 1, 0, 2, 1), -1.0)

    def test_decay_linear(self):
        value = dlm.decay_linear(100, 1, 1, -1, 0)
        npt.assert_almost_equal(value, 1/100)
        params = dlm.decay_linear(100, 1, 1, -1, 0, return_params=True)
        npt.assert_array_equal(params, (1,0,0))

    def test_decay_exponential(self):
        value = dlm.decay_exponential(2, 1, 1, -1, 0)
        npt.assert_almost_equal(value, 1/np.e)
        params = dlm.decay_exponential(100, 1, 1, -1, 0, return_params=True)
        npt.assert_array_equal(params, (np.e,-1,0))

    def test_pw_linear(self):
        pts = [(0, 0), (1, 2), (3, 4)]
        fx = lambda x: dlm.pw_linear(x, pts, 0, 6)
        fxExp = lambda x: dlm.pw_linear(x, pts, 0, 6, dlm.DecayType.EXPONENTIAL)
        npt.assert_almost_equal(fx(0), 0)
        npt.assert_almost_equal(fx(1), 2)
        npt.assert_almost_equal(fx(0.5), 1)
        npt.assert_almost_equal(fx(0.75), 1.5)
        npt.assert_almost_equal(fx(-2), 0)
        npt.assert_almost_equal(fx(2), 3)
        npt.assert_almost_equal(fx(3), 4)
        npt.assert_almost_equal(fx(4), 14/3)
        npt.assert_almost_equal(fx(1001), 5.996)
        npt.assert_almost_equal(fxExp(3.000001), 4, 6)
        npt.assert_almost_equal(fxExp(31), 6.0, 5)
        npt.assert_almost_equal(fxExp(-0.000001), 0, 5)
        npt.assert_almost_equal(fxExp(-30), 0)
        x=0
        (LHS,RHS) = dlm.pw_linear(x, pts, 0, 6, return_params=True)
        npt.assert_array_almost_equal(LHS, (0,0,0))
        npt.assert_array_almost_equal(RHS, (-2,2,6))

